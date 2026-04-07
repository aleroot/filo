#include "LlamaCppProvider.hpp"

#include "core/utils/JsonUtils.hpp"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace core::llm::providers {

namespace {

constexpr int kDefaultPredictTokens = 512;
constexpr int kPromptMarginTokens = 32;

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }

    std::string normalized;
    for (const unsigned char ch : std::string_view(raw)) {
        if (std::isspace(ch) || ch == '_' || ch == '-') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }

    return normalized == "1"
        || normalized == "true"
        || normalized == "yes"
        || normalized == "on";
}

void discard_llama_log(enum ggml_log_level, const char*, void*) {
    // Embedded llama.cpp logs are noisy inside the TUI. Suppress them by default.
}

void configure_llama_logging() {
    if (env_flag_enabled("FILO_LLAMACPP_STDIO_LOGS")) {
        return;
    }

    // Prevent common/chat helpers from queueing stdout/stderr log entries.
    common_log_set_verbosity_thold(-1);

    // Route ggml/llama logging away from stderr as well.
    llama_log_set(discard_llama_log, nullptr);
}

void initialize_llama_backend() {
    static std::once_flag once;
    std::call_once(once, []() {
        configure_llama_logging();
        ggml_backend_load_all();
        llama_backend_init();
    });
}

// Note: implemented as a free function so it can be called both from the
// anonymous-namespace helpers below and via LlamaCppProvider::build_tool_parameters_schema.
std::string build_tool_parameters_schema_impl(const core::llm::Tool& tool) {
    std::string schema;
    schema.reserve(512);
    schema += R"({"type":"object","properties":{)";

    for (std::size_t i = 0; i < tool.function.parameters.size(); ++i) {
        const auto& parameter = tool.function.parameters[i];
        schema += '"';
        schema += core::utils::escape_json_string(parameter.name);
        schema += R"(":{"type":")";
        schema += core::utils::escape_json_string(parameter.type);
        schema += R"(","description":")";
        schema += core::utils::escape_json_string(parameter.description);
        schema += R"("})";
        if (i + 1 < tool.function.parameters.size()) {
            schema += ',';
        }
    }

    schema += R"(},"required":[)";
    bool first_required = true;
    for (const auto& parameter : tool.function.parameters) {
        if (!parameter.required) {
            continue;
        }
        if (!first_required) {
            schema += ',';
        }
        first_required = false;
        schema += '"';
        schema += core::utils::escape_json_string(parameter.name);
        schema += '"';
    }
    schema += "]}";
    return schema;
}

common_chat_msg to_common_message(const core::llm::Message& message) {
    common_chat_msg converted;
    converted.role = message.role;
    converted.content = message_text_for_display(message);
    converted.tool_name = message.name;
    converted.tool_call_id = message.tool_call_id;

    converted.tool_calls.reserve(message.tool_calls.size());
    for (const auto& tool_call : message.tool_calls) {
        converted.tool_calls.push_back(common_chat_tool_call{
            .name = tool_call.function.name,
            .arguments = tool_call.function.arguments,
            .id = tool_call.id,
        });
    }

    return converted;
}

common_chat_tool to_common_tool(const core::llm::Tool& tool) {
    return common_chat_tool{
        .name = tool.function.name,
        .description = tool.function.description,
        .parameters = build_tool_parameters_schema_impl(tool),
    };
}

struct PreparedChat {
    common_chat_params chat_params;
    common_chat_parser_params parser_params;
};

PreparedChat prepare_chat(const core::llm::ChatRequest& request,
                          const llama_model* model,
                          const std::string& chat_template_override) {
    auto templates = common_chat_templates_init(model, chat_template_override);
    if (!templates) {
        throw std::runtime_error("failed to initialize llama.cpp chat templates");
    }

    common_chat_templates_inputs inputs;
    inputs.messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
        inputs.messages.push_back(to_common_message(message));
    }

    inputs.tools.reserve(request.tools.size());
    for (const auto& tool : request.tools) {
        inputs.tools.push_back(to_common_tool(tool));
    }

    inputs.tool_choice = request.tools.empty()
        ? COMMON_CHAT_TOOL_CHOICE_NONE
        : COMMON_CHAT_TOOL_CHOICE_AUTO;
    inputs.parallel_tool_calls = !request.tools.empty();
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    inputs.enable_thinking = false;

    PreparedChat prepared{
        .chat_params = common_chat_templates_apply(templates.get(), inputs),
        .parser_params = common_chat_parser_params{},
    };

    prepared.parser_params = common_chat_parser_params(prepared.chat_params);
    prepared.parser_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
    prepared.parser_params.parse_tool_calls = !request.tools.empty();
    if (!prepared.chat_params.parser.empty()) {
        prepared.parser_params.parser.load(prepared.chat_params.parser);
    }

    return prepared;
}

common_params_sampling make_sampling_params(const PreparedChat& prepared,
                                            const llama_vocab* vocab,
                                            bool has_tools,
                                            int top_k,
                                            float top_p,
                                            float temperature,
                                            int seed) {
    common_params_sampling sampling;
    sampling.seed = static_cast<uint32_t>(seed);
    sampling.top_k = top_k;
    sampling.top_p = top_p;
    sampling.temp = temperature;
    sampling.no_perf = true;
    if (!prepared.chat_params.grammar.empty()) {
        sampling.grammar = common_grammar{
            has_tools ? COMMON_GRAMMAR_TYPE_TOOL_CALLS : COMMON_GRAMMAR_TYPE_USER,
            prepared.chat_params.grammar,
        };
    }
    sampling.grammar_lazy = prepared.chat_params.grammar_lazy;
    sampling.grammar_triggers = prepared.chat_params.grammar_triggers;
    sampling.generation_prompt = prepared.chat_params.generation_prompt;

    for (const auto& preserved_token : prepared.chat_params.preserved_tokens) {
        const auto ids = common_tokenize(
            vocab,
            preserved_token,
            /* add_special = */ false,
            /* parse_special = */ true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids.front());
        }
    }

    return sampling;
}

std::string make_tool_call_id(int& next_tool_call_id) {
    return "call_llamacpp_" + std::to_string(++next_tool_call_id);
}

std::vector<core::llm::ToolCall> to_tool_deltas(const std::vector<common_chat_msg_diff>& diffs) {
    std::vector<core::llm::ToolCall> tools;
    for (const auto& diff : diffs) {
        if (diff.tool_call_index == std::string::npos) {
            continue;
        }
        tools.push_back(core::llm::ToolCall{
            .index = static_cast<int>(diff.tool_call_index),
            .id = diff.tool_call_delta.id,
            .type = "function",
            .function = {
                .name = diff.tool_call_delta.name,
                .arguments = diff.tool_call_delta.arguments,
            },
        });
    }
    return tools;
}

class StructuredOutputBridge {
public:
    explicit StructuredOutputBridge(common_chat_parser_params parser_params)
        : parser_params_(std::move(parser_params)) {}

    std::pair<std::string, std::vector<core::llm::ToolCall>> ingest(
        const std::string& text_added,
        bool is_final) {

        std::vector<common_chat_msg_diff> diffs;
        try {
            update_chat_message(text_added, !is_final, diffs, true);
        } catch (const std::exception&) {
            if (is_final) {
                throw;
            }
            return {};
        }

        std::string text_delta;
        for (const auto& diff : diffs) {
            if (!diff.content_delta.empty()) {
                text_delta += diff.content_delta;
            }
        }
        return {std::move(text_delta), to_tool_deltas(diffs)};
    }

private:
    void update_chat_message(const std::string& text_added,
                             bool is_partial,
                             std::vector<common_chat_msg_diff>& diffs,
                             bool filter_tool_calls) {
        generated_text_ += text_added;
        auto previous_message = chat_message_;
        auto new_message = common_chat_parse(generated_text_, is_partial, parser_params_);

        if (new_message.empty()) {
            return;
        }

        new_message.set_tool_call_ids(
            generated_tool_call_ids_,
            [&]() { return make_tool_call_id(next_tool_call_id_); });
        chat_message_ = new_message;

        auto all_diffs = common_chat_msg_diff::compute_diffs(previous_message, chat_message_);
        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
            return;
        }

        for (auto& diff : all_diffs) {
            for (std::size_t i = 0; i < chat_message_.tool_calls.size(); ++i) {
                if (sent_tool_call_names_.contains(i) || chat_message_.tool_calls[i].name.empty()) {
                    continue;
                }
                if (diff.tool_call_index != i || !diff.tool_call_delta.arguments.empty()) {
                    common_chat_msg_diff header;
                    header.tool_call_index = i;
                    header.tool_call_delta.id = chat_message_.tool_calls[i].id;
                    header.tool_call_delta.name = chat_message_.tool_calls[i].name;
                    diffs.push_back(std::move(header));
                    sent_tool_call_names_.insert(i);
                }
            }

            if (diff.tool_call_index == std::string::npos) {
                diffs.push_back(std::move(diff));
                continue;
            }

            const std::size_t tool_index = diff.tool_call_index;
            if (sent_tool_call_names_.contains(tool_index)) {
                if (!diff.tool_call_delta.arguments.empty()) {
                    diff.tool_call_delta.name.clear();
                    diff.tool_call_delta.id.clear();
                    diffs.push_back(std::move(diff));
                }
            } else if (!diff.tool_call_delta.arguments.empty() || !is_partial) {
                diff.tool_call_delta.name = chat_message_.tool_calls[tool_index].name;
                diff.tool_call_delta.id = chat_message_.tool_calls[tool_index].id;
                diffs.push_back(std::move(diff));
                sent_tool_call_names_.insert(tool_index);
            }
        }

        if (!is_partial) {
            for (std::size_t i = 0; i < chat_message_.tool_calls.size(); ++i) {
                if (sent_tool_call_names_.contains(i) || chat_message_.tool_calls[i].name.empty()) {
                    continue;
                }
                common_chat_msg_diff header;
                header.tool_call_index = i;
                header.tool_call_delta.id = chat_message_.tool_calls[i].id;
                header.tool_call_delta.name = chat_message_.tool_calls[i].name;
                diffs.push_back(std::move(header));
                sent_tool_call_names_.insert(i);
            }
        }
    }

    common_chat_parser_params parser_params_;
    std::string generated_text_;
    common_chat_msg chat_message_;
    std::vector<std::string> generated_tool_call_ids_;
    std::set<std::size_t> sent_tool_call_names_;
    int next_tool_call_id_ = 0;
};

bool trim_stop_sequence(std::string& output, const std::vector<std::string>& stop_sequences) {
    for (const auto& stop : stop_sequences) {
        if (stop.empty() || output.size() < stop.size()) {
            continue;
        }
        if (output.ends_with(stop)) {
            output.resize(output.size() - stop.size());
            return true;
        }
    }
    return false;
}

} // namespace

// ─── Public static helpers ───────────────────────────────────────────────────

std::string LlamaCppProvider::build_tool_parameters_schema(const core::llm::Tool& tool) {
    return build_tool_parameters_schema_impl(tool);
}

// ─── Lifecycle ───────────────────────────────────────────────────────────────

LlamaCppProvider::~LlamaCppProvider() {
    // Signal any in-flight inference thread to stop early.  The thread captured
    // shared_from_this() so the provider object (and its model / mutexes) remains
    // valid until the thread actually exits — we do not join here because the
    // detached thread will release its shared_ptr when it exits naturally.
    abort_requested_.store(true, std::memory_order_relaxed);
}

LlamaCppProvider::LlamaCppProvider(const core::config::ProviderConfig& config)
    : default_model_(config.model) {
    // Universal local-model settings (shared across all local engines)
    const core::config::LocalModelConfig local = config.local.value_or(core::config::LocalModelConfig{});
    model_path_             = local.model_path;
    chat_template_override_ = local.chat_template;
    temperature_            = local.temperature.value_or(0.2f);
    top_p_                  = local.top_p.value_or(0.95f);
    top_k_                  = local.top_k.value_or(40);
    seed_                   = local.seed.value_or(LLAMA_DEFAULT_SEED);

    // llama.cpp engine-specific settings
    const core::config::LlamaCppConfig ll = local.llamacpp.value_or(core::config::LlamaCppConfig{});
    context_size_  = ll.context_size.value_or(0);
    batch_size_    = ll.batch_size.value_or(0);
    threads_       = ll.threads.value_or(0);
    threads_batch_ = ll.threads_batch.value_or(0);
    gpu_layers_    = ll.gpu_layers.value_or(0);
    use_mmap_      = ll.use_mmap.value_or(true);
    use_mlock_     = ll.use_mlock.value_or(false);
}

void LlamaCppProvider::stream_response(
    const ChatRequest& request,
    std::function<void(const StreamChunk&)> callback) {

    ChatRequest req = request;
    if (req.model.empty()) {
        req.model = default_model_;
    }

    degrade_historical_image_inputs(req);
    if (latest_user_message_has_image_input(req)) {
        callback(StreamChunk::make_error(
            "\n[llama.cpp error: image input is not supported by the embedded llama.cpp provider]"));
        return;
    }

    // Capture a shared_ptr to ourselves so that the provider (and all its
    // members: inference_mutex_, state_mutex_, model_) stays alive for the
    // entire lifetime of the detached thread.  Without this, the provider
    // could be destroyed while the thread is still running, causing a crash.
    auto self = shared_from_this();
    std::thread([self, req, callback = std::move(callback)]() mutable {
        std::scoped_lock inference_lock(self->inference_mutex_);

        try {
            initialize_llama_backend();

            if (self->abort_requested_.load(std::memory_order_relaxed)) {
                callback(StreamChunk::make_final());
                return;
            }

            std::string load_error;
            if (!self->ensure_model_loaded(load_error)) {
                callback(StreamChunk::make_error("\n[llama.cpp error: " + load_error + "]"));
                return;
            }

            llama_model* model_raw = nullptr;
            {
                std::lock_guard lock(self->state_mutex_);
                model_raw = self->model_.get();
                self->last_model_ = self->effective_model_label(req);
            }

            const PreparedChat prepared = prepare_chat(req, model_raw, self->chat_template_override_);
            const std::string& prompt = prepared.chat_params.prompt;

            const llama_vocab* vocab = llama_model_get_vocab(model_raw);
            const int32_t required_prompt_tokens = -llama_tokenize(
                vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);

            if (required_prompt_tokens <= 0) {
                callback(StreamChunk::make_error("\n[llama.cpp error: failed to tokenize prompt]"));
                return;
            }

            std::vector<llama_token> prompt_tokens(required_prompt_tokens);
            if (llama_tokenize(
                    vocab,
                    prompt.c_str(),
                    static_cast<int32_t>(prompt.size()),
                    prompt_tokens.data(),
                    static_cast<int32_t>(prompt_tokens.size()),
                    true,
                    true) < 0) {
                callback(StreamChunk::make_error("\n[llama.cpp error: prompt tokenization failed]"));
                return;
            }

            const int n_predict = std::max(1, req.max_tokens.value_or(kDefaultPredictTokens));
            const int resolved_n_ctx = std::max(
                self->context_size_,
                required_prompt_tokens + n_predict + kPromptMarginTokens);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = static_cast<uint32_t>(resolved_n_ctx);
            ctx_params.n_batch = static_cast<uint32_t>(std::max<int32_t>(
                1,
                std::min(
                    resolved_n_ctx,
                    self->batch_size_ > 0
                        ? std::max(self->batch_size_, required_prompt_tokens)
                        : required_prompt_tokens)));
            ctx_params.no_perf = true;

            if (self->threads_ > 0) {
                ctx_params.n_threads = self->threads_;
            }
            if (self->threads_batch_ > 0) {
                ctx_params.n_threads_batch = self->threads_batch_;
            } else if (self->threads_ > 0) {
                ctx_params.n_threads_batch = self->threads_;
            }

            llama_context_ptr ctx(llama_init_from_model(model_raw, ctx_params));
            if (!ctx) {
                callback(StreamChunk::make_error("\n[llama.cpp error: failed to create inference context]"));
                return;
            }

            common_params_sampling sampling = make_sampling_params(
                prepared,
                vocab,
                !req.tools.empty(),
                self->top_k_,
                self->top_p_,
                req.temperature.value_or(self->temperature_),
                self->seed_);
            common_sampler_ptr sampler(common_sampler_init(model_raw, sampling));
            if (!sampler) {
                callback(StreamChunk::make_error("\n[llama.cpp error: failed to initialize sampler]"));
                return;
            }

            StructuredOutputBridge output_bridge(prepared.parser_params);

            llama_batch batch =
                llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));

            if (llama_decode(ctx.get(), batch) != 0) {
                callback(StreamChunk::make_error("\n[llama.cpp error: initial prompt evaluation failed]"));
                return;
            }

            int completion_tokens = 0;
            for (; completion_tokens < n_predict; ++completion_tokens) {
                // Check for early-exit request (e.g. provider is being destroyed).
                if (self->abort_requested_.load(std::memory_order_relaxed)) {
                    callback(StreamChunk::make_final());
                    return;
                }

                const llama_token token = common_sampler_sample(sampler.get(), ctx.get(), -1);
                if (llama_vocab_is_eog(vocab, token)) {
                    break;
                }
                common_sampler_accept(sampler.get(), token, true);

                char piece_buffer[256];
                const int piece_len = llama_token_to_piece(
                    vocab, token, piece_buffer, sizeof(piece_buffer), 0, true);

                if (piece_len < 0) {
                    callback(StreamChunk::make_error("\n[llama.cpp error: failed to decode output token]"));
                    return;
                }

                std::string output_piece(piece_buffer, piece_len);
                bool hit_stop = trim_stop_sequence(output_piece, prepared.chat_params.additional_stops);

                const auto [content_delta, tool_deltas] =
                    output_bridge.ingest(output_piece, hit_stop);
                if (!content_delta.empty() || !tool_deltas.empty()) {
                    StreamChunk chunk;
                    chunk.content = std::move(content_delta);
                    chunk.tools = std::move(tool_deltas);
                    callback(chunk);
                }
                if (hit_stop) {
                    break;
                }

                llama_token sampled_token = token;
                llama_batch next_batch = llama_batch_get_one(&sampled_token, 1);
                if (llama_decode(ctx.get(), next_batch) != 0) {
                    callback(StreamChunk::make_error("\n[llama.cpp error: token decode failed]"));
                    return;
                }
            }

            const auto [final_content_delta, final_tool_deltas] = output_bridge.ingest("", true);
            if (!final_content_delta.empty() || !final_tool_deltas.empty()) {
                StreamChunk chunk;
                chunk.content = std::move(final_content_delta);
                chunk.tools = std::move(final_tool_deltas);
                callback(chunk);
            }

            self->set_last_usage(required_prompt_tokens, completion_tokens);
            callback(StreamChunk::make_final());
        } catch (const std::exception& e) {
            callback(StreamChunk::make_error("\n[llama.cpp error: " + std::string(e.what()) + "]"));
        }
    }).detach();
}

std::string LlamaCppProvider::get_last_model() const {
    std::lock_guard lock(state_mutex_);
    return last_model_;
}

ProviderCapabilities LlamaCppProvider::capabilities() const {
    return ProviderCapabilities{
        .supports_tool_calls = true,
        .is_local = true,
    };
}

bool LlamaCppProvider::should_estimate_cost() const {
    return false;
}

bool LlamaCppProvider::ensure_model_loaded(std::string& error) {
    if (model_path_.empty()) {
        error = "provider is missing a GGUF model_path";
        return false;
    }

    std::lock_guard lock(state_mutex_);
    if (model_) {
        return true;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers_;
    model_params.use_mmap = use_mmap_;
    model_params.use_mlock = use_mlock_;

    auto* raw_model = llama_model_load_from_file(model_path_.c_str(), model_params);
    if (raw_model == nullptr) {
        error = "failed to load model from '" + model_path_ + "'";
        return false;
    }

    model_.reset(raw_model);
    return true;
}

std::string LlamaCppProvider::effective_model_label(const ChatRequest& request) const {
    if (!request.model.empty()) {
        return request.model;
    }
    if (!default_model_.empty()) {
        return default_model_;
    }
    return std::filesystem::path(model_path_).filename().string();
}

} // namespace core::llm::providers
