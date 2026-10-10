#include "QwenProvider.hpp"
#include "QwenRequestBudget.hpp"

#include "core/llm/QwenModelTraits.hpp"
#include "core/llm/ModelMetadata.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>

namespace core::llm::providers {
namespace {
std::optional<ModelInfo> cached_model_info(std::string_view catalog, std::string_view model) {
    const auto snapshot = ModelCatalogAvailability::instance().snapshot(catalog);
    auto fallback = ModelRegistry::instance().lookup(model);
    const int endpoint_window = qwen_endpoint_context_window(model);
    if (endpoint_window > 0) {
        auto scoped = fallback ? std::make_shared<ModelInfo>(*fallback) : std::make_shared<ModelInfo>();
        scoped->canonical_id = std::string(model);
        scoped->context_window = endpoint_window;
        fallback = std::move(scoped);
    }
    return resolve_model_metadata(model, snapshot.models, fallback).model;
}
bool thinking_disabled(std::string_view effort) {
    const auto normalized = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_view(effort));
    return normalized == "off" || normalized == "none" || normalized == "disabled";
}
} // namespace

QwenProvider::QwenProvider(Options options)
    : QwenProvider(std::move(options), std::make_shared<State>()) {}

QwenProvider::QwenProvider(Options options, std::shared_ptr<State> state)
    : HttpLLMProvider(options.base_url, options.credentials, options.model,
                      make_protocol(options, state), config::ApiType::DashScope,
                      options.provider_name, {}, options.model_selector,
                      options.service_id, options.transport)
    , options_(std::move(options)), state_(std::move(state)) {}

std::unique_ptr<protocols::ApiProtocolBase> QwenProvider::make_protocol(
    const Options& options, const std::shared_ptr<State>& state) {
    using namespace protocols;
    const std::string catalog = options.service_id.empty() ? options.provider_name : options.service_id;
    const DashScopeRequestLifecycle lifecycle{
        .prepare = [state, catalog, default_effort = options.default_effort](ChatRequest& request) {
            // The transport has already resolved the model, refreshed its
            // catalog, and normalized media. Reuse that snapshot; no second
            // discovery, transport hook, or generic request metadata is needed.
            const auto model = cached_model_info(catalog, request.model);
            std::size_t prompt_estimate;
            {
                std::lock_guard lock(state->mutex);
                request.max_tokens = state->requested_max_tokens;
                if (state->required_thinking_models.contains(request.model)
                    && thinking_disabled(request.effort.empty() ? default_effort : request.effort)) {
                    request.effort = "low";
                }
                if (state->model != request.model) {
                    state->last_context_estimate = 0;
                    state->last_prompt_tokens = 0;
                }
                state->model = request.model;
                state->context_estimate = QwenRequestBudget::estimate_history(request.messages);
                prompt_estimate = calibrated_estimate(*state, state->context_estimate);
            }
            QwenRequestBudget::prepare(request, model ? &*model : nullptr, prompt_estimate);
        },
        .failure = [state](int status, std::string_view message) {
            if (status != 400) return;
            std::lock_guard lock(state->mutex);
            state->status = status;
            state->error_message = message;
        },
        .prompt_usage = [state](int32_t tokens) {
            std::lock_guard lock(state->mutex);
            state->prompt_tokens = tokens;
        },
    };
    const std::string effort = options.default_effort.empty() && (options.token_plan || options.responses)
        ? "high" : options.default_effort;
    if (options.responses) {
        return std::make_unique<DashScopeResponsesProtocol>(DashScopeResponsesProtocol::Options{
            .default_effort = effort,
            .enable_hosted_tools = options.token_plan,
            .deployment = options.token_plan ? DashScopeDeployment::TokenPlan : DashScopeDeployment::Standard,
            .lifecycle = lifecycle,
        });
    }
    if (options.token_plan) {
        return std::make_unique<DashScopeTokenPlanProtocol>(DashScopeTokenPlanProtocol::Options{
            .thinking_budget = options.thinking_budget,
            .default_effort = effort,
            .lifecycle = lifecycle,
        });
    }
    return std::make_unique<DashScopeProtocol>(options.thinking_budget, effort,
                                             DashScopeDeployment::Standard, lifecycle);
}

void QwenProvider::stream_response(const ChatRequest& request,
                                  std::function<void(const StreamChunk&)> callback) {
    cancelled_.store(false, std::memory_order_release);
    ChatRequest effective = request;

    for (int attempt = 0; attempt < 2; ++attempt) {
        if (cancelled_.load(std::memory_order_acquire)) {
            callback(StreamChunk::make_final());
            return;
        }
        {
            std::lock_guard lock(state_->mutex);
            state_->status = 0;
            state_->error_message.clear();
            state_->prompt_tokens = 0;
            state_->requested_max_tokens = request.max_tokens;
        }
        std::optional<StreamChunk> error;
        bool emitted_output = false;
        HttpLLMProvider::stream_response(effective, [&](const StreamChunk& chunk) {
            if (chunk.reset_attempt) {
                {
                    std::lock_guard lock(state_->mutex);
                    state_->prompt_tokens = 0;
                    state_->status = 0;
                    state_->error_message.clear();
                }
                emitted_output = false;
                callback(chunk);
                return;
            }
            if (chunk.is_error && !emitted_output) {
                error = chunk;
                return;
            }
            if (error) return;
            if (chunk.is_final && !chunk.is_error) record_context_usage();
            emitted_output = emitted_output || !chunk.content.empty()
                || !chunk.reasoning_content.empty() || !chunk.tools.empty();
            callback(chunk);
        });
        if (!error) return;
        if (attempt == 0 && !cancelled_.load(std::memory_order_acquire) && repair_request(effective)) continue;
        callback(*error);
        return;
    }
}

bool QwenProvider::repair_request(ChatRequest& request) {
    std::string message;
    {
        std::lock_guard lock(state_->mutex);
        if (state_->status != 400) return false;
        message = core::utils::str::to_lower_ascii_copy(state_->error_message);
        request.model = state_->model;
    }
    if (thinking_disabled(request.effort.empty() ? options_.default_effort : request.effort)
        && !HttpLLMProvider::reasoning_capabilities(request.model).supports(ReasoningCapability::Required)
        && message.find("enable_thinking") != std::string::npos
        && (message.find("must be true") != std::string::npos
            || message.find("restricted to true") != std::string::npos)) {
        std::lock_guard lock(state_->mutex);
        state_->required_thinking_models.insert(request.model);
        request.effort = "low";
        return true;
    }
    if (request_has_image_input(request) || request_has_video_input(request)) {
        degrade_historical_media_inputs(request, {.images = true, .videos = true});
        return true;
    }
    return false;
}

void QwenProvider::record_context_usage() {
    std::lock_guard lock(state_->mutex);
    if (state_->prompt_tokens <= 0) return;
    state_->last_context_estimate = state_->context_estimate;
    state_->last_prompt_tokens = state_->prompt_tokens;
}

std::size_t QwenProvider::calibrated_estimate(const State& state, std::size_t estimate) noexcept {
    return state.last_prompt_tokens > 0 && estimate >= state.last_context_estimate
        ? std::max(estimate, static_cast<std::size_t>(state.last_prompt_tokens) + estimate - state.last_context_estimate)
        : estimate;
}

std::size_t QwenProvider::estimate_context_tokens(const std::vector<Message>& messages) const noexcept {
    std::lock_guard lock(state_->mutex);
    return calibrated_estimate(*state_, QwenRequestBudget::estimate_history(messages));
}

int QwenProvider::max_context_size() const noexcept {
    const auto model = options_.model.empty() ? get_last_model() : options_.model;
    const auto catalog = options_.service_id.empty() ? options_.provider_name : options_.service_id;
    if (const auto info = cached_model_info(catalog, model); info && info->context_window > 0) return info->context_window;
    const int fallback = qwen_endpoint_context_window(model);
    return fallback > 0 ? fallback : HttpLLMProvider::max_context_size();
}

ReasoningCapabilities QwenProvider::reasoning_capabilities(std::string_view model) const noexcept {
    const auto capabilities = HttpLLMProvider::reasoning_capabilities(model);
    std::lock_guard lock(state_->mutex);
    if (!state_->required_thinking_models.contains(std::string(model))) return capabilities;
    auto bits = capabilities.bits();
    bits &= ~static_cast<ReasoningCapabilities::Storage>(ReasoningCapability::Disable);
    bits |= static_cast<ReasoningCapabilities::Storage>(ReasoningCapability::Required);
    return ReasoningCapabilities::from_bits(bits);
}

void QwenProvider::cancel() {
    cancelled_.store(true, std::memory_order_release);
    HttpLLMProvider::cancel();
}

void QwenProvider::reset_conversation_state() {
    HttpLLMProvider::reset_conversation_state();
    std::lock_guard lock(state_->mutex);
    state_->last_context_estimate = 0;
    state_->last_prompt_tokens = 0;
}

std::shared_ptr<LLMProvider> QwenProvider::fork_for_parallel_request() const {
    auto fork = std::make_shared<QwenProvider>(options_);
    std::lock_guard lock(state_->mutex);
    fork->state_->required_thinking_models = state_->required_thinking_models;
    return fork;
}
} // namespace core::llm::providers
