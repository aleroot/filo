#include "Prompter.hpp"

#include "core/agent/Agent.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/cli/TrustFlagResolver.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/logging/Logger.hpp"
#include "core/session/SessionData.hpp"
#include "core/session/SessionHandoff.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/tools/ApplyPatchTool.hpp"
#include "core/tools/AskUserQuestionTool.hpp"
#include "core/tools/CreateDirectoryTool.hpp"
#include "core/tools/DeleteFileTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/GetTimeTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/MoveFileTool.hpp"
#include "core/tools/ReadFileTool.hpp"
#include "core/tools/ReplaceTool.hpp"
#include "core/tools/SearchReplaceTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/ToolPolicy.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/utils/JsonWriter.hpp"
#include "core/utils/JsonUtils.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "core/tools/PythonInterpreterTool.hpp"
#include "core/tools/SkillLoader.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace exec::prompter {

namespace {

enum class OutputFormat {
    Text,
    Json,
    StreamJson,
};

enum class InputFormat {
    Text,
    StreamJson,
};

struct ToolNameStats {
    int calls = 0;
    int success = 0;
    int failed = 0;
};

struct ToolStats {
    int total_calls = 0;
    int success_calls = 0;
    int failed_calls = 0;
    std::map<std::string, ToolNameStats, std::less<>> by_name;
};

[[nodiscard]] std::string to_lower_copy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] std::string trim_copy(std::string_view input) {
    const auto start = input.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return std::string(input.substr(start, end - start + 1));
}

[[nodiscard]] std::optional<OutputFormat> parse_output_format(std::string_view value) {
    const std::string lowered = to_lower_copy(trim_copy(value));
    if (lowered.empty() || lowered == "text") {
        return OutputFormat::Text;
    }
    if (lowered == "json") {
        return OutputFormat::Json;
    }
    if (lowered == "stream-json" || lowered == "stream_json") {
        return OutputFormat::StreamJson;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<InputFormat> parse_input_format(std::string_view value) {
    const std::string lowered = to_lower_copy(trim_copy(value));
    if (lowered.empty() || lowered == "text") {
        return InputFormat::Text;
    }
    if (lowered == "stream-json" || lowered == "stream_json") {
        return InputFormat::StreamJson;
    }
    return std::nullopt;
}

[[nodiscard]] bool stdin_has_data() {
#if defined(_WIN32)
    return _isatty(_fileno(stdin)) == 0;
#else
    return isatty(STDIN_FILENO) == 0;
#endif
}

[[nodiscard]] std::string read_stdin_all(std::istream& in = std::cin) {
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

[[nodiscard]] std::string random_uuid_v4() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dis(0, 15);

    const auto hex = [](int value) -> char {
        static constexpr std::string_view kHex = "0123456789abcdef";
        return kHex[static_cast<std::size_t>(value & 0x0f)];
    };

    std::string out;
    out.reserve(36);
    for (int i = 0; i < 8; ++i) out.push_back(hex(dis(gen)));
    out.push_back('-');
    for (int i = 0; i < 4; ++i) out.push_back(hex(dis(gen)));
    out.push_back('-');

    // UUID v4 + RFC 4122 variant
    out.push_back('4');
    for (int i = 0; i < 3; ++i) out.push_back(hex(dis(gen)));
    out.push_back('-');

    out.push_back("89ab"[static_cast<std::size_t>(dis(gen) & 0x3)]);
    for (int i = 0; i < 3; ++i) out.push_back(hex(dis(gen)));
    out.push_back('-');

    for (int i = 0; i < 12; ++i) out.push_back(hex(dis(gen)));
    return out;
}

[[nodiscard]] std::string maybe_abbreviate_home(std::string path) {
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        const std::string_view home_sv{home};
        if (std::string_view(path).starts_with(home_sv)) {
            return "~" + path.substr(home_sv.size());
        }
    }
    return path;
}

[[nodiscard]] std::string normalize_newlines(std::string input) {
    std::string output;
    output.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '\r') {
            if (i + 1 < input.size() && input[i + 1] == '\n') {
                ++i;
            }
            output.push_back('\n');
            continue;
        }
        output.push_back(input[i]);
    }
    return output;
}

[[nodiscard]] std::string compose_prompt(const std::optional<std::string>& prompt,
                                         std::string_view stdin_data) {
    const std::string prompt_text = prompt.has_value() ? trim_copy(*prompt) : std::string{};
    const std::string piped_text = trim_copy(stdin_data);

    if (prompt_text.empty()) {
        return piped_text;
    }

    if (piped_text.empty()) {
        return prompt_text;
    }

    std::string combined;
    combined.reserve(prompt_text.size() + piped_text.size() + 64);
    combined += prompt_text;
    combined += "\n\n";
    combined += "Input from stdin:\n";
    combined += piped_text;
    return combined;
}

[[nodiscard]] std::filesystem::path normalize_path(std::filesystem::path path) {
    std::error_code ec;
    if (path.empty()) {
        return path;
    }

    if (!path.is_absolute()) {
        path = std::filesystem::absolute(path, ec);
        if (ec) {
            return path.lexically_normal();
        }
    }

    auto canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        return path.lexically_normal();
    }
    return canonical;
}

[[nodiscard]] bool same_directory(std::string_view lhs, const std::filesystem::path& rhs) {
    if (lhs.empty()) {
        return false;
    }
    const auto left_norm = normalize_path(std::filesystem::path(lhs));
    const auto right_norm = normalize_path(rhs);
    return left_norm == right_norm;
}

[[nodiscard]] std::optional<core::session::SessionData>
load_most_recent_for_project(const core::session::SessionStore& store,
                             const std::filesystem::path& cwd) {
    const auto infos = store.list();
    for (const auto& info : infos) {
        auto data_opt = store.load_by_id(info.session_id);
        if (!data_opt.has_value()) {
            continue;
        }
        if (same_directory(data_opt->working_dir, cwd)) {
            return data_opt;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<core::session::SessionData>
resolve_resume_request(const RunOptions& options,
                       const core::session::SessionStore& store,
                       const std::filesystem::path& cwd,
                       std::string& error) {
    error.clear();

    if (options.continue_last && options.resume_session.has_value()) {
        error = "--continue cannot be combined with --resume in prompter mode.";
        return std::nullopt;
    }

    if (options.continue_last) {
        auto resumed = load_most_recent_for_project(store, cwd);
        if (!resumed.has_value()) {
            error = "No resumable session found for the current project.";
        }
        return resumed;
    }

    if (!options.resume_session.has_value()) {
        return std::nullopt;
    }

    const auto& req = *options.resume_session;
    auto resumed = req.empty() ? store.load_most_recent() : store.load(req);
    if (!resumed.has_value()) {
        error = req.empty()
            ? "No resumable session found."
            : std::format("Could not find session '{}'", req);
    }
    return resumed;
}

void register_default_tools(core::tools::ToolManager& tool_manager) {
    tool_manager.register_tool(std::make_shared<core::tools::GetTimeTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ShellTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ApplyPatchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::FileSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ReadFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ReplaceTool>());
    tool_manager.register_tool(std::make_shared<core::tools::GrepSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::SearchReplaceTool>());
    tool_manager.register_tool(std::make_shared<core::tools::DeleteFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::MoveFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::CreateDirectoryTool>());

    auto ask_user_tool = std::make_shared<core::tools::AskUserQuestionTool>();
    ask_user_tool->setQuestionCallback([](core::tools::QuestionRequest request) {
        if (request.promise) {
            request.promise->set_value(std::nullopt);
        }
    });
    tool_manager.register_tool(ask_user_tool);

#ifdef FILO_ENABLE_PYTHON
    tool_manager.register_tool(std::make_shared<core::tools::PythonInterpreterTool>());
    core::tools::SkillLoader::discover_and_register(tool_manager);
#endif
}

[[nodiscard]] std::string format_double(double value, int precision = 6) {
    return std::format("{:.{}f}", value, precision);
}

[[nodiscard]] std::string build_stats_json(const core::session::SessionStats::Snapshot& snapshot,
                                           const core::llm::TokenUsage& totals,
                                           double cost_usd,
                                           const ToolStats& tool_stats) {
    core::utils::JsonWriter writer(2048);
    {
        auto _obj = writer.object();

        writer.key("models");
        {
            auto _models = writer.object();
            bool first_model = true;
            for (const auto& model : snapshot.per_model) {
                if (!first_model) {
                    writer.comma();
                }
                first_model = false;

                writer.key(model.model);
                {
                    auto _model = writer.object();
                    writer.kv_num("calls", model.call_count).comma();
                    writer.key("tokens");
                    {
                        auto _tokens = writer.object();
                        writer.kv_num("input", model.prompt_tokens).comma()
                              .kv_num("output", model.completion_tokens).comma()
                              .kv_num("total", model.prompt_tokens + model.completion_tokens);
                    }
                    writer.comma().key("cost_usd").raw(format_double(model.cost_usd));
                }
            }
        }

        writer.comma().key("usage");
        {
            auto _usage = writer.object();
            writer.kv_num("prompt_tokens", totals.prompt_tokens).comma()
                  .kv_num("completion_tokens", totals.completion_tokens).comma()
                  .kv_num("total_tokens", totals.total_tokens).comma()
                  .key("cost_usd").raw(format_double(cost_usd));
        }

        writer.comma().key("tools");
        {
            auto _tools = writer.object();
            writer.kv_num("total_calls", tool_stats.total_calls).comma()
                  .kv_num("success", tool_stats.success_calls).comma()
                  .kv_num("failed", tool_stats.failed_calls).comma()
                  .key("by_name");

            {
                auto _by_name = writer.object();
                bool first_name = true;
                for (const auto& [name, counts] : tool_stats.by_name) {
                    if (!first_name) {
                        writer.comma();
                    }
                    first_name = false;

                    writer.key(name);
                    {
                        auto _counts = writer.object();
                        writer.kv_num("calls", counts.calls).comma()
                              .kv_num("success", counts.success).comma()
                              .kv_num("failed", counts.failed);
                    }
                }
            }
        }
    }

    return std::move(writer).take();
}

[[nodiscard]] std::string make_session_start_event(std::string_view session_id,
                                                   std::string_view provider,
                                                   std::string_view model,
                                                   std::string_view mode,
                                                   bool resumed) {
    core::utils::JsonWriter writer(512);
    {
        auto _obj = writer.object();
        writer.kv_str("type", "system").comma()
              .kv_str("subtype", "session_start").comma()
              .kv_str("uuid", random_uuid_v4()).comma()
              .kv_str("session_id", session_id).comma()
              .kv_str("timestamp", core::session::SessionStore::now_iso8601()).comma()
              .kv_str("provider", provider).comma()
              .kv_str("model", model).comma()
              .kv_str("mode", mode).comma()
              .kv_bool("resumed", resumed);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string make_assistant_delta_event(std::string_view session_id,
                                                     std::string_view delta) {
    core::utils::JsonWriter writer(512 + delta.size());
    {
        auto _obj = writer.object();
        writer.kv_str("type", "assistant").comma()
              .kv_str("subtype", "content_delta").comma()
              .kv_str("uuid", random_uuid_v4()).comma()
              .kv_str("session_id", session_id).comma()
              .kv_str("timestamp", core::session::SessionStore::now_iso8601()).comma()
              .kv_str("delta", delta);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string make_tool_event(std::string_view session_id,
                                          std::string_view subtype,
                                          std::string_view tool_name,
                                          std::string_view payload,
                                          bool include_success,
                                          bool success = false) {
    core::utils::JsonWriter writer(768 + payload.size());
    {
        auto _obj = writer.object();
        writer.kv_str("type", "tool").comma()
              .kv_str("subtype", subtype).comma()
              .kv_str("uuid", random_uuid_v4()).comma()
              .kv_str("session_id", session_id).comma()
              .kv_str("timestamp", core::session::SessionStore::now_iso8601()).comma()
              .kv_str("name", tool_name);

        if (!payload.empty()) {
            writer.comma().kv_str("payload", payload);
        }

        if (include_success) {
            writer.comma().kv_bool("success", success);
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string make_assistant_message_event(std::string_view session_id,
                                                        std::string_view model,
                                                        std::string_view message) {
    core::utils::JsonWriter writer(1024 + message.size());
    {
        auto _obj = writer.object();
        writer.kv_str("type", "assistant").comma()
              .kv_str("uuid", random_uuid_v4()).comma()
              .kv_str("session_id", session_id).comma()
              .kv_str("timestamp", core::session::SessionStore::now_iso8601()).comma()
              .key("message");
        {
            auto _msg = writer.object();
            writer.kv_str("id", random_uuid_v4()).comma()
                  .kv_str("type", "message").comma()
                  .kv_str("role", "assistant").comma()
                  .kv_str("model", model).comma()
                  .key("content");
            {
                auto _arr = writer.array();
                {
                    auto _item = writer.object();
                    writer.kv_str("type", "text").comma()
                          .kv_str("text", message);
                }
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string make_result_event(std::string_view session_id,
                                            bool is_error,
                                            std::int64_t duration_ms,
                                            std::string_view result,
                                            std::string_view stats_json) {
    core::utils::JsonWriter writer(2048 + result.size() + stats_json.size());
    {
        auto _obj = writer.object();
        writer.kv_str("type", "result").comma()
              .kv_str("subtype", is_error ? "error" : "success").comma()
              .kv_str("uuid", random_uuid_v4()).comma()
              .kv_str("session_id", session_id).comma()
              .kv_str("timestamp", core::session::SessionStore::now_iso8601()).comma()
              .kv_bool("is_error", is_error).comma()
              .kv_num("duration_ms", duration_ms).comma()
              .kv_str("result", result).comma()
              .key("stats").raw(stats_json);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string join_json_array(const std::vector<std::string>& events) {
    std::string out;
    std::size_t total = 2;
    for (const auto& e : events) {
        total += e.size() + 1;
    }
    out.reserve(total);

    out.push_back('[');
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i > 0) {
            out.push_back(',');
        }
        out += events[i];
    }
    out.push_back(']');
    return out;
}

[[nodiscard]] bool ends_with_line_break(std::string_view text) {
    return !text.empty() && (text.back() == '\n' || text.back() == '\r');
}

[[nodiscard]] RuntimeContext build_runtime_context(std::string& error) {
    error.clear();

    auto& config_manager = core::config::ConfigManager::get_instance();
    const auto& config = config_manager.get_config();

    auto& provider_manager = core::llm::ProviderManager::get_instance();

    std::vector<std::string> provider_names;
    provider_names.reserve(config.providers.size());

    for (const auto& [name, pconfig] : config.providers) {
        if (auto provider = core::llm::ProviderFactory::create_provider(name, pconfig)) {
            provider_manager.register_provider(name, provider);
            provider_names.push_back(name);
        }
    }

    if (provider_names.empty()) {
        error = "No providers could be initialized from configuration.";
        return {};
    }

    std::ranges::sort(provider_names);

    std::string provider_name = config.default_provider;
    if (!std::ranges::contains(provider_names, provider_name)) {
        provider_name = provider_names.front();
    }

    std::shared_ptr<core::llm::LLMProvider> provider;
    try {
        provider = provider_manager.get_provider(provider_name);
    } catch (const std::exception& ex) {
        error = std::format("Failed to initialize provider '{}': {}", provider_name, ex.what());
        return {};
    }

    std::string model_name;
    if (const auto it = config.providers.find(provider_name);
        it != config.providers.end()) {
        model_name = it->second.model;
    }

    RuntimeContext runtime;
    runtime.provider = std::move(provider);
    runtime.provider_name = provider_name;
    runtime.model_name = model_name;
    runtime.default_mode = config.default_mode.empty() ? "BUILD" : config.default_mode;
    return runtime;
}

} // namespace

bool should_run(bool explicit_prompter,
                bool prompt_provided,
                bool has_stdin_data,
                bool output_format_provided,
                bool input_format_provided,
                bool include_partial_messages,
                bool continue_last) {
    return explicit_prompter
        || prompt_provided
        || has_stdin_data
        || output_format_provided
        || input_format_provided
        || include_partial_messages
        || continue_last;
}

RunDiagnostics run_for_test(const RunOptions& options,
                            const RuntimeContext& runtime,
                            const StreamInput& input,
                            std::ostream& out,
                            std::ostream& err,
                            std::optional<std::filesystem::path> session_dir_override) {
    RunDiagnostics diagnostics;

    if (!runtime.provider) {
        diagnostics.exit_code = 1;
        err << "Prompter mode failed: no provider is available.\n";
        return diagnostics;
    }

    const auto output_format_opt = parse_output_format(options.output_format);
    if (!output_format_opt.has_value()) {
        diagnostics.exit_code = 2;
        err << "Invalid --output-format. Supported values: text, json, stream-json.\n";
        return diagnostics;
    }

    const auto input_format_opt = parse_input_format(options.input_format);
    if (!input_format_opt.has_value()) {
        diagnostics.exit_code = 2;
        err << "Invalid --input-format. Supported values: text, stream-json.\n";
        return diagnostics;
    }

    const auto output_format = *output_format_opt;
    const auto input_format = *input_format_opt;

    if (options.include_partial_messages && output_format != OutputFormat::StreamJson) {
        diagnostics.exit_code = 2;
        err << "--include-partial-messages requires --output-format stream-json.\n";
        return diagnostics;
    }

    if (input_format == InputFormat::StreamJson) {
        diagnostics.exit_code = 2;
        err << "--input-format stream-json is not implemented yet in prompter mode.\n";
        return diagnostics;
    }

    std::string stdin_data = input.has_stdin_data
        ? normalize_newlines(input.stdin_data)
        : std::string{};

    diagnostics.final_prompt = compose_prompt(options.prompt, stdin_data);
    if (trim_copy(diagnostics.final_prompt).empty()) {
        diagnostics.exit_code = 2;
        err << "Prompter mode requires a prompt via --prompt/-p or stdin input.\n";
        return diagnostics;
    }

    auto& tool_manager = core::tools::ToolManager::get_instance();
    register_default_tools(tool_manager);

    auto agent_session_context = core::context::make_session_context(
        core::workspace::Workspace::get_instance().snapshot(),
        core::context::SessionTransport::cli);
    auto agent = std::make_shared<core::agent::Agent>(
        runtime.provider,
        tool_manager,
        agent_session_context);
    if (!runtime.default_mode.empty()) {
        agent->set_mode(runtime.default_mode);
    }
    if (!runtime.model_name.empty()) {
        agent->set_active_model(runtime.model_name);
    }

    const auto trust_resolution =
        core::cli::resolve_trust_flags(options.yolo, options.trusted_tools);

    if (trust_resolution.trust_all_tools) {
        agent->set_permission_profile(core::agent::PermissionProfile::Autonomous);
    } else if (!trust_resolution.trusted_tool_names.empty()) {
        std::unordered_set<std::string> trusted_sensitive_tools{
            trust_resolution.trusted_tool_names.begin(),
            trust_resolution.trusted_tool_names.end(),
        };
        agent->set_permission_profile(core::agent::PermissionProfile::Interactive);
        agent->set_permission_fn(
            [trusted_tools = std::move(trusted_sensitive_tools)](
                std::string_view tool_name,
                std::string_view /*tool_args*/) {
                const auto canonical =
                    core::tools::policy::canonical_tool_name(tool_name);
                return trusted_tools.contains(canonical);
            });
    }

    core::budget::BudgetTracker::get_instance().reset_session();
    core::session::SessionStats::get_instance().reset();

    const auto session_dir = session_dir_override.value_or(
        core::session::SessionStore::default_sessions_dir());
    core::session::SessionStore store(session_dir);

    std::string session_id = core::session::SessionStore::generate_id();
    std::string created_at = core::session::SessionStore::now_iso8601();
    std::vector<core::session::SessionTodoItem> session_todos;

    std::optional<core::session::SessionData> resumed_session;
    {
        std::string resume_error;
        resumed_session = resolve_resume_request(
            options,
            store,
            std::filesystem::current_path(),
            resume_error);

        if ((options.continue_last || options.resume_session.has_value())
            && !resumed_session.has_value()) {
            diagnostics.exit_code = 1;
            err << "Prompter mode: " << resume_error << "\n";
            return diagnostics;
        }
    }

    if (resumed_session.has_value()) {
        diagnostics.used_resumed_session = true;
        diagnostics.resumed_session_id = resumed_session->session_id;
        if (options.continue_last) {
            session_todos = resumed_session->todos;
            agent->load_history(
                {},
                core::session::build_handoff_summary(*resumed_session),
                resumed_session->mode);
        } else {
            session_id = resumed_session->session_id;
            created_at = resumed_session->created_at;
            session_todos = resumed_session->todos;
            agent->load_history(
                resumed_session->messages,
                resumed_session->context_summary,
                resumed_session->mode);
        }
    }

    // Keep one stable external session id for the entire prompter run even if
    // the internal persisted session rotates between tool-loop steps.
    const std::string emitted_session_id = session_id;

    agent->set_efficiency_decision_fn(
        [agent, &store, &session_id, &created_at, &runtime, &session_todos](
            const core::session::SessionEfficiencyDecision&) {
            auto snap_messages = agent->get_history();
            auto snap_mode = agent->get_mode();
            auto snap_context = agent->get_context_summary();

            core::session::SessionData archived;
            archived.session_id = session_id;
            archived.created_at = created_at;
            archived.last_active_at = core::session::SessionStore::now_iso8601();
            archived.working_dir = std::filesystem::current_path().string();
            archived.provider = runtime.provider_name;
            archived.model = runtime.model_name;
            archived.mode = snap_mode;
            archived.context_summary = snap_context;
            archived.messages = snap_messages;
            archived.todos = session_todos;

            const auto& budget = core::budget::BudgetTracker::get_instance();
            const auto total = budget.session_total();
            archived.stats.prompt_tokens = total.prompt_tokens;
            archived.stats.completion_tokens = total.completion_tokens;
            archived.stats.cost_usd = budget.session_cost_usd();
            const auto stats_snapshot = core::session::SessionStats::get_instance().snapshot();
            archived.stats.turn_count = stats_snapshot.turn_count;
            archived.stats.tool_calls_total = stats_snapshot.tool_calls_total;
            archived.stats.tool_calls_success = stats_snapshot.tool_calls_success;
            archived.handoff_summary = core::session::build_handoff_summary(archived);

            std::string save_error;
            if (!store.save(archived, &save_error)) {
                core::logging::warn(
                    "Skipping prompter session rotation for {} because archival save failed: {}",
                    archived.session_id,
                    save_error);
                return;
            }

            agent->compact_history(archived.handoff_summary);
            core::budget::BudgetTracker::get_instance().reset_session();
            core::session::SessionStats::get_instance().reset();

            session_id = core::session::SessionStore::generate_id();
            created_at = core::session::SessionStore::now_iso8601();
        });

    std::mutex emit_mutex;
    std::vector<std::string> buffered_events;

    auto emit_text = [&](std::string_view text) {
        std::lock_guard lock(emit_mutex);
        out << text;
        out.flush();
        diagnostics.rendered_output.append(text);
    };

    auto emit_line = [&](std::string_view line) {
        std::lock_guard lock(emit_mutex);
        out << line << '\n';
        out.flush();
        diagnostics.rendered_output.append(line);
        diagnostics.rendered_output.push_back('\n');
    };

    auto emit_event = [&](std::string event_json) {
        if (output_format == OutputFormat::StreamJson) {
            emit_line(event_json);
        } else if (output_format == OutputFormat::Json) {
            std::lock_guard lock(emit_mutex);
            buffered_events.push_back(std::move(event_json));
        }
    };

    if (output_format != OutputFormat::Text) {
        const std::string mode_name = agent->get_mode();
        emit_event(make_session_start_event(
            emitted_session_id,
            runtime.provider_name,
            runtime.model_name,
            mode_name,
            diagnostics.used_resumed_session));
    }

    const auto history_before = agent->get_history().size();

    ToolStats tool_stats;
    std::mutex state_mutex;
    std::string streamed_text;

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    const auto start_time = std::chrono::steady_clock::now();

    agent->send_message(
        diagnostics.final_prompt,
        [&](const std::string& chunk) {
            {
                std::lock_guard lock(state_mutex);
                streamed_text += chunk;
            }

            if (output_format == OutputFormat::Text) {
                emit_text(chunk);
            } else if (output_format == OutputFormat::StreamJson
                       && options.include_partial_messages
                       && !chunk.empty()) {
                emit_event(make_assistant_delta_event(emitted_session_id, chunk));
            }
        },
        [&](const std::string& tool_name, const std::string& tool_payload) {
            {
                std::lock_guard lock(state_mutex);
                auto& stats = tool_stats.by_name[tool_name];
                stats.calls += 1;
                tool_stats.total_calls += 1;
            }

            if (output_format == OutputFormat::StreamJson) {
                emit_event(make_tool_event(
                    emitted_session_id,
                    "start",
                    tool_name,
                    tool_payload,
                    false));
            }
        },
        [&]() {
            std::lock_guard lock(done_mutex);
            done = true;
            done_cv.notify_all();
        },
        core::agent::Agent::TurnCallbacks{
            .on_step_begin = {},
            .on_tool_start = {},
            .on_tool_finish = [&](const core::llm::ToolCall& tool_call,
                                  const core::llm::Message& result) {
                const bool success = result.content.find("\"error\"") == std::string::npos;

                {
                    std::lock_guard lock(state_mutex);
                    auto& stats = tool_stats.by_name[tool_call.function.name];
                    if (success) {
                        stats.success += 1;
                        tool_stats.success_calls += 1;
                    } else {
                        stats.failed += 1;
                        tool_stats.failed_calls += 1;
                    }
                }

                if (output_format == OutputFormat::StreamJson) {
                    emit_event(make_tool_event(
                        emitted_session_id,
                        "finish",
                        tool_call.function.name,
                        result.content,
                        true,
                        success));
                }
            },
        });

    {
        std::unique_lock lock(done_mutex);
        constexpr auto kRunTimeout = std::chrono::minutes(30);
        if (!done_cv.wait_for(lock, kRunTimeout, [&] { return done; })) {
            diagnostics.exit_code = 1;
            err << "Prompter mode timed out while waiting for the agent response.\n";
            return diagnostics;
        }
    }

    const auto finished_at = std::chrono::steady_clock::now();
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        finished_at - start_time).count();

    if (output_format == OutputFormat::Text
        && !diagnostics.rendered_output.empty()
        && !ends_with_line_break(diagnostics.rendered_output)) {
        emit_text("\n");
    }

    const auto history_after = agent->get_history();
    if (history_after.size() >= history_before) {
        for (const auto& msg : std::span<const core::llm::Message>(
                 history_after.data() + history_before,
                 history_after.size() - history_before)) {
            if (msg.role == "assistant" && !msg.content.empty()) {
                diagnostics.final_text_response += msg.content;
            }
        }
    }

    if (diagnostics.final_text_response.empty()) {
        std::lock_guard lock(state_mutex);
        diagnostics.final_text_response = streamed_text;
    }

    const auto totals = core::budget::BudgetTracker::get_instance().session_total();
    const auto cost_usd = core::budget::BudgetTracker::get_instance().session_cost_usd();
    const auto snapshot = core::session::SessionStats::get_instance().snapshot();
    const auto stats_json = build_stats_json(snapshot, totals, cost_usd, tool_stats);
    const bool request_failed = snapshot.api_calls_total > snapshot.api_calls_success;

    if (output_format != OutputFormat::Text) {
        emit_event(make_assistant_message_event(
            emitted_session_id,
            runtime.model_name,
            diagnostics.final_text_response));

        emit_event(make_result_event(
            emitted_session_id,
            request_failed,
            duration_ms,
            diagnostics.final_text_response,
            stats_json));
    }

    if (output_format == OutputFormat::Json) {
        std::vector<std::string> events_copy;
        {
            std::lock_guard lock(emit_mutex);
            events_copy = buffered_events;
        }
        const std::string json_array = join_json_array(events_copy);
        emit_line(json_array);
    }

    core::session::SessionData data;
    data.session_id = session_id;
    data.created_at = created_at;
    data.last_active_at = core::session::SessionStore::now_iso8601();
    data.working_dir = std::filesystem::current_path().string();
    data.provider = runtime.provider_name;
    data.model = runtime.model_name;
    data.mode = agent->get_mode();
    data.context_summary = agent->get_context_summary();
    data.messages = history_after;
    data.todos = session_todos;
    data.stats.prompt_tokens = totals.prompt_tokens;
    data.stats.completion_tokens = totals.completion_tokens;
    data.stats.cost_usd = cost_usd;
    data.stats.turn_count = snapshot.turn_count;
    data.stats.tool_calls_total = snapshot.tool_calls_total;
    data.stats.tool_calls_success = snapshot.tool_calls_success;
    data.handoff_summary = core::session::build_handoff_summary(data);

    std::string save_error;
    if (!store.save(data, &save_error)) {
        core::logging::warn("Prompter session save failed: {}", save_error);
    }

    diagnostics.session_id = session_id;
    diagnostics.session_file_path = maybe_abbreviate_home(store.compute_path(data).string());
    diagnostics.exit_code = request_failed ? 1 : 0;
    return diagnostics;
}

int run(const RunOptions& options) {
    std::string runtime_error;
    RuntimeContext runtime = build_runtime_context(runtime_error);
    if (!runtime_error.empty()) {
        std::cerr << "Prompter mode failed: " << runtime_error << "\n";
        return 1;
    }

    StreamInput input;
    input.has_stdin_data = stdin_has_data();
    if (input.has_stdin_data) {
        input.stdin_data = read_stdin_all();
    }

    auto diagnostics = run_for_test(options, runtime, input, std::cout, std::cerr);
    return diagnostics.exit_code;
}

} // namespace exec::prompter
