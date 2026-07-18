#pragma once

#include "../context/SessionContext.hpp"
#include "../llm/LLMProvider.hpp"
#include "../llm/Models.hpp"
#include "../tools/ToolManager.hpp"
#include "SubagentEvents.hpp"

#include <atomic>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace core::config {
struct AppConfig;
}

namespace core::agent {

class SubagentOrchestrator {
public:
    static constexpr std::string_view kTaskToolName = "task";

    struct RunContext {
        std::string active_provider_name;
        std::string active_model;
        std::string parent_mode;
        const core::context::SessionContext& session_context;
        std::function<bool(const std::string&, const std::string&)> permission_check;
        std::string parent_tool_call_id;
        std::function<void(const SubagentEvent&)> on_subagent_event;
        /// Polled by DelegatedAgentRunner while the worker runs; returning
        /// true cancels the delegated task (e.g. parent agent stop request).
        std::function<bool()> cancellation_requested;
    };

    struct ExecutionRequest {
        std::string_view worker;
        std::string_view worker_label = "worker";
        std::string_view parent_mode;
        std::shared_ptr<core::llm::LLMProvider> inherited_provider = {};
        std::string_view inherited_provider_name;
        std::string_view inherited_model;
        bool prefer_inherited_provider = false;
        std::string_view provider_override;
        std::string_view model_override;
        std::optional<int> max_steps_override = std::nullopt;
    };

    struct ExecutionPlan {
        std::string worker_name;
        std::string worker_description;
        std::string worker_prompt;
        std::shared_ptr<core::llm::LLMProvider> provider;
        std::string provider_name;
        std::string model_name;
        std::optional<core::llm::ResponseFormat> response_format;
        std::vector<core::llm::Tool> tools;
        bool allow_task_tool = false;
        int max_steps = 0;

        [[nodiscard]] std::vector<std::string> allowed_tool_names() const;
    };

    explicit SubagentOrchestrator(core::tools::ToolManager& tool_manager,
                                  const core::config::AppConfig* app_config = nullptr);

    [[nodiscard]] core::llm::Tool task_tool_definition() const;
    [[nodiscard]] std::expected<ExecutionPlan, std::string> build_execution_plan(
        const ExecutionRequest& request) const;
    [[nodiscard]] std::string available_worker_names() const;

    [[nodiscard]] std::string execute_task(
        std::string_view json_args,
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        const RunContext& context);

    void clear_sessions();
    void reload_profiles(const core::config::AppConfig& app_config);

private:
    struct Profile {
        std::string name;
        std::string description;
        std::string prompt;
        std::string provider_override;
        std::string model_override;
        std::optional<core::llm::ResponseFormat> response_format;
        std::unordered_set<std::string> allowed_tools;
        bool use_allow_list = false;
        bool allow_task_tool = false;
        int max_steps = 8;
        bool enabled = true;
    };

    struct TaskSession {
        std::string id;
        std::string profile_name;
        std::string description;
        std::string model;
        std::string context_summary;
        std::vector<core::llm::Message> history;
        std::mutex mutex;
    };

    [[nodiscard]] std::optional<Profile> find_profile(std::string_view name) const;
    [[nodiscard]] std::string build_task_description() const;
    [[nodiscard]] std::string create_task_id();
    [[nodiscard]] static std::vector<Profile> make_default_profiles();
    void apply_config_overrides_unlocked(const core::config::AppConfig& app_config);
    [[nodiscard]] std::shared_ptr<TaskSession> get_or_create_session(
        const std::string& description,
        const Profile& profile,
        std::string_view requested_task_id,
        std::string_view initial_model,
        std::string& error_out);

    [[nodiscard]] std::vector<core::llm::Tool> build_tools_for_profile(
        const Profile& profile,
        std::string_view parent_mode) const;

    [[nodiscard]] std::string render_success_json(
        const TaskSession& session,
        const ExecutionPlan& plan,
        std::string_view result_text,
        int steps,
        int tool_calls,
        int failed_tool_calls) const;

    [[nodiscard]] static std::string render_error_json(std::string_view error);
    [[nodiscard]] static std::string normalize_agent_name(std::string_view name);
    [[nodiscard]] static std::string normalize_mode(std::string_view mode);

    core::tools::ToolManager& tool_manager_;
    std::vector<Profile> profiles_;
    mutable std::mutex profiles_mutex_;
    const core::config::AppConfig* app_config_ = nullptr;

    std::atomic<uint64_t> next_task_id_{1};

    mutable std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<TaskSession>> sessions_;
};

} // namespace core::agent
