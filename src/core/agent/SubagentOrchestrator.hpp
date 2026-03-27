#pragma once

#include "../llm/LLMProvider.hpp"
#include "../llm/Models.hpp"
#include "../tools/ToolManager.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
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
        std::string active_model;
        std::string parent_mode;
        std::function<bool(const std::string&, const std::string&)> permission_check;
    };

    explicit SubagentOrchestrator(core::tools::ToolManager& tool_manager,
                                  const core::config::AppConfig* app_config = nullptr);

    [[nodiscard]] core::llm::Tool task_tool_definition() const;

    [[nodiscard]] std::string execute_task(
        std::string_view json_args,
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        const RunContext& context);

    void clear_sessions();

private:
    struct Profile {
        std::string name;
        std::string description;
        std::string prompt;
        std::string provider_override;
        std::string model_override;
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
        std::vector<core::llm::Message> history;
        int consecutive_failure_rounds = 0;
        std::mutex mutex;
    };

    struct StepResult {
        bool ok = false;
        std::string text;
        std::string reasoning_content;
        std::vector<core::llm::ToolCall> tool_calls;
        std::string error;
    };

    struct LoopResult {
        bool ok = false;
        std::string text;
        int steps = 0;
        int tool_calls = 0;
        int failed_tool_calls = 0;
        std::string error;
    };

    [[nodiscard]] const Profile* find_profile(std::string_view name) const;
    [[nodiscard]] std::string available_profile_names() const;
    [[nodiscard]] std::string build_task_description() const;
    [[nodiscard]] std::string create_task_id();
    void apply_config_overrides(const core::config::AppConfig& app_config);
    [[nodiscard]] std::shared_ptr<TaskSession> get_or_create_session(
        const std::string& description,
        const Profile& profile,
        std::string_view requested_task_id,
        std::string_view parent_mode,
        std::string_view initial_model,
        std::string& error_out);

    [[nodiscard]] std::vector<core::llm::Tool> build_tools_for_profile(
        const Profile& profile,
        std::string_view parent_mode) const;

    [[nodiscard]] StepResult run_model_step(
        TaskSession& session,
        const Profile& profile,
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        std::string_view parent_mode);

    [[nodiscard]] LoopResult run_task_loop(
        TaskSession& session,
        const Profile& profile,
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        const RunContext& context);

    [[nodiscard]] std::string render_success_json(
        const TaskSession& session,
        const Profile& profile,
        const LoopResult& result) const;

    [[nodiscard]] static std::string render_error_json(std::string_view error);
    [[nodiscard]] static std::string normalize_agent_name(std::string_view name);
    [[nodiscard]] static std::string normalize_mode(std::string_view mode);
    [[nodiscard]] static bool is_tool_error_payload(std::string_view payload);
    [[nodiscard]] static std::string build_system_prompt(const Profile& profile,
                                                         std::string_view parent_mode);

    core::tools::ToolManager& tool_manager_;
    std::vector<Profile> profiles_;

    std::atomic<uint64_t> next_task_id_{1};

    mutable std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<TaskSession>> sessions_;
};

} // namespace core::agent
