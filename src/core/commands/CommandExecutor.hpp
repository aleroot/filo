#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <memory>
#include <optional>
#include "../../core/agent/Agent.hpp"
#include "../../core/config/ConfigManager.hpp"
#include "../../core/history/PromptHistoryStore.hpp"
#include "../../core/session/SessionData.hpp"

namespace core::commands {

struct CommandOperationResult {
    bool ok = false;
    std::string message;
};

/**
 * @brief Context passed to every command execution containing the UI state.
 */
struct CommandContext {
    std::string text;
    std::function<void()> clear_input_fn;
    std::function<void(const std::string&)> append_history_fn;
    std::shared_ptr<core::agent::Agent> agent;
    std::function<void()> clear_screen_fn = {};
    std::function<void()> quit_fn = {};
    std::function<std::string()> model_status_fn = {};
    std::function<std::string(std::string_view)> switch_model_fn = {};
    std::function<std::string()> profile_status_fn = {};
    std::function<std::string(std::string_view)> switch_profile_fn = {};
    std::function<std::string()> effort_status_fn = {};
    std::function<std::string(std::string_view)> switch_effort_fn = {};
    std::function<bool()> open_model_picker_fn = {};
    std::function<bool()> open_settings_picker_fn = {};
    std::function<bool()> open_sessions_picker_fn = {};
    std::function<void(std::string_view)> resume_session_fn = {};
    std::function<void(std::vector<std::string>, std::function<void(std::optional<std::string>)>)> open_provider_picker_fn = {};
    std::function<void(std::function<void(std::optional<std::string>)>)> open_review_picker_fn = {};
    std::function<std::string()> settings_status_fn = {};
    std::function<bool()> yolo_mode_enabled_fn = {};
    std::function<void(bool)> set_yolo_mode_enabled_fn = {};
    std::function<std::string()> fork_session_fn = {};
    std::function<void(std::function<void()>)> suspend_tui_fn = {};
    std::function<std::string()> latest_assistant_output_fn = {};
    std::function<std::optional<std::string>(std::string_view)> copy_to_clipboard_fn = {};
    std::shared_ptr<core::history::PromptHistoryStore> history_store_fn = {};
    std::function<void()> clear_history_fn = {};
    std::function<void(const std::string&)> send_user_message_fn = {};
    // Toggle and describe long-running /review activity in the TUI.
    std::function<void(bool, const std::string&)> set_review_activity_fn = {};
    std::function<void(const std::string&,
                       const std::string&,
                       const std::vector<std::string>&)> send_user_skill_message_fn = {};
    std::function<std::vector<core::session::SessionTodoItem>()> list_todos_fn = {};
    std::function<CommandOperationResult(std::string_view)> add_todo_fn = {};
    std::function<CommandOperationResult(std::string_view, bool)> set_todo_completed_fn = {};
    std::function<CommandOperationResult(std::string_view)> remove_todo_fn = {};
    std::function<CommandOperationResult()> clear_completed_todos_fn = {};
    std::function<std::vector<core::config::McpServerConfig>()> list_mcp_servers_fn = {};
    std::function<CommandOperationResult(const core::config::McpServerConfig&,
                                         core::config::SettingsScope)> add_mcp_server_fn = {};
    std::function<CommandOperationResult(std::string_view,
                                         core::config::SettingsScope)> remove_mcp_server_fn = {};
};

struct CommandDescriptor {
    std::string name;
    std::vector<std::string> aliases;
    std::string description;
    bool accepts_arguments = false;
};

struct ActiveCommandToken {
    std::size_t replace_begin = 0;
    std::size_t replace_end = 0;
    std::string token;
};

struct CommandCompletion {
    std::string text;
    std::size_t cursor = 0;
};

/// Copies @p text to the system clipboard.
/// Returns std::nullopt on success, or an error message string on failure.
/// Supports macOS (pbcopy), Linux/Wayland (wl-copy), Linux/X11 (xclip/xsel),
/// and falls back to OSC 52 for remote/SSH sessions.
std::optional<std::string> copy_text_to_clipboard(std::string_view text);

std::optional<ActiveCommandToken> find_active_command(std::string_view input,
                                                      std::size_t cursor);

CommandCompletion apply_command_completion(std::string_view input,
                                           const ActiveCommandToken& active,
                                           std::string_view replacement);

/**
 * @brief Base class for executable CLI slash commands.
 */
class Command {
public:
    virtual ~Command() = default;
    virtual std::string get_name() const = 0;
    virtual std::vector<std::string> get_aliases() const { return {}; }
    virtual std::string get_description() const = 0;
    virtual bool accepts_arguments() const { return false; }
    virtual void execute(const CommandContext& ctx) = 0;
};

/**
 * @brief Registry and Executor for slash commands.
 */
class CommandExecutor {
public:
    CommandExecutor();
    
    void register_command(std::unique_ptr<Command> cmd);
    std::vector<CommandDescriptor> describe_commands() const;
    
    /**
     * @brief Parses and executes a command if the input string starts with a mapped trigger.
     * @return true if a command was successfully executed, false if it's normal text.
     */
    bool try_execute(const std::string& input, const CommandContext& ctx);

private:
    std::vector<std::unique_ptr<Command>> commands_;
};

} // namespace core::commands
