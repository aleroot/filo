#include "MainApp.hpp"
#include "Autocomplete.hpp"
#include "ActivityTimer.hpp"
#include "Constants.hpp"
#include "HistoryComponent.hpp"
#include "PickerState.hpp"
#include "Conversation.hpp"
#include "KeyInput.hpp"
#include "SessionReplay.hpp"
#include "Text.hpp"
#include "PromptInput.hpp"
#include "PromptComponents.hpp"
#include "TuiTheme.hpp"
#include "core/session/SessionData.hpp"
#include "core/session/SessionHandoff.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/history/PromptHistoryStore.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include "core/llm/ProviderManager.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/providers/RouterProvider.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/routing/RouterEngine.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/GetTimeTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/ApplyPatchTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/ReadFileTool.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/ReplaceTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/SearchReplaceTool.hpp"
#include "core/tools/DeleteFileTool.hpp"
#include "core/tools/MoveFileTool.hpp"
#include "core/tools/AskUserQuestionTool.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "core/tools/PythonInterpreterTool.hpp"
#include "core/tools/SkillLoader.hpp"
#endif
#include "core/logging/Logger.hpp"
#include "core/agent/Agent.hpp"
#include "core/agent/PermissionGate.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/mcp/McpConnectionManager.hpp"
#include "core/context/ContextMentions.hpp"
#include "core/commands/CommandExecutor.hpp"
#include "core/commands/SkillCommandLoader.hpp"
#include <atomic>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <format>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <future>
#include <algorithm>
#include <cctype>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <system_error>
#include <cstdint>
#include <cstdio>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif
#include <cpr/cpr.h>

using namespace ftxui;

namespace tui {
namespace {

constexpr std::string_view kEnableTerminalInputModes =
    "\x1B[?2004h"  // Bracketed paste
    "\x1B[>1u"     // Kitty keyboard protocol
    "\x1B[>4;2m";  // modifyOtherKeys level 2

constexpr std::string_view kDisableTerminalInputModes =
    "\x1B[?2004l"
    "\x1B[<u"
    "\x1B[>4;0m";

constexpr auto kExitConfirmWindow = std::chrono::milliseconds(3000);
constexpr auto kStreamChunkPauseBreakThreshold = std::chrono::milliseconds(1500);
constexpr std::string_view kStreamChunkResumeMarker = "💡";

struct PromptActivitySnapshot {
    std::string message_id;
    std::string label;
};

std::optional<PromptActivitySnapshot> active_prompt_activity(
    const std::vector<UiMessage>& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        const auto& message = *it;
        if (message.type != MessageType::Assistant || !message.pending) {
            continue;
        }

        bool has_executing_tools = false;
        bool has_completed_tools = false;
        for (const auto& tool : message.tools) {
            if (tool.status == ToolActivity::Status::Executing) {
                has_executing_tools = true;
            } else if (tool.status == ToolActivity::Status::Succeeded
                       || tool.status == ToolActivity::Status::Failed) {
                has_completed_tools = true;
            }
        }

        const bool show_thinking = message.thinking
            || (message.pending && has_completed_tools && !has_executing_tools);
        if (!show_thinking) {
            return std::nullopt;
        }

        const bool show_analyzing = has_completed_tools
            && !has_executing_tools
            && !message.thinking;
        return PromptActivitySnapshot{
            .message_id = message.id,
            .label = show_analyzing ? "Analyzing..." : "Thinking...",
        };
    }
    return std::nullopt;
}

class TerminalInputModeGuard {
public:
    TerminalInputModeGuard() {
        std::cout << kEnableTerminalInputModes << std::flush;
    }

    ~TerminalInputModeGuard() {
        std::cout << kDisableTerminalInputModes << std::flush;
    }
};

std::string shell_single_quote(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('\'');
    for (const char ch : value) {
        if (ch == '\'') {
            out += "'\\''";
        } else {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

bool command_exists_in_path(std::string_view command) {
    if (command.empty()) {
        return false;
    }
#if defined(_WIN32)
    (void)command;
    return false;
#else
    const std::string probe =
        "command -v " + shell_single_quote(command) + " >/dev/null 2>&1";
    return std::system(probe.c_str()) == 0;
#endif
}

std::optional<std::string> run_command_capture(std::string_view command) {
#if defined(_WIN32)
    (void)command;
    return std::nullopt;
#else
    std::array<char, 4096> buffer{};
    std::string output;
    const std::string command_str(command);
    FILE* pipe = ::popen(command_str.c_str(), "r");
    if (pipe == nullptr) {
        return std::nullopt;
    }

    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = ::pclose(pipe);
    if (status != 0) {
        return std::nullopt;
    }
    return output;
#endif
}

bool file_has_content(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        return false;
    }
    return std::filesystem::file_size(path, ec) > 0;
}

void remove_file_if_exists(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

std::filesystem::path make_clipboard_output_path(std::string_view extension) {
    static std::atomic<std::uint64_t> counter{0};
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "filo" / "clipboard";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir / std::format("clipboard-{}-{}.{}", now_ms, seq, extension);
}

bool write_clipboard_image_with_command(const std::string& command,
                                        const std::filesystem::path& out_path) {
#if defined(_WIN32)
    (void)command;
    (void)out_path;
    return false;
#else
    const int status = std::system(command.c_str());
    if (status != 0) {
        remove_file_if_exists(out_path);
        return false;
    }
    if (!file_has_content(out_path)) {
        remove_file_if_exists(out_path);
        return false;
    }
    return true;
#endif
}

std::optional<std::filesystem::path> read_clipboard_image_to_temp() {
#if defined(__APPLE__)
    {
        const auto path = make_clipboard_output_path("png");
        if (command_exists_in_path("pngpaste")) {
            const std::string cmd =
                "pngpaste " + shell_single_quote(path.string()) + " >/dev/null 2>&1";
            if (write_clipboard_image_with_command(cmd, path)) {
                return path;
            }
        }
    }
    {
        const auto path = make_clipboard_output_path("png");
        if (command_exists_in_path("osascript")) {
            const std::string cmd =
                "osascript "
                "-e " + shell_single_quote("set png_data to the clipboard as «class PNGf»") + " "
                "-e " + shell_single_quote(
                    std::string("set file_ref to open for access POSIX file \"")
                    + path.string() + "\" with write permission") + " "
                "-e " + shell_single_quote("set eof file_ref to 0") + " "
                "-e " + shell_single_quote("write png_data to file_ref") + " "
                "-e " + shell_single_quote("close access file_ref")
                + " >/dev/null 2>&1";
            if (write_clipboard_image_with_command(cmd, path)) {
                return path;
            }
        }
    }
#endif

#if defined(__linux__)
    if (command_exists_in_path("wl-paste")) {
        const auto types = run_command_capture("wl-paste --list-types 2>/dev/null");
        if (types.has_value()) {
            if (types->find("image/png") != std::string::npos) {
                const auto path = make_clipboard_output_path("png");
                const std::string cmd =
                    "wl-paste --type image/png > " + shell_single_quote(path.string()) + " 2>/dev/null";
                if (write_clipboard_image_with_command(cmd, path)) {
                    return path;
                }
            }
            if (types->find("image/jpeg") != std::string::npos) {
                const auto path = make_clipboard_output_path("jpg");
                const std::string cmd =
                    "wl-paste --type image/jpeg > " + shell_single_quote(path.string()) + " 2>/dev/null";
                if (write_clipboard_image_with_command(cmd, path)) {
                    return path;
                }
            }
        }
    }

    if (command_exists_in_path("xclip")) {
        const auto targets =
            run_command_capture("xclip -selection clipboard -t TARGETS -o 2>/dev/null");
        if (targets.has_value()) {
            if (targets->find("image/png") != std::string::npos) {
                const auto path = make_clipboard_output_path("png");
                const std::string cmd =
                    "xclip -selection clipboard -t image/png -o > "
                    + shell_single_quote(path.string()) + " 2>/dev/null";
                if (write_clipboard_image_with_command(cmd, path)) {
                    return path;
                }
            }
            if (targets->find("image/jpeg") != std::string::npos) {
                const auto path = make_clipboard_output_path("jpg");
                const std::string cmd =
                    "xclip -selection clipboard -t image/jpeg -o > "
                    + shell_single_quote(path.string()) + " 2>/dev/null";
                if (write_clipboard_image_with_command(cmd, path)) {
                    return path;
                }
            }
        }
    }
#endif

    return std::nullopt;
}

std::optional<std::string> read_clipboard_text() {
#if defined(__APPLE__)
    if (command_exists_in_path("pbpaste")) {
        if (auto out = run_command_capture("pbpaste 2>/dev/null"); out.has_value()) {
            return out;
        }
    }
#endif

#if defined(__linux__)
    if (command_exists_in_path("wl-paste")) {
        if (auto out = run_command_capture("wl-paste --no-newline --type text/plain 2>/dev/null");
            out.has_value()) {
            return out;
        }
    }
    if (command_exists_in_path("xclip")) {
        if (auto out = run_command_capture("xclip -selection clipboard -o 2>/dev/null");
            out.has_value()) {
            return out;
        }
    }
    if (command_exists_in_path("xsel")) {
        if (auto out = run_command_capture("xsel --clipboard --output 2>/dev/null");
            out.has_value()) {
            return out;
        }
    }
#endif

    return std::nullopt;
}

void insert_text_at_cursor(std::string& text, int& cursor, std::string_view chunk) {
    cursor = std::clamp(cursor, 0, static_cast<int>(text.size()));
    text.insert(static_cast<std::size_t>(cursor), chunk);
    cursor += static_cast<int>(chunk.size());
}

void insert_token_with_spacing(std::string& text, int& cursor, std::string_view token) {
    cursor = std::clamp(cursor, 0, static_cast<int>(text.size()));
    const bool need_leading_space =
        cursor > 0 && !std::isspace(static_cast<unsigned char>(text[static_cast<std::size_t>(cursor - 1)]));
    const bool need_trailing_space =
        cursor == static_cast<int>(text.size())
        || !std::isspace(static_cast<unsigned char>(text[static_cast<std::size_t>(cursor)]));

    std::string chunk;
    chunk.reserve(token.size() + 2);
    if (need_leading_space) {
        chunk.push_back(' ');
    }
    chunk.append(token);
    if (need_trailing_space) {
        chunk.push_back(' ');
    }

    insert_text_at_cursor(text, cursor, chunk);
}

bool is_gui_editor_command(std::string_view lowered_command) {
    static constexpr std::array<std::string_view, 5> kGuiEditors{
        "code", "cursor", "subl", "zed", "atom"
    };
    for (const auto editor : kGuiEditors) {
        if (lowered_command.find(editor) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

bool has_wait_flag(std::string_view lowered_command) {
    return lowered_command.find("--wait") != std::string_view::npos
        || lowered_command.find(" -w") != std::string_view::npos
        || lowered_command.starts_with("-w");
}

bool is_vi_family_command(std::string_view lowered_command) {
    static constexpr std::array<std::string_view, 3> kViCommands{
        "vi", "vim", "nvim"
    };
    for (const auto cmd : kViCommands) {
        if (lowered_command.find(cmd) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

std::string build_external_editor_command(const std::string& file_path) {
    const char* visual = std::getenv("VISUAL");
    const char* editor = std::getenv("EDITOR");

    std::string command;
    if (visual != nullptr && *visual != '\0') {
        command = visual;
    } else if (editor != nullptr && *editor != '\0') {
        command = editor;
    } else {
        command = "vi";
    }

    const std::string lowered = to_lower_ascii(command);
    if (is_gui_editor_command(lowered) && !has_wait_flag(lowered)) {
        command += lowered.find("subl") != std::string::npos ? " -w" : " --wait";
    }
    if (is_vi_family_command(lowered) && lowered.find("-i") == std::string::npos) {
        command += " -i NONE";
    }

    command += " " + shell_single_quote(file_path);
    return command;
}

int normalize_exit_code(int status) {
    if (status == -1) {
        return -1;
    }
#if defined(_WIN32)
    return status;
#else
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return status;
#endif
}

} // namespace

RunResult run(RunOptions opts) {
    cpr::Session dummy_session;  // initialise curl

    auto& config_manager     = core::config::ConfigManager::get_instance();
    auto config              = config_manager.get_config();
    auto& provider_manager   = core::llm::ProviderManager::get_instance();
    const auto settings_working_dir = std::filesystem::current_path();

    enum class ModelSelectionMode {
        Manual,
        Router,
        Auto,   // Smart routing with AutoClassifier task classification
    };

    auto parse_model_selection_mode = [](std::string_view mode) {
        std::string normalized;
        normalized.reserve(mode.size());
        for (const char c : mode) {
            if (!std::isspace(static_cast<unsigned char>(c))
                && c != '-'
                && c != '_') {
                normalized.push_back(static_cast<char>(
                    std::tolower(static_cast<unsigned char>(c))));
            }
        }
        if (normalized == "router") return ModelSelectionMode::Router;
        if (normalized == "auto")   return ModelSelectionMode::Auto;
        return ModelSelectionMode::Manual;
    };

    enum class ApprovalMode {
        Prompt,
        Yolo,
    };

    auto parse_approval_mode = [](std::string_view value) {
        std::string normalized;
        normalized.reserve(value.size());
        for (const char ch : value) {
            if (std::isspace(static_cast<unsigned char>(ch))
                || ch == '-'
                || ch == '_') {
                continue;
            }
            normalized.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(ch))));
        }
        return normalized == "yolo" ? ApprovalMode::Yolo : ApprovalMode::Prompt;
    };

    std::unordered_set<std::string> registered_provider_names;
    std::unordered_map<std::string, std::string> provider_default_models;

    for (const auto& [name, pconfig] : config.providers) {
        if (auto provider = core::llm::ProviderFactory::create_provider(name, pconfig)) {
            provider_manager.register_provider(name, provider);
            registered_provider_names.insert(name);
            provider_default_models[name] = pconfig.model;
        }
    }

    std::vector<std::string> sorted_provider_names(
        registered_provider_names.begin(),
        registered_provider_names.end());
    std::ranges::sort(sorted_provider_names);

    if (registered_provider_names.empty()) {
        core::logging::error("Fatal: No providers could be initialised from configuration.");
        return {};
    }

    std::string manual_provider_name = config.default_provider;
    if (!registered_provider_names.contains(manual_provider_name)) {
        manual_provider_name = *registered_provider_names.begin();
    }

    std::string manual_model_name;
    if (const auto it = config.providers.find(manual_provider_name);
        it != config.providers.end()) {
        manual_model_name = it->second.model;
    }

    auto router_engine = std::make_shared<core::llm::routing::RouterEngine>(
        config.router,
        registered_provider_names);
    auto router_provider = std::make_shared<core::llm::providers::RouterProvider>(
        provider_manager,
        router_engine,
        provider_default_models);

    const bool router_available = config.router.enabled && !router_engine->list_policies().empty();

    ModelSelectionMode model_selection_mode = parse_model_selection_mode(config.default_model_selection);
    if ((model_selection_mode == ModelSelectionMode::Router
         || model_selection_mode == ModelSelectionMode::Auto)
        && !router_available) {
        model_selection_mode = ModelSelectionMode::Manual;
    }

    std::string active_router_policy = router_engine->active_policy();
    std::string active_provider_name;
    std::string active_model_name;

    std::shared_ptr<core::llm::LLMProvider> llm_provider;
    if (model_selection_mode == ModelSelectionMode::Router
        || model_selection_mode == ModelSelectionMode::Auto) {
        llm_provider = router_provider;
        active_provider_name = (model_selection_mode == ModelSelectionMode::Auto) ? "auto" : "router";
        active_model_name = active_router_policy.empty()
            ? "policy/<unset>"
            : "policy/" + active_router_policy;
    } else {
        try {
            llm_provider = provider_manager.get_provider(manual_provider_name);
        } catch (const std::exception& e) {
            core::logging::error(
                "Fatal: Could not initialise default provider '{}'. Error: {}",
                manual_provider_name,
                e.what());
            return {};
        }
        active_provider_name = manual_provider_name;
        active_model_name = manual_model_name;
    }

    bool ui_show_banner = visibility_setting_enabled(config.ui_banner, true);
    bool ui_show_footer = visibility_setting_enabled(config.ui_footer, true);
    bool ui_show_model_info = visibility_setting_enabled(config.ui_model_info, true);
    bool ui_show_context_usage = visibility_setting_enabled(config.ui_context_usage, true);
    bool ui_show_timestamps = visibility_setting_enabled(config.ui_timestamps, true);
    bool ui_show_spinner = visibility_setting_enabled(config.ui_spinner, true);
    bool tool_output_expanded = false;

    // ── Tool registration ───────────────────────────────────────────────────
    auto& tool_manager = core::tools::ToolManager::get_instance();
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
    
    // AskUserQuestion tool - needs callback for UI
    auto ask_user_tool = std::make_shared<core::tools::AskUserQuestionTool>();
    tool_manager.register_tool(ask_user_tool);
    
#ifdef FILO_ENABLE_PYTHON
    tool_manager.register_tool(std::make_shared<core::tools::PythonInterpreterTool>());
    core::tools::SkillLoader::discover_and_register(tool_manager);
#endif

    // ── MCP client connections ───────────────────────────────────────────────
    core::mcp::McpConnectionManager::get_instance().connect_all(
        config,
        tool_manager,
        llm_provider,
        model_selection_mode == ModelSelectionMode::Manual ? manual_model_name : std::string{});

    // ── Agent ────────────────────────────────────────────────────────────────
    auto agent_session_context = core::context::make_session_context(
        core::workspace::Workspace::get_instance().snapshot(),
        core::context::SessionTransport::cli);
    auto agent = std::make_shared<core::agent::Agent>(
        llm_provider,
        tool_manager,
        agent_session_context);
    agent->set_auto_compact_threshold(config.auto_compact_threshold);

    // Set the active model for budget tracking
    if (model_selection_mode == ModelSelectionMode::Router
        || model_selection_mode == ModelSelectionMode::Auto) {
        agent->set_active_model(active_model_name);
    } else if (!manual_model_name.empty()) {
        agent->set_active_model(manual_model_name);
    }

    // ── UI state ─────────────────────────────────────────────────────────────
    auto provider_setup_hint = [&](std::string_view provider_name) -> std::string {
        const auto it = config.providers.find(std::string(provider_name));
        if (it == config.providers.end()) return "";
        if (provider_name.starts_with("grok") && it->second.api_key.empty()) {
            return "Set XAI_API_KEY to start chatting with Grok.\n"
                   "Get a key at: console.x.ai  (sign up → API Keys)\n"
                   "Grok presets: grok (default), grok-4, grok-4-fast, grok-reasoning, grok-fast, grok-mini\n";
        }
        return "";
    };

    auto startup_history_message = [&]() {
        if (ui_show_banner) {
            return std::string{};
        }
        return std::format(
            "Filo AI Agent  —  provider: {}  —  model: {}  —  MCP servers: {}\n{}",
            active_provider_name,
            active_model_name.empty() ? "<provider default>" : active_model_name,
            core::mcp::McpConnectionManager::get_instance().connected_count(),
            provider_setup_hint(active_provider_name));
    };

    std::vector<UiMessage> ui_messages;
    if (const auto message = startup_history_message(); !message.empty()) {
        ui_messages.push_back(make_system_message(message));
    }

    std::string input_text;
    int input_cursor_position = 0;
    int ctrl_d_press_count = 0;
    std::chrono::steady_clock::time_point ctrl_d_deadline =
        std::chrono::steady_clock::time_point::min();
    std::mutex  ui_mutex;
    std::mutex stream_chunk_timing_mutex;
    std::unordered_map<std::size_t, std::chrono::steady_clock::time_point> stream_chunk_last_at;
    std::atomic_bool assistant_turn_active{false};
    ActivityTimerRegistry turn_activity_timers;
    
    // Rate limit tracking for status bar and notifications
    struct RateLimitState {
        int32_t requests_limit = 0;
        int32_t requests_remaining = 0;
        int32_t tokens_limit = 0;
        int32_t tokens_remaining = 0;
        std::vector<core::llm::protocols::UsageWindow> usage_windows;
        bool quota_notified_low = false;
        bool quota_notified_critical = false;
        std::chrono::steady_clock::time_point last_update;
    };
    RateLimitState rate_limit_state;
    auto screen = ScreenInteractive::Fullscreen();
    // Enable mouse tracking for scrolling and focus.
    screen.TrackMouse(true);
    // Auto-copy any text selected via left-click drag to the system clipboard.
    // This is the cross-platform solution: on Linux the Shift+right-click bypass
    // works in xterm-family terminals, but on macOS iTerm2 uses Option/Alt instead.
    // Wiring SelectionChange ensures copy works identically on both platforms.
    screen.SelectionChange([&screen]() {
        const std::string selection = screen.GetSelection();
        if (!selection.empty()) {
            core::commands::copy_text_to_clipboard(selection);
        }
    });
    TerminalInputModeGuard terminal_input_mode_guard;
    std::atomic<std::size_t> animation_tick = 0;
    std::mutex animation_mutex;
    std::condition_variable animation_cv;
    core::commands::CommandExecutor cmd_executor;
    // Load layered prompt skills (global → project-local) before describe_commands()
    // so skill commands appear in the autocomplete index.
    core::commands::SkillCommandLoader::discover_and_register(cmd_executor);
    const auto mention_index = build_mention_index(std::filesystem::current_path());
    const auto command_index = cmd_executor.describe_commands();

    // ── Picker state (shared structure for mention and command pickers) ───────
    // See tui/PickerState.hpp for the struct definition.
    PickerState mention_picker;
    PickerState command_picker;

    // ── Prompt history ────────────────────────────────────────────────────────
    // Persistent history store for input prompts.
    auto history_store = std::make_shared<core::history::PromptHistoryStore>(
        core::history::PromptHistoryStore::default_history_path());
    // Load existing history (silent failure is OK).
    static_cast<void>(history_store->load(nullptr));
    
    // Persistent prompt history with navigation.
    core::history::PersistentPromptHistory prompt_history(history_store);

    ApprovalMode approval_mode = parse_approval_mode(config.default_approval_mode);

    // Session allow-list: tool calls matching these keys are auto-approved
    // without prompting (populated via the "don't ask again" option).
    std::unordered_set<std::string> session_allowed;

    // Permission overlay state
    struct PermissionState {
        bool                                            active = false;
        std::string                                     tool_name;
        std::string                                     args_preview;
        ToolDiffPreview                                 diff_preview;
        int                                             selected = 0;  // 0-3
        std::string                                     allow_key;
        std::string                                     allow_label;
        std::shared_ptr<std::promise<bool>>             promise;
    };
    PermissionState perm_state;
    std::mutex permission_prompt_mutex;
    std::condition_variable permission_prompt_cv;
    bool permission_prompt_in_flight = false;

    struct ModelPickerState {
        bool active = false;
        int selected = 0; // 0=manual, 1=router, 2=auto
    };
    ModelPickerState model_picker_state;

    struct ProviderPickerState {
        bool active = false;
        int selected = 0;
        std::vector<std::string> providers;
        std::function<void(std::optional<std::string>)> on_select;
    };
    ProviderPickerState provider_picker_state;

    struct LocalModelPickerState {
        bool active = false;
        int selected = 0;
        std::filesystem::path current_dir;
        std::vector<tui::LocalModelEntry> entries;
    };
    LocalModelPickerState local_model_picker_state;

    struct SessionPickerState {
        bool active = false;
        int selected = 0;
        std::vector<core::session::SessionInfo> sessions;
    };
    SessionPickerState session_picker_state;

    struct ConversationSearchState {
        bool active = false;
        std::string query;
        int selected = 0;
        std::vector<tui::ConversationSearchHit> hits;
    };
    ConversationSearchState conversation_search_state;

    // Question dialog state (AskUserQuestion tool)
    tui::QuestionDialogState question_dialog_state;

    struct SettingsChoice {
        std::string value;
        std::string label;
    };

    struct SettingsDefinition {
        core::config::ManagedSettingKey key;
        std::string label;
        std::string description;
        std::vector<SettingsChoice> choices;
    };

    struct SettingsPanelState {
        bool active = false;
        int selected = 0;
        core::config::SettingsScope scope = core::config::SettingsScope::User;
        std::string status_message;
    };
    SettingsPanelState settings_panel_state;

    int current_mode_idx = 0;
    std::vector<std::pair<std::string, Color>> modes = {
        {"BUILD",    Color::Blue},
        {"DEBUG",    ColorWarn},
        {"RESEARCH", Color::Green},
        {"EXECUTE",  Color::Red}
    };
    auto normalize_mode = [](std::string mode) {
        std::erase_if(mode, [](unsigned char c) { return !std::isalpha(c); });
        if (mode.empty()) mode = "BUILD";
        std::ranges::transform(mode, mode.begin(), ::toupper);
        return mode;
    };
    {
        const std::string desired_mode = normalize_mode(config.default_mode);
        for (std::size_t i = 0; i < modes.size(); ++i) {
            if (modes[i].first == desired_mode) {
                current_mode_idx = static_cast<int>(i);
                break;
            }
        }
    }
    agent->set_mode(modes[current_mode_idx].first);

    std::vector<SettingsDefinition> settings_definitions;
    {
        std::vector<SettingsChoice> mode_choices;
        mode_choices.reserve(modes.size());
        for (const auto& [mode_name, _] : modes) {
            mode_choices.push_back(SettingsChoice{
                .value = mode_name,
                .label = mode_name,
            });
        }

        std::vector<SettingsChoice> router_policy_choices;
        for (const auto& policy_name : router_engine->list_policies()) {
            router_policy_choices.push_back(SettingsChoice{
                .value = policy_name,
                .label = policy_name,
            });
        }

        const auto visibility_choices = []() {
            return std::vector<SettingsChoice>{
                SettingsChoice{.value = "show", .label = "Show"},
                SettingsChoice{.value = "hide", .label = "Hide"},
            };
        };

        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::DefaultMode,
            .label = "General · Start Mode",
            .description = "The agent mode Filo should start in for this scope.",
            .choices = std::move(mode_choices),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::DefaultApprovalMode,
            .label = "General · Approval Mode",
            .description = "PROMPT asks before sensitive tools. YOLO auto-approves them.",
            .choices = {
                SettingsChoice{.value = "prompt", .label = "PROMPT"},
                SettingsChoice{.value = "yolo", .label = "YOLO"},
            },
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::DefaultRouterPolicy,
            .label = "Routing · Default Policy",
            .description = "Pick the router policy new Router/Auto sessions should start from.",
            .choices = std::move(router_policy_choices),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiBanner,
            .label = "UI · Startup Banner",
            .description = "Show or hide the Filo banner on startup and after /clear.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiFooter,
            .label = "UI · Footer",
            .description = "Show or hide the footer with session status and context info.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiModelInfo,
            .label = "UI · Model Badge",
            .description = "Show or hide the active provider/model badge in the footer.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiContextUsage,
            .label = "UI · Context Meter",
            .description = "Show or hide the context usage indicator in the footer.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiTimestamps,
            .label = "UI · Message Timestamps",
            .description = "Show or hide timestamps above user messages.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiSpinner,
            .label = "UI · Activity Spinner",
            .description = "Show or hide the animated spinner while Filo is working.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::AutoCompactThreshold,
            .label = "Session · Auto-Compaction",
            .description = "Total tokens before the conversation is summarised. 0 = disabled.",
            .choices = {
                SettingsChoice{.value = "0",      .label = "Disabled"},
                SettingsChoice{.value = "25000",  .label = "25k (Fast)"},
                SettingsChoice{.value = "50000",  .label = "50k (Balanced)"},
                SettingsChoice{.value = "100000", .label = "100k (Long)"},
                SettingsChoice{.value = "200000", .label = "200k (Pro)"},
            },
        });
    }

    // ── Session management ────────────────────────────────────────────────────
    auto session_store = std::make_shared<core::session::SessionStore>(
        core::session::SessionStore::default_sessions_dir());

    std::string session_id          = core::session::SessionStore::generate_id();
    std::string session_created_at  = core::session::SessionStore::now_iso8601();
    std::string session_file_path;  // computed after first save

    // If resuming, compute the path now so we can return it in RunResult.
    auto compute_session_path = [&]() {
        core::session::SessionData tmp;
        tmp.session_id  = session_id;
        tmp.created_at  = session_created_at;
        session_file_path = session_store->compute_path(tmp).string();
    };

    // Handle --resume / opts.resume_session_id
    if (opts.resume_session_id.has_value()) {
        const auto& req = *opts.resume_session_id;
        auto data_opt = req.empty()
            ? session_store->load_most_recent()
            : session_store->load(req);
        if (data_opt.has_value()) {
            const auto& data = *data_opt;
            session_id         = data.session_id;
            session_created_at = data.created_at;

            // Restore agent state.
            agent->load_history(data.messages, data.context_summary, data.mode);

            // Align mode picker.
            for (std::size_t i = 0; i < modes.size(); ++i) {
                if (modes[i].first == data.mode) {
                    current_mode_idx = static_cast<int>(i);
                    break;
                }
            }

            // Restore session file path.
            session_file_path = session_store->compute_path(data).string();

            ui_messages = build_resumed_ui_messages(
                data,
                SessionReplayOptions{.include_continue_hint = true});
        } else {
            // Session not found — warn and continue with a fresh session.
            ui_messages.push_back(make_warning_message(
                std::format("Session '{}' not found. Starting a fresh session.",
                    req.empty() ? std::string("most recent") : req)));
        }
    }

    compute_session_path();

    // ── Helper lambdas ───────────────────────────────────────────────────────

    auto navigate_history_prev = [&]() -> bool {
        return prompt_history.navigate_prev(input_text, input_cursor_position);
    };

    auto navigate_history_next = [&]() -> bool {
        return prompt_history.navigate_next(input_text, input_cursor_position);
    };

    auto clear_screen = [&]() {
        {
            std::lock_guard lock(stream_chunk_timing_mutex);
            stream_chunk_last_at.clear();
        }
        turn_activity_timers.clear();
        assistant_turn_active.store(false, std::memory_order_relaxed);
        {
            std::lock_guard lock(ui_mutex);
            ui_messages.clear();
            if (const auto message = startup_history_message(); !message.empty()) {
                ui_messages.push_back(make_system_message(message));
            }
        }
        animation_cv.notify_one();
        agent->clear_history();
        core::budget::BudgetTracker::get_instance().reset_session();
        core::session::SessionStats::get_instance().reset();
        // Start a fresh session after /clear.
        session_id         = core::session::SessionStore::generate_id();
        session_created_at = core::session::SessionStore::now_iso8601();
        compute_session_path();
    };

    auto append_history = [&](const std::string& str) {
        std::lock_guard lock(ui_mutex);
        if (ui_messages.empty() || ui_messages.back().type != MessageType::System) {
            ui_messages.push_back(make_system_message(str));
        } else {
            ui_messages.back().text += str;
        }
        screen.PostEvent(Event::Custom);
    };

    auto has_active_animation = [&]() -> bool {
        if (assistant_turn_active.load(std::memory_order_relaxed)) {
            return true;
        }
        std::lock_guard lock(ui_mutex);
        return conversation_uses_animation(ui_messages, ui_show_spinner);
    };

    auto refresh_conversation_search_locked = [&]() {
        conversation_search_state.hits.clear();

        const std::string query = std::string(trim_ascii(conversation_search_state.query));
        if (query.empty()) {
            conversation_search_state.selected = 0;
            return;
        }

        const std::string lowered_query = to_lower_ascii(query);
        constexpr std::size_t kMaxSearchHits = 256;

        for (std::size_t i = 0; i < ui_messages.size(); ++i) {
            const auto& message = ui_messages[i];
            const std::string searchable = search_text_for_message(message);
            if (searchable.empty()) {
                continue;
            }

            const std::string lowered_text = to_lower_ascii(searchable);
            const std::size_t pos = lowered_text.find(lowered_query);
            if (pos == std::string::npos) {
                continue;
            }

            conversation_search_state.hits.push_back(tui::ConversationSearchHit{
                .message_index = static_cast<int>(i),
                .role = search_role_label(message.type),
                .snippet = build_search_snippet(searchable, pos, lowered_query.size()),
            });

            if (conversation_search_state.hits.size() >= kMaxSearchHits) {
                break;
            }
        }

        if (conversation_search_state.hits.empty()) {
            conversation_search_state.selected = 0;
            return;
        }

        conversation_search_state.selected = std::clamp(
            conversation_search_state.selected,
            0,
            static_cast<int>(conversation_search_state.hits.size()) - 1);
    };
    
    // Helper to check and notify on quota changes (called periodically from render loop)
    auto check_and_notify_quota = [&]() {
        const auto utilization_from_remaining = [](int32_t remaining, int32_t limit) -> float {
            if (limit <= 0) return -1.0f;
            const int32_t used = std::max<int32_t>(0, limit - remaining);
            const float ratio = static_cast<float>(used) / static_cast<float>(limit);
            return std::clamp(ratio, 0.0f, 1.5f);
        };

        const float req_util = utilization_from_remaining(
            rate_limit_state.requests_remaining, rate_limit_state.requests_limit);
        const float tok_util = utilization_from_remaining(
            rate_limit_state.tokens_remaining, rate_limit_state.tokens_limit);
        float unified_util = 0.0f;
        for (const auto& w : rate_limit_state.usage_windows) {
            unified_util = std::max(unified_util, w.utilization);
        }
        const float api_util = std::max(req_util, tok_util);
        const float max_util = std::max(unified_util, api_util);

        // Maps short labels to human-readable window names for notification messages.
        auto window_name = [](std::string_view label) -> std::string {
            if (label == "4h") return "4-hour window";
            if (label == "5h") return "5-hour window";
            if (label == "7d") return "7-day window";
            return std::string(label) + " window";
        };

        auto append_trigger_details = [&](std::string& msg, float threshold) {
            for (const auto& w : rate_limit_state.usage_windows) {
                if (w.utilization >= threshold) {
                    msg += std::format("    {}: {:.0f}% used\n",
                                       window_name(w.label), w.utilization * 100.0f);
                }
            }
            if (req_util >= threshold) {
                msg += std::format("    Request window: {:.0f}% used ({}/{})\n",
                                   req_util * 100.0f,
                                   rate_limit_state.requests_remaining,
                                   rate_limit_state.requests_limit);
            }
            if (tok_util >= threshold) {
                msg += std::format("    Token window: {:.0f}% used ({}/{})\n",
                                   tok_util * 100.0f,
                                   rate_limit_state.tokens_remaining,
                                   rate_limit_state.tokens_limit);
            }
        };

        // Check for critical quota (90%+ used)
        if (max_util >= 0.90f) {
            if (!rate_limit_state.quota_notified_critical) {
                rate_limit_state.quota_notified_critical = true;
                std::string msg = "\n\xe2\x9a\xa0  CRITICAL: Rate limit quota critical (90%+ used).\n";
                append_trigger_details(msg, 0.90f);
                append_history(msg);
            }
        }
        // Check for low quota (75%+ used)
        else if (max_util >= 0.75f) {
            if (!rate_limit_state.quota_notified_low) {
                rate_limit_state.quota_notified_low = true;
                std::string msg = "\n\xe2\x84\xb9  Warning: Rate limit quota running low (75%+ used).\n";
                append_trigger_details(msg, 0.75f);
                append_history(msg);
            }
        }
        // Reset notifications when quota recovers (below 50%)
        else if ((unified_util <= 0.0f || unified_util < 0.50f)
                 && (api_util <= 0.0f || api_util < 0.50f)) {
            rate_limit_state.quota_notified_low = false;
            rate_limit_state.quota_notified_critical = false;
        }
    };

    auto is_yolo_mode_enabled = [&]() -> bool {
        std::lock_guard lock(ui_mutex);
        return approval_mode == ApprovalMode::Yolo;
    };

    auto set_yolo_mode_enabled = [&](bool enabled) {
        {
            std::lock_guard lock(ui_mutex);
            approval_mode = enabled ? ApprovalMode::Yolo : ApprovalMode::Prompt;
        }
        screen.PostEvent(Event::Custom);
    };

    auto router_policy_label = [&]() {
        return active_router_policy.empty()
            ? std::string("policy/<unset>")
            : std::string("policy/") + active_router_policy;
    };

    auto refresh_status_labels = [&]() {
        if (model_selection_mode == ModelSelectionMode::Router) {
            active_provider_name = "router";
            if (router_provider) {
                active_router_policy = router_provider->active_policy();
                const std::string routed_model = router_provider->get_last_model();
                const std::string guardrail_summary = router_provider->last_guardrail_summary();
                active_model_name = router_policy_label();
                if (!routed_model.empty()) {
                    active_model_name += " -> " + routed_model;
                }
                if (!guardrail_summary.empty()) {
                    active_model_name += " [reserve fallback]";
                }
            } else {
                active_model_name = router_policy_label();
            }
            return;
        }

        if (model_selection_mode == ModelSelectionMode::Auto) {
            active_provider_name = "auto";
            if (router_provider) {
                const std::string summary = router_provider->last_route_summary();
                const std::string routed_model = router_provider->get_last_model();
                const std::string guardrail_summary = router_provider->last_guardrail_summary();
                // Format: "auto · [Debugging·0.72] → grok-reasoning" when we have routing info
                if (!routed_model.empty()) {
                    active_model_name = "smart -> " + routed_model;
                } else if (!summary.empty()) {
                    active_model_name = "smart -> " + summary;
                } else {
                    active_model_name = "smart routing";
                }
                if (!guardrail_summary.empty()) {
                    active_model_name += " [reserve fallback]";
                }
            } else {
                active_model_name = "smart routing";
            }
            return;
        }

        active_provider_name = manual_provider_name;
        active_model_name = manual_model_name;
    };

    auto sync_mcp_sampling_backend = [&](std::shared_ptr<core::llm::LLMProvider> provider,
                                         std::string sampling_model) {
        core::mcp::McpConnectionManager::get_instance().update_sampling_backend(
            std::move(provider),
            std::move(sampling_model));
    };

    auto resume_session = [&](const core::session::SessionData& data) {
        {
            std::lock_guard lock(stream_chunk_timing_mutex);
            stream_chunk_last_at.clear();
        }
        turn_activity_timers.clear();
        assistant_turn_active.store(false, std::memory_order_relaxed);
        {
            std::lock_guard lock(ui_mutex);
            session_id         = data.session_id;
            session_created_at = data.created_at;

            // Restore agent state.
            agent->load_history(data.messages, data.context_summary, data.mode);

            // Align mode picker.
            for (std::size_t i = 0; i < modes.size(); ++i) {
                if (modes[i].first == data.mode) {
                    current_mode_idx = static_cast<int>(i);
                    break;
                }
            }

            // Restore session file path.
            session_file_path = session_store->compute_path(data).string();

            ui_messages = build_resumed_ui_messages(data);
            
            // Re-sync TUI status labels to the resumed session's provider/model
            manual_provider_name = data.provider;
            manual_model_name    = data.model;
            // If the resumed session used a provider we still have, switch to it.
            if (registered_provider_names.contains(data.provider)) {
                try {
                    auto p = provider_manager.get_provider(data.provider);
                    agent->set_provider(p);
                    agent->set_active_model(data.model);
                    model_selection_mode = ModelSelectionMode::Manual;
                    sync_mcp_sampling_backend(p, data.model);
                } catch(...) {}
            }
            refresh_status_labels();
        }
        animation_cv.notify_one();
        screen.PostEvent(Event::Custom);
    };

    auto open_sessions_picker = [&]() -> bool {
        {
            std::lock_guard lock(ui_mutex);
            session_picker_state.sessions = session_store->list();
            if (session_picker_state.sessions.empty()) {
                return false;
            }
            session_picker_state.active = true;
            session_picker_state.selected = 0;
        }
        screen.PostEvent(Event::Custom);
        return true;
    };

    auto fork_session = [&]() -> std::string {
        const auto snap_messages = agent->get_history();
        const std::string snap_mode = agent->get_mode();
        const std::string snap_context = agent->get_context_summary();

        std::string old_session_id;
        std::string provider_name;
        std::string model_name;
        {
            std::lock_guard lock(ui_mutex);
            old_session_id = session_id;
            provider_name = active_provider_name;
            model_name = active_model_name;
        }

        const std::string new_session_id = core::session::SessionStore::generate_id();
        const std::string new_created_at = core::session::SessionStore::now_iso8601();

        core::session::SessionData data;
        data.session_id      = new_session_id;
        data.created_at      = new_created_at;
        data.last_active_at  = core::session::SessionStore::now_iso8601();
        data.working_dir     = std::filesystem::current_path().string();
        data.provider        = provider_name;
        data.model           = model_name;
        data.mode            = snap_mode;
        data.context_summary = snap_context;
        data.messages        = snap_messages;

        const auto& budget = core::budget::BudgetTracker::get_instance();
        const auto total = budget.session_total();
        data.stats.prompt_tokens = total.prompt_tokens;
        data.stats.completion_tokens = total.completion_tokens;
        data.stats.cost_usd = budget.session_cost_usd();

        const auto stats_snapshot = core::session::SessionStats::get_instance().snapshot();
        data.stats.turn_count = stats_snapshot.turn_count;
        data.stats.tool_calls_total = stats_snapshot.tool_calls_total;
        data.stats.tool_calls_success = stats_snapshot.tool_calls_success;
        data.handoff_summary = core::session::build_handoff_summary(data);

        std::string error;
        if (!session_store->save(data, &error)) {
            if (error.empty()) {
                error = "unknown save error";
            }
            return std::format("Failed to fork session: {}", error);
        }

        {
            std::lock_guard lock(ui_mutex);
            session_id = new_session_id;
            session_created_at = new_created_at;
            session_file_path = session_store->compute_path(data).string();
        }
        screen.PostEvent(Event::Custom);
        return std::format("Forked session {} into {}.", old_session_id, new_session_id);
    };

    agent->set_efficiency_decision_fn(
        [agent, session_store, &ui_mutex, &session_id, &session_created_at,
         &session_file_path, &active_provider_name, &active_model_name,
         &ui_messages, &screen](const core::session::SessionEfficiencyDecision& decision) {
            auto snap_messages = agent->get_history();
            auto snap_mode = agent->get_mode();
            auto snap_context = agent->get_context_summary();

            core::session::SessionData archived;
            {
                std::lock_guard lock(ui_mutex);
                archived.session_id = session_id;
                archived.created_at = session_created_at;
                archived.provider = active_provider_name;
                archived.model = active_model_name;
            }
            archived.last_active_at = core::session::SessionStore::now_iso8601();
            archived.working_dir = std::filesystem::current_path().string();
            archived.mode = snap_mode;
            archived.context_summary = snap_context;
            archived.messages = snap_messages;
            archived.handoff_summary = core::session::build_handoff_summary(archived);

            const auto& budget = core::budget::BudgetTracker::get_instance();
            const auto total = budget.session_total();
            archived.stats.prompt_tokens = total.prompt_tokens;
            archived.stats.completion_tokens = total.completion_tokens;
            archived.stats.cost_usd = budget.session_cost_usd();
            const auto stats_snapshot = core::session::SessionStats::get_instance().snapshot();
            archived.stats.turn_count = stats_snapshot.turn_count;
            archived.stats.tool_calls_total = stats_snapshot.tool_calls_total;
            archived.stats.tool_calls_success = stats_snapshot.tool_calls_success;

            std::string save_error;
            if (!session_store->save(archived, &save_error)) {
                core::logging::warn(
                    "Skipping TUI session rotation for {} because archival save failed: {}",
                    archived.session_id,
                    save_error);
                {
                    std::lock_guard lock(ui_mutex);
                    ui_messages.push_back(make_warning_message(std::format(
                        "Filo skipped an internal session rotation because it could not archive the current segment.\nSession: {}\nReason: {}\nYour full context is still intact and no history was compacted.",
                        archived.session_id,
                        save_error.empty() ? std::string("unknown archival error.") : save_error)));
                }
                screen.PostEvent(ftxui::Event::Custom);
                return;
            }

            agent->compact_history(archived.handoff_summary);
            core::budget::BudgetTracker::get_instance().reset_session();
            core::session::SessionStats::get_instance().reset();

            const std::string old_session_id = archived.session_id;
            const std::string new_session_id = core::session::SessionStore::generate_id();
            const std::string new_created_at = core::session::SessionStore::now_iso8601();

            {
                std::lock_guard lock(ui_mutex);
                core::session::SessionData new_segment;
                new_segment.session_id = new_session_id;
                new_segment.created_at = new_created_at;
                session_id = new_session_id;
                session_created_at = new_created_at;
                session_file_path = session_store->compute_path(new_segment).string();
                ui_messages.push_back(make_system_message(std::format(
                    "Filo rotated the internal session to keep the working set lean.\nPrevious segment: {}  New segment: {}\nReason: {}\nContext was preserved through an internal handoff, so you can continue normally.",
                    old_session_id,
                    new_session_id,
                    decision.reason.empty() ? std::string("session growth exceeded the efficiency budget.") : decision.reason)));
            }
            screen.PostEvent(ftxui::Event::Custom);
        });

    auto activate_manual_mode = [&]() -> std::string {
        try {
            auto provider = provider_manager.get_provider(manual_provider_name);
            agent->set_provider(provider);
            model_selection_mode = ModelSelectionMode::Manual;
            agent->set_active_model(manual_model_name);
            sync_mcp_sampling_backend(provider, manual_model_name);
            refresh_status_labels();

            std::string message = std::format(
                "Switched to Manual mode: {} ({})",
                manual_provider_name,
                manual_model_name.empty() ? "<provider default>" : manual_model_name);
            const std::string hint = provider_setup_hint(manual_provider_name);
            if (!hint.empty()) {
                message += "\n        " + hint;
            }
            return message;
        } catch (const std::exception& e) {
            return std::format(
                "Failed to activate Manual mode using '{}': {}",
                manual_provider_name,
                e.what());
        }
    };

    auto activate_router_mode = [&]() -> std::string {
        if (!router_available) {
            return "Router mode is unavailable: add at least one policy in config.router.policies.";
        }
        if (!router_provider) {
            return "Router mode is unavailable: router provider was not initialised.";
        }

        model_selection_mode = ModelSelectionMode::Router;
        active_router_policy = router_provider->active_policy();
        agent->set_provider(router_provider);
        agent->set_active_model(router_policy_label());
        sync_mcp_sampling_backend(router_provider, {});
        refresh_status_labels();

        return std::format(
            "Switched to Router mode ({})",
            router_policy_label());
    };

    auto activate_auto_mode = [&]() -> std::string {
        if (!router_available) {
            return "Auto mode is unavailable: configure config.router with strategy=smart and at least one policy.";
        }
        if (!router_provider) {
            return "Auto mode is unavailable: router provider was not initialised.";
        }

        model_selection_mode = ModelSelectionMode::Auto;
        active_router_policy = router_provider->active_policy();
        agent->set_provider(router_provider);
        agent->set_active_model("auto");
        sync_mcp_sampling_backend(router_provider, {});
        refresh_status_labels();

        return std::format(
            "Switched to Auto mode — task-aware smart routing via policy '{}'",
            active_router_policy.empty() ? "<unset>" : active_router_policy);
    };

    // Keep status labels coherent with startup mode.
    refresh_status_labels();

    struct MentionSnapshot {
        std::optional<core::context::ActiveMention> active;
        std::vector<MentionSuggestion> suggestions;
    };

    struct CommandSnapshot {
        std::optional<core::commands::ActiveCommandToken> active;
        std::vector<CommandSuggestion> suggestions;
    };

    auto current_mention_snapshot = [&]() -> MentionSnapshot {
        MentionSnapshot snapshot;
        snapshot.active = core::context::find_active_mention(
            input_text,
            static_cast<std::size_t>(std::max(input_cursor_position, 0)));
        if (snapshot.active.has_value()) {
            snapshot.suggestions = search_mention_index(
                mention_index,
                snapshot.active->raw_path,
                kMaxAutocompleteSuggestions);
        }
        return snapshot;
    };

    auto sync_mention_picker = [&](const MentionSnapshot& snapshot) {
        const std::string_view key = snapshot.active.has_value()
            ? (snapshot.active->quoted ? "q|" : "u|")
            : "";
        const std::string full_key = snapshot.active.has_value()
            ? std::string(key) + snapshot.active->raw_path
            : "";
        mention_picker.sync(full_key, static_cast<int>(snapshot.suggestions.size()));
    };

    auto current_command_snapshot = [&]() -> CommandSnapshot {
        CommandSnapshot snapshot;
        snapshot.active = core::commands::find_active_command(
            input_text,
            static_cast<std::size_t>(std::max(input_cursor_position, 0)));
        if (snapshot.active.has_value()) {
            snapshot.suggestions = search_command_index(
                command_index,
                snapshot.active->token,
                kMaxAutocompleteSuggestions);
        }
        return snapshot;
    };

    auto sync_command_picker = [&](const CommandSnapshot& snapshot) {
        const std::string key = snapshot.active.has_value() ? snapshot.active->token : "";
        command_picker.sync(key, static_cast<int>(snapshot.suggestions.size()));
    };

    auto accept_selected_mention = [&]() -> bool {
        auto snapshot = current_mention_snapshot();
        sync_mention_picker(snapshot);
        if (!snapshot.active.has_value() || snapshot.suggestions.empty()) {
            return false;
        }

        const auto& suggestion =
            snapshot.suggestions[static_cast<std::size_t>(mention_picker.selected)];
        const auto completed = core::context::apply_mention_completion(
            input_text, *snapshot.active, suggestion.insertion_text);
        input_text = completed.text;
        input_cursor_position = static_cast<int>(completed.cursor);
        mention_picker.key.clear();
        mention_picker.selected = 0;
        screen.PostEvent(Event::Custom);
        return true;
    };

    auto accept_selected_command = [&]() -> bool {
        auto snapshot = current_command_snapshot();
        sync_command_picker(snapshot);
        if (!snapshot.active.has_value() || snapshot.suggestions.empty()) {
            return false;
        }

        const auto& suggestion =
            snapshot.suggestions[static_cast<std::size_t>(command_picker.selected)];
        const auto completed = core::commands::apply_command_completion(
            input_text, *snapshot.active, suggestion.insertion_text);
        input_text = completed.text;
        input_cursor_position = static_cast<int>(completed.cursor);
        if (suggestion.accepts_arguments
            && input_cursor_position == static_cast<int>(input_text.size())) {
            input_text.push_back(' ');
            ++input_cursor_position;
        }
        command_picker.key.clear();
        command_picker.selected = 0;
        screen.PostEvent(Event::Custom);
        return true;
    };

    auto has_pending_mention_completion = [&]() -> bool {
        auto snapshot = current_mention_snapshot();
        sync_mention_picker(snapshot);
        if (!snapshot.active.has_value() || snapshot.suggestions.empty()) {
            return false;
        }

        const auto& suggestion =
            snapshot.suggestions[static_cast<std::size_t>(mention_picker.selected)];
        return suggestion.insertion_text != snapshot.active->raw_path;
    };

    auto has_pending_command_completion = [&]() -> bool {
        auto snapshot = current_command_snapshot();
        sync_command_picker(snapshot);
        if (!snapshot.active.has_value() || snapshot.suggestions.empty()) {
            return false;
        }

        const auto& suggestion =
            snapshot.suggestions[static_cast<std::size_t>(command_picker.selected)];
        return suggestion.insertion_text != snapshot.active->token;
    };

    auto describe_models = [&]() -> std::string {
        std::vector<std::string> provider_names;
        provider_names.reserve(config.providers.size());
        for (const auto& [name, provider_cfg] : config.providers) {
            std::string entry = name;
            if (!provider_cfg.model.empty()) {
                entry += " (" + provider_cfg.model + ")";
            }
            provider_names.push_back(std::move(entry));
        }
        std::ranges::sort(provider_names);

        const auto policy_names = router_engine->list_policies();

        auto join_values = [](const std::vector<std::string>& values) {
            std::string out;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i > 0) out += ", ";
                out += values[i];
            }
            return out.empty() ? std::string("<none>") : out;
        };

        const std::string mode =
            model_selection_mode == ModelSelectionMode::Router ? "Router" :
            model_selection_mode == ModelSelectionMode::Auto   ? "Auto"   : "Manual";
        const std::string manual_info = std::format(
            "{} ({})",
            manual_provider_name,
            manual_model_name.empty() ? "<provider default>" : manual_model_name);
        const std::string router_info = router_available
            ? router_policy_label()
            : "<unavailable>";

        return std::format(
            "Mode: {}\n"
            "        Manual: {}\n"
            "        Router: {}\n"
            "        Providers: {}\n"
            "        Policies: {}",
            mode,
            manual_info,
            router_info,
            join_values(provider_names),
            join_values(policy_names));
    };

    auto persist_model_preferences = [&]() -> std::optional<std::string> {
        const std::string mode =
            model_selection_mode == ModelSelectionMode::Router ? "router" :
            model_selection_mode == ModelSelectionMode::Auto   ? "auto"   : "manual";
        std::string error;
        const std::string specific_model =
            (model_selection_mode == ModelSelectionMode::Manual && !manual_model_name.empty())
                ? manual_model_name
                : "";
        if (!config_manager.persist_model_defaults(manual_provider_name, mode, specific_model, &error)) {
            return std::format("Could not persist model defaults: {}", error);
        }
        return std::nullopt;
    };

    auto with_persisted_model_preferences = [&](std::string message) -> std::string {
        if (!message.starts_with("Switched")) {
            return message;
        }
        if (const auto persist_error = persist_model_preferences(); persist_error.has_value()) {
            message += std::format("\n        ⚠  {}", *persist_error);
        }
        return message;
    };

    auto apply_model_selector = [&](std::string_view requested,
                                    bool persist_selection) -> std::string {
        auto trim_ascii = [](std::string_view s) -> std::string_view {
            const auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string_view::npos) return {};
            const auto end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };
        const std::string_view trimmed = trim_ascii(requested);
        if (trimmed.empty()) {
            return describe_models();
        }

        std::string normalized;
        normalized.reserve(trimmed.size());
        for (const char c : trimmed) {
            if (c == '-' || c == '_' || std::isspace(static_cast<unsigned char>(c))) continue;
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }

        auto finalize = [&](std::string message) {
            return persist_selection ? with_persisted_model_preferences(std::move(message))
                                     : message;
        };

        if (normalized == "manual") {
            return finalize(activate_manual_mode());
        }
        if (normalized == "router") {
            return finalize(activate_router_mode());
        }
        if (normalized == "auto") {
            return finalize(activate_auto_mode());
        }

        std::string policy_name(trimmed);
        if (policy_name.starts_with("policy/")) {
            policy_name = policy_name.substr(7);
        }
        if (router_engine->has_policy(policy_name)) {
            if (!router_available) {
                return "Router policy exists but router mode is disabled in configuration.";
            }
            if (!router_engine->set_active_policy(policy_name)) {
                return std::format("Could not activate router policy '{}'.", policy_name);
            }
            active_router_policy = policy_name;
            return finalize(activate_router_mode());
        }

        // Allow explicit provider + model overrides in a single command:
        //   /model claude sonnet
        //   /model claude opus
        //   /model claude claude-opus-4-6
        if (const auto split = trimmed.find_first_of(" \t");
            split != std::string_view::npos) {
            const std::string_view provider_name = trim_ascii(trimmed.substr(0, split));
            const std::string_view model_name = trim_ascii(trimmed.substr(split + 1));
            if (!provider_name.empty() && !model_name.empty()) {
                const auto provider_it = config.providers.find(std::string(provider_name));
                if (provider_it != config.providers.end()) {
                    manual_provider_name = provider_it->first;
                    manual_model_name = std::string(model_name);
                    return finalize(activate_manual_mode());
                }
            }
        }

        const auto it = config.providers.find(std::string(trimmed));
        if (it == config.providers.end()) {
            return std::format("Unknown model selector '{}'.\n        {}",
                               trimmed, describe_models());
        }

        manual_provider_name = it->first;
        manual_model_name = it->second.model;
        return finalize(activate_manual_mode());
    };

    auto switch_provider = [&](std::string_view requested) -> std::string {
        return apply_model_selector(requested, true);
    };

    auto open_model_picker = [&]() -> bool {
        {
            std::lock_guard lock(ui_mutex);
            model_picker_state.active = true;
            model_picker_state.selected =
                model_selection_mode == ModelSelectionMode::Manual ? 0 :
                model_selection_mode == ModelSelectionMode::Auto   ? 2 : 1;
        }
        screen.PostEvent(Event::Custom);
        return true;
    };

    // Build a sorted listing for the local model file browser.
    // Directories come first (sorted), then .gguf files (sorted).
    // ".." is prepended unless dir is a filesystem root.
    auto list_gguf_entries = [](const std::filesystem::path& dir)
        -> std::vector<tui::LocalModelEntry>
    {
        std::vector<tui::LocalModelEntry> result;

        // ".." entry to navigate up
        const auto parent = dir.parent_path();
        if (parent != dir) {
            result.push_back({"..", parent, true});
        }

        std::vector<tui::LocalModelEntry> subdirs, files;
        std::error_code ec;
        for (const auto& fs_entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) { ec.clear(); continue; }
            const auto& p = fs_entry.path();
            const std::string fname = p.filename().string();
            if (fname.empty() || fname.front() == '.') continue; // skip hidden

            std::error_code ec2;
            if (fs_entry.is_directory(ec2) && !ec2) {
                subdirs.push_back({fname + "/", p, true});
            } else if (!fs_entry.is_directory(ec2)) {
                std::string ext = p.extension().string();
                std::ranges::transform(ext, ext.begin(),
                    [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                if (ext == ".gguf") {
                    files.push_back({fname, p, false});
                }
            }
        }
        std::ranges::sort(subdirs, {}, &tui::LocalModelEntry::name);
        std::ranges::sort(files,   {}, &tui::LocalModelEntry::name);
        for (auto& d : subdirs) result.push_back(std::move(d));
        for (auto& f : files)   result.push_back(std::move(f));
        return result;
    };

    auto open_local_model_picker = [&]() {
        std::filesystem::path start_dir;
        // Start at parent of the currently configured local model if there is one.
        {
            const auto it = config.providers.find("local");
            if (it != config.providers.end() && it->second.local && !it->second.local->model_path.empty()) {
                const auto parent = std::filesystem::path(it->second.local->model_path).parent_path();
                std::error_code ec;
                if (std::filesystem::is_directory(parent, ec) && !ec) {
                    start_dir = parent;
                }
            }
        }
        if (start_dir.empty()) {
            const char* home = std::getenv("HOME");
            start_dir = (home && *home) ? std::filesystem::path(home) : std::filesystem::current_path();
        }
        {
            std::lock_guard lock(ui_mutex);
            local_model_picker_state.current_dir = start_dir;
            local_model_picker_state.entries     = list_gguf_entries(start_dir);
            local_model_picker_state.selected    = 0;
            local_model_picker_state.active      = true;
            model_picker_state.active            = false; // close model picker
        }
        screen.PostEvent(Event::Custom);
    };

    auto select_local_model = [&](const std::filesystem::path& gguf_path) -> std::string {
        const std::string model_path_str = gguf_path.string();
        const std::string model_label    = gguf_path.stem().string();

        core::config::ProviderConfig local_cfg;
        local_cfg.api_type = core::config::ApiType::LlamaCppLocal;
        local_cfg.model    = model_label;
        local_cfg.local    = core::config::LocalModelConfig{
            .model_path = model_path_str,
        };

        auto new_provider = core::llm::ProviderFactory::create_provider("local", local_cfg);
        if (!new_provider) {
            return "Local model support is not compiled in. Rebuild with FILO_ENABLE_LLAMACPP=ON.";
        }
        provider_manager.register_provider("local", std::move(new_provider));
        config.providers["local"] = local_cfg;

        manual_provider_name = "local";
        manual_model_name    = model_label;

        std::string result = activate_manual_mode();
        if (result.starts_with("Switched")) {
            std::string persist_error;
            if (!config_manager.persist_local_provider(model_path_str, model_label, &persist_error)) {
                result += std::format("\n        \xe2\x9a\xa0  Could not persist: {}", persist_error);
            }
        }
        return result;
    };

    auto managed_setting_value = [&](const core::config::ManagedSettings& settings,
                                     core::config::ManagedSettingKey key)
        -> std::optional<std::string> {
        switch (key) {
            case core::config::ManagedSettingKey::DefaultMode:
                return settings.default_mode;
            case core::config::ManagedSettingKey::DefaultApprovalMode:
                return settings.default_approval_mode;
            case core::config::ManagedSettingKey::DefaultRouterPolicy:
                return settings.default_router_policy;
            case core::config::ManagedSettingKey::UiBanner:
                return settings.ui_banner;
            case core::config::ManagedSettingKey::UiFooter:
                return settings.ui_footer;
            case core::config::ManagedSettingKey::UiModelInfo:
                return settings.ui_model_info;
            case core::config::ManagedSettingKey::UiContextUsage:
                return settings.ui_context_usage;
            case core::config::ManagedSettingKey::UiTimestamps:
                return settings.ui_timestamps;
            case core::config::ManagedSettingKey::UiSpinner:
                return settings.ui_spinner;
            case core::config::ManagedSettingKey::AutoCompactThreshold:
                return settings.auto_compact_threshold;
        }
        return std::nullopt;
    };

    auto effective_setting_value = [&](core::config::ManagedSettingKey key) -> std::string {
        const auto& effective = config_manager.get_config();
        switch (key) {
            case core::config::ManagedSettingKey::DefaultMode:
                return effective.default_mode;
            case core::config::ManagedSettingKey::DefaultApprovalMode:
                return effective.default_approval_mode.empty()
                    ? std::string("prompt")
                    : effective.default_approval_mode;
            case core::config::ManagedSettingKey::DefaultRouterPolicy:
                return effective.router.default_policy;
            case core::config::ManagedSettingKey::UiBanner:
                return effective.ui_banner;
            case core::config::ManagedSettingKey::UiFooter:
                return effective.ui_footer;
            case core::config::ManagedSettingKey::UiModelInfo:
                return effective.ui_model_info;
            case core::config::ManagedSettingKey::UiContextUsage:
                return effective.ui_context_usage;
            case core::config::ManagedSettingKey::UiTimestamps:
                return effective.ui_timestamps;
            case core::config::ManagedSettingKey::UiSpinner:
                return effective.ui_spinner;
            case core::config::ManagedSettingKey::AutoCompactThreshold:
                return std::to_string(effective.auto_compact_threshold);
        }
        return {};
    };

    auto settings_scope_label = [](core::config::SettingsScope scope) {
        return scope == core::config::SettingsScope::User
            ? std::string("User")
            : std::string("Workspace");
    };

    auto setting_choice_label = [&](const SettingsDefinition& definition,
                                    std::string_view value) -> std::string {
        for (const auto& choice : definition.choices) {
            if (choice.value == value) {
                return choice.label;
            }
        }
        return std::string(value);
    };

    auto apply_effective_settings = [&](const core::config::AppConfig& before,
                                        const core::config::AppConfig& after)
        -> std::optional<std::string> {
        config.default_mode = after.default_mode;
        config.default_approval_mode = after.default_approval_mode;
        config.router.default_policy = after.router.default_policy;
        config.ui_banner = after.ui_banner;
        config.ui_footer = after.ui_footer;
        config.ui_model_info = after.ui_model_info;
        config.ui_context_usage = after.ui_context_usage;
        config.ui_timestamps = after.ui_timestamps;
        config.ui_spinner = after.ui_spinner;

        if (before.default_mode != after.default_mode) {
            const std::string desired_mode = normalize_mode(after.default_mode);
            for (std::size_t i = 0; i < modes.size(); ++i) {
                if (modes[i].first == desired_mode) {
                    current_mode_idx = static_cast<int>(i);
                    agent->set_mode(modes[current_mode_idx].first);
                    break;
                }
            }
        }

        if (before.default_approval_mode != after.default_approval_mode) {
            set_yolo_mode_enabled(parse_approval_mode(after.default_approval_mode) == ApprovalMode::Yolo);
        }

        if (before.router.default_policy != after.router.default_policy
            && !after.router.default_policy.empty()) {
            if (!router_engine->set_active_policy(after.router.default_policy)) {
                return std::format(
                    "Router policy '{}' is not available in this session.",
                    after.router.default_policy);
            }
            active_router_policy = router_engine->active_policy();
            refresh_status_labels();
        }

        ui_show_banner = visibility_setting_enabled(after.ui_banner, true);
        ui_show_footer = visibility_setting_enabled(after.ui_footer, true);
        ui_show_model_info = visibility_setting_enabled(after.ui_model_info, true);
        ui_show_context_usage = visibility_setting_enabled(after.ui_context_usage, true);
        ui_show_timestamps = visibility_setting_enabled(after.ui_timestamps, true);
        ui_show_spinner = visibility_setting_enabled(after.ui_spinner, true);
        animation_cv.notify_one();

        return std::nullopt;
    };

    auto persist_settings_value = [&](int definition_index,
                                      std::optional<std::string> value) {
        if (definition_index < 0
            || definition_index >= static_cast<int>(settings_definitions.size())) {
            return;
        }

        const auto scope = settings_panel_state.scope;
        const auto& definition =
            settings_definitions[static_cast<std::size_t>(definition_index)];
        const auto before = config_manager.get_config();
        std::string error;
        if (!config_manager.persist_managed_setting(scope,
                                                    definition.key,
                                                    value,
                                                    settings_working_dir,
                                                    &error)) {
            std::lock_guard lock(ui_mutex);
            settings_panel_state.status_message = std::format(
                "Could not save {} setting: {}",
                settings_scope_label(scope),
                error);
            screen.PostEvent(Event::Custom);
            return;
        }

        const auto after = config_manager.get_config();
        std::string status = std::format(
            "{} setting saved: {} -> {}",
            settings_scope_label(scope),
            definition.label,
            value.has_value()
                ? setting_choice_label(definition, *value)
                : std::string("inherit"));
        if (const auto live_apply_warning = apply_effective_settings(before, after);
            live_apply_warning.has_value()) {
            status += std::format(" ({})", *live_apply_warning);
        }

        {
            std::lock_guard lock(ui_mutex);
            settings_panel_state.status_message = std::move(status);
        }
        screen.PostEvent(Event::Custom);
    };

    auto open_settings_picker = [&]() -> bool {
        if (settings_definitions.empty()) {
            return false;
        }
        {
            std::lock_guard lock(ui_mutex);
            settings_panel_state.active = true;
            settings_panel_state.selected = std::clamp(
                settings_panel_state.selected,
                0,
                static_cast<int>(settings_definitions.size()) - 1);
            settings_panel_state.status_message =
                "Edit scoped app preferences here. Use /model for provider and session routing choices.";
        }
        screen.PostEvent(Event::Custom);
        return true;
    };

    auto settings_status = [&]() -> std::string {
        const auto& effective = config_manager.get_config();
        const auto user_path = config_manager.get_settings_path(
            core::config::SettingsScope::User,
            settings_working_dir);
        const auto workspace_path = config_manager.get_settings_path(
            core::config::SettingsScope::Workspace,
            settings_working_dir);

        return std::format(
            "Effective settings\n"
            "Start mode: {}\n"
            "Approval mode: {}\n"
            "Default router policy: {}\n"
            "Startup banner: {}\n"
            "Footer: {}\n"
            "Footer model badge: {}\n"
            "Footer context meter: {}\n"
            "Message timestamps: {}\n"
            "Activity spinner: {}\n"
            "User settings file: {}\n"
            "Workspace settings file: {}\n"
            "Model defaults are managed separately via `/model`.",
            effective.default_mode,
            effective.default_approval_mode.empty() ? std::string("prompt")
                                                    : effective.default_approval_mode,
            effective.router.default_policy.empty() ? std::string("<unset>")
                                                    : effective.router.default_policy,
            effective.ui_banner,
            effective.ui_footer,
            effective.ui_model_info,
            effective.ui_context_usage,
            effective.ui_timestamps,
            effective.ui_spinner,
            user_path.string(),
            workspace_path.string());
    };

    // ── Permission gate wiring ───────────────────────────────────────────────

    // Notify fn: called from the worker thread when a permission is requested.
    core::agent::PermissionGate::get_instance().set_notify_fn([&]() {
        screen.PostEvent(Event::Custom);  // wake up the render loop
    });

    // Permission function given to the Agent.
    auto permission_fn = [&](std::string_view tool_name,
                             std::string_view args) -> bool {
        bool yolo_enabled = false;
        {
            std::lock_guard lock(ui_mutex);
            yolo_enabled = approval_mode == ApprovalMode::Yolo;
        }
        if (yolo_enabled) {
            {
                std::lock_guard lock(ui_mutex);
                // Keep YOLO approvals inside the current tool card instead of
                // flooding the system-history stream.
                for (auto msg_it = ui_messages.rbegin(); msg_it != ui_messages.rend(); ++msg_it) {
                    if (msg_it->type != MessageType::Assistant || !msg_it->pending) {
                        continue;
                    }

                    bool marked = false;
                    for (auto tool_it = msg_it->tools.rbegin();
                         tool_it != msg_it->tools.rend();
                         ++tool_it) {
                        if (tool_it->name == tool_name
                            && tool_it->args == args
                            && !tool_it->auto_approved) {
                            tool_it->auto_approved = true;
                            marked = true;
                            break;
                        }
                    }
                    if (marked) {
                        break;
                    }
                }
            }
            screen.PostEvent(Event::Custom);
            return true;
        }

        // Check session allow-list ("don't ask again for this").
        const std::string allow_key = make_allow_key(tool_name, args);
        bool is_allowed = false;
        {
            std::lock_guard lock(ui_mutex);
            is_allowed = session_allowed.count(allow_key) > 0;
        }
        if (is_allowed) {
            // Auto-approved via session allow-list - no status message needed
            // (tool activity card already shows status progression)
            return true;
        }

        // Serialize prompt-based approvals across concurrent agent threads to
        // prevent multiple permission overlays from racing and overwriting state.
        {
            std::unique_lock slot_lock(permission_prompt_mutex);
            permission_prompt_cv.wait(slot_lock, [&]() {
                return !permission_prompt_in_flight;
            });
            permission_prompt_in_flight = true;
        }
        struct PermissionPromptSlotGuard {
            std::mutex* mutex = nullptr;
            std::condition_variable* cv = nullptr;
            bool* in_flight = nullptr;
            ~PermissionPromptSlotGuard() {
                if (mutex == nullptr || cv == nullptr || in_flight == nullptr) {
                    return;
                }
                {
                    std::lock_guard lock(*mutex);
                    *in_flight = false;
                }
                cv->notify_one();
            }
        } slot_guard{
            .mutex = &permission_prompt_mutex,
            .cv = &permission_prompt_cv,
            .in_flight = &permission_prompt_in_flight
        };

        // Block this worker thread on a promise resolved by the TUI.
        auto prom = std::make_shared<std::promise<bool>>();
        auto fut  = prom->get_future();

        {
            std::lock_guard lock(ui_mutex);
            perm_state.active       = true;
            perm_state.tool_name    = std::string(tool_name);
            // Keep full arguments so the permission panel can parse/render
            // a structured, user-friendly preview instead of truncated JSON.
            perm_state.args_preview = std::string(args);
            perm_state.diff_preview = build_tool_diff_preview(
                tool_name,
                args,
                kPermissionDiffPreviewMaxLines);
            perm_state.selected     = 0;
            perm_state.allow_key    = allow_key;
            perm_state.allow_label  = make_allow_label(tool_name, args);
            perm_state.promise      = prom;
        }
        screen.PostEvent(Event::Custom);
        return fut.get();
    };

    // ── AskUserQuestion callback ─────────────────────────────────────────────
    ask_user_tool->setQuestionCallback([&](core::tools::QuestionRequest request) {
        // Convert core::tools types to tui types
        std::vector<tui::QuestionDialogItem> dialog_questions;
        for (const auto& q : request.questions) {
            tui::QuestionDialogItem item;
            item.question = q.question;
            item.header = q.header;
            item.multi_select = q.multi_select;
            item.body = q.body;
            for (const auto& opt : q.options) {
                item.options.push_back({opt.label, opt.description});
            }
            dialog_questions.push_back(std::move(item));
        }
        
        {
            std::lock_guard lock(ui_mutex);
            question_dialog_state.active = true;
            question_dialog_state.questions = std::move(dialog_questions);
            question_dialog_state.current_question_index = 0;
            question_dialog_state.selected_option = 0;
            question_dialog_state.answers.clear();
            question_dialog_state.multi_selected.clear();
            question_dialog_state.show_other_input = false;
            question_dialog_state.other_input_text.clear();
            question_dialog_state.promise = request.promise;
        }
        screen.PostEvent(Event::Custom);
        
        // Wait for the result (UI will resolve the promise)
    });

    agent->set_permission_fn(permission_fn);

    // ── Loop-break callback ──────────────────────────────────────────────────
    agent->set_loop_break_fn([&](int rounds) {
        append_history(std::format(
            "\n\xe2\x9a\xa0  Agent paused after {} consecutive tool failures"
            " \xe2\x80\x94 provide guidance or /clear to start fresh.\n", rounds));
    });

    auto update_assistant_message = [&](std::size_t assistant_index, auto&& updater) {
        {
            std::lock_guard lock(ui_mutex);
            if (assistant_index >= ui_messages.size()) {
                return;
            }

            auto& message = ui_messages[assistant_index];
            if (message.type != MessageType::Assistant) {
                return;
            }

            updater(message);
        }
        animation_cv.notify_one();
        screen.PostEvent(Event::Custom);
    };

    // ── Agent turn submission ─────────────────────────────────────────────────
    // Extracted so that SkillCommand can inject an expanded prompt as a full
    // agent turn (with user message card + tool cards) via send_user_message_fn.
    auto submit_agent_turn = [&](const std::string& text) {
        std::string timestamp = current_time_str();
        std::size_t assistant_index = 0;
        std::string assistant_message_id;
        {
            std::lock_guard lock(ui_mutex);
            ui_messages.push_back(make_user_message(text, timestamp));
            ui_messages.push_back(make_assistant_message("", "", true));
            assistant_index = ui_messages.size() - 1;
            assistant_message_id = ui_messages.back().id;
        }
        turn_activity_timers.start(assistant_message_id);
        animation_cv.notify_one();

        const auto expanded_prompt =
            core::context::expand_prompt(text, std::filesystem::current_path());

        core::llm::Message user_message;
        user_message.role = "user";
        user_message.content = expanded_prompt.display_text;
        if (core::llm::message_has_image_input(expanded_prompt.content_parts)) {
            user_message.content_parts = expanded_prompt.content_parts;
        }

        std::thread([user_message = std::move(user_message),
                     agent,
                     assistant_index,
                     session_store,
                     &ui_mutex,
                     &session_id,
                     &session_created_at,
                     &active_provider_name,
                     &active_model_name,
                     &update_assistant_message,
                     &assistant_turn_active,
                     &turn_activity_timers,
                     &stream_chunk_timing_mutex,
                     &stream_chunk_last_at,
                     assistant_message_id = std::move(assistant_message_id)]() {
            agent->send_message(user_message,
                [assistant_index,
                 &update_assistant_message,
                 &assistant_turn_active,
                 &stream_chunk_timing_mutex,
                 &stream_chunk_last_at](const std::string& chunk) {
                    const auto now = std::chrono::steady_clock::now();
                    bool resumed_after_pause = false;
                    {
                        std::lock_guard lock(stream_chunk_timing_mutex);
                        if (const auto it = stream_chunk_last_at.find(assistant_index);
                            it != stream_chunk_last_at.end()) {
                            resumed_after_pause =
                                (now - it->second) >= kStreamChunkPauseBreakThreshold;
                            it->second = now;
                        } else {
                            stream_chunk_last_at.emplace(assistant_index, now);
                        }
                    }

                    update_assistant_message(assistant_index, [&](UiMessage& message) {
                        if (assistant_turn_active.load(std::memory_order_relaxed)) {
                            message.pending = true;
                        }
                        if (message.thinking) {
                            message.thinking = false;
                            message.show_lightbulb = true;
                        }

                        if (message.text.empty()) {
                            message.text = chunk;
                        } else {
                            if (resumed_after_pause) {
                                if (!ends_with_line_break(message.text)) {
                                    message.text.push_back('\n');
                                }
                                message.text += kStreamChunkResumeMarker;
                                if (!starts_with_line_break(chunk)) {
                                    message.text.push_back('\n');
                                }
                            }
                            message.text += chunk;
                        }
                    });
                },
                [](const std::string&, const std::string&) {},
                // done_callback — update UI and auto-save session.
                [assistant_index, agent, session_store,
                 &ui_mutex, &session_id, &session_created_at,
                 &active_provider_name, &active_model_name,
                 &update_assistant_message,
                 &assistant_turn_active,
                 &turn_activity_timers,
                 &stream_chunk_timing_mutex,
                 &stream_chunk_last_at,
                 assistant_message_id]() {
                    {
                        std::lock_guard lock(stream_chunk_timing_mutex);
                        stream_chunk_last_at.erase(assistant_index);
                    }
                    turn_activity_timers.stop(assistant_message_id);
                    assistant_turn_active.store(false, std::memory_order_relaxed);
                    const bool was_stopped = agent->is_stop_requested();
                    update_assistant_message(assistant_index, [&](UiMessage& message) {
                        message.pending = false;
                        message.thinking = false;
                        message.stopped = was_stopped;
                        if (!message.text.empty() || !message.tools.empty()) {
                            message.show_lightbulb = true;
                        }
                    });
                    // Snapshot agent state now (before the save thread runs) to
                    // avoid a race where the user types a new message and
                    // send_message() appends it to history_ before get_history()
                    // is called inside the detached thread.
                    auto snap_messages = agent->get_history();
                    auto snap_mode     = agent->get_mode();
                    auto snap_context  = agent->get_context_summary();
                    std::string sid;
                    std::string created_at;
                    std::string provider_name;
                    std::string model_name;
                    {
                        std::lock_guard lock(ui_mutex);
                        sid = session_id;
                        created_at = session_created_at;
                        provider_name = active_provider_name;
                        model_name = active_model_name;
                    }

                    // Auto-save the session (detached; file I/O stays off the hot path).
                    std::thread([session_store, sid, created_at,
                                 provider_name, model_name,
                                 snap_messages = std::move(snap_messages),
                                 snap_mode, snap_context]() {
                        core::session::SessionData data;
                        data.session_id        = sid;
                        data.created_at        = created_at;
                        data.last_active_at    = core::session::SessionStore::now_iso8601();
                        data.working_dir       = std::filesystem::current_path().string();
                        data.provider          = provider_name;
                        data.model             = model_name;
                        data.mode              = snap_mode;
                        data.context_summary   = snap_context;
                        data.messages          = snap_messages;

                        const auto& budget = core::budget::BudgetTracker::get_instance();
                        const auto  total  = budget.session_total();
                        data.stats.prompt_tokens     = total.prompt_tokens;
                        data.stats.completion_tokens = total.completion_tokens;
                        data.stats.cost_usd          = budget.session_cost_usd();

                        const auto snap = core::session::SessionStats::get_instance().snapshot();
                        data.stats.turn_count         = snap.turn_count;
                        data.stats.tool_calls_total   = snap.tool_calls_total;
                        data.stats.tool_calls_success = snap.tool_calls_success;
                        data.handoff_summary          = core::session::build_handoff_summary(data);

                        session_store->save(data);
                    }).detach();
                },
                core::agent::Agent::TurnCallbacks{
                    .on_step_begin = [assistant_index, &update_assistant_message, &assistant_turn_active]() {
                        assistant_turn_active.store(true, std::memory_order_relaxed);
                        update_assistant_message(assistant_index, [&](UiMessage& message) {
                            message.pending = true;
                            message.thinking = true;
                        });
                    },
                    .on_tool_start =
                        [assistant_index, &update_assistant_message](
                            const core::llm::ToolCall& tool_call) {
                            update_assistant_message(assistant_index, [&](UiMessage& message) {
                                message.pending = true;
                                message.thinking = false;
                                message.show_lightbulb = true;
                                message.tools.push_back(make_tool_activity(
                                    tool_call.id,
                                    tool_call.function.name,
                                    tool_call.function.arguments,
                                    summarize_tool_arguments(
                                        tool_call.function.name,
                                        tool_call.function.arguments)));
                                message.tools.back().status = ToolActivity::Status::Executing;
                            });
                        },
                    .on_tool_finish =
                        [assistant_index, &update_assistant_message](
                            const core::llm::ToolCall& tool_call,
                            const core::llm::Message& result) {
                            update_assistant_message(assistant_index, [&](UiMessage& message) {
                                auto* tool = find_tool_activity(message, tool_call.id);

                                if (tool == nullptr) {
                                    message.tools.push_back(make_tool_activity(
                                        tool_call.id,
                                        tool_call.function.name,
                                        tool_call.function.arguments,
                                        summarize_tool_arguments(
                                            tool_call.function.name,
                                            tool_call.function.arguments)));
                                    tool = &message.tools.back();
                                }

                                apply_tool_result(*tool, result.content);

                                // Check if all tools are now done but we're still waiting
                                // for the model's response. Re-activate thinking indicator
                                // so the user knows the model is processing.
                                bool has_pending_tools = false;
                                for (const auto& t : message.tools) {
                                    if (t.status == ToolActivity::Status::Pending ||
                                        t.status == ToolActivity::Status::Executing) {
                                        has_pending_tools = true;
                                        break;
                                    }
                                }
                                if (!has_pending_tools && message.pending) {
                                    message.thinking = true;
                                    message.show_lightbulb = true;
                                }
                            });
                        }
                }
            );
        }).detach();
    };

    // ── Input component ──────────────────────────────────────────────────────
    auto input_option = InputOption();
    input_option.transform = [](InputState state) {
        if (state.is_placeholder) {
            return state.element | color(Color::GrayDark);
        }
        return state.element | color(tui::ColorYellowBright);
    };
    input_option.multiline = false;
    input_option.cursor_position = &input_cursor_position;

    input_option.on_enter = [&]() {
        // If permission overlay is active, ignore normal input submission
        {
            std::lock_guard lock(ui_mutex);
            if (perm_state.active
                || question_dialog_state.active
                || model_picker_state.active
                || provider_picker_state.active
                || local_model_picker_state.active
                || conversation_search_state.active
                || settings_panel_state.active) return;
        }
        if (input_text.empty()) return;
        std::string text = input_text;
        input_text.clear();
        prompt_history.save(text);

        core::commands::CommandContext ctx{
            .text             = text,
            .clear_input_fn   = []() {},
            .append_history_fn = append_history,
            .agent            = agent,
            .clear_screen_fn  = clear_screen,
            .quit_fn          = screen.ExitLoopClosure(),
            .model_status_fn  = describe_models,
            .switch_model_fn  = switch_provider,
            .open_model_picker_fn = open_model_picker,
            .open_settings_picker_fn = open_settings_picker,
            .open_sessions_picker_fn = open_sessions_picker,
            .resume_session_fn = [&](std::string_view id_or_idx) {
                auto data_opt = id_or_idx.empty()
                    ? session_store->load_most_recent()
                    : session_store->load(id_or_idx);

                if (!data_opt.has_value()) {
                    append_history(std::format(
                        "\n\xe2\x9c\x97  Session '{}' not found. Use /sessions to list available sessions.\n",
                        id_or_idx.empty() ? std::string("most recent") : std::string(id_or_idx)));
                    return;
                }
                resume_session(*data_opt);
            },
            .open_provider_picker_fn = [&](std::vector<std::string> providers,
                                           std::function<void(std::optional<std::string>)> on_select) {
                std::lock_guard lock(ui_mutex);
                provider_picker_state.active   = true;
                provider_picker_state.selected  = 0;
                provider_picker_state.providers = std::move(providers);
                provider_picker_state.on_select = std::move(on_select);
                screen.PostEvent(Event::Custom);
            },
            .settings_status_fn = settings_status,
            .yolo_mode_enabled_fn = is_yolo_mode_enabled,
            .set_yolo_mode_enabled_fn = set_yolo_mode_enabled,
            .fork_session_fn = fork_session,
            .suspend_tui_fn   = [&](std::function<void()> task) {
                // Suspends the TUI loop and restores standard I/O for the duration of the task.
                auto closure = screen.WithRestoredIO(task);
                closure();
            },
            .latest_assistant_output_fn = [&]() {
                std::lock_guard lock(ui_mutex);
                for (auto it = ui_messages.rbegin(); it != ui_messages.rend(); ++it) {
                    if (it->type != MessageType::Assistant || it->pending) {
                        continue;
                    }

                    if (!it->text.empty()) {
                        return it->text;
                    }
                }
                return std::string{};
            },
            .history_store_fn = history_store,
            .clear_history_fn = [&]() {
                prompt_history.clear();
            },
            // Skill commands (SkillCommand) use this to inject an expanded
            // prompt as a full agent turn with TUI message cards and tool
            // activity panels, identical to normal user input.
            .send_user_message_fn = submit_agent_turn,
        };

        if (cmd_executor.try_execute(text, ctx)) return;

        submit_agent_turn(text);
    };

    auto open_external_editor = [&]() -> bool {
        {
            std::lock_guard lock(ui_mutex);
            if (perm_state.active
                || question_dialog_state.active
                || model_picker_state.active
                || provider_picker_state.active
                || local_model_picker_state.active
                || conversation_search_state.active
                || settings_panel_state.active) {
                return true;
            }
        }

        std::array<char, 32> temp_dir_template{};
        constexpr std::string_view kTemplate = "/tmp/filo-edit-XXXXXX";
        std::ranges::copy(kTemplate, temp_dir_template.begin());

        char* temp_dir = ::mkdtemp(temp_dir_template.data());
        if (temp_dir == nullptr) {
            append_history("\n\xe2\x9c\x97  Could not create a temporary buffer file for external editor.\n");
            return true;
        }

        const std::filesystem::path temp_dir_path(temp_dir);
        const std::filesystem::path buffer_path = temp_dir_path / "buffer.txt";

        auto cleanup_temp_files = [&]() {
            std::error_code ec;
            std::filesystem::remove(buffer_path, ec);
            std::filesystem::remove(temp_dir_path, ec);
        };

        {
            std::ofstream out(buffer_path, std::ios::binary | std::ios::trunc);
            if (!out) {
                cleanup_temp_files();
                append_history("\n\xe2\x9c\x97  Could not write temporary editor buffer.\n");
                return true;
            }
            out << input_text;
        }

        const std::string command = build_external_editor_command(buffer_path.string());
        int command_status = -1;
        screen.WithRestoredIO([&]() {
            command_status = std::system(command.c_str());
        })();

        const int exit_code = normalize_exit_code(command_status);
        if (exit_code != 0) {
            cleanup_temp_files();
            append_history(std::format(
                "\n\xe2\x9c\x97  External editor exited with status {}.\n",
                exit_code));
            return true;
        }

        std::ifstream in(buffer_path, std::ios::binary);
        if (!in) {
            cleanup_temp_files();
            append_history("\n\xe2\x9c\x97  Could not reload edited text from temporary buffer.\n");
            return true;
        }

        std::string edited_text(
            (std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());
        input_text = normalize_newlines(std::move(edited_text));
        input_cursor_position = static_cast<int>(input_text.size());

        cleanup_temp_files();
        screen.PostEvent(Event::Custom);
        return true;
    };

    auto execute_selected_command = [&]() -> bool {
        auto snapshot = current_command_snapshot();
        sync_command_picker(snapshot);
        if (!snapshot.active.has_value() || snapshot.suggestions.empty()) {
            return false;
        }

        const auto& suggestion =
            snapshot.suggestions[static_cast<std::size_t>(command_picker.selected)];
        const auto completed = core::commands::apply_command_completion(
            input_text, *snapshot.active, suggestion.insertion_text);
        input_text = completed.text;
        input_cursor_position = static_cast<int>(completed.cursor);
        command_picker.key.clear();
        command_picker.selected = 0;
        input_option.on_enter();
        return true;
    };

    auto ctrl_d_warning_active = [&]() -> bool {
        if (ctrl_d_press_count <= 0) {
            return false;
        }
        if (std::chrono::steady_clock::now() > ctrl_d_deadline) {
            ctrl_d_press_count = 0;
            ctrl_d_deadline = std::chrono::steady_clock::time_point::min();
            return false;
        }
        return true;
    };

    auto reset_ctrl_d_warning = [&]() {
        ctrl_d_press_count = 0;
        ctrl_d_deadline = std::chrono::steady_clock::time_point::min();
    };

    auto input_component = PromptInput(&input_text,
        "Ask anything  (F2 mode  @ file context  / command)",
        input_option);

    auto history_component = Make<HistoryComponent>(
        [&]() {
            std::vector<UiMessage> messages;
            {
                std::lock_guard lock(ui_mutex);
                messages = ui_messages;
            }

            if (const auto activity = active_prompt_activity(messages);
                activity.has_value()) {
                const auto elapsed = turn_activity_timers
                    .elapsed(activity->message_id)
                    .value_or(std::chrono::seconds::zero());
                for (auto& message : messages) {
                    if (message.id == activity->message_id) {
                        message.activity_elapsed = format_elapsed_compact(elapsed);
                        break;
                    }
                }
            }

            return messages;
        },
        std::cref(animation_tick),
        [&]() {
            return ConversationRenderOptions{
                .show_timestamps = ui_show_timestamps,
                .show_spinner    = ui_show_spinner,
                .expand_tool_results = tool_output_expanded,
                .tool_result_preview_max_lines = kToolResultPreviewMaxLines,
                // scroll_pos set by component
            };
        }
    );

    // ── Event handling ───────────────────────────────────────────────────────
    auto component = CatchEvent(input_component, [&](Event event) {
        // ── Permission overlay keypresses ────────────────────────────────
        // Resolve the promise OUTSIDE the lock to avoid locking issues when
        // the worker thread wakes up and might try to re-acquire ui_mutex.
        std::shared_ptr<std::promise<bool>> perm_prom;
        std::optional<bool> perm_answer;
        bool enable_yolo_from_permission = false;
        bool enable_always_allow = false;
        std::string always_allow_key;
        std::string always_allow_label;
        bool perm_was_active = false;
        {
            std::lock_guard lock(ui_mutex);
            if (perm_state.active) {
                perm_was_active = true;
                // ── Option indices:
                //   0 = Yes, once
                //   1 = Yes, don't ask again for this
                //   2 = Yes, enable YOLO
                //   3 = No, suggest something
                if (event == Event::ArrowUp) {
                    perm_state.selected = (perm_state.selected + 3) % 4;
                } else if (event == Event::ArrowDown) {
                    perm_state.selected = (perm_state.selected + 1) % 4;
                } else if (event == Event::Character('1')
                           || event == Event::Character('y')
                           || event == Event::Character('Y')) {
                    // Yes, once
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = true;
                    perm_state.active = false;
                } else if (event == Event::Character('2')
                           || event == Event::Character('a')
                           || event == Event::Character('A')) {
                    // Yes, don't ask again for this
                    always_allow_key   = perm_state.allow_key;
                    always_allow_label = perm_state.allow_label;
                    enable_always_allow = true;
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = true;
                    perm_state.active = false;
                } else if (event == Event::Character('3')
                           || is_ctrl_y_event(event)) {
                    // Yes, enable YOLO
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = true;
                    enable_yolo_from_permission = true;
                    perm_state.active = false;
                } else if (event == Event::Return) {
                    const int sel = perm_state.selected;
                    always_allow_key   = perm_state.allow_key;
                    always_allow_label = perm_state.allow_label;
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = sel != 3;
                    enable_always_allow         = sel == 1;
                    enable_yolo_from_permission = sel == 2;
                    perm_state.active = false;
                } else if (event == Event::Character('4')
                           || event == Event::Character('n')
                           || event == Event::Character('N')
                           || event == Event::Escape) {
                    // No, suggest something
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = false;
                    perm_state.active = false;
                }
                // else: absorb all other keys while overlay is active
            }
        }
        if (perm_was_active) {
            // Resolve the promise FIRST (before re-acquiring ui_mutex) to avoid
            // a race where the worker thread wakes up and tries to re-enter the
            // permission check while we're still holding the lock for session_allowed.
            if (perm_prom && perm_answer.has_value()) {
                perm_prom->set_value(*perm_answer);
            }
            if (enable_always_allow && !always_allow_key.empty()) {
                {
                    std::lock_guard lock(ui_mutex);
                    session_allowed.insert(always_allow_key);
                }
                // Session allow-list updated - no status message needed
                (void)always_allow_label;
            }
            if (enable_yolo_from_permission) {
                set_yolo_mode_enabled(true);
                append_history(
                    "\n\xe2\x9a\xa0  Approval mode set to YOLO: sensitive tools will auto-run.\n");
            }
            if (perm_prom && perm_answer.has_value() && !*perm_answer) {
                append_history(
                    "\n\xe2\x84\xb9  Tool call rejected. Share a suggestion and Filo will adapt.\n");
            }
            return true;
        }

        // ── Question dialog input handling ────────────────────────────────────────
        bool question_dialog_was_active = false;
        bool question_dialog_submitted = false;
        bool question_dialog_dismissed = false;
        std::shared_ptr<std::promise<std::optional<std::vector<std::pair<std::string, std::string>>>>> question_promise;
        std::optional<std::vector<std::pair<std::string, std::string>>> question_result;
        {
            std::lock_guard lock(ui_mutex);
            if (question_dialog_state.active) {
                question_dialog_was_active = true;
                auto& state = question_dialog_state;
                const auto& current_q = state.questions[state.current_question_index];
                const int option_count = static_cast<int>(current_q.options.size());
                
                if (event == Event::ArrowUp) {
                    state.selected_option = (state.selected_option + option_count - 1) % option_count;
                } else if (event == Event::ArrowDown) {
                    state.selected_option = (state.selected_option + 1) % option_count;
                } else if (event == Event::Character(' ') && current_q.multi_select) {
                    // Toggle selection for multi-select
                    auto it = std::find(state.multi_selected.begin(), state.multi_selected.end(), state.selected_option);
                    if (it != state.multi_selected.end()) {
                        state.multi_selected.erase(it);
                    } else {
                        state.multi_selected.push_back(state.selected_option);
                    }
                } else if (event == Event::Return) {
                    if (current_q.options[state.selected_option].label == "Other") {
                        // Show Other input (not implemented inline - just dismiss for now)
                        question_dialog_dismissed = true;
                        state.active = false;
                        question_promise = std::move(state.promise);
                    } else {
                        // Submit answer
                        state.answers.push_back({current_q.question, current_q.options[state.selected_option].label});
                        
                        // Move to next question or finish
                        if (state.current_question_index + 1 < static_cast<int>(state.questions.size())) {
                            state.current_question_index++;
                            state.selected_option = 0;
                        } else {
                            // All questions answered
                            question_dialog_submitted = true;
                            state.active = false;
                            question_result = state.answers;
                            question_promise = std::move(state.promise);
                        }
                    }
                } else if (event == Event::Escape) {
                    question_dialog_dismissed = true;
                    state.active = false;
                    question_promise = std::move(state.promise);
                } else {
                    // Number keys 1-5 for quick select
                    for (int n = 1; n <= std::min(option_count, 5); ++n) {
                        if (event == Event::Character(static_cast<char>('0' + n))) {
                            state.selected_option = n - 1;
                            if (!current_q.multi_select) {
                                // Auto-submit on number press for single-select
                                if (current_q.options[state.selected_option].label == "Other") {
                                    question_dialog_dismissed = true;
                                    state.active = false;
                                    question_promise = std::move(state.promise);
                                } else {
                                    state.answers.push_back({current_q.question, current_q.options[state.selected_option].label});
                                    if (state.current_question_index + 1 < static_cast<int>(state.questions.size())) {
                                        state.current_question_index++;
                                        state.selected_option = 0;
                                    } else {
                                        question_dialog_submitted = true;
                                        state.active = false;
                                        question_result = state.answers;
                                        question_promise = std::move(state.promise);
                                    }
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        if (question_dialog_was_active) {
            if (question_promise) {
                if (question_dialog_submitted && question_result.has_value()) {
                    question_promise->set_value(*question_result);
                } else if (question_dialog_dismissed) {
                    question_promise->set_value(std::nullopt);
                }
            }
            return true;
        }

        bool settings_panel_was_active = false;
        std::optional<int> settings_cycle_direction;
        std::optional<int> settings_selected_index;
        std::optional<core::config::SettingsScope> settings_scope;
        bool settings_reset = false;
        {
            std::lock_guard lock(ui_mutex);
            if (settings_panel_state.active) {
                settings_panel_was_active = true;
                if (event == Event::ArrowUp) {
                    settings_panel_state.selected =
                        (settings_panel_state.selected
                         + static_cast<int>(settings_definitions.size()) - 1)
                        % static_cast<int>(settings_definitions.size());
                } else if (event == Event::ArrowDown) {
                    settings_panel_state.selected =
                        (settings_panel_state.selected + 1)
                        % static_cast<int>(settings_definitions.size());
                } else if (event == Event::ArrowLeft) {
                    settings_cycle_direction = -1;
                } else if (event == Event::ArrowRight || event == Event::Return) {
                    settings_cycle_direction = 1;
                } else if (event == Event::Tab) {
                    settings_panel_state.scope =
                        settings_panel_state.scope == core::config::SettingsScope::User
                            ? core::config::SettingsScope::Workspace
                            : core::config::SettingsScope::User;
                    settings_panel_state.status_message = std::format(
                        "{} scope selected.",
                        settings_scope_label(settings_panel_state.scope));
                } else if (event == Event::Backspace || event == Event::Delete) {
                    settings_reset = true;
                } else if (event == Event::Escape) {
                    settings_panel_state.active = false;
                } else {
                    for (int n = 1;
                         n <= std::min(static_cast<int>(settings_definitions.size()), 9);
                         ++n) {
                        if (event == Event::Character(static_cast<char>('0' + n))) {
                            settings_panel_state.selected = n - 1;
                            break;
                        }
                    }
                }

                if (settings_cycle_direction.has_value() || settings_reset) {
                    settings_selected_index = settings_panel_state.selected;
                    settings_scope = settings_panel_state.scope;
                }
            }
        }
        if (settings_panel_was_active) {
            if (settings_selected_index.has_value()
                && settings_scope.has_value()
                && *settings_selected_index >= 0
                && *settings_selected_index < static_cast<int>(settings_definitions.size())) {
                const auto& definition =
                    settings_definitions[static_cast<std::size_t>(*settings_selected_index)];
                const auto scoped_value = managed_setting_value(
                    config_manager.get_settings_overlay(*settings_scope),
                    definition.key);
                if (settings_reset) {
                    if (scoped_value.has_value()) {
                        persist_settings_value(*settings_selected_index, std::nullopt);
                    } else {
                        std::lock_guard lock(ui_mutex);
                        settings_panel_state.status_message =
                            "This scope is already inheriting the effective value.";
                    }
                } else if (settings_cycle_direction.has_value() && !definition.choices.empty()) {
                    const std::string current_value = scoped_value.value_or(
                        effective_setting_value(definition.key));
                    int current_index = 0;
                    for (std::size_t i = 0; i < definition.choices.size(); ++i) {
                        if (definition.choices[i].value == current_value) {
                            current_index = static_cast<int>(i);
                            break;
                        }
                    }
                    const int next_index =
                        (current_index
                         + *settings_cycle_direction
                         + static_cast<int>(definition.choices.size()))
                        % static_cast<int>(definition.choices.size());
                    persist_settings_value(
                        *settings_selected_index,
                        definition.choices[static_cast<std::size_t>(next_index)].value);
                }
            }
            return true;
        }

        bool session_picker_was_active = false;
        std::optional<int> session_choice;
        std::optional<int> session_delete_idx;
        {
            std::lock_guard lock(ui_mutex);
            if (session_picker_state.active) {
                session_picker_was_active = true;
                const int count = static_cast<int>(session_picker_state.sessions.size());
                if (event == Event::ArrowUp) {
                    if (count > 0) {
                        session_picker_state.selected = (session_picker_state.selected + count - 1) % count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (count > 0) {
                        session_picker_state.selected = (session_picker_state.selected + 1) % count;
                    }
                } else if (event == Event::Return) {
                    if (count > 0) {
                        session_choice = session_picker_state.selected;
                        session_picker_state.active = false;
                    }
                } else if (event == Event::Backspace || event == Event::Delete) {
                    if (count > 0) {
                        session_delete_idx = session_picker_state.selected;
                    }
                } else if (event == Event::Escape) {
                    session_picker_state.active = false;
                }
            }
        }
        if (session_picker_was_active) {
            if (session_delete_idx.has_value()) {
                std::string error;
                std::string sid;
                {
                    std::lock_guard lock(ui_mutex);
                    sid = session_picker_state.sessions[static_cast<size_t>(*session_delete_idx)].session_id;
                }
                if (session_store->remove(sid, &error)) {
                    std::lock_guard lock(ui_mutex);
                    session_picker_state.sessions = session_store->list();
                    if (session_picker_state.sessions.empty()) {
                        session_picker_state.active = false;
                    } else {
                        session_picker_state.selected = std::min(
                            session_picker_state.selected,
                            static_cast<int>(session_picker_state.sessions.size()) - 1);
                    }
                } else {
                    append_history(std::format("\n\xe2\x9c\x97  Failed to delete session: {}\n", error));
                }
                return true;
            }
            if (session_choice.has_value()) {
                core::session::SessionInfo info;
                {
                    std::lock_guard lock(ui_mutex);
                    info = session_picker_state.sessions[static_cast<size_t>(*session_choice)];
                }
                auto data_opt = session_store->load_by_id(info.session_id);
                if (data_opt) {
                    resume_session(*data_opt);
                } else {
                    append_history(std::format("\n\xe2\x9c\x97  Failed to load session {}.\n", info.session_id));
                }
            }
            return true;
        }

        bool model_picker_was_active = false;
        std::optional<int> model_choice;
        bool open_local_picker_from_model = false;
        {
            std::lock_guard lock(ui_mutex);
            if (model_picker_state.active) {
                model_picker_was_active = true;
                if (event == Event::ArrowUp) {
                    model_picker_state.selected = (model_picker_state.selected + 2) % 3;
                } else if (event == Event::ArrowDown) {
                    model_picker_state.selected = (model_picker_state.selected + 1) % 3;
                } else if (event == Event::Character('1')) {
                    model_choice = 0;
                    model_picker_state.active = false;
                } else if (event == Event::Character('2')) {
                    model_choice = 1;
                    model_picker_state.active = false;
                } else if (event == Event::Character('3')) {
                    model_choice = 2;
                    model_picker_state.active = false;
                } else if (event == Event::Return) {
                    model_choice = model_picker_state.selected;
                    model_picker_state.active = false;
                } else if (event == Event::Escape) {
                    model_picker_state.active = false;
                } else if (event == Event::Character('l') || event == Event::Character('L')) {
                    open_local_picker_from_model = true;
                    model_picker_state.active = false;
                }
            }
        }
        if (open_local_picker_from_model) {
            open_local_model_picker();
            return true;
        }
        if (model_picker_was_active) {
            if (model_choice.has_value()) {
                std::string result =
                    (*model_choice == 0) ? activate_manual_mode() :
                    (*model_choice == 2) ? activate_auto_mode()   :
                                          activate_router_mode();
                const bool success = result.starts_with("Switched");
                if (success) {
                    if (const auto persist_error = persist_model_preferences();
                        persist_error.has_value()) {
                        result += std::format("\n        \xe2\x9a\xa0  {}", *persist_error);
                    }
                }
                append_history(std::format(
                    "\n{}\n",
                    success
                        ? "\xe2\x9c\x93  " + result
                        : "\xe2\x9c\x97  " + result));
            }
            return true;
        }

        bool provider_picker_was_active = false;
        std::optional<int> provider_choice;
        std::function<void(std::optional<std::string>)> provider_on_select;
        std::vector<std::string> provider_list;
        {
            std::lock_guard lock(ui_mutex);
            if (provider_picker_state.active) {
                provider_picker_was_active = true;
                const int count = static_cast<int>(provider_picker_state.providers.size());
                if (event == Event::ArrowUp) {
                    if (count > 0) {
                        provider_picker_state.selected = (provider_picker_state.selected + count - 1) % count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (count > 0) {
                        provider_picker_state.selected = (provider_picker_state.selected + 1) % count;
                    }
                } else if (event == Event::Return) {
                    if (count > 0) {
                        provider_choice     = provider_picker_state.selected;
                        provider_list       = provider_picker_state.providers;
                        provider_on_select  = std::move(provider_picker_state.on_select);
                        provider_picker_state.active = false;
                    }
                } else if (event == Event::Escape) {
                    provider_on_select = std::move(provider_picker_state.on_select);
                    provider_picker_state.active = false;
                } else {
                    // Quick-select by number key
                    for (int n = 1; n <= std::min(count, 9); ++n) {
                        if (event == Event::Character(static_cast<char>('0' + n))) {
                            provider_choice     = n - 1;
                            provider_list       = provider_picker_state.providers;
                            provider_on_select  = std::move(provider_picker_state.on_select);
                            provider_picker_state.active = false;
                            break;
                        }
                    }
                }
            }
        }
        if (provider_picker_was_active) {
            if (provider_choice.has_value() && provider_on_select) {
                provider_on_select(provider_list[static_cast<std::size_t>(*provider_choice)]);
            } else if (!provider_choice.has_value() && provider_on_select) {
                provider_on_select(std::nullopt);
            }
            return true;
        }

        // ── Local model picker ────────────────────────────────────────────────
        bool local_model_picker_was_active = false;
        std::optional<std::filesystem::path> local_model_selected_path;
        std::filesystem::path navigate_into_dir;
        bool navigate_up = false;
        {
            std::lock_guard lock(ui_mutex);
            if (local_model_picker_state.active) {
                local_model_picker_was_active = true;
                const int count = static_cast<int>(local_model_picker_state.entries.size());
                if (event == Event::ArrowUp) {
                    if (count > 0) {
                        local_model_picker_state.selected =
                            (local_model_picker_state.selected + count - 1) % count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (count > 0) {
                        local_model_picker_state.selected =
                            (local_model_picker_state.selected + 1) % count;
                    }
                } else if (event == Event::Return) {
                    if (count > 0) {
                        const auto& entry =
                            local_model_picker_state.entries[
                                static_cast<std::size_t>(local_model_picker_state.selected)];
                        if (entry.is_directory) {
                            navigate_into_dir = entry.path;
                        } else {
                            local_model_selected_path = entry.path;
                            local_model_picker_state.active = false;
                        }
                    }
                } else if (event == Event::Escape) {
                    const auto parent = local_model_picker_state.current_dir.parent_path();
                    if (parent == local_model_picker_state.current_dir) {
                        local_model_picker_state.active = false;
                    } else {
                        navigate_up = true;
                    }
                }
            }
        }
        if (!navigate_into_dir.empty()) {
            auto new_entries = list_gguf_entries(navigate_into_dir);
            {
                std::lock_guard lock(ui_mutex);
                local_model_picker_state.current_dir = navigate_into_dir;
                local_model_picker_state.entries     = std::move(new_entries);
                local_model_picker_state.selected    = 0;
            }
            screen.PostEvent(Event::Custom);
            return true;
        }
        if (navigate_up) {
            const auto parent = [&]() {
                std::lock_guard lock(ui_mutex);
                return local_model_picker_state.current_dir.parent_path();
            }();
            auto new_entries = list_gguf_entries(parent);
            {
                std::lock_guard lock(ui_mutex);
                local_model_picker_state.current_dir = parent;
                local_model_picker_state.entries     = std::move(new_entries);
                local_model_picker_state.selected    = 0;
            }
            screen.PostEvent(Event::Custom);
            return true;
        }
        if (local_model_picker_was_active) {
            if (local_model_selected_path.has_value()) {
                const std::string result = select_local_model(*local_model_selected_path);
                const bool success = result.starts_with("Switched");
                append_history(std::format(
                    "\n{}\n",
                    success ? "\xe2\x9c\x93  " + result
                            : "\xe2\x9c\x97  " + result));
            }
            return true;
        }

        bool search_panel_was_active = false;
        std::optional<int> search_jump_index;
        {
            std::lock_guard lock(ui_mutex);
            if (conversation_search_state.active) {
                search_panel_was_active = true;
                bool query_changed = false;

                if (event == Event::Escape || is_ctrl_f_event(event)) {
                    conversation_search_state.active = false;
                } else if (event == Event::ArrowUp) {
                    if (!conversation_search_state.hits.empty()) {
                        const int count = static_cast<int>(conversation_search_state.hits.size());
                        conversation_search_state.selected =
                            (conversation_search_state.selected + count - 1) % count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (!conversation_search_state.hits.empty()) {
                        const int count = static_cast<int>(conversation_search_state.hits.size());
                        conversation_search_state.selected =
                            (conversation_search_state.selected + 1) % count;
                    }
                } else if (event == Event::Return) {
                    if (!conversation_search_state.hits.empty()) {
                        const auto& hit = conversation_search_state.hits[
                            static_cast<std::size_t>(conversation_search_state.selected)];
                        search_jump_index = hit.message_index;
                        conversation_search_state.active = false;
                    }
                } else if (event == Event::Backspace || event == Event::Delete) {
                    if (erase_last_utf8_codepoint(conversation_search_state.query)) {
                        query_changed = true;
                    }
                } else if (event.is_character()) {
                    const std::string input = event.character();
                    if (!input.empty()) {
                        const unsigned char first =
                            static_cast<unsigned char>(input.front());
                        const bool is_control = input.size() == 1 && std::iscntrl(first);
                        if (!is_control) {
                            conversation_search_state.query += input;
                            query_changed = true;
                        }
                    }
                }

                if (query_changed) {
                    refresh_conversation_search_locked();
                }
            }
        }
        if (search_panel_was_active) {
            if (search_jump_index.has_value() && *search_jump_index >= 0) {
                std::size_t message_count = 0;
                {
                    std::lock_guard lock(ui_mutex);
                    message_count = ui_messages.size();
                }
                history_component->JumpToMessage(
                    static_cast<std::size_t>(*search_jump_index),
                    message_count);
            }
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (is_ctrl_f_event(event)) {
            {
                std::lock_guard lock(ui_mutex);
                conversation_search_state.active = true;
                conversation_search_state.selected = 0;
                refresh_conversation_search_locked();
            }
            screen.PostEvent(Event::Custom);
            return true;
        }

        if (event == Event::Escape
            && assistant_turn_active.load(std::memory_order_relaxed)) {
            agent->request_stop();
            return true;
        }

        auto command_snapshot = current_command_snapshot();
        sync_command_picker(command_snapshot);
        if (!command_snapshot.suggestions.empty() && !command_picker.suppressed) {
            if (event == Event::Return && has_pending_command_completion()) {
                if (command_snapshot.active.has_value()
                    && trim_ascii(input_text) == command_snapshot.active->token) {
                    return execute_selected_command();
                }
                return accept_selected_command();
            }
            if (event == Event::Tab) {
                return accept_selected_command();
            }
            if (event == Event::ArrowDown) {
                command_picker.navigate_down(static_cast<int>(command_snapshot.suggestions.size()));
                return true;
            }
            // Escape dismisses the command picker so user can navigate history.
            // suppress() keeps the key so sync() won't re-show the picker
            // until the input text actually changes.
            if (event == Event::Escape) {
                command_picker.suppress();
                return true;
            }
            if (event == Event::ArrowUp) {
                command_picker.navigate_up(static_cast<int>(command_snapshot.suggestions.size()));
                return true;
            }
        }

        auto mention_snapshot = current_mention_snapshot();
        sync_mention_picker(mention_snapshot);
        if (!mention_snapshot.suggestions.empty()) {
            if (event == Event::Return && has_pending_mention_completion()) {
                return accept_selected_mention();
            }
            if (event == Event::Tab) {
                return accept_selected_mention();
            }
            if (event == Event::ArrowDown) {
                mention_picker.navigate_down(static_cast<int>(mention_snapshot.suggestions.size()));
                return true;
            }
            if (event == Event::ArrowUp) {
                mention_picker.navigate_up(static_cast<int>(mention_snapshot.suggestions.size()));
                return true;
            }
        }

        if (event == Event::F2) {
            current_mode_idx = (current_mode_idx + 1) % static_cast<int>(modes.size());
            agent->set_mode(modes[current_mode_idx].first);
            return true;
        }
        if (is_ctrl_l_event(event)) {  // Ctrl+L — clear screen (same as /clear)
            clear_screen();
            return true;
        }
        if (is_ctrl_x_event(event)) {  // Ctrl+X — open external editor
            return open_external_editor();
        }
        if (is_ctrl_v_event(event)) {  // Ctrl+V — paste clipboard content / image
            if (const auto image_path = read_clipboard_image_to_temp(); image_path.has_value()) {
                const std::string mention = std::format(
                    "@\"{}\"",
                    image_path->string());
                insert_token_with_spacing(input_text, input_cursor_position, mention);
                append_history(std::format(
                    "\n\xe2\x84\xb9  Pasted clipboard image as context: {}\n",
                    image_path->string()));
                screen.PostEvent(Event::Custom);
                return true;
            }

            if (const auto text = read_clipboard_text(); text.has_value() && !text->empty()) {
                insert_text_at_cursor(
                    input_text,
                    input_cursor_position,
                    normalize_newlines(*text));
                screen.PostEvent(Event::Custom);
                return true;
            }

            append_history(
                "\n\xe2\x9c\x97  Clipboard paste is unavailable (no compatible clipboard provider found).\n");
            screen.PostEvent(Event::Custom);
            return true;
        }
        if (is_ctrl_y_event(event)) {  // Ctrl+Y — toggle YOLO approvals
            const bool enable_yolo = !is_yolo_mode_enabled();
            set_yolo_mode_enabled(enable_yolo);
            append_history(enable_yolo
                ? "\n\xe2\x9a\xa0  Approval mode set to YOLO: sensitive tools will auto-run.\n"
                : "\n\xe2\x84\xb9  Approval mode set to PROMPT: sensitive tools require confirmation.\n");
            return true;
        }
        if (is_ctrl_o_event(event)) {  // Ctrl+O — toggle tool output expansion
            tool_output_expanded = !tool_output_expanded;
            append_history(tool_output_expanded
                ? "\n\xe2\x84\xb9  Tool output view set to EXPANDED: full command output is visible.\n"
                : "\n\xe2\x84\xb9  Tool output view set to COMPACT: long command output is collapsed.\n");
            return true;
        }
        if (is_ctrl_c_event(event)) {  // Ctrl+C — stop current LLM generation
            agent->request_stop();
            return true;
        }

        // ── History navigation (Gemini CLI compat) ───────────────────────────
        // Ctrl+P → previous history entry
        if (event == Event::Special({16})) {
            return navigate_history_prev();
        }
        // Ctrl+N → next history entry
        if (event == Event::Special({14})) {
            return navigate_history_next();
        }
        // Up/Down arrows with no active autocomplete → navigate history
        if (event == Event::ArrowUp) {
            return navigate_history_prev();
        }
        if (event == Event::ArrowDown) {
            return navigate_history_next();
        }

        // ── Ctrl+D: delete-right in prompt, double-press to exit on empty ────
        if (is_ctrl_d_event(event)) {
            if (!input_text.empty()) {
                // Normalize terminal-specific Ctrl+D variants through PromptInput's
                // canonical control-byte handler.
                reset_ctrl_d_warning();
                input_component->OnEvent(Event::Special({4}));
                return true;
            }

            if (ctrl_d_warning_active()) {
                screen.ExitLoopClosure()();
                return true;
            }

            ctrl_d_press_count = 1;
            ctrl_d_deadline = std::chrono::steady_clock::now() + kExitConfirmWindow;
            screen.PostEvent(Event::Custom);
            return true;
        }

        // ── History panel scroll ─────────────────────────────────────────────
        // Route mouse wheel events to the history component (even when input focused)
        if (event.is_mouse()) {
            if (history_component->HandleWheel(event)) return true;
        }

        // Page Up / Page Down also scroll history (large step).
        if (event == Event::PageUp) {
            history_component->ScrollPageUp();
            return true;
        }
        if (event == Event::PageDown) {
            history_component->ScrollPageDown();
            return true;
        }

        return false;
    });
    auto main_container = Container::Vertical({
        history_component | flex,
        component
    });
    // Focus the input AFTER the container is built so TakeFocus() can propagate
    // up through the component tree correctly.
    input_component->TakeFocus();

    // ── Renderer ─────────────────────────────────────────────────────────────
    auto renderer = Renderer(main_container, [&]() {
        refresh_status_labels();
        auto command_snapshot = current_command_snapshot();
        sync_command_picker(command_snapshot);
        auto mention_snapshot = current_mention_snapshot();
        sync_mention_picker(mention_snapshot);
        
        bool                   perm_active = false;
        bool                   model_picker_active = false;
        bool                   provider_picker_active = false;
        bool                   settings_panel_active = false;
        std::string            perm_tool, perm_args, perm_allow_label;
        std::string            settings_panel_status;
        ToolDiffPreview        perm_diff;
        int                    perm_selected = 0;
        int                    model_picker_selected = 0;
        int                    provider_picker_selected = 0;
        int                    settings_panel_selected = 0;
        std::vector<std::string> provider_picker_providers;
        core::config::SettingsScope settings_panel_scope = core::config::SettingsScope::User;
        bool                            local_model_picker_active   = false;
        int                             local_model_picker_selected = 0;
        std::string                     local_model_picker_dir;
        std::vector<tui::LocalModelEntry> local_model_picker_entries;
        bool                            session_picker_active = false;
        int                             session_picker_selected = 0;
        std::vector<core::session::SessionInfo> session_picker_sessions;
        bool                            conversation_search_active = false;
        int                             conversation_search_selected = 0;
        std::string                     conversation_search_query;
        std::vector<tui::ConversationSearchHit> conversation_search_hits;
        {
            std::lock_guard lock(ui_mutex);
            if (conversation_search_state.active) {
                refresh_conversation_search_locked();
            }
            perm_active       = perm_state.active;
            perm_tool         = perm_state.tool_name;
            perm_args         = perm_state.args_preview;
            perm_diff         = perm_state.diff_preview;
            perm_selected     = perm_state.selected;
            perm_allow_label  = perm_state.allow_label;
            model_picker_active   = model_picker_state.active;
            model_picker_selected = model_picker_state.selected;
            provider_picker_active    = provider_picker_state.active;
            provider_picker_selected  = provider_picker_state.selected;
            provider_picker_providers = provider_picker_state.providers;
            settings_panel_active = settings_panel_state.active;
            settings_panel_selected = settings_panel_state.selected;
            settings_panel_scope = settings_panel_state.scope;
            settings_panel_status = settings_panel_state.status_message;
            local_model_picker_active   = local_model_picker_state.active;
            local_model_picker_selected = local_model_picker_state.selected;
            local_model_picker_dir      = local_model_picker_state.current_dir.string();
            local_model_picker_entries  = local_model_picker_state.entries;
            session_picker_active = session_picker_state.active;
            session_picker_selected = session_picker_state.selected;
            session_picker_sessions = session_picker_state.sessions;
            conversation_search_active = conversation_search_state.active;
            conversation_search_selected = conversation_search_state.selected;
            conversation_search_query = conversation_search_state.query;
            conversation_search_hits = conversation_search_state.hits;
        }
        
        // Question dialog state (copy for rendering)
        QuestionDialogState question_state_copy;
        {
            std::lock_guard lock(ui_mutex);
            question_state_copy = question_dialog_state;
        }
        
        auto history_el = history_component->Render() | flex;

        Element banner_el = emptyElement();
        if (ui_show_banner) {
            banner_el = render_startup_banner_panel(
                active_provider_name,
                active_model_name.empty() ? "<provider default>" : active_model_name,
                core::mcp::McpConnectionManager::get_instance().connected_count(),
                provider_setup_hint(active_provider_name));
        }

        // ── Permission overlay ───────────────────────────────────────────
        Element bottom_el;
        if (settings_panel_active) {
            std::vector<SettingsPanelRow> settings_rows;
            settings_rows.reserve(settings_definitions.size());
            const auto& scope_overlay = config_manager.get_settings_overlay(settings_panel_scope);
            for (const auto& definition : settings_definitions) {
                const auto scoped_value = managed_setting_value(scope_overlay, definition.key);
                const std::string effective_value = effective_setting_value(definition.key);
                settings_rows.push_back(SettingsPanelRow{
                    .label = definition.label,
                    .value = setting_choice_label(
                        definition,
                        scoped_value.value_or(effective_value)),
                    .description = definition.description,
                    .inherited = !scoped_value.has_value(),
                });
            }

            bottom_el = render_settings_panel(
                settings_scope_label(settings_panel_scope),
                config_manager.get_settings_path(settings_panel_scope, settings_working_dir).string(),
                settings_rows,
                settings_panel_selected,
                settings_panel_status);
        } else if (question_state_copy.active) {
            bottom_el = render_question_dialog_panel(question_state_copy);
        } else if (conversation_search_active) {
            bottom_el = render_conversation_search_panel(
                conversation_search_query,
                conversation_search_hits,
                conversation_search_selected);
        } else if (session_picker_active) {
            bottom_el = render_session_picker_panel(session_picker_sessions, session_picker_selected);
        } else if (provider_picker_active) {
            bottom_el = render_provider_selection_panel(provider_picker_providers,
                                                        provider_picker_selected);
        } else if (local_model_picker_active) {
            bottom_el = render_local_model_picker_panel(
                local_model_picker_dir,
                local_model_picker_entries,
                local_model_picker_selected);
        } else if (model_picker_active) {
            const std::string manual_description = std::format(
                "Use a fixed preset: {} ({})",
                manual_provider_name,
                manual_model_name.empty() ? "<provider default>" : manual_model_name);
            const std::string router_description = router_available
                ? "Policy-driven routing: uses your configured fallback/load-balance rules."
                : "Router unavailable: configure config.router.policies first.";
            const std::string auto_description = router_available
                ? "Intelligent routing: classifies each request and picks the best model tier."
                : "Auto unavailable: requires a configured router with strategy=smart.";
            constexpr bool kLocalModelAvailable =
#ifdef FILO_ENABLE_LLAMACPP
                true;
#else
                false;
#endif
            bottom_el = render_model_selection_panel(
                model_picker_selected,
                manual_description,
                router_description,
                auto_description,
                router_policy_label(),
                router_available,
                kLocalModelAvailable);
        } else if (perm_active) {
            bottom_el = render_permission_prompt_panel(
                perm_tool,
                perm_args,
                perm_diff,
                perm_allow_label,
                perm_selected);
        } else {
            Element input_el = component->Render() | color(tui::ColorYellowBright) | xflex;

            if (command_snapshot.active.has_value() && !command_snapshot.suggestions.empty() && !command_picker.suppressed) {
                bottom_el = render_command_prompt_panel(
                    command_snapshot.suggestions,
                    command_picker.selected,
                    std::move(input_el),
                    input_text);
            } else if (mention_snapshot.active.has_value() && !mention_snapshot.suggestions.empty()) {
                bottom_el = render_mention_prompt_panel(
                    mention_snapshot.suggestions,
                    mention_picker.selected,
                    std::move(input_el),
                    input_text);
            } else {
                bottom_el = render_default_prompt_panel(std::move(input_el), input_text);
            }
        }

        // ── Status bar ───────────────────────────────────────────────────
        std::string budget_str = core::budget::BudgetTracker::get_instance().status_string();
        const int32_t ctx_pct  = core::budget::BudgetTracker::get_instance()
                                     .context_remaining_pct(active_model_name);

        Color ctx_color = Color::Green;
        if (ctx_pct >= 0 && ctx_pct < 25)       ctx_color = Color::Red;
        else if (ctx_pct >= 0 && ctx_pct < 50)  ctx_color = ColorWarn;
        
        // Get rate limit info from the current provider (for Anthropic/OAuth users)
        auto rate_limit_info = llm_provider->get_last_rate_limit_info();
        
        // Update our local rate limit state for quota notifications
        {
            std::lock_guard lock(ui_mutex);
            rate_limit_state.requests_limit     = rate_limit_info.requests_limit;
            rate_limit_state.requests_remaining = rate_limit_info.requests_remaining;
            rate_limit_state.tokens_limit       = rate_limit_info.tokens_limit;
            rate_limit_state.tokens_remaining   = rate_limit_info.tokens_remaining;
            rate_limit_state.usage_windows    = rate_limit_info.usage_windows;
        }
        // Check for quota notifications
        check_and_notify_quota();

        // Detect subscription/OAuth mode: from config auth_type (reliable before first
        // response) or from received unified windows (works in router/auto mode).
        const bool is_subscription = [&]() -> bool {
            if (auto it = config.providers.find(active_provider_name); it != config.providers.end()) {
                if (it->second.auth_type.starts_with("oauth_")) return true;
            }
            return !rate_limit_info.usage_windows.empty();
        }();

        // Format current working directory for display
        auto format_cwd = []() -> std::string {
            try {
                auto cwd = std::filesystem::current_path();
                const char* home = std::getenv("HOME");
                if (home != nullptr) {
                    std::string cwd_str = cwd.string();
                    std::string home_str(home);
                    if (cwd_str == home_str) {
                        return "~";
                    }
                    if (cwd_str.starts_with(home_str + "/")) {
                        return "~" + cwd_str.substr(home_str.length());
                    }
                }
                return cwd.string();
            } catch (...) {
                return "";
            }
        };

        // Build right side of status bar: current folder + context left
        Element right_el;
        if (ctrl_d_warning_active()) {
            right_el = text(" Press Ctrl+D again to quit ") | color(Color::GrayDark);
        } else {
            Elements right_items;
            // Current working directory
            std::string cwd_str = format_cwd();
            if (!cwd_str.empty()) {
                right_items.push_back(text(" " + cwd_str + " ") | color(Color::GrayLight));
            }
            // Context left percentage
            if (ui_show_context_usage && ctx_pct >= 0) {
                right_items.push_back(text(std::format(" {}% Context Left ", ctx_pct)) | color(ctx_color));
            }
            if (right_items.empty()) {
                right_el = text("");
            } else {
                right_el = hbox(std::move(right_items));
            }
        }

        // For subscription users: show token counts but hide the dollar cost
        // (window utilization replaces it in rate_limit_el).
        Element budget_el = text("");
        if (!budget_str.empty()) {
            if (is_subscription) {
                auto total = core::budget::BudgetTracker::get_instance().session_total();
                if (total.has_data()) {
                    auto fmt_k = [](int32_t n) -> std::string {
                        if (n >= 1000) return std::format("{:.1f}k", n / 1000.0);
                        return std::to_string(n);
                    };
                    // ↑ = prompt tokens going UP to the LLM, ↓ = completion tokens coming DOWN from LLM
                    budget_el = text("  \xe2\x86\x91" + fmt_k(total.prompt_tokens)
                                     + " \xe2\x86\x93" + fmt_k(total.completion_tokens))
                                | color(Color::GrayLight);
                }
            } else {
                budget_el = text("  " + budget_str) | color(Color::GrayLight);
            }
        }

        // Build rate limit status element.
        // Subscription users with unified windows: iterate and display each window.
        // OpenAI-style subscription (no unified windows): show token window utilization %.
        // API key users: show raw remaining counts.
        Element rate_limit_el = text("");
        if (!rate_limit_info.usage_windows.empty()) {
            const float max_util = rate_limit_info.max_window_utilization();
            Color util_color = Color::Green;
            if (max_util >= 0.90f)       util_color = Color::Red;
            else if (max_util >= 0.75f)  util_color = ColorWarn;
            else if (max_util >= 0.50f)  util_color = Color::Yellow;

            std::string util_str;
            for (const auto& w : rate_limit_info.usage_windows) {
                util_str += std::format(" {}:{:.0f}%", w.label, w.utilization * 100.0f);
            }
            rate_limit_el = text(util_str) | color(util_color);
        } else if (rate_limit_info.requests_limit > 0 || rate_limit_info.tokens_limit > 0
                   || rate_limit_info.requests_remaining > 0 || rate_limit_info.tokens_remaining > 0) {
            if (is_subscription && rate_limit_info.tokens_limit > 0) {
                // Non-Anthropic subscription: show token window utilization %.
                const float token_util = std::clamp(
                    1.0f - static_cast<float>(rate_limit_info.tokens_remaining)
                           / static_cast<float>(rate_limit_info.tokens_limit),
                    0.0f, 1.0f);
                Color util_color = Color::Green;
                if (token_util >= 0.90f)       util_color = Color::Red;
                else if (token_util >= 0.75f)  util_color = ColorWarn;
                else if (token_util >= 0.50f)  util_color = Color::Yellow;
                rate_limit_el = text(std::format(" T:{:.0f}%", token_util * 100.0f))
                                | color(util_color);
            } else {
                // API key users: show raw remaining counts.
                Color rate_color = Color::Green;
                if (rate_limit_info.requests_remaining < 10 || rate_limit_info.tokens_remaining < 1000) {
                    rate_color = Color::Red;
                } else if (rate_limit_info.requests_remaining < 25 || rate_limit_info.tokens_remaining < 5000) {
                    rate_color = ColorWarn;
                }
                rate_limit_el = text(std::format(" R:{} T:{}",
                                                 rate_limit_info.requests_remaining,
                                                 rate_limit_info.tokens_remaining))
                                | color(rate_color);
            }
        }

        Element guardrail_el = text("");
        const bool router_managed_mode =
            model_selection_mode == ModelSelectionMode::Router
            || model_selection_mode == ModelSelectionMode::Auto;
        if (router_managed_mode && router_provider) {
            const std::string guardrail_summary = router_provider->last_guardrail_summary();
            if (!guardrail_summary.empty()) {
                guardrail_el = text(" GR " + compact_single_line(guardrail_summary, 72) + " ")
                               | bgcolor(ColorWarn) | color(Color::Black);
            }
        }
        
        Elements left_items;
        left_items.push_back(
            text(std::string(" ") + modes[current_mode_idx].first + " ")
                | ftxui::bold | bgcolor(modes[current_mode_idx].second) | color(Color::White));
        if (ui_show_model_info) {
            left_items.push_back(
                text(" " + active_provider_name + " · "
                     + (active_model_name.empty() ? "<provider default>" : active_model_name)
                     + " ")
                    | bgcolor(ColorYellowDark) | color(Color::Black));
        }
        left_items.push_back(budget_el);
        left_items.push_back(rate_limit_el);
        left_items.push_back(guardrail_el);
        auto left_el = ui_show_footer
            ? hbox(std::move(left_items)) | xflex
            : text("");

        auto status_el = (ui_show_footer || ctrl_d_warning_active())
            ? hbox({
                left_el,
                right_el,
            }) | xflex
            : text("");

        Elements window_rows;
        window_rows.reserve(4);
        if (ui_show_banner) {
            window_rows.push_back(std::move(banner_el));
        }
        window_rows.push_back(std::move(history_el));
        window_rows.push_back(std::move(bottom_el));
        window_rows.push_back(std::move(status_el));

        return UiWindow(
            text(std::format(" {} ", kAppVersion)) | color(ColorYellowBright) | ftxui::bold,
            vbox(std::move(window_rows))
        ) | color(ColorYellowBright);
    });

    std::atomic<bool> animation_running = true;
    std::thread animation_thread([&]() {
        while (animation_running.load(std::memory_order_relaxed)) {
            {
                std::unique_lock lock(animation_mutex);
                animation_cv.wait(lock, [&]() {
                    return !animation_running.load(std::memory_order_relaxed)
                        || has_active_animation();
                });
            }
            if (!animation_running.load(std::memory_order_relaxed)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kAnimationIntervalMs));
            if (!has_active_animation()) {
                continue;
            }
            animation_tick.fetch_add(1, std::memory_order_relaxed);
            screen.PostEvent(Event::Custom);
        }
    });

    screen.Loop(renderer);

    animation_running.store(false, std::memory_order_relaxed);
    animation_cv.notify_one();
    if (animation_thread.joinable()) {
        animation_thread.join();
    }

    // Shut down MCP connections gracefully.
    core::mcp::McpConnectionManager::get_instance().shutdown_all();

    return RunResult{
        .session_id        = session_id,
        .session_file_path = session_file_path,
    };
}

} // namespace tui
