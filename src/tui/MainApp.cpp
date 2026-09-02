#include "MainApp.hpp"
#include "Autocomplete.hpp"
#include "ActivityTimer.hpp"
#include "Constants.hpp"
#include "HistoryComponent.hpp"
#include "PickerState.hpp"
#include "Conversation.hpp"
#include "CodeBlockRunner.hpp"
#include "CodeBlockRunServices.hpp"
#include "FileSystemPicker.hpp"
#include "FileSystemPickerView.hpp"
#include "KeyInput.hpp"
#include "SessionReplay.hpp"
#include "SessionPicker.hpp"
#include "ThreadRuntime.hpp"
#include "SelectionClipboardCopier.hpp"
#include "Text.hpp"
#include "editor/ExternalEditorController.hpp"
#include "PromptInput.hpp"
#include "PromptComponents.hpp"
#include "QuestionDialogController.hpp"
#include "RewindActions.hpp"
#include "RewindPicker.hpp"
#include "RemoteActivityPanel.hpp"
#include "UsageDetailsPanel.hpp"
#include "AgentsVisualizerPanel.hpp"
#include "TuiTheme.hpp"
#include "core/session/SessionData.hpp"
#include "core/session/ThreadCatalog.hpp"
#include "core/commands/GoalExecutor.hpp"
#include "core/goal/GoalEngine.hpp"
#include "core/session/GoalManager.hpp"
#include "core/session/SessionHandoff.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/session/ActiveSessionLease.hpp"
#include "core/memory/MemoryStore.hpp"
#include "core/scm/ScmFactory.hpp"
#include "core/history/PromptHistoryStore.hpp"
#include "core/utils/PathUtils.hpp"
#include <ftxui/component/component.hpp>
#include <ftxui/component/app.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include "core/llm/ModelCatalogDiscovery.hpp"
#include "core/llm/ModelMetadata.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderCatalogGrouping.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/providers/RouterProvider.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/config/ModelDefaultsPersistence.hpp"
#include "core/config/SessionModelOverride.hpp"
#include "core/auth/AuthenticationManager.hpp"
#include "core/llm/routing/RouterEngine.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/BuiltinToolRegistry.hpp"
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
#include "core/tools/ActivateSkillTool.hpp"
#include "core/tools/SkillRegistry.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "core/tools/PythonInterpreterTool.hpp"
#include "core/tools/SkillLoader.hpp"
#endif
#include "core/logging/Logger.hpp"
#include "core/landrun/LandrunSettings.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"
#include "core/agent/Agent.hpp"
#include "core/memory/MemorySystem.hpp"
#include "core/agent/PermissionGate.hpp"
#include "core/permissions/PermissionSystem.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/budget/TokenUsageFormatters.hpp"
#include "core/mcp/McpConnectionManager.hpp"
#include "core/mcp/McpOAuth.hpp"
#include "core/auth/ui/ConsoleAuthUI.hpp"
#include "core/mcp/RemoteActivity.hpp"
#include "core/context/ContextMentions.hpp"
#include "core/context/SteeringLoader.hpp"
#include "core/commands/CommandExecutor.hpp"
#include "core/commands/SkillCommandLoader.hpp"
#include "core/commands/SkillTurnResolver.hpp"
#include "core/utils/StringUtils.hpp"
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <charconv>
#include <memory>
#include <format>
#include <chrono>
#include <csignal>
#include <thread>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <expected>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <future>
#include <algorithm>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <system_error>
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
constexpr int kReviewActivitySpinnerCharset = 12; // Compact circle spinner (single-cell frames).
constexpr std::size_t kMaxStderrPanelLines = 20;

std::string join_context_source_labels(const std::vector<std::string>& source_labels) {
    std::string joined;
    for (const auto& label : source_labels) {
        if (label.empty()) {
            continue;
        }
        if (!joined.empty()) {
            joined += ", ";
        }
        joined += label;
    }
    return compact_single_line(joined, 120);
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

class SigintIgnoreGuard {
public:
    SigintIgnoreGuard() : previous_(std::signal(SIGINT, SIG_IGN)) {}
    ~SigintIgnoreGuard() { std::signal(SIGINT, previous_); }

    SigintIgnoreGuard(const SigintIgnoreGuard&) = delete;
    SigintIgnoreGuard& operator=(const SigintIgnoreGuard&) = delete;

private:
    using Handler = void (*)(int);
    Handler previous_ = SIG_DFL;
};

struct DirectShellState {
    std::unique_ptr<core::tools::shell::IShellExecutor> executor{
        core::tools::shell::make_shell_executor()};
    std::mutex run_mutex;
    mutable std::mutex active_mutex;
    std::condition_variable active_cv;
    std::optional<core::commands::ActiveTerminalInfo> active_command;
    std::chrono::steady_clock::time_point active_started_at{};
    std::size_t workers_in_flight = 0;

    void begin_worker() {
        std::lock_guard lock(active_mutex);
        ++workers_in_flight;
    }

    void finish_worker() {
        {
            std::lock_guard lock(active_mutex);
            if (workers_in_flight > 0) {
                --workers_in_flight;
            }
        }
        active_cv.notify_all();
    }

    void mark_active(std::string session_id,
                     std::string command,
                     std::string working_dir) {
        std::lock_guard lock(active_mutex);
        active_started_at = std::chrono::steady_clock::now();
        active_command = core::commands::ActiveTerminalInfo{
            .session_id = std::move(session_id),
            .command = std::move(command),
            .working_dir = std::move(working_dir),
            .tool_call_id = {},
            .elapsed = std::chrono::seconds{0},
        };
    }

    void clear_active() {
        {
            std::lock_guard lock(active_mutex);
            active_command.reset();
            active_started_at = {};
        }
        active_cv.notify_all();
    }

    [[nodiscard]] std::optional<core::commands::ActiveTerminalInfo>
    active_info(std::chrono::steady_clock::time_point now) const {
        std::lock_guard lock(active_mutex);
        if (!active_command.has_value()) {
            return std::nullopt;
        }

        auto info = *active_command;
        info.elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - active_started_at);
        return info;
    }

    [[nodiscard]] bool has_active_for(std::string_view session_id) const {
        std::lock_guard lock(active_mutex);
        return active_command.has_value() && active_command->session_id == session_id;
    }

    bool interrupt_active_for(std::string_view session_id) {
        std::lock_guard lock(active_mutex);
        if (!active_command.has_value() || active_command->session_id != session_id) {
            return false;
        }
        if (!executor) {
            return false;
        }
        return executor->interrupt();
    }

    void interrupt_active() {
        std::lock_guard lock(active_mutex);
        if (active_command.has_value() && executor) {
            static_cast<void>(executor->interrupt());
        }
    }

    void wait_until_idle() {
        std::unique_lock lock(active_mutex);
        active_cv.wait(lock, [&]() {
            return !active_command.has_value() && workers_in_flight == 0;
        });
    }

    core::tools::shell::IShellExecutor::Result run(std::string_view command,
                                                   std::string_view working_dir,
                                                   std::string session_id) {
        std::lock_guard lock(run_mutex);
        mark_active(std::move(session_id), std::string(command), std::string(working_dir));

        struct ActiveGuard {
            DirectShellState* state = nullptr;
            ~ActiveGuard() {
                if (state) {
                    state->clear_active();
                }
            }
        } guard{this};

        return executor->run(command, working_dir);
    }
};

constexpr std::size_t kDirectShellHistoryMaxOutputLength = 10'000;

std::string markdown_code_fence_for(std::string_view value) {
    std::size_t longest_run = 0;
    std::size_t current_run = 0;
    for (const char ch : value) {
        if (ch == '`') {
            ++current_run;
            longest_run = std::max(longest_run, current_run);
        } else {
            current_run = 0;
        }
    }

    return std::string(std::max<std::size_t>(3, longest_run + 1), '`');
}

std::string make_direct_shell_history_content(std::string_view command,
                                              std::string_view result_text) {
    std::string model_content = std::string(trim_ascii(result_text));
    if (model_content.empty()) {
        model_content = "(Command produced no output)";
    } else if (model_content.size() > kDirectShellHistoryMaxOutputLength) {
        model_content =
            model_content.substr(0, kDirectShellHistoryMaxOutputLength)
            + "\n... (truncated)";
    }

    const std::string command_fence = markdown_code_fence_for(command);
    const std::string result_fence = markdown_code_fence_for(model_content);

    return std::format(
        "I ran the following shell command:\n"
        "{}sh\n"
        "{}\n"
        "{}\n\n"
        "This produced the following result:\n"
        "{}\n"
        "{}\n"
        "{}",
        command_fence,
        command,
        command_fence,
        result_fence,
        model_content,
        result_fence);
}

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

std::vector<ReviewBaseRef> filter_review_base_refs(const std::vector<ReviewBaseRef>& refs,
                                                   std::string_view query) {
    const std::string lowered_query = to_lower_ascii(std::string(trim_ascii(query)));
    if (lowered_query.empty()) {
        return refs;
    }

    std::vector<ReviewBaseRef> filtered;
    for (const auto& ref : refs) {
        const std::string lowered_name = to_lower_ascii(ref.name);
        const std::string lowered_description = to_lower_ascii(ref.description);
        if (lowered_name.find(lowered_query) != std::string::npos
            || lowered_description.find(lowered_query) != std::string::npos) {
            filtered.push_back(ref);
        }
    }
    return filtered;
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

} // namespace

RunResult run(RunOptions opts) {
    cpr::Session dummy_session;  // initialise curl

    auto& config_manager     = core::config::ConfigManager::get_instance();
    auto config              = config_manager.get_config();
    auto& provider_manager   = core::llm::ProviderManager::get_instance();
    const auto settings_working_dir = std::filesystem::current_path();
    const auto authentication_manager =
        core::auth::AuthenticationManager::create_with_defaults(
            config_manager.get_config_dir());

    core::config::ModelDefaultsPersistence model_defaults{config_manager};
    if (opts.startup_model.has_value()) {
        if (auto applied = core::config::apply_session_model_override(
                config, *opts.startup_model); !applied) {
            core::logging::error("{}", applied.error());
            return {};
        }
    }

    using ModelSelectionMode = tui::ModelSelectionMode;
    using ModelSelectionSnapshot = tui::ModelSelectionSnapshot;

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

    core::llm::ProviderDescriptorSet registered_providers;
    std::unordered_map<std::string, std::string> provider_default_models;

    for (const auto& [name, pconfig] : config.providers) {
        if (auto provider = core::llm::ProviderFactory::create_provider(name, pconfig)) {
            const auto caps = provider->capabilities();
            provider_manager.register_provider(name, provider);
            registered_providers.insert({name, caps.is_local});
            provider_default_models[name] = pconfig.model;
        }
    }

    std::vector<std::string> sorted_provider_names;
    sorted_provider_names.reserve(registered_providers.size());
    for (const auto& provider : registered_providers) {
        sorted_provider_names.push_back(provider.name);
    }
    std::ranges::sort(sorted_provider_names);

    if (registered_providers.empty()) {
        core::logging::error("Fatal: No providers could be initialised from configuration.");
        return {};
    }

    std::string manual_provider_name = config.default_provider;
    if (!core::llm::contains_provider(registered_providers, manual_provider_name)) {
        manual_provider_name = registered_providers.begin()->name;
    }

    std::string manual_model_name;
    if (const auto it = config.providers.find(manual_provider_name);
        it != config.providers.end()) {
        manual_model_name = it->second.model;
    }

    auto router_engine = std::make_shared<core::llm::routing::RouterEngine>(
        config.router,
        registered_providers);
    auto router_provider_template = std::make_shared<core::llm::providers::RouterProvider>(
        provider_manager,
        router_engine,
        provider_default_models);
    auto router_provider = router_provider_template;

    bool router_available = config.router.enabled && !router_engine->list_policies().empty();

    ModelSelectionMode model_selection_mode = parse_model_selection_mode(config.default_model_selection);
    if ((model_selection_mode == ModelSelectionMode::Router
         || model_selection_mode == ModelSelectionMode::Auto)
        && !router_available) {
        model_selection_mode = ModelSelectionMode::Manual;
    }

    std::string active_router_policy = router_engine->active_policy();
    std::string active_provider_name;
    std::string active_model_name;
    std::string session_effort_value; // empty => auto/provider default

    std::optional<ModelSelectionSnapshot> previous_model_selection;

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
        if (active_model_name.empty() && llm_provider) {
            active_model_name = llm_provider->get_last_model();
        }
    }

    // Providers decide whether live model discovery is supported. For Kimi
    // Code this also refreshes the account-specific K3 context limit.
    core::llm::request_model_catalog_discovery(
        llm_provider,
        {.timeout_ms = 3000});

    bool ui_show_banner = visibility_setting_enabled(config.ui_banner, true);
    bool ui_show_footer = visibility_setting_enabled(config.ui_footer, true);
    bool ui_show_model_info = visibility_setting_enabled(config.ui_model_info, true);
    bool ui_show_context_usage = visibility_setting_enabled(config.ui_context_usage, true);
    bool ui_show_timestamps = visibility_setting_enabled(config.ui_timestamps, true);
    std::atomic_bool ui_show_spinner{
        visibility_setting_enabled(config.ui_spinner, true)};
    bool ui_show_reasoning = visibility_setting_enabled(config.ui_reasoning, true);
    bool tool_output_expanded = false;

    // One MemorySystem for this TUI process: Agent, MemoryTool, /memory, and
    // thread agents all share it. Not a singleton — owned here and passed down.
    auto memory_system = core::memory::make_memory_system(
        core::memory::MemoryConfig{.tool_recovery = config.tool_recovery});

    // ── Tool registration ───────────────────────────────────────────────────
    auto& tool_manager = core::tools::ToolManager::get_instance();
    std::shared_ptr<core::tools::AskUserQuestionTool> ask_user_tool;
    auto tool_options = core::tools::agent_builtin_tool_options();
    tool_options.ask_user_question_tool_out = &ask_user_tool;
    tool_options.memory_store = memory_system->semantic();
    core::tools::register_builtin_tools(tool_manager, std::move(tool_options));

    // ── MCP client connections ───────────────────────────────────────────────
    core::mcp::McpConnectionManager::get_instance().connect_all(
        config,
        tool_manager,
        llm_provider,
        model_selection_mode == ModelSelectionMode::Manual ? manual_model_name : std::string{});

    // ── Agent ────────────────────────────────────────────────────────────────
    // One session-keyed metrics registry is injected into every Agent. The
    // process composition root also reads it for the exit report.
    auto session_stats_registry =
        core::session::SessionStatsRegistry::shared_instance();
    auto workspace_leases =
        std::make_shared<core::scm::WorkspaceLeaseRegistry>();
    auto agent_session_context = core::context::make_session_context(
        core::workspace::Workspace::get_instance().snapshot(),
        core::context::SessionTransport::cli);
    agent_session_context.steering_policy = opts.steering_policy;
    auto steering_context = core::context::load_project_steering_context(
        agent_session_context.workspace_view().primary(),
        agent_session_context.steering_policy);
    std::string context_sources_label =
        join_context_source_labels(steering_context.source_labels);
    auto agent = std::make_shared<core::agent::Agent>(
        llm_provider,
        tool_manager,
        agent_session_context,
        core::agent::ToolResultStore::default_root(),
        std::shared_ptr<core::power::SleepInhibitor>{},
        session_stats_registry,
        &core::budget::BudgetTracker::get_instance(),
        memory_system,
        workspace_leases);
    agent->set_active_provider_name(active_provider_name);
    agent->set_auto_compact_threshold(
        config.auto_compact_threshold,
        !config.auto_compact_threshold_explicit);
    agent->set_effort_level(session_effort_value);

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
        return format_provider_setup_hint(
            provider_name, it->second.api_key, it->second.auth_type);
    };

    auto startup_history_message = [&]() {
        if (ui_show_banner) {
            return std::string{};
        }
        std::string summary = format_runtime_status_summary(
            active_provider_name,
            active_model_name,
            core::mcp::McpConnectionManager::get_instance().connected_count());
        if (!context_sources_label.empty()) {
            summary += "  —  ";
            summary += context_sources_label;
        }
        if (const auto hint = provider_setup_hint(active_provider_name); !hint.empty()) {
            summary.push_back('\n');
            summary += hint;
        }
        return summary;
    };

    auto selected_messages = std::make_shared<std::vector<UiMessage>>();
    if (const auto message = startup_history_message(); !message.empty()) {
        append_ui_message(*selected_messages, make_system_message(message));
    }

    std::string input_text;
    int input_cursor_position = 0;
    DoubleEscapeState double_escape_state;
    // Double-press-to-quit confirmation for Ctrl+D on the main thread (with an
    // empty prompt). Secondary threads use one empty-prompt press to close the
    // tab. Stores the key label that armed app exit so the footer hint names
    // the right key, plus the expiry deadline. Ctrl+C quits immediately when idle.
    std::string quit_confirm_key;
    std::chrono::steady_clock::time_point quit_confirm_deadline =
        std::chrono::steady_clock::time_point::min();
    std::mutex  ui_mutex;
    auto direct_shell_state = std::make_shared<DirectShellState>();
    std::atomic<std::size_t> direct_shell_animation_count{0};
    struct PendingAuthenticationRecovery {
        ThreadRuntime::Ptr runtime;
        core::llm::AuthenticationRecoveryRequest request;
        core::auth::AuthenticationProviderDescriptor provider;
        std::string retry_text;
        core::agent::Agent::TurnCallbacks retry_callbacks;
    };
    struct AuthenticationRecoveryState {
        std::optional<PendingAuthenticationRecovery> active;
        std::deque<PendingAuthenticationRecovery> queued;
        std::unordered_set<std::string> providers;
        int selected = 0;
    };
    AuthenticationRecoveryState authentication_recovery_state;
    
    // Rate limit tracking for status bar and notifications
    struct RateLimitState {
        core::llm::protocols::RateLimitInfo latest;
        bool quota_notified_limited = false;
        bool quota_notified_low = false;
        bool quota_notified_critical = false;
    };
    RateLimitState rate_limit_state;
    struct StderrPanelState {
        bool active = false;
        std::vector<std::string> lines;
    };
    StderrPanelState stderr_panel_state;
    struct RemoteActivityPanelState {
        bool active = false;
        std::size_t selected = 0;
    };
    RemoteActivityPanelState remote_activity_panel_state;
    Box remote_activity_pill_box{0, -1, 0, -1};
    bool usage_details_panel_active = false;
    Box usage_status_box{0, -1, 0, -1};
    bool agents_visualizer_panel_active = false;
    Box agents_status_box{0, -1, 0, -1};
    std::vector<Box> agents_tab_hitboxes;
    std::size_t agents_visualizer_selected_file = 0;
    int agents_visualizer_scroll_offset = 0;
    std::vector<Box> thread_tab_hitboxes;
    std::vector<std::string> thread_tab_session_ids;
    SelectionClipboardCopier selection_clipboard_copier;
    auto screen = App::Fullscreen();
    // Enable mouse tracking for scrolling and focus.
    screen.TrackMouse(true);
    // By default FTXUI exits the main loop on Ctrl+C even when the component
    // consumes the event (force_handle_ctrl_c_ defaults to true). Disable that
    // so the CatchEvent handler can use Ctrl+C to stop the active terminal
    // command / generation instead of terminating the whole app. If a future
    // event handler ever fails to consume Ctrl+C, FTXUI still exits gracefully
    // because the "!handled" fallback in its dispatch loop remains active.
    screen.ForceHandleCtrlC(false);
    // Auto-copy any text selected via left-click drag to the system clipboard.
    // This is the cross-platform solution: on Linux the Shift+right-click bypass
    // works in xterm-family terminals, but on macOS iTerm2 uses Option/Alt instead.
    // Keep the FTXUI callback non-blocking; selection changes can fire rapidly
    // while dragging/scrolling, and clipboard providers may spawn processes.
    screen.SelectionChange([&screen, &selection_clipboard_copier]() {
        selection_clipboard_copier.enqueue(screen.GetSelection());
    });
    std::atomic_bool ui_accepting_events{true};
    // Every substantive UI mutation wakes the event loop and advances this
    // revision. Animation-only frames bypass wake_ui(), allowing the history
    // component to retain one immutable message snapshot across all ticks.
    std::atomic<std::size_t> ui_state_revision{1};
    std::mutex ui_event_post_mutex;
    auto wake_ui = [&]() {
        if (!ui_accepting_events.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard lock(ui_event_post_mutex);
        if (ui_accepting_events.load(std::memory_order_relaxed)) {
            ui_state_revision.fetch_add(1, std::memory_order_release);
            screen.PostEvent(Event::Custom);
        }
    };
    auto& remote_activity_hub =
        core::mcp::RemoteActivityHub::get_instance();
    // Owns the whole prompt-editor lifecycle: scratch buffer, backend
    // selection, worker thread and cancellation.
    editor::ExternalEditorController external_editor({
        .wake_ui = [&wake_ui]() { wake_ui(); },
        .with_terminal =
            [&screen](const std::function<void()>& body) { screen.WithRestoredIO(body)(); },
    });
    core::logging::Logger::get_instance().use_callback_sink(
        [&ui_mutex, &stderr_panel_state, &wake_ui](core::logging::Level level,
                                                  std::string line) {
            if (level < core::logging::Level::Error) {
                return;
            }
            {
                std::lock_guard lock(ui_mutex);
                stderr_panel_state.active = true;
                stderr_panel_state.lines.push_back(std::move(line));
                if (stderr_panel_state.lines.size() > kMaxStderrPanelLines) {
                    const auto excess =
                        stderr_panel_state.lines.size() - kMaxStderrPanelLines;
                    stderr_panel_state.lines.erase(
                        stderr_panel_state.lines.begin(),
                        stderr_panel_state.lines.begin()
                            + static_cast<std::vector<std::string>::difference_type>(excess));
                }
            }
            wake_ui();
        });
    TerminalInputModeGuard terminal_input_mode_guard;
    SigintIgnoreGuard sigint_ignore_guard;
    std::atomic<std::size_t> animation_tick = 0;
    std::mutex animation_mutex;
    std::condition_variable animation_cv;
    if (opts.remote_mcp_server_enabled) {
        // Invoked from daemon threads. Capturing run()-local state by reference
        // is safe only because detaching the callback below is a barrier that
        // waits for in-flight invocations; the daemon itself outlives run().
        remote_activity_hub.set_notify_callback([&wake_ui, &animation_cv]() {
            wake_ui();
            animation_cv.notify_one();
        });
    }
    std::function<void()> reset_history_view = [] {};
    core::commands::CommandExecutor cmd_executor;
    // Load layered prompt skills (global → project-local) before describe_commands()
    // so skill commands appear in the autocomplete index.
    core::commands::SkillCommandLoader::discover_and_register(cmd_executor);
    using MentionIndex = std::vector<MentionSuggestion>;
    std::shared_ptr<const MentionIndex> mention_index =
        std::make_shared<const MentionIndex>();
    std::mutex mention_index_mutex;
    auto mention_index_snapshot = [&]() {
        std::lock_guard lock(mention_index_mutex);
        return mention_index;
    };
    std::jthread mention_index_thread(
        [root = std::filesystem::current_path(),
         &mention_index,
         &mention_index_mutex,
         &wake_ui](std::stop_token stop_token) {
            auto built = std::make_shared<const MentionIndex>(
                build_mention_index(root, stop_token));
            if (stop_token.stop_requested()) {
                return;
            }
            {
                std::lock_guard lock(mention_index_mutex);
                mention_index = std::move(built);
            }
            wake_ui();
        });
    auto command_index = cmd_executor.describe_commands();
    if (opts.remote_mcp_server_enabled) {
        command_index.push_back(core::commands::CommandDescriptor{
            .name = "/remote",
            .description = "Open inbound MCP client and tool activity.",
            .accepts_arguments = false,
        });
        std::ranges::sort(command_index, {}, &core::commands::CommandDescriptor::name);
    }

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

    const bool command_line_yolo_enabled = opts.startup_trust.trust_all_tools;
    const bool startup_yolo_enabled = command_line_yolo_enabled
        || parse_approval_mode(config.default_approval_mode) == ApprovalMode::Yolo;

    // Session trust rules used for auto-approval in this session.
    // Rules are canonical strings (for example: shell:git, files:write,
    // tool:write_file) and are managed by `/tools` plus the permission overlay.
    std::unordered_set<std::string> startup_session_allow_rules;
    for (const auto& raw_rule : opts.startup_trust.session_allow_rules) {
        const auto normalized = core::permissions::normalize_session_allow_rule(raw_rule);
        if (!normalized.empty()) {
            startup_session_allow_rules.insert(normalized);
        }
    }

    // Permission overlay state
    struct PermissionState {
        bool                                            active = false;
        std::string                                     tool_name;
        std::string                                     args_preview;
        ToolDiffPreview                                 diff_preview;
        int                                             selected = 0;  // 0-3
        std::string                                     remember_rule;
        std::string                                     allow_label;
        std::shared_ptr<std::promise<bool>>             promise;
        /// Thread asking for permission and a display label; Ctrl+C must stop
        /// this thread's agent (not necessarily the currently visible one).
        ThreadRuntime::Ptr                              origin_runtime;
        std::string                                     origin_label;
    };
    PermissionState perm_state;
    std::mutex permission_prompt_mutex;
    std::condition_variable permission_prompt_cv;
    bool permission_prompt_in_flight = false;
    // Set at shutdown under ui_mutex; permission waiters treat it as an
    // immediate deny so they never block the idle barriers during exit.
    bool permission_prompt_shutdown = false;
    // AskUserQuestion overlays are also single-slot UI resources. Concurrent
    // threads wait here instead of displacing (and silently cancelling) the
    // question already visible to the user.
    std::mutex question_prompt_mutex;
    std::condition_variable question_prompt_cv;
    bool question_prompt_shutdown = false;

    struct ModelPickerState {
        bool active = false;
        int selected = 0; // 0=manual, 1=router, 2=auto
    };
    ModelPickerState model_picker_state;

    struct ModelProviderPickerState {
        bool active = false;
        int selected = 0;
        std::vector<tui::ModelProviderPickerRow> providers;
    };
    ModelProviderPickerState model_provider_picker_state;

    struct ProviderModelPickerState {
        bool active = false;
        int selected = 0;
        std::string provider_name;
        std::vector<tui::ModelPickerRow> models;
    };
    ProviderModelPickerState provider_model_picker_state;

    struct CommandOptionPickerState {
        bool active = false;
        int selected = 0;
        std::string command_name;
        std::string title;
        std::string current_value;
        std::string help_text;
        std::vector<tui::OptionPickerRow> options;
        std::function<std::string(std::string_view)> on_select;
        bool prefill_input = false;
    };
    CommandOptionPickerState command_option_picker_state;

    struct ProviderPickerState {
        bool active = false;
        int selected = 0;
        std::vector<std::string> providers;
        std::function<void(std::optional<std::string>)> on_select;
    };
    ProviderPickerState provider_picker_state;

    struct ReviewPickerState {
        bool active = false;
        int selected = 0;
        ReviewPickerMode mode = ReviewPickerMode::SelectTarget;
        std::string input_text;
        std::vector<ReviewBaseRef> base_refs;
        std::vector<ReviewBaseRef> filtered_base_refs;
        int selected_base_ref = 0;
        std::function<void(std::optional<std::string>)> on_select;
    };
    ReviewPickerState review_picker_state;

    auto refresh_review_base_refs_locked = [&]() {
        review_picker_state.filtered_base_refs =
            filter_review_base_refs(review_picker_state.base_refs, review_picker_state.input_text);
        if (review_picker_state.filtered_base_refs.empty()) {
            review_picker_state.selected_base_ref = 0;
        } else {
            review_picker_state.selected_base_ref = std::clamp(
                review_picker_state.selected_base_ref,
                0,
                static_cast<int>(review_picker_state.filtered_base_refs.size()) - 1);
        }
    };

    struct ReviewActivityState {
        bool active = false;
        std::string hint;
        std::chrono::steady_clock::time_point started_at =
            std::chrono::steady_clock::time_point::min();
    };
    ReviewActivityState review_activity_state;

    struct LocalModelPickerState {
        bool active = false;
        int selected = 0;
        std::filesystem::path current_dir;
        std::vector<tui::LocalModelEntry> entries;
    };
    LocalModelPickerState local_model_picker_state;

    // Reusable filesystem browser. The confirmation callback lives *beside* the
    // state rather than inside it: the component stays free of application
    // concerns, and the per-frame state snapshot stays a cheap value copy.
    FileSystemPickerState file_picker_state;
    std::function<void(const std::filesystem::path&)> file_picker_on_confirm;

    SessionPickerState session_picker_state;

    struct PromptsPickerState {
        bool active = false;
        int selected = 0;
        std::vector<std::string> prompts;  // newest-first
        std::string status_message;        // ephemeral feedback (e.g. copy confirmation)
    };
    PromptsPickerState prompts_picker_state;

    RewindPickerState rewind_picker_state;
    const auto code_run_script_directory = select_code_run_script_directory(
        opts.landrun_mode,
        opts.landrun_environment.runtime_root,
        std::filesystem::temp_directory_path());
    CodeBlockRunServices code_block_run_services(CodeBlockRunDependencies{
        .runner = core::code::make_interactive_code_runner(),
        .working_directory = [] { return std::filesystem::current_path(); },
        .script_directory = code_run_script_directory,
        .landrun_policy = [agent,
                           mode = opts.landrun_mode,
                           compiler = core::landrun::LandrunPolicyCompiler(
                               std::move(opts.landrun_environment))] {
            return compiler.build(agent->workspace_snapshot(), mode);
        },
        .with_restored_terminal = [&screen](std::function<void()> task) {
            auto closure = screen.WithRestoredIO(std::move(task));
            closure();
        },
        .copy_to_clipboard = core::commands::copy_text_to_clipboard,
        .transcript_directory = std::filesystem::temp_directory_path()
            / "filo" / "code-runs",
    });
    CodeBlockRunnerController code_block_runner(code_block_run_services);

    struct ConversationSearchState {
        bool active = false;
        std::string query;
        int selected = 0;
        std::vector<tui::ConversationSearchHit> hits;
    };
    ConversationSearchState conversation_search_state;

    QuestionDialogController question_dialog;

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
    std::vector<std::pair<std::string, Color>> modes;
    modes.reserve(core::agent::kAgentModes.size());
    for (const auto &descriptor : core::agent::kAgentModes) {
      const Color mode_color = [&]() -> Color {
        switch (descriptor.mode) {
        case core::agent::AgentMode::Auto:
          return Color::Magenta;
        case core::agent::AgentMode::Build:
          return Color::Blue;
        case core::agent::AgentMode::Debug:
          return ColorWarn;
        case core::agent::AgentMode::Research:
          return Color::Green;
        case core::agent::AgentMode::Execute:
          return Color::Red;
        }
        return Color::Blue;
      }();
      modes.emplace_back(descriptor.name, mode_color);
    }
    auto normalize_mode = [](std::string mode) {
      return std::string(
          core::agent::to_string(core::agent::agent_mode_from_string(mode)));
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
        // Driven by the editor catalog, so a new backend shows up here (and only
        // on the platforms that ship it) without touching the settings pane.
        std::vector<SettingsChoice> prompt_editor_choices;
        for (const auto& descriptor : editor::builtin_prompt_editors().descriptors()) {
            prompt_editor_choices.push_back(SettingsChoice{
                .value = std::string(descriptor.id),
                .label = std::string(descriptor.label),
            });
        }
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::PromptEditor,
            .label = "General · Prompt Editor",
            .description = "Editor opened by Ctrl+G for the current prompt draft.",
            .choices = std::move(prompt_editor_choices),
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
            .description = "Show or hide timestamps on user messages.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiSpinner,
            .label = "UI · Activity Spinner",
            .description = "Show or hide the animated spinner while Filo is working.",
            .choices = visibility_choices(),
        });
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::UiReasoning,
            .label = "UI · Reasoning",
            .description = "Show or hide the model's collapsible thinking/analyzing disclosure.",
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
        settings_definitions.push_back(SettingsDefinition{
            .key = core::config::ManagedSettingKey::ContextCompression,
            .label = "Session · Tool Compression",
            .description = "How aggressively tool outputs are compacted before entering context.",
            .choices = {
                SettingsChoice{.value = "off",   .label = "Off"},
                SettingsChoice{.value = "light", .label = "Light"},
                SettingsChoice{.value = "full",  .label = "Full"},
                SettingsChoice{.value = "ultra", .label = "Ultra"},
            },
        });
    }

    // ── Session management ────────────────────────────────────────────────────
    auto session_store = std::make_shared<core::session::SessionStore>(
        core::session::SessionStore::default_sessions_dir());

    // Base for auto-generated thread tab names: the primary workspace's leaf
    // directory. Mutable because /workspace change rebases it and retitles
    // auto-named threads (see ThreadRuntimeRegistry::retitle_auto_named);
    // user-renamed threads are never touched.
    const auto project_base_name_for = [](const std::filesystem::path& root) {
        const auto leaf = root.filename().string();
        return leaf.empty() ? std::string{"thread"} : leaf;
    };
    std::string project_thread_base_name = [&] {
        try {
            return project_base_name_for(std::filesystem::current_path());
        } catch (...) {
            return std::string{"thread"};
        }
    }();

    std::string session_id          = core::session::SessionStore::generate_id();
    std::string session_name;       // optional user-assigned name (/rename)
    std::string session_created_at  = core::session::SessionStore::now_iso8601();
    std::string session_file_path;  // computed after first save
    core::session::GoalManager goal_manager{&core::session::SessionStore::now_iso8601};
    // Graph-based goal engine (/goal plan|run|graph|replan). Constructed lazily
    // on first use so it can capture the fully-built command context.
    std::shared_ptr<core::goal::GoalEngine> goal_engine;
    std::string pending_goal_graph_snapshot;
    // If resuming, compute the path now so we can return it in RunResult.
    auto compute_session_path = [&]() {
        core::session::SessionData tmp;
        tmp.session_id  = session_id;
        tmp.created_at  = session_created_at;
        session_file_path = session_store->compute_path(tmp).string();
    };

    // Resolve the startup session from --resume or --continue.
    // --resume wins over --continue if both are somehow set.
    std::optional<core::session::SessionData> startup_data;
    std::string missing_session_label;   // non-empty → warn when not found
    if (opts.resume_session_id.has_value()) {
        const auto& req = *opts.resume_session_id;
        startup_data = req.empty()
            ? session_store->load_most_recent()
            : session_store->load(req);
        missing_session_label = req.empty()
            ? std::string("most recent")
            : std::string(req);
    } else if (opts.continue_last) {
        // --continue scopes to the current project: only sessions whose
        // working_dir matches the current directory are candidates.  If none
        // exist we start a fresh session silently (no warning).
        startup_data = session_store->load_most_recent_for_project(
            std::filesystem::current_path().string());
    }

    core::session::ActiveSessionLeaseManager session_leases{*session_store};
    if (startup_data.has_value()) {
        auto lease = session_leases.reserve(*startup_data);
        if (lease) {
            lease->commit();
        } else {
            append_ui_message(*selected_messages, make_warning_message(std::format(
                "Session '{}' was not resumed because {}. Starting a fresh session instead.",
                startup_data->session_id,
                lease.error())));
            startup_data.reset();
            missing_session_label.clear();
        }
    }

    if (startup_data.has_value()) {
        const auto& data = *startup_data;
        session_id         = data.session_id;
        session_name       = data.name;
        session_created_at = data.created_at;
        goal_manager.restore(data.goal);
        // Deferred: the engine is built on first /goal use, so stash the blob
        // and rehydrate it then.
        pending_goal_graph_snapshot =
            data.goal_graph.has_value() ? data.goal_graph->snapshot : std::string{};
        agent->restore_todos(data.todos);

        // Restore agent state.
        agent->load_history(data.messages, data.context_summary, data.mode);
        agent->set_session_goal(goal_manager.active_goal());

        // Align mode picker.
        for (std::size_t i = 0; i < modes.size(); ++i) {
            if (modes[i].first == data.mode) {
                current_mode_idx = static_cast<int>(i);
                break;
            }
        }

        // Restore session file path.
        session_file_path = session_store->compute_path(data).string();

        *selected_messages = build_resumed_ui_messages(
            data,
            SessionReplayOptions{.include_continue_hint = true});

        // Warn when a resumed session originated in a different project than
        // the current directory: the agent's history references the old paths
        // but tools will act on the current directory. This is almost never
        // intended, so surface it prominently instead of resuming silently.
        if (auto notice = core::session::SessionStore::working_dir_mismatch_notice(
                data.working_dir, std::filesystem::current_path().string());
            notice.has_value()) {
            append_ui_message(*selected_messages, make_warning_message(std::move(*notice)));
        }
    } else if (!missing_session_label.empty()) {
        // --resume pointed at something that doesn't exist — warn the user.
        append_ui_message(*selected_messages, make_warning_message(
            std::format("Session '{}' not found. Starting a fresh session.",
                missing_session_label)));
    }

    compute_session_path();
    if (!session_leases.retain()) {
        core::session::SessionData fresh_session;
        fresh_session.session_id = session_id;
        fresh_session.created_at = session_created_at;
        auto lease = session_leases.reserve(fresh_session);
        if (!lease) {
            core::logging::error(
                "Could not reserve fresh session '{}': {}",
                session_id, lease.error());
            return {};
        }
        lease->commit();
    }
    core::budget::BudgetTracker::get_instance().reset_session(session_id);
    session_stats_registry->reset(session_id);
    agent->set_session_id(session_id);

    ThreadRuntimeRegistry thread_runtimes;
    auto current_runtime = std::make_shared<ThreadRuntime>(
        ThreadRuntimeMetadata{
            .session_id = session_id,
            .thread_name = "main",
            .session_name = session_name,
            .created_at = session_created_at,
            .file_path = session_file_path,
            .provider = active_provider_name,
            .model = active_model_name,
            .model_selection = ModelSelectionSnapshot{
                .mode = model_selection_mode,
                .manual_provider_name = manual_provider_name,
                .manual_model_name = manual_model_name,
                .router_policy = active_router_policy,
            },
            .yolo_enabled = startup_yolo_enabled,
            .permission_rules = {},
            .goal = startup_data.has_value() ? startup_data->goal : std::nullopt,
            .goal_graph = startup_data.has_value() ? startup_data->goal_graph : std::nullopt,
        },
        agent,
        selected_messages,
        session_leases.retain(session_id));
    // Stable runtime identity for ordering. Labels and session ids can change
    // independently, but the process's initial thread must always stay first.
    const auto main_runtime = current_runtime;
    if (!thread_runtimes.insert(current_runtime)
        || !thread_runtimes.select(session_id)) {
        core::logging::error("Could not initialize the thread runtime registry.");
        return {};
    }

    auto allocate_project_thread_name = [&]() {
        std::unordered_set<std::string> used_names;
        for (const auto& runtime : thread_runtimes.snapshot()) {
            if (const auto name = runtime->metadata().thread_name; !name.empty()) {
                used_names.insert(name);
            }
        }
        if (!used_names.contains(project_thread_base_name)) {
            return project_thread_base_name;
        }
        for (std::size_t ordinal = 2;; ++ordinal) {
            auto candidate = std::format("{} {}", project_thread_base_name, ordinal);
            if (!used_names.contains(candidate)) {
                return candidate;
            }
        }
    };

    auto sync_runtime_metadata = [&]() {
        std::optional<core::session::SessionGoalGraph> graph;
        if (goal_engine) {
            const auto status = goal_engine->status();
            if (status.has_graph) {
                graph = core::session::SessionGoalGraph{
                    .plan_version = status.plan_version,
                    .run_state = std::string(core::goal::to_string(status.run_state)),
                    .snapshot = goal_engine->snapshot_json(),
                    .updated_at = core::session::SessionStore::now_iso8601(),
                };
            }
        }
        current_runtime->mutate_metadata([&](ThreadRuntimeMetadata& metadata) {
            metadata.session_id = session_id;
            metadata.session_name = session_name;
            metadata.created_at = session_created_at;
            metadata.file_path = session_file_path;
            metadata.provider = active_provider_name;
            metadata.model = active_model_name;
            metadata.model_selection = ModelSelectionSnapshot{
                .mode = model_selection_mode,
                .manual_provider_name = manual_provider_name,
                .manual_model_name = manual_model_name,
                .router_policy = active_router_policy,
            };
            metadata.previous_model_selection = previous_model_selection;
            metadata.goal = goal_manager.current();
            metadata.goal_graph = std::move(graph);
        });
    };

    // ── Helper lambdas ───────────────────────────────────────────────────────

    auto navigate_history_prev = [&]() -> bool {
        return prompt_history.navigate_prev(input_text, input_cursor_position);
    };

    auto navigate_history_next = [&]() -> bool {
        return prompt_history.navigate_next(input_text, input_cursor_position);
    };

    auto clear_screen = [&]() {
        if (agent->turn_in_progress()
            || (direct_shell_state
                && direct_shell_state->has_active_for(session_id))) {
            {
                std::lock_guard lock(ui_mutex);
                append_ui_message(*selected_messages, make_warning_message(
                    "Stop all active work before clearing the session. "
                    "Filo kept the current history intact."));
            }
            wake_ui();
            return;
        }
        current_runtime->reset_turn_state();
        {
            std::lock_guard lock(ui_mutex);
            selected_messages->clear();
            review_activity_state.active = false;
            review_activity_state.hint.clear();
            review_activity_state.started_at = std::chrono::steady_clock::time_point::min();
            if (const auto message = startup_history_message(); !message.empty()) {
                append_ui_message(*selected_messages, make_system_message(message));
            }
            goal_manager.clear();
            goal_engine.reset();
            pending_goal_graph_snapshot.clear();
        }
        reset_history_view();
        animation_cv.notify_one();
        agent->set_session_goal(std::nullopt);
        agent->clear_todos();
        agent->clear_history();
        core::budget::BudgetTracker::get_instance().reset_session(session_id);
        session_stats_registry->reset(session_id);

        // Start a fresh session after /clear, with a proper exclusive lease.
        const std::string new_session_id = core::session::SessionStore::generate_id();
        const std::string new_created_at = core::session::SessionStore::now_iso8601();
        core::session::SessionData fresh;
        fresh.session_id = new_session_id;
        fresh.created_at = new_created_at;
        auto next_lease = session_leases.reserve(fresh);
        if (!next_lease) {
            {
                std::lock_guard lock(ui_mutex);
                append_ui_message(*selected_messages, make_warning_message(std::format(
                    "Could not start a fresh session: {}. History was still cleared, "
                    "but the previous session id was kept.",
                    next_lease.error())));
            }
            wake_ui();
            return;
        }

        const std::string previous_session_id = session_id;
        session_id         = new_session_id;
        session_name.clear();
        session_created_at = new_created_at;
        agent->set_session_id(session_id);
        compute_session_path();
        next_lease->commit();
        current_runtime->set_lease(session_leases.retain(session_id));
        sync_runtime_metadata();
        if (!thread_runtimes.rekey(previous_session_id, session_id)) {
            core::logging::warn(
                "Could not rekey cleared thread runtime {} to {}.",
                previous_session_id,
                session_id);
        }
        session_leases.release(previous_session_id);
        wake_ui();
    };

    auto append_history = [&](const std::string& str) {
        std::lock_guard lock(ui_mutex);
        if (selected_messages->empty()
            || selected_messages->back().type != MessageType::System) {
            append_ui_message(*selected_messages, make_system_message(str));
        } else {
            selected_messages->back().text += str;
        }
        wake_ui();
    };

    auto list_active_terminals = [&]() {
        std::vector<core::commands::ActiveTerminalInfo> terminals;
        const auto now = std::chrono::steady_clock::now();
        for (const auto& command : core::tools::ShellTool::active_commands()) {
            if (command.session_id != session_id) {
                continue;
            }
            terminals.push_back({
                .session_id = command.session_id,
                .command = command.command,
                .working_dir = command.working_dir,
                .tool_call_id = command.tool_call_id,
                .elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - command.started_at),
            });
        }
        if (const auto direct_shell = direct_shell_state->active_info(now);
            direct_shell.has_value() && direct_shell->session_id == session_id) {
            terminals.push_back(*direct_shell);
        }
        return terminals;
    };

    // Single source of truth for "is there anything Ctrl+C/Esc could stop?".
    // Both stop_active_terminal() and the Ctrl+C quit-when-idle gate rely on
    // this predicate so they can never disagree about what "idle" means.
    auto has_stoppable_activity = [&]() -> bool {
        // Account for hidden threads too: with parallel sessions, work can be
        // running in a thread that is not currently selected, and Ctrl+C
        // must never quit while any of it is active.
        return !list_active_terminals().empty()
            || !thread_runtimes.running_session_ids().empty();
    };

    auto stop_active_terminal = [&]() -> core::commands::CommandOperationResult {
        if (!has_stoppable_activity()) {
            return {.ok = false, .message = "Nothing is currently running."};
        }
        const bool had_terminal = !list_active_terminals().empty();
        const bool had_direct_shell =
            direct_shell_state && direct_shell_state->has_active_for(session_id);

        // Stop the current thread's agent, plus every hidden thread that is
        // still working, so a single stop gesture settles the whole session.
        agent->request_stop();
        std::size_t hidden_stopped = 0;
        for (const auto& runtime : thread_runtimes.snapshot()) {
            if (runtime->turn_active() && runtime->agent() != agent) {
                runtime->agent()->request_stop();
                ++hidden_stopped;
            }
        }
        if (had_direct_shell && direct_shell_state) {
            [[maybe_unused]] const bool interrupted =
                direct_shell_state->interrupt_active_for(session_id);
        }
        std::string message = had_terminal
            ? "Stop requested for the active terminal command."
            : "Stop requested for the active generation and any running subagents.";
        if (hidden_stopped > 0) {
            message += std::format(
                " Also stopping {} hidden thread{}.",
                hidden_stopped,
                hidden_stopped == 1 ? "" : "s");
        }
        return {.ok = true, .message = message};
    };

    auto append_assistant_output = [&](const std::string& str) {
        std::lock_guard lock(ui_mutex);
        append_ui_message(
            *selected_messages,
            make_assistant_message(str, current_time_str(), false));
        wake_ui();
    };

    auto animation_cadence = [&]() -> std::optional<AnimationCadence> {
        // Race fix: this lambda runs on the animation thread while thread
        // swaps (resume_session / start_new_thread) repoint `current_runtime`
        // and `selected_messages` on the UI thread. Read both pointers under
        // ui_mutex so the animation thread never observes a torn swap; all
        // reads below then go through the local strong references.
        ThreadRuntime::Ptr runtime_snapshot;
        std::shared_ptr<std::vector<UiMessage>> messages_snapshot;
        {
            std::lock_guard lock(ui_mutex);
            runtime_snapshot = current_runtime;
            messages_snapshot = selected_messages;
        }
        const bool assistant_active =
            runtime_snapshot && runtime_snapshot->turn_active();
        bool remote_activity_active = false;
        bool remote_completion_fresh = false;
        if (opts.remote_mcp_server_enabled) {
            const auto remote_snapshot = remote_activity_hub.snapshot(false);
            remote_activity_active = std::ranges::any_of(
                remote_snapshot.activities,
                [](const core::mcp::RemoteToolActivity& activity) {
                    return activity.status == core::mcp::RemoteToolStatus::running;
                });
            if (!remote_snapshot.activities.empty()) {
                const auto& latest = remote_snapshot.activities.front();
                remote_completion_fresh =
                    latest.finished_at != std::chrono::steady_clock::time_point{}
                    && std::chrono::steady_clock::now() - latest.finished_at
                        <= std::chrono::seconds{2};
            }
        }
        std::lock_guard lock(ui_mutex);
        const bool review_active = review_activity_state.active;
        bool conversation_animation_active =
            direct_shell_animation_count.load(std::memory_order_relaxed) > 0
            || remote_activity_active;
        if (!assistant_active && !review_active && !conversation_animation_active) {
            // Defensive fallback for restored or externally-updated activity
            // cards that are not owned by the normal assistant/shell counters.
            conversation_animation_active = messages_snapshot
                && conversation_uses_animation(*messages_snapshot, true);
        }
        const auto cadence = select_animation_cadence(
            ui_show_spinner.load(std::memory_order_relaxed),
            assistant_active,
            review_active,
            conversation_animation_active);
        if (!cadence.has_value() && remote_completion_fresh) {
            return AnimationCadence{
                .period = std::chrono::seconds(1),
                .advance_frame = false,
            };
        }
        if (!cadence.has_value() && ui_show_banner) {
            return AnimationCadence{
                .period = std::chrono::seconds(1),
                .advance_frame = false,
            };
        }
        return cadence;
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

        for (std::size_t i = 0; i < selected_messages->size(); ++i) {
            const auto& message = (*selected_messages)[i];
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
            rate_limit_state.latest.requests_remaining,
            rate_limit_state.latest.requests_limit);
        const float tok_util = utilization_from_remaining(
            rate_limit_state.latest.tokens_remaining,
            rate_limit_state.latest.tokens_limit);
        float unified_util = 0.0f;
        for (const auto& w : rate_limit_state.latest.usage_windows) {
            unified_util = std::max(unified_util, w.utilization);
        }
        const float api_util = std::max(req_util, tok_util);
        const float max_util = std::max(unified_util, api_util);

        // Maps short labels to human-readable window names for notification messages.
        auto window_name = [](std::string_view label) -> std::string {
            if (label == "4h") return "4-hour window";
            if (label == "5h") return "5-hour window";
            if (label == "7d") return "7-day window";
            if (label == "30d") return "30-day window";
            return std::string(label) + " window";
        };

        auto append_trigger_details = [&](std::string& msg, float threshold) {
            for (const auto& w : rate_limit_state.latest.usage_windows) {
                if (w.utilization >= threshold) {
                    msg += std::format("    {}: {:.0f}% used\n",
                                       window_name(w.label), w.utilization * 100.0f);
                }
            }
            if (req_util >= threshold) {
                msg += std::format("    Request window: {:.0f}% used ({}/{})\n",
                                   req_util * 100.0f,
                                   rate_limit_state.latest.requests_remaining,
                                   rate_limit_state.latest.requests_limit);
            }
            if (tok_util >= threshold) {
                msg += std::format("    Token window: {:.0f}% used ({}/{})\n",
                                   tok_util * 100.0f,
                                   rate_limit_state.latest.tokens_remaining,
                                   rate_limit_state.latest.tokens_limit);
            }
        };

        // Providers that expose no usage windows or numeric limits (notably
        // the QwenCloud Token Plan) can still report a hard 429 rate-limit
        // state. Surface it with its own one-time notification and skip the
        // percentage tiers, which have no data to work with here.
        const bool hard_limited =
            (rate_limit_state.latest.is_rate_limited
             || rate_limit_state.latest.unified_status == "rate_limited")
            && rate_limit_state.latest.usage_windows.empty()
            && rate_limit_state.latest.requests_limit <= 0
            && rate_limit_state.latest.tokens_limit <= 0;

        if (hard_limited) {
            if (!rate_limit_state.quota_notified_limited) {
                rate_limit_state.quota_notified_limited = true;
                std::string msg = "\n\xe2\x9b\x94  Rate limited by the provider";
                if (rate_limit_state.latest.retry_after > 0) {
                    msg += std::format(" (retry in {}s)",
                                       rate_limit_state.latest.retry_after);
                }
                msg += ". The request was blocked. Try again after the "
                       "provider's limit resets.\n";
                append_history(msg);
            }
            return;
        }
        // No longer hard-limited: re-arm the notification for the next time.
        rate_limit_state.quota_notified_limited = false;

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
        return command_line_yolo_enabled
            || current_runtime->metadata().yolo_enabled;
    };

    auto set_yolo_mode_enabled = [&](bool enabled) {
        {
            std::lock_guard lock(ui_mutex);
            current_runtime->mutate_metadata(
                [enabled, command_line_yolo_enabled](ThreadRuntimeMetadata& metadata) {
                    metadata.yolo_enabled = command_line_yolo_enabled || enabled;
                });
        }
        wake_ui();
        return command_line_yolo_enabled || enabled;
    };

    auto list_tool_rules = [&]() -> std::vector<std::string> {
        std::vector<std::string> rules;
        {
            std::lock_guard lock(ui_mutex);
            const auto metadata = current_runtime->metadata();
            rules.assign(metadata.permission_rules.begin(), metadata.permission_rules.end());
            rules.insert(
                rules.end(),
                startup_session_allow_rules.begin(),
                startup_session_allow_rules.end());
        }
        std::sort(rules.begin(), rules.end());
        rules.erase(std::unique(rules.begin(), rules.end()), rules.end());
        return rules;
    };

    auto add_tool_rule = [&](std::string_view raw_rule)
        -> core::commands::CommandOperationResult {
        const std::string normalized =
            core::permissions::normalize_session_allow_rule(raw_rule);
        if (normalized.empty()) {
            return {
                .ok = false,
                .message = "Trust rule cannot be empty.",
            };
        }

        bool inserted = false;
        {
            std::lock_guard lock(ui_mutex);
            if (!startup_session_allow_rules.contains(normalized)) {
                current_runtime->mutate_metadata([&](ThreadRuntimeMetadata& metadata) {
                    inserted = metadata.permission_rules.insert(normalized).second;
                });
            }
        }

        wake_ui();
        return {
            .ok = true,
            .message = inserted
                ? std::format(
                    "Added `{}` ({})",
                    normalized,
                    core::permissions::describe_session_allow_rule(normalized))
                : std::format(
                    "Rule already active: `{}` ({})",
                    normalized,
                    core::permissions::describe_session_allow_rule(normalized)),
        };
    };

    auto remove_tool_rule = [&](std::string_view raw_rule)
        -> core::commands::CommandOperationResult {
        const std::string normalized =
            core::permissions::normalize_session_allow_rule(raw_rule);
        if (normalized.empty()) {
            return {
                .ok = false,
                .message = "Trust rule cannot be empty.",
            };
        }
        if (startup_session_allow_rules.contains(normalized)) {
            return {
                .ok = false,
                .message = std::format(
                    "Rule `{}` comes from the command line and applies to every thread.",
                    normalized),
            };
        }

        bool removed = false;
        {
            std::lock_guard lock(ui_mutex);
            current_runtime->mutate_metadata([&](ThreadRuntimeMetadata& metadata) {
                removed = metadata.permission_rules.erase(normalized) > 0;
            });
        }

        wake_ui();
        return removed
            ? core::commands::CommandOperationResult{
                .ok = true,
                .message = std::format(
                    "Removed `{}` ({})",
                    normalized,
                    core::permissions::describe_session_allow_rule(normalized)),
            }
            : core::commands::CommandOperationResult{
                .ok = false,
                .message = std::format(
                    "Rule not found: `{}`",
                    normalized),
            };
    };

    auto clear_tool_rules = [&]() -> core::commands::CommandOperationResult {
        std::size_t removed = 0;
        {
            std::lock_guard lock(ui_mutex);
            current_runtime->mutate_metadata([&](ThreadRuntimeMetadata& metadata) {
                removed = metadata.permission_rules.size();
                metadata.permission_rules.clear();
            });
        }
        wake_ui();
        return {
            .ok = true,
            .message = removed == 0
                ? std::string("No trust rules were set.")
                : std::format("Cleared {} trust rule{}.", removed, removed == 1 ? "" : "s"),
        };
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

    std::function<void(const ThreadRuntime::Ptr&)> configure_runtime_agent;
    std::function<void(const ThreadRuntime::Ptr&)> configure_runtime_efficiency;
    using SaveResult = std::optional<std::string>;
    std::function<std::future<SaveResult>(ThreadRuntime::Ptr)> save_runtime_snapshot;

    auto make_thread_agent = [&](const core::session::SessionData& data)
        -> std::expected<std::shared_ptr<core::agent::Agent>, std::string> {
        std::shared_ptr<core::llm::LLMProvider> base_provider;
        try {
            if (!data.provider.empty()
                && core::llm::contains_provider(registered_providers, data.provider)) {
                base_provider = provider_manager.get_provider(data.provider);
            } else {
                base_provider = current_runtime->agent()->get_provider();
            }
        } catch (const std::exception& error) {
            return std::unexpected(error.what());
        }
        if (!base_provider) {
            return std::unexpected("No provider is available for the thread.");
        }

        auto isolated_provider = base_provider->fork_for_parallel_request();
        if (!isolated_provider) {
            return std::unexpected(std::format(
                "Provider '{}' cannot create an isolated concurrent request. "
                "The current thread is still running; wait for it to finish or "
                "select a provider with parallel-request support.",
                data.provider.empty() ? active_provider_name : data.provider));
        }

        auto context = core::context::make_session_context(
            core::workspace::Workspace::get_instance().snapshot(),
            core::context::SessionTransport::cli,
            data.session_id);
        context.steering_policy = agent ? agent->session_context_snapshot().steering_policy : opts.steering_policy;
        auto created = std::make_shared<core::agent::Agent>(
            std::move(isolated_provider),
            tool_manager,
            std::move(context),
            core::agent::ToolResultStore::default_root(),
            std::shared_ptr<core::power::SleepInhibitor>{},
            session_stats_registry,
            &core::budget::BudgetTracker::get_instance(),
            memory_system,
            workspace_leases);
        created->set_active_provider_name(
            data.provider.empty() ? active_provider_name : data.provider);
        created->set_auto_compact_threshold(
            config.auto_compact_threshold,
            !config.auto_compact_threshold_explicit);
        created->set_effort_level(session_effort_value);
        created->set_active_model(data.model.empty() ? active_model_name : data.model);
        created->load_history(data.messages, data.context_summary, data.mode);
        created->restore_todos(data.todos);
        return created;
    };

    // Human label for a thread: its name when named, else its short id.
    // Used by permission/question overlays to identify hidden requesters.
    auto thread_display_label = [](const ThreadRuntime::Ptr& runtime) -> std::string {
        if (!runtime) {
            return {};
        }
        const auto metadata = runtime->metadata();
        return metadata.thread_name.empty() ? metadata.session_id : metadata.thread_name;
    };

    // Snapshot a live thread's in-memory state into SessionData so resume
    // paths never fall back to stale on-disk data for a runtime that is
    // already loaded (single source of truth = the runtime itself).
    auto live_session_data = [&](const ThreadRuntime::Ptr& runtime)
        -> std::optional<core::session::SessionData> {
        if (!runtime) {
            return std::nullopt;
        }
        const auto metadata = runtime->metadata();
        core::session::SessionData data;
        data.session_id = metadata.session_id;
        data.name = metadata.session_name;
        data.created_at = metadata.created_at;
        data.last_active_at =
            core::session::SessionStore::to_iso8601(runtime->last_activity());
        data.working_dir = std::filesystem::current_path().string();
        data.provider = metadata.provider;
        data.model = metadata.model;
        data.mode = runtime->agent()->get_mode();
        data.context_summary = runtime->agent()->get_context_summary();
        data.messages = runtime->agent()->get_history();
        data.goal = metadata.goal;
        data.goal_graph = metadata.goal_graph;
        data.todos = runtime->agent()->get_todos();
        return data;
    };

    auto resume_session = [&](const core::session::SessionData& data)
        -> std::optional<std::string> {
        if (data.session_id == session_id) {
            return std::nullopt;
        }

        sync_runtime_metadata();
        if (save_runtime_snapshot && !agent->get_history().empty()) {
            save_runtime_snapshot(current_runtime);
        }

        auto next_lease = session_leases.reserve(data);
        if (!next_lease) return next_lease.error();

        auto target_runtime = thread_runtimes.find(data.session_id);
        if (!target_runtime) {
            auto next_agent = make_thread_agent(data);
            if (!next_agent) {
                return next_agent.error();
            }
            auto next_messages = std::make_shared<std::vector<UiMessage>>(
                build_resumed_ui_messages(data));
            if (auto notice = core::session::SessionStore::working_dir_mismatch_notice(
                    data.working_dir, std::filesystem::current_path().string());
                notice.has_value()) {
                append_ui_message(*next_messages, make_warning_message(std::move(*notice)));
            }
            next_lease->commit();
            target_runtime = std::make_shared<ThreadRuntime>(
                ThreadRuntimeMetadata{
                    .session_id = data.session_id,
                    .thread_name = allocate_project_thread_name(),
                    .auto_thread_name = true,
                    .session_name = data.name,
                    .created_at = data.created_at,
                    .file_path = session_store->compute_path(data).string(),
                    .provider = data.provider,
                    .model = data.model,
                    .model_selection = ModelSelectionSnapshot{
                        .mode = ModelSelectionMode::Manual,
                        .manual_provider_name = data.provider,
                        .manual_model_name = data.model,
                    },
                    .yolo_enabled = startup_yolo_enabled,
                    .permission_rules = {},
                    .goal = data.goal,
                    .goal_graph = data.goal_graph,
                },
                *next_agent,
                std::move(next_messages),
                session_leases.retain(data.session_id));
            if (!thread_runtimes.insert(target_runtime)) {
                return std::string("Could not register the resumed thread runtime.");
            }
            if (configure_runtime_agent) {
                configure_runtime_agent(target_runtime);
            }
        } else {
            next_lease->commit();
        }

        const auto metadata = target_runtime->metadata();
        {
            std::lock_guard lock(ui_mutex);
            current_runtime = target_runtime;
            agent = target_runtime->agent();
            llm_provider = agent->get_provider();
            session_effort_value = agent->get_effort_level();
            router_provider = std::dynamic_pointer_cast<
                core::llm::providers::RouterProvider>(llm_provider);
            selected_messages = target_runtime->messages();
            session_id         = metadata.session_id;
            session_name       = metadata.session_name;
            session_created_at = metadata.created_at;
            session_file_path  = metadata.file_path;
            goal_manager.restore(data.goal);
            pending_goal_graph_snapshot =
                data.goal_graph.has_value() ? data.goal_graph->snapshot : std::string{};
            goal_engine.reset(); // rebuilt against the resumed session
            agent->set_session_goal(goal_manager.active_goal());

            // Align mode picker.
            for (std::size_t i = 0; i < modes.size(); ++i) {
                if (modes[i].first == agent->get_mode()) {
                    current_mode_idx = static_cast<int>(i);
                    break;
                }
            }

            // Restore the selector state owned by this runtime. A thread may
            // be in Manual, Router, or Auto mode independently of every other
            // live thread.
            model_selection_mode = metadata.model_selection.mode;
            manual_provider_name = metadata.model_selection.manual_provider_name;
            manual_model_name = metadata.model_selection.manual_model_name;
            active_router_policy = metadata.model_selection.router_policy;
            previous_model_selection = metadata.previous_model_selection;
            refresh_status_labels();
        }
        static_cast<void>(thread_runtimes.select(session_id));
        sync_mcp_sampling_backend(
            agent->get_provider(),
            model_selection_mode == ModelSelectionMode::Manual
                ? manual_model_name
                : std::string{});
        reset_history_view();
        animation_cv.notify_one();
        wake_ui();
        return std::nullopt;
    };

    auto active_thread_catalogue = [&]() {
        std::vector<core::session::SessionInfo> catalogue;
        for (const auto& runtime : thread_runtimes.snapshot()) {
            const auto metadata = runtime->metadata();
            core::session::SessionInfo live;
            live.session_id = metadata.session_id;
            live.name = metadata.thread_name;
            live.created_at = metadata.created_at;
            live.last_active_at =
                core::session::SessionStore::to_iso8601(runtime->last_activity());
            live.working_dir = std::filesystem::current_path().string();
            live.provider = metadata.provider;
            live.model = metadata.model;
            live.mode = runtime->agent()->get_mode();
            live.preview = core::session::first_user_message_preview(
                runtime->agent()->get_history());
            live.turn_count =
                session_stats_registry->snapshot(metadata.session_id).turn_count;
            live.path = metadata.file_path;
            catalogue.push_back(std::move(live));
        }
        core::session::order_active_threads(catalogue, main_runtime->session_id());
        return catalogue;
    };

    auto open_threads_picker = [&]() -> bool {
        auto catalogue = active_thread_catalogue();
        std::string current_id;
        {
            std::lock_guard lock(ui_mutex);
            current_id = session_id;
            open_thread_picker(session_picker_state, std::move(catalogue), current_id);
        }
        wake_ui();
        return true;
    };

    auto open_sessions_picker = [&]() -> bool {
        auto catalogue = session_store->list();
        if (catalogue.empty()) {
            return false;
        }
        std::string current_id;
        {
            std::lock_guard lock(ui_mutex);
            current_id = session_id;
            open_session_picker(session_picker_state, std::move(catalogue), current_id);
        }
        wake_ui();
        return true;
    };

    auto open_prompts_picker = [&]() -> bool {
        {
            std::lock_guard lock(ui_mutex);
            static_cast<void>(history_store->load());
            prompts_picker_state.prompts = history_store->entries_newest_first();
            if (prompts_picker_state.prompts.empty()) {
                return false;
            }
            prompts_picker_state.active = true;
            prompts_picker_state.selected = 0;
            prompts_picker_state.status_message.clear();
        }
        prompt_history.reload();
        wake_ui();
        return true;
    };

    auto open_review_picker = [&](std::function<void(std::optional<std::string>)> on_select) {
        auto base_refs = core::scm::ScmFactory::create(settings_working_dir)->list_branch_refs();
        {
            std::lock_guard lock(ui_mutex);
            review_picker_state.active = true;
            review_picker_state.selected = 0;
            review_picker_state.mode = ReviewPickerMode::SelectTarget;
            review_picker_state.input_text.clear();
            review_picker_state.base_refs = std::move(base_refs);
            review_picker_state.selected_base_ref = 0;
            refresh_review_base_refs_locked();
            review_picker_state.on_select = std::move(on_select);
        }
        wake_ui();
    };

    auto set_review_activity = [&](bool active, const std::string& hint) {
        {
            std::lock_guard lock(ui_mutex);
            review_activity_state.active = active;
            if (active) {
                review_activity_state.hint = hint;
                review_activity_state.started_at = std::chrono::steady_clock::now();
            } else {
                review_activity_state.hint.clear();
                review_activity_state.started_at = std::chrono::steady_clock::time_point::min();
            }
        }
        animation_cv.notify_one();
        wake_ui();
    };

    using SessionBranchIds = std::pair<std::string, std::string>;
    auto branch_session = [&](const std::vector<core::llm::Message>& branch_messages,
                              std::string branch_context)
        -> std::expected<SessionBranchIds, std::string> {
        if (agent->turn_in_progress()
            || (direct_shell_state
                && direct_shell_state->has_active_for(session_id))) {
            return std::unexpected(
                "Stop all active work before replacing or branching its history.");
        }

        const auto original_messages = agent->get_history();
        const std::string snap_mode = agent->get_mode();
        const std::string snap_context = agent->get_context_summary();

        std::string old_session_id;
        std::string old_session_name;
        std::string old_created_at;
        std::string provider_name;
        std::string model_name;
        std::optional<core::session::SessionGoal> snap_goal;
        auto snap_todos = agent->get_todos();
        {
            std::lock_guard lock(ui_mutex);
            old_session_id = session_id;
            old_session_name = session_name;
            old_created_at = session_created_at;
            provider_name = active_provider_name;
            model_name = active_model_name;
            snap_goal = goal_manager.current();
        }

        core::session::SessionData original;
        original.session_id      = old_session_id;
        original.name            = old_session_name;
        original.created_at      = old_created_at;
        original.last_active_at  = core::session::SessionStore::now_iso8601();
        original.working_dir     = std::filesystem::current_path().string();
        original.provider        = provider_name;
        original.model           = model_name;
        original.mode            = snap_mode;
        original.context_summary = snap_context;
        original.messages        = original_messages;
        original.goal            = snap_goal;
        original.todos           = snap_todos;

        const auto& budget = core::budget::BudgetTracker::get_instance();
        const auto total = budget.session_total(old_session_id);
        original.stats.prompt_tokens = total.prompt_tokens;
        original.stats.completion_tokens = total.completion_tokens;
        original.stats.cost_usd = budget.session_cost_usd(old_session_id);

        const auto stats_snapshot =
            session_stats_registry->snapshot(old_session_id);
        original.stats.turn_count = stats_snapshot.turn_count;
        original.stats.tool_calls_total = stats_snapshot.tool_calls_total;
        original.stats.tool_calls_success = stats_snapshot.tool_calls_success;
        original.handoff_summary = core::session::build_handoff_summary(original);

        const std::string new_session_id = core::session::SessionStore::generate_id();
        const std::string new_created_at = core::session::SessionStore::now_iso8601();
        auto branch = original;
        branch.session_id = new_session_id;
        branch.name.clear();  // the fork starts unnamed; the original keeps its name
        branch.created_at = new_created_at;
        branch.last_active_at = new_created_at;
        branch.context_summary = std::move(branch_context);
        branch.messages = branch_messages;
        branch.handoff_summary = core::session::build_handoff_summary(branch);

        auto branch_lease = session_leases.reserve(branch);
        if (!branch_lease) {
            return std::unexpected(std::format(
                "Failed to reserve branch session: {}", branch_lease.error()));
        }

        std::string error;
        const auto save_generation = current_runtime->request_save();
        {
            std::lock_guard save_lock(current_runtime->save_mutex());
            if (!current_runtime->is_latest_save(save_generation)) {
                return std::unexpected("A newer session snapshot superseded the branch request.");
            }
            if (!session_store->save(original, &error)) {
                return std::unexpected(std::format(
                    "Failed to preserve session: {}",
                    error.empty() ? std::string("unknown save error") : error));
            }
            error.clear();
            if (!session_store->save(branch, &error)) {
                return std::unexpected(std::format(
                    "Failed to create session branch: {}",
                    error.empty() ? std::string("unknown save error") : error));
            }
        }

        {
            std::lock_guard lock(ui_mutex);
            session_id = new_session_id;
            session_name.clear();
            session_created_at = new_created_at;
            session_file_path = session_store->compute_path(branch).string();
            branch_lease->commit();
        }
        agent->set_session_id(new_session_id);
        current_runtime->set_lease(session_leases.retain(new_session_id));
        sync_runtime_metadata();
        if (!thread_runtimes.rekey(old_session_id, new_session_id)) {
            return std::unexpected("Failed to rekey the branch runtime.");
        }
        session_leases.release(old_session_id);
        wake_ui();
        return SessionBranchIds{old_session_id, new_session_id};
    };

    auto fork_session = [&]() -> std::string {
        const auto result = branch_session(
            agent->get_history(),
            agent->get_context_summary());
        if (!result) return result.error();
        return std::format(
            "Forked session {} into {}.",
            result->first,
            result->second);
    };

    configure_runtime_efficiency = [&](const ThreadRuntime::Ptr& runtime) {
      auto runtime_agent = runtime->agent();
      runtime_agent->set_efficiency_decision_fn(
        [runtime, runtime_agent, session_store, session_stats_registry,
         &thread_runtimes, &ui_mutex, &session_id, &session_name, &session_created_at,
         &session_file_path, &active_provider_name, &active_model_name,
         &selected_messages, &wake_ui, &goal_manager,
         &session_leases](const core::session::SessionEfficiencyDecision& decision) {
            // Rotation rewrites session identity and selected presentation
            // state. Defer it while this runtime is hidden; the next visible
            // turn may evaluate the same pressure again without data loss.
            if (thread_runtimes.current() != runtime) {
                return;
            }
            auto snap_messages = runtime_agent->get_history();
            auto snap_mode = runtime_agent->get_mode();
            auto snap_context = runtime_agent->get_context_summary();
            auto snap_todos = runtime_agent->get_todos();

            core::session::SessionData archived;
            {
                std::lock_guard lock(ui_mutex);
                archived.session_id = session_id;
                archived.name = session_name;
                archived.created_at = session_created_at;
                archived.provider = active_provider_name;
                archived.model = active_model_name;
                archived.goal = goal_manager.current();
                archived.todos = std::move(snap_todos);
            }
            archived.last_active_at = core::session::SessionStore::now_iso8601();
            archived.working_dir = std::filesystem::current_path().string();
            archived.mode = snap_mode;
            archived.context_summary = snap_context;
            archived.messages = snap_messages;
            archived.handoff_summary = core::session::build_handoff_summary(archived);

            const auto& budget = core::budget::BudgetTracker::get_instance();
            const auto total = budget.session_total(archived.session_id);
            archived.stats.prompt_tokens = total.prompt_tokens;
            archived.stats.completion_tokens = total.completion_tokens;
            archived.stats.cost_usd = budget.session_cost_usd(archived.session_id);
            const auto stats_snapshot =
                session_stats_registry->snapshot(archived.session_id);
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
                    append_ui_message(*selected_messages, make_warning_message(std::format(
                        "Filo skipped an internal session rotation because it could not archive the current segment.\nSession: {}\nReason: {}\nYour full context is still intact and no history was compacted.",
                        archived.session_id,
                        save_error.empty() ? std::string("unknown archival error.") : save_error)));
                }
                wake_ui();
                return;
            }

            const std::string old_session_id = archived.session_id;
            const std::string new_session_id = core::session::SessionStore::generate_id();
            const std::string new_created_at = core::session::SessionStore::now_iso8601();
            core::session::SessionData new_segment;
            new_segment.session_id = new_session_id;
            new_segment.created_at = new_created_at;
            auto new_segment_lease = session_leases.reserve(new_segment);
            if (!new_segment_lease) {
                core::logging::warn(
                    "Skipping TUI session rotation because the new segment could not be reserved: {}",
                    new_segment_lease.error());
                return;
            }

            runtime_agent->compact_history(archived.handoff_summary);
            // Runtime-local identity: always safe — this agent is the only
            // consumer of its own session id.
            runtime_agent->set_session_id(new_session_id);

            // TOCTOU guard: the selected-thread check at the top of this
            // lambda is advisory only. Re-validate under ui_mutex that this
            // runtime still owns the shared session identity before touching
            // process-wide state; a thread switch in between would otherwise
            // let rotation overwrite another thread's globals.
            bool still_selected = false;
            {
                std::lock_guard lock(ui_mutex);
                still_selected = (session_id == old_session_id);
                if (still_selected) {
                    session_id = new_session_id;
                    session_created_at = new_created_at;
                    session_file_path = session_store->compute_path(new_segment).string();
                }
            }
            if (!still_selected) {
                core::logging::warn(
                    "Thread selection changed mid-rotation; the rotated segment "
                    "continues as a background thread ({})",
                    runtime->session_id());
            } else {
                core::budget::BudgetTracker::get_instance().reset_session(old_session_id);
                // The old segment's stats were persisted into `archived.stats`
                // above; drop them so the registry does not grow per rotation.
                session_stats_registry->reset(old_session_id);
            }

            {
                std::lock_guard lock(ui_mutex);
                const std::string reason = decision.reason.empty()
                    ? std::string("session growth exceeded the efficiency budget.")
                    : decision.reason;
                append_ui_message(*selected_messages, make_system_disclosure_message(
                    "Internal session rotated to keep the working set lean (context preserved).",
                    std::format(
                        "Previous segment: {}\nNew segment: {}\nReason: {}",
                        old_session_id,
                        new_session_id,
                        reason)));
                new_segment_lease->commit();
            }
            runtime->set_lease(session_leases.retain(new_session_id));
            runtime->mutate_metadata([&](ThreadRuntimeMetadata& metadata) {
                metadata.session_id = new_session_id;
                metadata.created_at = new_created_at;
                metadata.file_path = session_store->compute_path(new_segment).string();
            });
            if (!thread_runtimes.rekey(old_session_id, new_session_id)) {
                core::logging::warn(
                    "Could not rekey rotated thread runtime {} to {}.",
                    old_session_id,
                    new_session_id);
            }
            session_leases.release(old_session_id);
            wake_ui();
        });
    };
    configure_runtime_efficiency(current_runtime);

    std::function<std::string()> apply_active_profile_live;
    std::function<std::string()> reload_mcp_live;

    auto provider_for_current_thread = [&](std::string_view provider_name)
        -> std::expected<std::shared_ptr<core::llm::LLMProvider>, std::string> {
        std::shared_ptr<core::llm::LLMProvider> provider;
        try {
            provider = provider_manager.get_provider(std::string(provider_name));
        } catch (const std::exception& error) {
            return std::unexpected(error.what());
        }

        // Preserve the legacy provider instance in the ordinary one-thread
        // case. Once concurrent runtimes exist, every model switch must use a
        // private provider, just as a separate Filo process would.
        if (thread_runtimes.snapshot().size() == 1) {
            return provider;
        }
        auto isolated = provider->fork_for_parallel_request();
        if (!isolated) {
            return std::unexpected(std::format(
                "Provider '{}' cannot isolate concurrent thread requests.",
                provider_name));
        }
        return isolated;
    };

    auto activate_manual_mode = [&]() -> std::string {
        auto selected_provider = provider_for_current_thread(manual_provider_name);
        if (!selected_provider) {
            return std::format(
                "Failed to activate Manual mode using '{}': {}",
                manual_provider_name,
                selected_provider.error());
        }
        try {
            auto provider = *selected_provider;
            agent->set_provider(provider);
            agent->set_active_provider_name(manual_provider_name);
            llm_provider = provider;
            model_selection_mode = ModelSelectionMode::Manual;
            agent->set_active_model(manual_model_name);
            sync_mcp_sampling_backend(provider, manual_model_name);
            refresh_status_labels();
            sync_runtime_metadata();

            std::string message = std::format(
                "Switched to Manual mode: {} ({})",
                manual_provider_name,
                manual_model_name.empty() ? "<provider default>" : manual_model_name);
            const std::string hint = provider_setup_hint(manual_provider_name);
            if (!hint.empty()) {
                message += "\n   " + hint;
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
        if (!router_provider_template) {
            return "Router mode is unavailable: router provider was not initialised.";
        }

        auto selected_provider = std::dynamic_pointer_cast<
            core::llm::providers::RouterProvider>(
                router_provider_template->fork_for_parallel_request());
        if (!selected_provider) {
            return "Router mode could not create an isolated provider for this thread.";
        }
        if (!active_router_policy.empty()
            && !selected_provider->set_active_policy(active_router_policy)) {
            return std::format(
                "Router policy '{}' is no longer available.", active_router_policy);
        }
        router_provider = selected_provider;
        model_selection_mode = ModelSelectionMode::Router;
        active_router_policy = selected_provider->active_policy();
        agent->set_provider(router_provider);
        agent->set_active_provider_name("router");
        llm_provider = router_provider;
        agent->set_active_model(router_policy_label());
        sync_mcp_sampling_backend(router_provider, {});
        refresh_status_labels();
        sync_runtime_metadata();

        return std::format(
            "Switched to Router mode ({})",
            router_policy_label());
    };

    auto activate_auto_mode = [&]() -> std::string {
        if (!router_available) {
            return "Auto mode is unavailable: configure config.router with strategy=smart and at least one policy.";
        }
        if (!router_provider_template) {
            return "Auto mode is unavailable: router provider was not initialised.";
        }

        auto selected_provider = std::dynamic_pointer_cast<
            core::llm::providers::RouterProvider>(
                router_provider_template->fork_for_parallel_request());
        if (!selected_provider) {
            return "Auto mode could not create an isolated provider for this thread.";
        }
        if (!active_router_policy.empty()
            && !selected_provider->set_active_policy(active_router_policy)) {
            return std::format(
                "Router policy '{}' is no longer available.", active_router_policy);
        }
        router_provider = selected_provider;
        model_selection_mode = ModelSelectionMode::Auto;
        active_router_policy = selected_provider->active_policy();
        agent->set_provider(router_provider);
        agent->set_active_provider_name("auto");
        llm_provider = router_provider;
        agent->set_active_model("auto");
        sync_mcp_sampling_backend(router_provider, {});
        refresh_status_labels();
        sync_runtime_metadata();

        return std::format(
            "Switched to Auto mode — task-aware smart routing via policy '{}'",
            active_router_policy.empty() ? "<unset>" : active_router_policy);
    };

    reload_mcp_live = [&]() -> std::string {
        config = config_manager.get_config();
        core::mcp::McpConnectionManager::get_instance().connect_all(
            config,
            tool_manager,
            llm_provider,
            model_selection_mode == ModelSelectionMode::Manual
                ? manual_model_name
                : std::string{});

        refresh_status_labels();
        animation_cv.notify_one();
        wake_ui();
        return "Reloaded MCP servers live.";
    };

    apply_active_profile_live = [&]() -> std::string {
        const auto loaded = config_manager.get_config();
        config = loaded;

        core::llm::ProviderDescriptorSet next_registered_providers;
        std::unordered_map<std::string, std::string> next_provider_models;
        for (const auto& [name, pconfig] : config.providers) {
            if (auto provider = core::llm::ProviderFactory::create_provider(name, pconfig)) {
                const auto caps = provider->capabilities();
                provider_manager.register_provider(name, provider);
                next_registered_providers.insert({name, caps.is_local});
                next_provider_models[name] = pconfig.model;
            }
        }

        if (next_registered_providers.empty()) {
            return "the selected profile leaves no usable providers.";
        }

        registered_providers = std::move(next_registered_providers);
        provider_default_models = std::move(next_provider_models);

        router_engine = std::make_shared<core::llm::routing::RouterEngine>(
            config.router,
            registered_providers);
        router_provider_template = std::make_shared<core::llm::providers::RouterProvider>(
            provider_manager,
            router_engine,
            provider_default_models);
        router_provider = router_provider_template;
        router_available = config.router.enabled && !router_engine->list_policies().empty();
        active_router_policy = router_engine->active_policy();

        manual_provider_name = config.default_provider;
        if (!core::llm::contains_provider(registered_providers, manual_provider_name)) {
            manual_provider_name = registered_providers.begin()->name;
        }
        if (const auto it = config.providers.find(manual_provider_name);
            it != config.providers.end()) {
            manual_model_name = it->second.model;
        } else {
            manual_model_name.clear();
        }

        const ModelSelectionMode preferred_mode = parse_model_selection_mode(
            config.default_model_selection);
        if (preferred_mode == ModelSelectionMode::Router) {
            const std::string result = activate_router_mode();
            if (!result.starts_with("Switched")) {
                activate_manual_mode();
            }
        } else if (preferred_mode == ModelSelectionMode::Auto) {
            const std::string result = activate_auto_mode();
            if (!result.starts_with("Switched")) {
                activate_manual_mode();
            }
        } else {
            activate_manual_mode();
        }

        {
            std::string desired_mode = normalize_mode(config.default_mode);
            for (std::size_t i = 0; i < modes.size(); ++i) {
                if (modes[i].first == desired_mode) {
                    current_mode_idx = static_cast<int>(i);
                    agent->set_mode(modes[current_mode_idx].first);
                    break;
                }
            }
        }

        set_yolo_mode_enabled(
            parse_approval_mode(config.default_approval_mode) == ApprovalMode::Yolo);

        ui_show_banner = visibility_setting_enabled(config.ui_banner, true);
        ui_show_footer = visibility_setting_enabled(config.ui_footer, true);
        ui_show_model_info = visibility_setting_enabled(config.ui_model_info, true);
        ui_show_context_usage = visibility_setting_enabled(config.ui_context_usage, true);
        ui_show_timestamps = visibility_setting_enabled(config.ui_timestamps, true);
        ui_show_spinner = visibility_setting_enabled(config.ui_spinner, true);
        ui_show_reasoning = visibility_setting_enabled(config.ui_reasoning, true);
        agent->set_auto_compact_threshold(
            config.auto_compact_threshold,
            !config.auto_compact_threshold_explicit);
        agent->reload_subagent_profiles(config);

        core::mcp::McpConnectionManager::get_instance().connect_all(
            config,
            tool_manager,
            llm_provider,
            model_selection_mode == ModelSelectionMode::Manual
                ? manual_model_name
                : std::string{});

        refresh_status_labels();
        animation_cv.notify_one();
        wake_ui();

        return "Applied profile live.";
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
            const auto index = mention_index_snapshot();
            snapshot.suggestions = search_mention_index(
                *index,
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
        wake_ui();
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
        wake_ui();
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
        auto join_values = [](const std::vector<std::string>& values) {
            std::string out;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i > 0) out += ", ";
                out += values[i];
            }
            return out.empty() ? std::string("<none>") : out;
        };

        std::vector<std::string> provider_names;
        std::vector<std::string> live_provider_models;
        provider_names.reserve(config.providers.size());
        for (const auto& [name, provider_cfg] : config.providers) {
            std::string entry = name;
            if (!provider_cfg.model.empty()) {
                entry += " (" + provider_cfg.model + ")";
            }
            provider_names.push_back(std::move(entry));

            auto snapshot = core::llm::ModelCatalogAvailability::instance().snapshot(name);
            if (!core::llm::is_local_provider(registered_providers, name)
                && snapshot.refresh_due()) {
                try {
                    core::llm::request_model_catalog_discovery(
                        provider_manager.get_provider(name),
                        {.timeout_ms = 1000});
                    snapshot = core::llm::ModelCatalogAvailability::instance().snapshot(name);
                } catch (const std::exception&) {
                }
            }
            if (!snapshot.models.empty()) {
                std::vector<std::string> ids;
                ids.reserve(std::min<std::size_t>(snapshot.models.size(), 12));
                for (const auto& model : snapshot.models) {
                    if (ids.size() == 12) break;
                    ids.push_back(model.canonical_id);
                }
                std::string live = name + ": " + join_values(ids);
                if (snapshot.models.size() > ids.size()) {
                    live += std::format(", +{} more", snapshot.models.size() - ids.size());
                }
                live_provider_models.push_back(std::move(live));
            } else if (snapshot.refresh_in_progress) {
                live_provider_models.push_back(name + ": discovery running");
            }
        }
        std::ranges::sort(provider_names);
        std::ranges::sort(live_provider_models);

        const auto policy_names = router_engine->list_policies();

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
            "        Live models: {}\n"
            "        Policies: {}",
            mode,
            manual_info,
            router_info,
            join_values(provider_names),
            join_values(live_provider_models),
            join_values(policy_names));
    };

    auto persist_model_preferences = [&]() {
        const std::string mode =
            model_selection_mode == ModelSelectionMode::Router ? "router" :
            model_selection_mode == ModelSelectionMode::Auto   ? "auto"   : "manual";
        const std::string specific_model =
            (model_selection_mode == ModelSelectionMode::Manual && !manual_model_name.empty())
                ? manual_model_name
                : "";
        return model_defaults.persist(manual_provider_name, mode, specific_model);
    };

    auto with_model_persistence_notice = [](
        std::string message,
        const core::config::ModelPersistenceResult& persistence) {
        using enum core::config::ModelPersistenceStatus;
        if (persistence.status == SessionOnly) {
            message += std::format("\n   ℹ  {}", persistence.detail);
        } else if (persistence.status == Failed) {
            message += std::format("\n   ⚠  {}", persistence.detail);
        }
        return message;
    };

    auto with_persisted_model_preferences = [&](std::string message) -> std::string {
        if (!message.starts_with("Switched")) {
            return message;
        }
        return with_model_persistence_notice(
            std::move(message), persist_model_preferences());
    };

    auto current_model_selection_snapshot = [&]() -> ModelSelectionSnapshot {
        return ModelSelectionSnapshot{
            .mode = model_selection_mode,
            .manual_provider_name = manual_provider_name,
            .manual_model_name = manual_model_name,
            .router_policy = active_router_policy,
        };
    };

    auto same_model_selection = [](const ModelSelectionSnapshot& a,
                                   const ModelSelectionSnapshot& b) {
        return a.mode == b.mode
            && a.manual_provider_name == b.manual_provider_name
            && a.manual_model_name == b.manual_model_name
            && a.router_policy == b.router_policy;
    };

    auto remember_previous_model_selection = [&](const ModelSelectionSnapshot& before) {
        const ModelSelectionSnapshot after = current_model_selection_snapshot();
        if (!same_model_selection(before, after)) {
            previous_model_selection = before;
            sync_runtime_metadata();
        }
    };

    auto finalize_model_switch = [&](const ModelSelectionSnapshot& before,
                                     std::string message) -> std::string {
        if (message.starts_with("Switched")) {
            remember_previous_model_selection(before);
        }
        return with_persisted_model_preferences(std::move(message));
    };

    auto restore_model_selection = [&](const ModelSelectionSnapshot& target) -> std::string {
        switch (target.mode) {
            case ModelSelectionMode::Manual: {
                if (!core::llm::contains_provider(registered_providers, target.manual_provider_name)) {
                    return std::format(
                        "Previous provider '{}' is no longer available.",
                        target.manual_provider_name);
                }
                manual_provider_name = target.manual_provider_name;
                manual_model_name = target.manual_model_name;
                return activate_manual_mode();
            }
            case ModelSelectionMode::Router: {
                if (!target.router_policy.empty()
                    && !router_engine->has_policy(target.router_policy)) {
                    return std::format(
                        "Previous router policy '{}' is no longer available.",
                        target.router_policy);
                }
                active_router_policy = target.router_policy;
                return activate_router_mode();
            }
            case ModelSelectionMode::Auto: {
                if (!target.router_policy.empty()
                    && !router_engine->has_policy(target.router_policy)) {
                    return std::format(
                        "Previous router policy '{}' is no longer available.",
                        target.router_policy);
                }
                active_router_policy = target.router_policy;
                return activate_auto_mode();
            }
        }
        return "Previous model selection could not be restored.";
    };

    auto switch_to_previous_model_selection = [&]() -> std::string {
        if (!previous_model_selection.has_value()) {
            return "No previous model selection to restore.";
        }

        const ModelSelectionSnapshot before = current_model_selection_snapshot();
        const ModelSelectionSnapshot target = *previous_model_selection;
        std::string message = restore_model_selection(target);
        if (message.starts_with("Switched")) {
            previous_model_selection = before;
            sync_runtime_metadata();
        }
        return with_persisted_model_preferences(std::move(message));
    };

    auto models_equivalent = [&](std::string_view lhs, std::string_view rhs) {
        if (lhs == rhs) {
            return true;
        }
        const auto& registry = core::llm::ModelRegistry::instance();
        const auto lhs_info = registry.lookup(lhs);
        const auto rhs_info = registry.lookup(rhs);
        return lhs_info && rhs_info
            && lhs_info->canonical_id == rhs_info->canonical_id;
    };

    auto compact_model_description = [](const core::llm::ModelInfo& info) {
        std::string description = info.display_name.empty()
            ? info.canonical_id
            : info.display_name;
        if (info.context_window > 0) {
            description += std::format(" · {}k context", info.context_window / 1000);
        }
        if (!info.aliases.empty()) {
            description += " · aliases: ";
            const std::size_t limit = std::min<std::size_t>(info.aliases.size(), 4);
            for (std::size_t i = 0; i < limit; ++i) {
                if (i > 0) description += ", ";
                description += info.aliases[i];
            }
            if (info.aliases.size() > limit) {
                description += std::format(", +{}", info.aliases.size() - limit);
            }
        }
        return description;
    };

    auto provider_model_rows = [&](std::string_view provider_name) {
        std::vector<tui::ModelPickerRow> rows;
        std::unordered_set<std::string> seen_selections;
        std::unordered_set<std::string> seen_services;

        const auto catalog_group =
            core::llm::provider_catalog_group_for(provider_name, sorted_provider_names);
        if (catalog_group.sources.empty()) {
            return rows;
        }

        auto add_row = [&](std::string_view service_id,
                           std::string id,
                           std::string selector,
                           std::string source_provider,
                           std::string description,
                           bool provider_default,
                           std::string_view category_label) {
            if (id.empty()) {
                id = "<provider default>";
            }
            const std::string selection_key =
                core::llm::provider_catalog_selection_key(service_id, id);
            if (!seen_selections.insert(selection_key).second) {
                return;
            }
            const bool active = model_selection_mode == ModelSelectionMode::Manual
                && manual_provider_name == source_provider
                && (selector.empty()
                    ? manual_model_name.empty()
                    : models_equivalent(manual_model_name, selector));
            if (!category_label.empty()) {
                if (!description.empty()) {
                    description += " ";
                }
                description += category_label;
            }
            if (provider_default && !description.empty()) {
                description += " Configured default.";
            } else if (provider_default) {
                description = "Configured default.";
            }
            rows.push_back(tui::ModelPickerRow{
                .id = std::move(id),
                .selector = std::move(selector),
                .source_provider = std::move(source_provider),
                .description = std::move(description),
                .active = active,
                .provider_default = provider_default,
            });
        };

        for (const auto& source : catalog_group.sources) {
            const auto& source_provider = source.provider_name;
            const auto provider_it = config.providers.find(source_provider);
            if (provider_it == config.providers.end()) {
                continue;
            }
            const auto& provider_cfg = provider_it->second;
            const std::string configured_default = provider_cfg.model;
            std::shared_ptr<core::llm::LLMProvider> source_llm_provider;
            const core::llm::ProviderCatalogSource* effective_source = &source;
            try {
                source_llm_provider =
                    provider_manager.get_provider(source_provider);
                if (const auto metadata = source_llm_provider->metadata()) {
                    if (const auto* resolved =
                            catalog_group.find_source_by_service_id(
                                metadata->service_id)) {
                        effective_source = resolved;
                    }
                }
            } catch (const std::exception&) {
            }
            if (!seen_services.insert(effective_source->service_id).second) {
                continue;
            }
            const std::string_view category_label =
                effective_source->category_label;

            const std::string catalog_id = source_llm_provider
                ? source_llm_provider->metadata()
                      .transform([&](const auto& metadata) {
                          return metadata.service_id.empty()
                              ? source_provider
                              : metadata.service_id;
                      })
                      .value_or(source_provider)
                : source_provider;
            auto snapshot = core::llm::ModelCatalogAvailability::instance().snapshot(catalog_id);
            if (!core::llm::is_local_provider(
                    registered_providers, source_provider)) {
                try {
                    snapshot = core::llm::request_model_catalog_snapshot(
                        source_llm_provider
                            ? source_llm_provider
                            : provider_manager.get_provider(source_provider),
                        source_provider,
                        {.timeout_ms = 2500},
                        std::chrono::milliseconds{3000});
                } catch (const std::exception&) {
                }
            }

            const std::string registry_key =
                core::llm::model_registry_provider_key(
                    source_provider, provider_cfg.api_type);
            auto registry_models =
                core::llm::ModelRegistry::instance().get_by_provider(
                    registry_key);
            std::erase_if(registry_models, [&](const auto& model) {
                return !effective_source->includes_registry_model(
                    model.canonical_id);
            });
            std::ranges::sort(
                registry_models, {}, &core::llm::ModelInfo::canonical_id);

            auto provider_models = snapshot.models;
            std::erase_if(provider_models, [&](const auto& model) {
                return !effective_source->includes_api_model(
                    model.canonical_id);
            });
            const auto resolved = core::llm::resolve_model_catalog(
                provider_models, registry_models);
            if (!configured_default.empty()) {
                const auto default_metadata =
                    core::llm::resolve_model_metadata(
                        configured_default,
                        provider_models,
                        core::llm::ModelRegistry::instance().lookup(
                            configured_default));
                add_row(
                    effective_source->service_id,
                    configured_default,
                    configured_default,
                    source_provider,
                    default_metadata.model
                        ? compact_model_description(*default_metadata.model)
                        : std::string{},
                    true,
                    category_label);
            }
            for (const auto& model : resolved.models) {
                add_row(
                    effective_source->service_id,
                    model.canonical_id,
                    model.canonical_id,
                    source_provider,
                    compact_model_description(model),
                    models_equivalent(
                        configured_default, model.canonical_id),
                    category_label);
            }
        }

        if (rows.empty()) {
            const auto& source_provider = catalog_group.sources.front().provider_name;
            add_row(catalog_group.sources.front().service_id,
                    "<provider default>",
                    "",
                    source_provider,
                    "Use the provider default configured by the backend.",
                    true,
                    catalog_group.sources.front().category_label);
        }

        return rows;
    };

    auto model_provider_rows = [&]() {
        std::vector<tui::ModelProviderPickerRow> rows;
        const auto catalog_groups =
            core::llm::provider_catalog_groups(sorted_provider_names);
        rows.reserve(catalog_groups.size());
        for (const auto& catalog_group : catalog_groups) {
            if (catalog_group.sources.empty()) {
                continue;
            }

            std::vector<std::string> default_models;
            default_models.reserve(catalog_group.sources.size());
            std::unordered_set<std::string> known_model_ids;
            for (const auto& source : catalog_group.sources) {
                const auto& source_provider = source.provider_name;
                const auto provider_it = config.providers.find(source_provider);
                if (provider_it == config.providers.end()) {
                    continue;
                }
                if (!provider_it->second.model.empty()) {
                    default_models.push_back(provider_it->second.model);
                }

                std::string catalog_id = source_provider;
                try {
                    const auto provider = provider_manager.get_provider(source_provider);
                    if (const auto metadata = provider->metadata();
                        metadata && !metadata->service_id.empty()) {
                        catalog_id = metadata->service_id;
                    }
                } catch (const std::exception&) {
                }
                const auto snapshot =
                    core::llm::ModelCatalogAvailability::instance().snapshot(catalog_id);
                const std::string registry_key =
                    core::llm::model_registry_provider_key(
                        source_provider, provider_it->second.api_type);
                auto registry_models =
                    core::llm::ModelRegistry::instance().get_by_provider(
                        registry_key);
                std::erase_if(registry_models, [&](const auto& model) {
                    return !source.includes_registry_model(
                        model.canonical_id);
                });
                auto provider_models = snapshot.models;
                std::erase_if(provider_models, [&](const auto& model) {
                    return !source.includes_api_model(model.canonical_id);
                });
                const auto resolved = core::llm::resolve_model_catalog(
                    provider_models, registry_models);
                for (const auto& model : resolved.models) {
                    known_model_ids.insert(model.canonical_id);
                }
            }

            std::string description;
            if (default_models.empty()) {
                description = "Default: <provider default>";
            } else if (default_models.size() == 1) {
                description = std::format("Default: {}", default_models.front());
            } else {
                description = "Defaults: ";
                for (std::size_t i = 0; i < default_models.size(); ++i) {
                    if (i > 0) {
                        description += ", ";
                    }
                    description += default_models[i];
                }
            }

            const std::size_t known_count = known_model_ids.size();
            if (known_count > 0) {
                description += std::format(" · {} known model{}", known_count, known_count == 1 ? "" : "s");
            }
            rows.push_back(tui::ModelProviderPickerRow{
                .name = catalog_group.provider_name,
                .description = std::move(description),
                .active = model_selection_mode == ModelSelectionMode::Manual
                    && catalog_group.contains_source_provider(manual_provider_name),
            });
        }
        return rows;
    };

    // The active wire protocol owns model-specific effort policy. UI code
    // consumes the provider-neutral capability value and never switches on
    // provider names or API families.
    auto reasoning_capabilities = [&](std::string_view provider_name,
                                      std::string_view model_name) {
        try {
            return provider_manager.get_provider(std::string(provider_name))
                ->reasoning_capabilities(model_name);
        } catch (...) {
            return core::llm::ReasoningCapabilities{};
        }
    };

    auto resolve_effective_effort = [&](std::string_view configured,
                                        const core::llm::ReasoningCapabilities& capabilities)
        -> std::string {
        if (configured.empty()) return "high (auto default)";
        if (configured == "max"
            && !capabilities.supports(core::llm::ReasoningCapability::MaxEffort)) {
            return "high (max unsupported on current model)";
        }
        if (configured == "xhigh"
            && !capabilities.supports(core::llm::ReasoningCapability::XHighEffort)) {
            return "high (xhigh unsupported on current model)";
        }
        if (configured == "ultra"
            && !capabilities.supports(core::llm::ReasoningCapability::UltraEffort)) {
            return capabilities.supports(core::llm::ReasoningCapability::MaxEffort)
                ? "max (ultra unsupported on current model)"
                : "high (ultra unsupported on current model)";
        }
        return std::string(configured);
    };

    auto describe_effort = [&]() -> std::string {
        const std::string configured = session_effort_value.empty()
            ? "auto"
            : session_effort_value;
        const bool manual_mode = model_selection_mode == ModelSelectionMode::Manual;
        const std::string model_for_status = manual_mode
            ? manual_model_name
            : (active_model_name.empty() ? manual_model_name : active_model_name);
        const std::string provider_for_status = manual_mode
            ? manual_provider_name
            : (active_provider_name.empty() ? manual_provider_name : active_provider_name);
        const auto capabilities = reasoning_capabilities(
            provider_for_status, model_for_status);
        const std::string effective = resolve_effective_effort(
            session_effort_value,
            capabilities);

        std::string applies_note =
            "Applies when the active provider protocol supports effort for this model.";
        if (model_selection_mode == ModelSelectionMode::Manual
            && !capabilities.supports_effort()) {
            applies_note = std::format(
                "Current manual provider '{}' does not support effort for this model.",
                manual_provider_name);
        } else if (model_selection_mode != ModelSelectionMode::Manual) {
            applies_note =
                "Router/auto mode applies effort only when the selected protocol reports support.";
        }

        return std::format(
            "Configured effort: {}\n"
            "        Effective for current model: {}\n"
            "        Active provider: {}\n"
            "        Active model: {}\n"
            "        {}\n"
            "        Levels: auto, off, low, medium, high, xhigh, max, ultra",
            configured,
            effective,
            provider_for_status,
            model_for_status.empty() ? std::string("<provider default>") : model_for_status,
            applies_note);
    };

    auto switch_effort = [&](std::string_view requested) -> std::string {
        auto trim_ascii = [](std::string_view s) -> std::string_view {
            const auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string_view::npos) return {};
            const auto end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };

        const std::string_view trimmed = trim_ascii(requested);
        if (trimmed.empty()) {
            return "Usage: /effort auto|off|low|medium|high|xhigh|max|ultra";
        }

        std::string normalized;
        normalized.reserve(trimmed.size());
        for (const char ch : trimmed) {
            if (ch == '-' || ch == '_' || std::isspace(static_cast<unsigned char>(ch))) continue;
            normalized.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(ch))));
        }

        if (normalized == "auto" || normalized == "unset"
            || normalized == "default") {
            session_effort_value.clear();
            agent->set_effort_level(session_effort_value);
            return "Set effort to auto (provider default, typically high).";
        }

        if (normalized == "off" || normalized == "none" || normalized == "disabled") {
            session_effort_value = "none";
            agent->set_effort_level(session_effort_value);
            return "Disabled reasoning for providers with switchable thinking.";
        }

        if (normalized != "low" && normalized != "medium"
            && normalized != "high" && normalized != "xhigh"
            && normalized != "max" && normalized != "ultra") {
            return "Unknown effort level. Use one of: auto, low, medium, high, xhigh, max, ultra.";
        }

        session_effort_value = normalized;
        agent->set_effort_level(session_effort_value);

        if (normalized == "max") {
            return "Set effort to max. Models without max support will automatically use high.";
        }
        if (normalized == "xhigh") {
            return "Set effort to xhigh. Unsupported models will automatically use high.";
        }
        if (normalized == "ultra") {
            return "Set effort to ultra. Unsupported models will use max or high.";
        }
        return std::format("Set effort to {}.", normalized);
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
        if (trimmed == "-") {
            return switch_to_previous_model_selection();
        }

        std::string normalized;
        normalized.reserve(trimmed.size());
        for (const char c : trimmed) {
            if (c == '-' || c == '_' || std::isspace(static_cast<unsigned char>(c))) continue;
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }

        auto finalize = [&](const ModelSelectionSnapshot& before,
                            std::string message) {
            if (persist_selection) {
                return finalize_model_switch(before, std::move(message));
            }
            if (message.starts_with("Switched")) {
                remember_previous_model_selection(before);
            }
            return message;
        };

        if (normalized == "manual") {
            const ModelSelectionSnapshot before = current_model_selection_snapshot();
            return finalize(before, activate_manual_mode());
        }
        if (normalized == "router") {
            const ModelSelectionSnapshot before = current_model_selection_snapshot();
            return finalize(before, activate_router_mode());
        }
        if (normalized == "auto") {
            const ModelSelectionSnapshot before = current_model_selection_snapshot();
            return finalize(before, activate_auto_mode());
        }

        std::string policy_name(trimmed);
        if (policy_name.starts_with("policy/")) {
            policy_name = policy_name.substr(7);
        }
        if (router_engine->has_policy(policy_name)) {
            if (!router_available) {
                return "Router policy exists but router mode is disabled in configuration.";
            }
            const ModelSelectionSnapshot before = current_model_selection_snapshot();
            active_router_policy = policy_name;
            return finalize(before, activate_router_mode());
        }

        // Allow explicit provider + model overrides in a single command:
        //   /model claude sonnet
        //   /model claude opus
        //   /model claude claude-opus-5
        if (const auto split = trimmed.find_first_of(" \t");
            split != std::string_view::npos) {
            const std::string_view provider_name = trim_ascii(trimmed.substr(0, split));
            const std::string_view model_name = trim_ascii(trimmed.substr(split + 1));
            if (!provider_name.empty() && !model_name.empty()) {
                const auto provider_it = config.providers.find(std::string(provider_name));
                if (provider_it != config.providers.end()) {
                    const ModelSelectionSnapshot before = current_model_selection_snapshot();
                    manual_provider_name = provider_it->first;
                    manual_model_name = std::string(model_name);
                    return finalize(before, activate_manual_mode());
                }
            }
        }

        const auto it = config.providers.find(std::string(trimmed));
        if (it == config.providers.end()) {
            return std::format("Unknown model selector '{}'.\n        {}",
                               trimmed, describe_models());
        }

        const ModelSelectionSnapshot before = current_model_selection_snapshot();
        manual_provider_name = it->first;
        manual_model_name = it->second.model;
        return finalize(before, activate_manual_mode());
    };

    auto switch_provider = [&](std::string_view requested) -> std::string {
        return apply_model_selector(requested, true);
    };

    auto profile_status = [&]() -> std::string {
        const auto& profiles = config_manager.get_profiles();
        const std::string active = config_manager.get_active_profile().empty()
            ? std::string("<none>")
            : config_manager.get_active_profile();

        if (profiles.empty()) {
            return std::format(
                "Active profile: {}\n"
                "        Available profiles: <none>\n"
                "        Define named profiles under `profiles` in your config.json.",
                active);
        }

        std::vector<std::string> names;
        names.reserve(profiles.size());
        for (const auto& [name, profile] : profiles) {
            (void)profile;
            names.push_back(name);
        }
        std::ranges::sort(names);

        auto join_values = [](const std::vector<std::string>& values) {
            std::string out;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i > 0) out += ", ";
                out += values[i];
            }
            return out;
        };

        std::string body = std::format("Active profile: {}", active);
        body += "\n        Available profiles:";
        for (const auto& name : names) {
            const auto it = profiles.find(name);
            if (it == profiles.end()) {
                continue;
            }
            std::string line = name;
            if (!it->second.description.empty()) {
                line += " - " + it->second.description;
            }
            if (!it->second.extends_from.empty()) {
                line += " (extends: " + join_values(it->second.extends_from) + ")";
            }
            body += "\n        - " + line;
        }
        body += "\n        Use `/profile <name>` to switch and apply profile immediately.";
        return body;
    };

    auto switch_profile = [&](std::string_view requested) -> std::string {
        auto trim_ascii = [](std::string_view s) -> std::string_view {
            const auto start = s.find_first_not_of(" \t\r\n");
            if (start == std::string_view::npos) return {};
            const auto end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };

        const std::string_view trimmed = trim_ascii(requested);
        if (trimmed.empty()) {
            return "Usage: /profile <name>|list|status|clear";
        }

        const std::string lowered = core::utils::str::to_lower_ascii_copy(trimmed);
        if (lowered == "list" || lowered == "status" || lowered == "ls") {
            return profile_status();
        }

        std::optional<std::string> next_profile;
        if (lowered == "clear"
            || lowered == "none"
            || lowered == "unset"
            || lowered == "off") {
            next_profile = std::nullopt;
        } else {
            next_profile = std::string(trimmed);
        }

        std::string error;
        if (!config_manager.persist_active_profile(next_profile, settings_working_dir, &error)) {
            return std::format("Could not switch profile: {}", error);
        }

        const std::string live_result = apply_active_profile_live
            ? apply_active_profile_live()
            : "live profile application is unavailable in this session.";

        if (!next_profile.has_value()) {
            if (!config_manager.get_active_profile().empty()) {
                if (live_result.starts_with("Applied")) {
                    return std::format(
                        "Could not clear active profile because '{}' is still enforced (via FILO_PROFILE or config active_profile).",
                        config_manager.get_active_profile());
                }
                return std::format(
                    "Could not clear active profile because '{}' is still enforced (via FILO_PROFILE or config active_profile), and {} Restart Filo to fully apply.",
                    config_manager.get_active_profile(),
                    live_result);
            }
            if (live_result.starts_with("Applied")) {
                return "Cleared active profile and applied base configuration live.";
            }
            return std::format(
                "Cleared active profile, but {} Restart Filo to fully apply.",
                live_result);
        }

        if (live_result.starts_with("Applied")) {
            return std::format(
                "Switched active profile to '{}' and applied it live.",
                config_manager.get_active_profile());
        }
        return std::format(
            "Switched active profile to '{}', but {} Restart Filo to fully apply.",
            config_manager.get_active_profile(),
            live_result);
    };

    auto open_model_picker = [&]() -> bool {
        auto providers = model_provider_rows();
        if (providers.empty()) {
            return false;
        }
        {
            std::lock_guard lock(ui_mutex);
            model_provider_picker_state.providers = std::move(providers);
            model_provider_picker_state.active = true;
            model_provider_picker_state.selected = 0;
            for (std::size_t i = 0; i < model_provider_picker_state.providers.size(); ++i) {
                if (model_provider_picker_state.providers[i].active) {
                    model_provider_picker_state.selected = static_cast<int>(i);
                    break;
                }
            }
            provider_model_picker_state.active = false;
            model_picker_state.active = false;
        }
        wake_ui();
        return true;
    };

    auto open_provider_model_picker = [&](std::string provider_name) -> bool {
        auto rows = provider_model_rows(provider_name);
        if (rows.empty()) {
            return false;
        }
        {
            std::lock_guard lock(ui_mutex);
            provider_model_picker_state.active = true;
            provider_model_picker_state.provider_name = std::move(provider_name);
            provider_model_picker_state.models = std::move(rows);
            provider_model_picker_state.selected = 0;
            for (std::size_t i = 0; i < provider_model_picker_state.models.size(); ++i) {
                if (provider_model_picker_state.models[i].active) {
                    provider_model_picker_state.selected = static_cast<int>(i);
                    break;
                }
            }
            model_provider_picker_state.active = false;
            model_picker_state.active = false;
        }
        wake_ui();
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
            model_provider_picker_state.active   = false;
            provider_model_picker_state.active   = false;
        }
        wake_ui();
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

        const ModelSelectionSnapshot before = current_model_selection_snapshot();
        manual_provider_name = "local";
        manual_model_name    = model_label;

        std::string result = activate_manual_mode();
        if (result.starts_with("Switched")) {
            remember_previous_model_selection(before);
            result = with_model_persistence_notice(
                std::move(result),
                model_defaults.persist_local(model_path_str, model_label));
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
            case core::config::ManagedSettingKey::PromptEditor:
                return settings.prompt_editor;
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
            case core::config::ManagedSettingKey::UiReasoning:
                return settings.ui_reasoning;
            case core::config::ManagedSettingKey::AutoCompactThreshold:
                return settings.auto_compact_threshold;
            case core::config::ManagedSettingKey::ContextCompression:
                return settings.context_compression;
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
            case core::config::ManagedSettingKey::PromptEditor:
                return effective.prompt_editor.empty()
                    ? std::string("system")
                    : effective.prompt_editor;
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
            case core::config::ManagedSettingKey::UiReasoning:
                return effective.ui_reasoning;
            case core::config::ManagedSettingKey::AutoCompactThreshold:
                return std::to_string(effective.auto_compact_threshold);
            case core::config::ManagedSettingKey::ContextCompression:
                return effective.context_compression.empty()
                    ? std::string("off")
                    : effective.context_compression;
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
        config.prompt_editor = after.prompt_editor;
        config.router.default_policy = after.router.default_policy;
        config.ui_banner = after.ui_banner;
        config.ui_footer = after.ui_footer;
        config.ui_model_info = after.ui_model_info;
        config.ui_context_usage = after.ui_context_usage;
        config.ui_timestamps = after.ui_timestamps;
        config.ui_spinner = after.ui_spinner;
        config.ui_reasoning = after.ui_reasoning;
        config.context_compression = after.context_compression;

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
            if (!router_engine->has_policy(after.router.default_policy)) {
                return std::format(
                    "Router policy '{}' is not available in this session.",
                    after.router.default_policy);
            }
            active_router_policy = after.router.default_policy;
            if (model_selection_mode == ModelSelectionMode::Router) {
                static_cast<void>(activate_router_mode());
            } else if (model_selection_mode == ModelSelectionMode::Auto) {
                static_cast<void>(activate_auto_mode());
            } else {
                sync_runtime_metadata();
            }
        }

        ui_show_banner = visibility_setting_enabled(after.ui_banner, true);
        ui_show_footer = visibility_setting_enabled(after.ui_footer, true);
        ui_show_model_info = visibility_setting_enabled(after.ui_model_info, true);
        ui_show_context_usage = visibility_setting_enabled(after.ui_context_usage, true);
        ui_show_timestamps = visibility_setting_enabled(after.ui_timestamps, true);
        ui_show_spinner = visibility_setting_enabled(after.ui_spinner, true);
        ui_show_reasoning = visibility_setting_enabled(after.ui_reasoning, true);
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
            wake_ui();
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
        wake_ui();
    };

    static constexpr std::array<std::string_view, 4> kCompressionModes{
        "off", "light", "full", "ultra"
    };
    static constexpr std::string_view kCompressionModeList = "off, light, full, ultra";

    auto normalize_compression_mode = [&](std::string_view requested)
        -> std::optional<std::string> {
        std::string normalized;
        normalized.reserve(requested.size());
        for (const char ch : requested) {
            if (ch == '-' || ch == '_' || std::isspace(static_cast<unsigned char>(ch))) {
                continue;
            }
            normalized.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(ch))));
        }
        if (normalized == "none" || normalized == "disabled") {
            normalized = "off";
        } else if (normalized == "on") {
            normalized = "light";
        } else if (normalized == "max") {
            normalized = "ultra";
        }

        if (std::ranges::find(kCompressionModes, normalized) != kCompressionModes.end()) {
            return normalized;
        }
        return std::nullopt;
    };

    auto describe_compression = [&]() -> std::string {
        const auto& effective = config_manager.get_config();
        const std::string mode = effective.context_compression.empty()
            ? std::string("off")
            : effective.context_compression;
        return std::format(
            "Compression: {}\n"
            "        Modes: {}\n"
            "        Use /compression <mode> or /compression to open the selector.",
            mode,
            kCompressionModeList);
    };

    auto switch_compression = [&](std::string_view requested) -> std::string {
        const auto normalized = normalize_compression_mode(requested);
        if (!normalized.has_value()) {
            return std::format("Unknown compression mode. Use one of: {}.", kCompressionModeList);
        }

        const auto before = config_manager.get_config();
        std::string error;
        if (!config_manager.persist_managed_setting(
                core::config::SettingsScope::Workspace,
                core::config::ManagedSettingKey::ContextCompression,
                *normalized,
                settings_working_dir,
                &error)) {
            return std::format("Could not save compression mode: {}", error);
        }

        const auto after = config_manager.get_config();
        if (const auto live_apply_warning = apply_effective_settings(before, after);
            live_apply_warning.has_value()) {
            return std::format(
                "Set compression to {}. Warning: {}",
                *normalized,
                *live_apply_warning);
        }
        return std::format(
            "Set compression to {}. Future tool outputs will use this mode.",
            *normalized);
    };

    // Replaces the primary working directory for /workspace change: chdirs
    // the process, rebases the process-wide default workspace (so new
    // threads inherit it), and rebases the active thread's session
    // workspace. Lives at the composition root because only it may touch
    // process-global state (cwd, the Workspace singleton); the agent itself
    // only owns its own SessionContext (see Agent::change_workspace_root).
    auto change_workspace_root = [&](std::string_view requested_path)
        -> core::commands::CommandOperationResult {
        if (requested_path.empty()) {
            return {.ok = false, .message = "Provide a directory to switch to."};
        }

        std::error_code ec;
        const auto resolved = core::workspace::SessionWorkspace::normalize_path(
            std::filesystem::path(requested_path));
        if (!std::filesystem::is_directory(resolved, ec)) {
            return {
                .ok = false,
                .message = std::format("'{}' is not an existing directory.", std::string(requested_path)),
            };
        }

        if (core::landrun::LandrunSettings::instance().enabled()) {
            const char* home_value = std::getenv("HOME");
            if (home_value && *home_value
                && core::landrun::is_landrun_path_within(
                       resolved,
                       core::workspace::SessionWorkspace::normalize_path(home_value))) {
                return {
                    .ok = false,
                    .message =
                        "Refusing to switch into a directory that would expose the entire "
                        "home directory under secure mode. Pick a project directory, or "
                        "restart with --sandbox off.",
                };
            }
        }

        std::filesystem::current_path(resolved, ec);
        if (ec) {
            return {
                .ok = false,
                .message = std::format("Could not switch to '{}': {}", resolved.string(), ec.message()),
            };
        }

        const auto previous = core::workspace::Workspace::get_instance().snapshot();
        core::workspace::Workspace::get_instance().initialize(
            resolved, previous.additional, previous.enforce, previous.scratch);

        // set_primary is a no-op when the target equals the current root
        // (e.g. the session was already rebased by a prior call); either way
        // the process and process-wide default above have been updated, so
        // report success regardless of whether the session-level root moved.
        if (agent) {
            agent->change_workspace_root(resolved);
            steering_context = core::context::load_project_steering_context(
                resolved, agent->session_context_snapshot().steering_policy);
            context_sources_label = join_context_source_labels(steering_context.source_labels);
        }

        // Auto-generated tab names follow the new primary workspace.
        project_thread_base_name = project_base_name_for(resolved);
        thread_runtimes.retitle_auto_named(project_thread_base_name,
                                           main_runtime->session_id());

        return {
            .ok = true,
            .message = std::format("Switched the working directory to '{}'.", resolved.string()),
        };
    };

    // ── Reusable filesystem browser ──────────────────────────────────────────
    // One entry point for every feature that needs a path from the user. The
    // caller supplies *what* it wants (`FileSystemPickerRequest`) and *what to
    // do with the answer*; overlay bookkeeping and the keyboard contract are
    // handled once, here and in FileSystemPicker.

    auto file_picker_quick_roots = [agent]() -> std::vector<FileSystemQuickRoot> {
        std::vector<FileSystemQuickRoot> roots;
        if (agent) {
            const auto workspace = agent->workspace_snapshot();
            if (!workspace.primary().empty()) {
                roots.push_back({.label = "Workspace", .path = workspace.primary()});
            }
            for (const auto& extra : workspace.additional()) {
                roots.push_back({.label = extra.filename().string(), .path = extra});
            }
        }
        if (const auto home = core::utils::path::home_directory(); !home.empty()) {
            roots.push_back({.label = "Home", .path = home});
        }
        return roots;
    };

    auto open_file_picker = [&](FileSystemPickerRequest request,
                               std::function<void(const std::filesystem::path&)> on_confirm) {
        if (request.quick_roots.empty()) {
            request.quick_roots = file_picker_quick_roots();
        }
        // An empty start directory is fine: the component falls back through
        // the working directory, the home directory, and finally the root.
        {
            std::lock_guard lock(ui_mutex);
            open_file_system_picker(file_picker_state, std::move(request));
            file_picker_on_confirm = std::move(on_confirm);
            // Only one bottom panel can own the keyboard at a time.
            command_option_picker_state.active = false;
            model_picker_state.active = false;
            model_provider_picker_state.active = false;
            provider_model_picker_state.active = false;
            local_model_picker_state.active = false;
            settings_panel_state.active = false;
        }
        wake_ui();
    };

    /// `/workspace add` and `/workspace change` used to prefill the command and
    /// leave the user to type an absolute path from memory. Both now browse.
    auto open_workspace_directory_picker = [&](std::string_view action) -> bool {
        const bool changing = action == "change";
        if (!changing && !agent) {
            return false;  // Nothing owns the workspace to extend.
        }

        FileSystemPickerRequest request{
            .title = changing ? "CHANGE WORKING DIRECTORY" : "ADD WORKSPACE FOLDER",
            // The panel documents its own keys; a caller hint should explain
            // the consequence of choosing, which the keys cannot convey.
            .hint = changing
                ? "The folder you choose becomes the primary working directory."
                : "The folder you choose gains session read/write access.",
            .target = FileSystemPickerTarget::Directory,
        };

        open_file_picker(std::move(request), [&, changing](const std::filesystem::path& chosen) {
            if (changing) {
                const auto result = change_workspace_root(chosen.string());
                append_history(std::format(
                    "\n{}\n",
                    result.ok ? "\xe2\x9c\x93  " + result.message
                              : "\xe2\x9c\x97  " + result.message));
                return;
            }
            const auto added = agent->grant_workspace_paths({chosen});
            append_history(std::format(
                "\n{}\n",
                added > 0
                    ? std::format("\xe2\x9c\x93  Added '{}' to the workspace.", chosen.string())
                    : std::format(
                        "\xe2\x9c\x97  '{}' was not added. It must be an existing directory "
                        "that isn't already in scope.",
                        chosen.string())));
        });
        return true;
    };

    /// Ctrl+B: pick a file from disk and drop it into the prompt as an
    /// `@mention`, so attaching context no longer requires drag-and-drop.
    auto open_attachment_picker = [&]() {
        const std::filesystem::path workspace_root =
            agent ? agent->workspace_snapshot().primary() : std::filesystem::path{};
        open_file_picker(
            FileSystemPickerRequest{
                .title = "ATTACH FILE",
                .hint = "The file you choose is inserted into the prompt "
                        "as an @mention.",
                .target = FileSystemPickerTarget::File,
                .start_directory = workspace_root,
            },
            [&, workspace_root](const std::filesystem::path& chosen) {
                // Workspace-relative mentions keep prompts short and portable;
                // files outside it fall back to the absolute path.
                std::error_code ec;
                const auto relative = workspace_root.empty()
                    ? std::filesystem::path{}
                    : std::filesystem::relative(chosen, workspace_root, ec);
                const bool inside = !ec && !relative.empty()
                    && !relative.generic_string().starts_with("..");
                const std::string mention_path =
                    inside ? relative.generic_string() : chosen.generic_string();

                std::lock_guard lock(ui_mutex);
                auto cursor = static_cast<std::size_t>(std::clamp(
                    input_cursor_position, 0, static_cast<int>(input_text.size())));
                // Keep the mention a standalone token when it lands after a word.
                if (cursor > 0
                    && !core::utils::ascii::is_space(
                        static_cast<unsigned char>(input_text[cursor - 1]))) {
                    input_text.insert(cursor, " ");
                    ++cursor;
                }
                // Reuse the mention formatter so quoting and trailing-space
                // rules match the `@` autocomplete exactly.
                const auto completed = core::context::apply_mention_completion(
                    input_text,
                    core::context::ActiveMention{
                        .replace_begin = cursor,
                        .replace_end = cursor,
                    },
                    mention_path);
                input_text = completed.text;
                input_cursor_position = static_cast<int>(completed.cursor);
            });
    };

    std::function<bool()> open_steering_picker;

    auto open_command_option_picker = [&](std::string_view command_name) -> bool {
        CommandOptionPickerState next;
        next.active = true;
        next.command_name = std::string(command_name);

        if (command_name == "/effort") {
            next.title = "EFFORT";
            next.current_value = session_effort_value.empty()
                ? std::string("auto")
                : session_effort_value;
            next.help_text = "Esc closes this panel.";
            next.options = {
                {.value = "auto", .label = "Auto", .description = "Use the provider default for the selected model."},
                {.value = "low", .label = "Low", .description = "Prefer faster, lighter reasoning when supported."},
                {.value = "medium", .label = "Medium", .description = "Use balanced reasoning effort when supported."},
                {.value = "high", .label = "High", .description = "Use deeper reasoning effort when supported."},
                {.value = "xhigh", .label = "XHigh", .description = "Use extra-high reasoning effort when supported."},
                {.value = "max", .label = "Max", .description = "Request maximum effort; unsupported models fall back to high."},
                {.value = "ultra", .label = "Ultra", .description = "Request maximum reasoning with automatic delegation when supported."},
            };
            next.on_select = switch_effort;
        } else if (command_name == "/compression" || command_name == "/compress") {
            next.title = "COMPRESSION";
            next.current_value = normalize_compression_mode(
                config_manager.get_config().context_compression).value_or("off");
            next.help_text = "Esc closes this panel.";
            next.options = {
                {.value = "off", .label = "Off", .description = "Keep tool outputs exact until the hard history-size clamp."},
                {.value = "light", .label = "Light", .description = "Summarize only oversized read_file and shell outputs."},
                {.value = "full", .label = "Full", .description = "Use read caching plus command-family summaries for common shell output."},
                {.value = "ultra", .label = "Ultra", .description = "Use the tightest built-in budgets for high token pressure."},
            };
            next.on_select = switch_compression;
        } else if (command_name == "/workspace" || command_name == "/dir" || command_name == "/dirs") {
            next.title = "WORKSPACE";
            next.help_text = "Enter opens selection. Esc closes this panel.";
            const auto policy = agent ? agent->session_context_snapshot().steering_policy : core::context::SteeringPolicy{};
            next.options = {
                {.value = "add", .label = "Add directory",
                 .description = "Grant this session read/write access to another directory."},
                {.value = "change", .label = "Change directory",
                 .description = "Switch the primary working directory (like restarting in a new folder)."},
                {.value = "steering", .label = "Steering files...",
                 .description = std::format("Inspect mode, switch to custom folder/file, or load/unload files (Current: {}).",
                                            core::context::format_steering_policy(policy))},
            };
            next.on_select = [&](std::string_view value) -> std::string {
                if (value == "steering") {
                    if (open_steering_picker) {
                        open_steering_picker();
                    }
                    return {};
                }
                open_workspace_directory_picker(value);
                return {};
            };
        } else if (command_name == "/steering" || command_name == "/agents" || command_name == "/steer") {
            if (open_steering_picker) {
                return open_steering_picker();
            }
            return false;
        } else {
            return false;
        }

        for (std::size_t i = 0; i < next.options.size(); ++i) {
            next.options[i].active = next.options[i].value == next.current_value;
            if (next.options[i].active) {
                next.selected = static_cast<int>(i);
            }
        }

        {
            std::lock_guard lock(ui_mutex);
            command_option_picker_state = std::move(next);
        }
        wake_ui();
        return true;
    };

    open_steering_picker = [&]() -> bool {
        const auto policy = agent ? agent->session_context_snapshot().steering_policy : core::context::SteeringPolicy{};
        const auto primary = agent_session_context.workspace_view().primary();
        steering_context = core::context::load_project_steering_context(primary, policy);

        CommandOptionPickerState next;
        next.active = true;
        next.command_name = "/steering";
        next.title = "STEERING FILES";
        next.current_value = policy.format();
        next.help_text = "Enter: toggle/apply  Esc: close";

        if (policy.mode == core::context::SteeringMode::None) {
            next.options.push_back({
                .value = "mode:default",
                .label = "Enable all steering",
                .description = "Switch back to default workspace steering discovery.",
            });
        } else {
            next.options.push_back({
                .value = "mode:none",
                .label = "Disable all steering",
                .description = "Unload and disable all project steering files (safe mode).",
            });
        }

        for (const auto& file : steering_context.files) {
            const std::string action_val = (file.enabled ? "unload:" : "load:") + file.label;
            const std::string status_label = file.enabled ? "[✓] " + file.label : "[✗] " + file.label + " (unloaded)";
            const std::string desc = file.enabled
                ? "Currently loaded in prompt context. Select to UNLOAD."
                : "Currently excluded from prompt context. Select to LOAD.";
            next.options.push_back({
                .value = action_val,
                .label = status_label,
                .description = desc,
                .active = file.enabled,
            });
        }

        next.options.push_back({
            .value = "view",
            .label = "View instruction contents...",
            .description = "Open the Agent Instructions viewer to inspect file contents.",
        });

        next.on_select = [&](std::string_view value) -> std::string {
            auto cur_policy = agent ? agent->session_context_snapshot().steering_policy : core::context::SteeringPolicy{};
            if (value == "mode:default") {
                cur_policy.mode = core::context::SteeringMode::Default;
            } else if (value == "mode:none") {
                cur_policy.mode = core::context::SteeringMode::None;
            } else if (value.starts_with("unload:")) {
                cur_policy.disable_source(value.substr(7));
            } else if (value.starts_with("load:")) {
                cur_policy.enable_source(value.substr(5));
            } else if (value == "view") {
                std::lock_guard lock(ui_mutex);
                agents_visualizer_panel_active = true;
                agents_visualizer_scroll_offset = 0;
                return {};
            }
            if (agent) {
                agent->update_session_context([&](core::context::SessionContext& sctx) {
                    sctx.steering_policy = cur_policy;
                });
                steering_context = core::context::load_project_steering_context(primary, cur_policy);
                context_sources_label = join_context_source_labels(steering_context.source_labels);
            }
            return {};
        };

        for (std::size_t i = 0; i < next.options.size(); ++i) {
            next.options[i].active = next.options[i].value == next.current_value;
            if (next.options[i].active) {
                next.selected = static_cast<int>(i);
            }
        }

        {
            std::lock_guard lock(ui_mutex);
            command_option_picker_state = std::move(next);
        }
        wake_ui();
        return true;
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
        wake_ui();
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
            "Tool compression: {}\n"
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
            effective.context_compression.empty() ? std::string("off")
                                                  : effective.context_compression,
            user_path.string(),
            workspace_path.string());
    };

    // ── Permission gate wiring ───────────────────────────────────────────────

    // Notify fn: called from the worker thread when a permission is requested.
    core::agent::PermissionGate::get_instance().set_notify_fn([&]() {
        wake_ui();  // wake up the render loop
    });

    // Permission function given to each Agent. Capturing the owning runtime
    // keeps auto-approval annotations on the correct hidden conversation.
    auto make_permission_fn = [&](const ThreadRuntime::Ptr& runtime) {
        return [&, runtime](std::string_view tool_name,
                            std::string_view args) -> bool {
        bool yolo_enabled = false;
        {
            yolo_enabled = command_line_yolo_enabled
                || runtime->metadata().yolo_enabled;
        }
        if (yolo_enabled) {
            {
                std::lock_guard lock(ui_mutex);
                // Keep YOLO approvals inside the current tool card instead of
                // flooding the system-history stream.
                auto messages = runtime->messages();
                for (auto msg_it = messages->rbegin(); msg_it != messages->rend(); ++msg_it) {
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
            wake_ui();
            return true;
        }

        // Check session trust rules ("don't ask again for this").
        bool is_allowed = false;
        {
            const auto metadata = runtime->metadata();
            const auto matches = [&](const auto& rules) {
                return std::ranges::any_of(rules, [&](const auto& allow_rule) {
                    return core::permissions::session_allow_rule_matches(
                        allow_rule,
                        tool_name,
                        args);
                });
            };
            is_allowed = matches(startup_session_allow_rules)
                || matches(metadata.permission_rules);
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
                return permission_prompt_shutdown || !permission_prompt_in_flight;
            });
            if (permission_prompt_shutdown) {
                // The app is winding down: deny instead of blocking exit.
                return false;
            }
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
            // The overlay is a fixed-height panel, so it clamps explicitly
            // rather than relying on the model to arrive pre-truncated.
            perm_state.diff_preview = clamp_diff_preview(
                build_tool_diff_preview(tool_name, args),
                kPermissionDiffPreviewMaxLines);
            perm_state.selected     = 0;
            perm_state.remember_rule =
                core::permissions::make_session_allow_rule(tool_name, args);
            perm_state.allow_label  = make_allow_label(tool_name, args);
            perm_state.promise      = prom;
            // Attribute the overlay to the requesting thread. A hidden thread
            // can reach the user too; Ctrl+C must stop the right agent.
            perm_state.origin_runtime = runtime;
            perm_state.origin_label = thread_display_label(runtime);
        }
        wake_ui();
        return fut.get();
        };
    };

    // ── AskUserQuestion callback ─────────────────────────────────────────────
    ask_user_tool->setQuestionCallback([&](core::tools::QuestionRequest request) {
        auto origin_runtime = thread_runtimes.find(request.session_id);
        std::unique_lock slot_lock(question_prompt_mutex);
        question_prompt_cv.wait(slot_lock, [&]() {
            return question_prompt_shutdown || !question_dialog.active();
        });
        if (question_prompt_shutdown) {
            request.promise->set_value(std::nullopt);
            return;
        }
        const bool origin_is_hidden = origin_runtime
            && thread_runtimes.current() != origin_runtime;
        auto displaced_promise = question_dialog.open(
            std::move(request),
            origin_is_hidden ? thread_display_label(origin_runtime) : std::string{});
        slot_lock.unlock();
        wake_ui();

        if (displaced_promise) {
            displaced_promise->set_value(std::nullopt);
        }
        
        // Wait for the result (UI will resolve the promise)
    });

    auto append_runtime_history = [&](const ThreadRuntime::Ptr& runtime,
                                      const std::string& str) {
        std::lock_guard lock(ui_mutex);
        auto messages = runtime->messages();
        if (messages->empty() || messages->back().type != MessageType::System) {
            append_ui_message(*messages, make_system_message(str));
        } else {
            messages->back().text += str;
        }
        animation_cv.notify_one();
        wake_ui();
    };

    auto update_ui_message = [&](const ThreadRuntime::Ptr& runtime,
                                 std::string_view message_id,
                                 auto&& updater) {
        {
            std::lock_guard lock(ui_mutex);
            auto messages = runtime->messages();
            auto it = std::ranges::find_if(*messages, [&](const UiMessage& message) {
                return message.id == message_id;
            });
            if (it == messages->end()) {
                return;
            }

            updater(*it);
        }
        animation_cv.notify_one();
        wake_ui();
    };

    auto update_live_assistant_message =
        [&](const ThreadRuntime::Ptr& runtime,
            const std::shared_ptr<LiveAssistantTimeline>& timeline,
            auto&& updater) {
        {
            std::lock_guard lock(ui_mutex);
            if (auto* message = timeline->current(*runtime->messages())) {
                updater(*message);
            }
        }
        animation_cv.notify_one();
        wake_ui();
    };

    save_runtime_snapshot = [&](ThreadRuntime::Ptr runtime)
        -> std::future<SaveResult> {
        auto completion_promise = std::make_shared<std::promise<SaveResult>>();
        auto completion_future = completion_promise->get_future();
        if (!runtime) {
            completion_promise->set_value(std::string{"No thread runtime to save."});
            return completion_future;
        }
        auto runtime_agent = runtime->agent();
        auto snap_messages = runtime_agent->get_history();
        // Stamp the live activity-phase duration onto the persisted assistant
        // messages so a resumed session can still render "Thought for Ns"
        // disclosures. The duration lives on the UI message; match it back to
        // the wire message by its reasoning text (both carry the same
        // accumulated chain of thought).
        {
            std::unordered_map<std::string, std::string> elapsed_by_reasoning;
            {
                std::lock_guard lock(ui_mutex);
                for (const auto& m : *runtime->messages()) {
                    if (m.type == MessageType::Assistant && m.finalized
                        && !m.reasoning_elapsed.empty() && !m.reasoning_text.empty()) {
                        elapsed_by_reasoning.emplace(
                            m.reasoning_text,
                            m.reasoning_elapsed);
                    }
                }
            }
            for (auto& m : snap_messages) {
                if (m.role == "assistant" && !m.reasoning_content.empty()) {
                    if (const auto it = elapsed_by_reasoning.find(m.reasoning_content);
                        it != elapsed_by_reasoning.end()) {
                        m.reasoning_elapsed = it->second;
                    }
                }
            }
        }
        const auto snap_mode = runtime_agent->get_mode();
        const auto snap_context = runtime_agent->get_context_summary();
        const auto working_dir = std::filesystem::current_path().string();

        if (thread_runtimes.current() == runtime) {
            std::lock_guard lock(ui_mutex);
            sync_runtime_metadata();
        }
        const auto metadata = runtime->metadata();
        auto snap_todos = runtime_agent->get_todos();

        const std::uint64_t generation = runtime->request_save();
        const auto& budget = core::budget::BudgetTracker::get_instance();
        const auto total = budget.session_total(metadata.session_id);
        const double cost = budget.session_cost_usd(metadata.session_id);
        const auto stats = session_stats_registry->snapshot(metadata.session_id);

        runtime->begin_save();
        try {
          std::thread([session_store, runtime, metadata,
                     working_dir,
                     snap_messages = std::move(snap_messages),
                     snap_mode, snap_context,
                     snap_todos = std::move(snap_todos),
                     total, cost, stats,
                     generation,
                     completion_promise]() {
            struct SaveCompletion {
                ThreadRuntime::Ptr runtime;
                ~SaveCompletion() { runtime->finish_save(); }
            } completion{runtime};
            const auto complete = [&](SaveResult result) noexcept {
                try {
                    completion_promise->set_value(std::move(result));
                } catch (...) {
                    // The writer must never let promise bookkeeping escape
                    // its detached thread entry point.
                }
            };
            try {
                // Keep this session exclusive until its detached snapshot completes.
                const auto session_lease = runtime->lease();
                (void)session_lease;

                core::session::SessionData data;
                data.session_id        = metadata.session_id;
                data.name              = metadata.session_name;
                data.created_at        = metadata.created_at;
                data.last_active_at    = core::session::SessionStore::now_iso8601();
                data.working_dir       = working_dir;
                data.provider          = metadata.provider;
                data.model             = metadata.model;
                data.mode              = snap_mode;
                data.context_summary   = snap_context;
                data.messages          = snap_messages;
                data.goal              = metadata.goal;
                data.goal_graph        = metadata.goal_graph;
                data.todos             = snap_todos;
                data.stats.prompt_tokens     = total.prompt_tokens;
                data.stats.completion_tokens = total.completion_tokens;
                data.stats.cost_usd          = cost;
                data.stats.turn_count         = stats.turn_count;
                data.stats.tool_calls_total   = stats.tool_calls_total;
                data.stats.tool_calls_success = stats.tool_calls_success;
                data.handoff_summary          = core::session::build_handoff_summary(data);

                std::lock_guard save_lock(runtime->save_mutex());
                if (!runtime->is_latest_save(generation)) {
                    complete(std::nullopt);
                    return;
                }

                std::string error;
                if (session_store->save(data, &error)) {
                    complete(std::nullopt);
                    return;
                }
                const std::string message = error.empty()
                    ? std::string{"unknown save error"}
                    : std::move(error);
                core::logging::warn(
                    "Failed to save session snapshot {}: {}",
                    metadata.session_id,
                    message);
                complete(message);
            } catch (const std::exception& error) {
                const std::string message = std::format(
                    "Session snapshot threw an exception: {}", error.what());
                core::logging::warn(
                    "Failed to save session snapshot {}: {}",
                    metadata.session_id,
                    message);
                complete(message);
            } catch (...) {
                const std::string message =
                    "Session snapshot threw an unknown exception.";
                core::logging::warn(
                    "Failed to save session snapshot {}: {}",
                    metadata.session_id,
                    message);
                complete(message);
            }
          }).detach();
        } catch (const std::exception& error) {
            runtime->finish_save();
            const std::string message = std::format(
                "Could not start session snapshot writer: {}", error.what());
            core::logging::warn(
                "Failed to start session snapshot writer {}: {}",
                metadata.session_id,
                message);
            try {
                completion_promise->set_value(message);
            } catch (...) {
            }
        }
        return completion_future;
    };

    auto save_session_snapshot = [&]() {
        save_runtime_snapshot(current_runtime);
    };

    configure_runtime_agent = [&](const ThreadRuntime::Ptr& runtime) {
        runtime->agent()->set_permission_fn(make_permission_fn(runtime));
        runtime->agent()->set_loop_break_fn([&, runtime](int rounds) {
            append_runtime_history(runtime, std::format(
                "\n\xe2\x9a\xa0  Agent paused after {} consecutive tool failures"
                " \xe2\x80\x94 provide guidance or /clear to start fresh.\n",
                rounds));
        });
        configure_runtime_efficiency(runtime);
    };
    configure_runtime_agent(current_runtime);

    // Archive the current conversation (if any) and open a blank thread.
    // Agentty-compatible: Ctrl+N / /new. Distinct from /clear which also
    // resets history but does not guarantee a durable snapshot first.
    auto start_new_thread = [&]() -> std::optional<std::string> {
        const bool has_history = !agent->get_history().empty();
        const std::string next_thread_name = allocate_project_thread_name();
        const std::string previous_title = thread_display_label(current_runtime);

        const std::string new_session_id = core::session::SessionStore::generate_id();
        const std::string new_created_at = core::session::SessionStore::now_iso8601();
        core::session::SessionData fresh;
        fresh.session_id = new_session_id;
        fresh.created_at = new_created_at;
        fresh.provider = active_provider_name;
        fresh.model = active_model_name;
        fresh.mode = agent->get_mode();
        auto next_lease = session_leases.reserve(fresh);
        if (!next_lease) {
            return std::format("Could not start a new thread: {}", next_lease.error());
        }

        auto next_agent = make_thread_agent(fresh);
        if (!next_agent) {
            return std::format("Could not start a new thread: {}", next_agent.error());
        }

        auto next_messages = std::make_shared<std::vector<UiMessage>>();
        std::string notice = "New thread started.";
        if (has_history) {
            notice = std::format(
                "Saved previous thread ({}) and started a new one. "
                "Use /threads to switch threads.",
                previous_title);
        }
        append_ui_message(*next_messages, make_system_message(
            std::format("\n»  {}\n", notice)));
        if (const auto message = startup_history_message(); !message.empty()) {
            append_ui_message(*next_messages, make_system_message(message));
        }

        next_lease->commit();
        auto next_runtime = std::make_shared<ThreadRuntime>(
            ThreadRuntimeMetadata{
                .session_id = new_session_id,
                .thread_name = next_thread_name,
                .auto_thread_name = true,
                .session_name = {},
                .created_at = new_created_at,
                .file_path = session_store->compute_path(fresh).string(),
                .provider = active_provider_name,
                .model = active_model_name,
                .model_selection = ModelSelectionSnapshot{
                    .mode = model_selection_mode,
                    .manual_provider_name = manual_provider_name,
                    .manual_model_name = manual_model_name,
                    .router_policy = active_router_policy,
                },
                .previous_model_selection = previous_model_selection,
                .yolo_enabled = startup_yolo_enabled,
                .permission_rules = {},
            },
            *next_agent,
            std::move(next_messages),
            session_leases.retain(new_session_id));
        if (!thread_runtimes.insert(next_runtime)) {
            return std::string("Could not register the new thread runtime.");
        }
        configure_runtime_agent(next_runtime);

        sync_runtime_metadata();
        if (has_history) {
            save_session_snapshot();
        }

        {
            std::lock_guard lock(ui_mutex);
            current_runtime = next_runtime;
            agent = next_runtime->agent();
            llm_provider = agent->get_provider();
            session_effort_value = agent->get_effort_level();
            router_provider = std::dynamic_pointer_cast<
                core::llm::providers::RouterProvider>(llm_provider);
            selected_messages = next_runtime->messages();
            review_activity_state.active = false;
            review_activity_state.hint.clear();
            review_activity_state.started_at =
                std::chrono::steady_clock::time_point::min();
            session_picker_state = {};
            goal_manager.clear();
            goal_engine.reset();
            pending_goal_graph_snapshot.clear();
            session_id = new_session_id;
            session_name.clear();
            session_created_at = new_created_at;
            session_file_path = session_store->compute_path(fresh).string();
        }

        static_cast<void>(thread_runtimes.select(new_session_id));
        reset_history_view();
        animation_cv.notify_one();
        core::budget::BudgetTracker::get_instance().reset_session(new_session_id);
        wake_ui();
        return std::nullopt;
    };

    auto thread_has_active_work = [&](const ThreadRuntime::Ptr& runtime) {
        if (!runtime) {
            return false;
        }
        const std::string target_session_id = runtime->session_id();
        const bool has_active_shell = std::ranges::any_of(
            core::tools::ShellTool::active_commands(),
            [&](const auto& command) {
                return command.session_id == target_session_id;
            });
        return runtime->turn_active()
            || runtime->queued_turn_count() > 0
            || runtime->workers_in_flight() > 0
            || has_active_shell
            || (direct_shell_state
                && direct_shell_state->has_active_for(target_session_id));
    };

    // Archive and release one live runtime without deleting its saved
    // session. The process's main/current/running threads are intentionally
    // protected so closing can never invalidate shared UI state or active
    // callbacks.
    auto close_thread = [&](std::string_view target_session_id)
        -> std::optional<std::string> {
        auto runtime = thread_runtimes.find(target_session_id);
        if (!runtime) {
            return std::format(
                "Active thread {} is no longer available.", target_session_id);
        }
        if (runtime == main_runtime) {
            return "The main thread cannot be closed.";
        }
        if (runtime == thread_runtimes.current()) {
            return "Switch to another thread before closing the current one.";
        }
        if (thread_has_active_work(runtime)) {
            return "Stop the thread's active work before closing it.";
        }

        if (!runtime->agent()->get_history().empty()) {
            auto save_result = save_runtime_snapshot(runtime).get();
            if (save_result.has_value()) {
                return std::format(
                    "Could not archive thread {}: {}",
                    thread_display_label(runtime),
                    *save_result);
            }
        }
        // Older superseded saves may still be winding down. Do not release
        // their runtime/lease until every writer has passed its completion
        // guard.
        runtime->wait_until_saved();
        if (!thread_runtimes.erase(target_session_id)) {
            return "The thread became active while it was being closed.";
        }

        runtime->set_lease(nullptr);
        session_leases.release(target_session_id);
        session_stats_registry->reset(target_session_id);
        core::budget::BudgetTracker::get_instance().reset_session(target_session_id);
        return std::nullopt;
    };

    // Ctrl+D on an empty secondary thread behaves like closing a terminal tab:
    // return to the persistent main runtime, then archive and release the tab
    // that was visible. The normal close path remains the single authority for
    // persistence, leases, stats, and budget cleanup.
    auto close_current_secondary_thread = [&]() -> std::optional<std::string> {
        auto closing_runtime = thread_runtimes.current();
        if (!closing_runtime || closing_runtime == main_runtime) {
            return "The main thread cannot be closed.";
        }
        if (thread_has_active_work(closing_runtime)) {
            return "Stop the thread's active work before closing it.";
        }

        const std::string closing_session_id = closing_runtime->session_id();
        const auto main_data = live_session_data(main_runtime);
        if (!main_data.has_value()) {
            return "The main thread is no longer available.";
        }
        if (const auto error = resume_session(*main_data); error.has_value()) {
            return std::format("Could not return to the main thread: {}", *error);
        }
        return close_thread(closing_session_id);
    };

    auto latest_completed_assistant_output = [&]() {
        std::lock_guard lock(ui_mutex);
        return latest_completed_assistant_source(*selected_messages);
    };

    auto open_code_blocks = [&](std::optional<std::size_t> one_based_block)
        -> core::commands::CommandOperationResult {
        if (current_runtime->turn_active()) {
            return {.ok = false, .message = "Stop the active turn before running response code."};
        }
        const std::string response = latest_completed_assistant_output();
        if (response.empty()) {
            return {.ok = false, .message = "There is no completed assistant response to inspect."};
        }

        auto opened = code_block_runner.open(response, one_based_block);
        if (!opened) {
            return {.ok = false, .message = std::move(opened.error())};
        }
        wake_ui();
        return {.ok = true, .message = {}};
    };

    auto open_rewind_menu = [&]() -> bool {
        if (current_runtime->turn_active()) {
            append_history("\nℹ  Stop the active turn before rewinding.\n");
            return true;
        }

        const auto history = agent->get_history();
        std::lock_guard lock(ui_mutex);
        const bool opened = open_rewind_picker(rewind_picker_state, history);
        wake_ui();
        return opened;
    };

    auto rewind_to_message = [&](const RewindPickerOption& target) {
        if (current_runtime->turn_active()) {
            append_history("\nℹ  Stop the active turn before rewinding.\n");
            return;
        }

        const auto history = agent->get_history();
        if (target.history_index >= history.size()
            || history[target.history_index].role != "user") {
            append_history("\n✗  The selected rewind point is no longer available.\n");
            return;
        }

        bool ui_target_exists = false;
        {
            std::lock_guard lock(ui_mutex);
            const auto user_count = std::ranges::count_if(
                *selected_messages,
                [](const UiMessage& message) { return message.type == MessageType::User; });
            ui_target_exists = target.user_ordinal < static_cast<std::size_t>(user_count);
        }
        if (!ui_target_exists) {
            append_history("\n✗  The selected rewind point is not present in the visible conversation.\n");
            return;
        }

        std::vector<core::llm::Message> prefix(
            history.begin(),
            history.begin() + static_cast<std::ptrdiff_t>(target.history_index));
        const auto mode = agent->get_mode();
        const auto branch = branch_session(prefix, {});
        if (!branch) {
            append_history(std::format("\n✗  {}\n", branch.error()));
            return;
        }

        agent->load_history(prefix, {}, mode);
        {
            std::lock_guard lock(ui_mutex);
            (void)truncate_ui_before_user_turn(*selected_messages, target.user_ordinal);
            input_text = target.prompt;
            input_cursor_position = static_cast<int>(input_text.size());
        }
        reset_history_view();
        append_history(std::format(
            "\n↶  Rewound into session {}. The selected prompt is ready to edit; "
            "the original remains in session {}.\n",
            branch->second,
            branch->first));
    };

    auto list_todos = [&]() {
        return agent->get_todos();
    };

    auto add_todo = [&](std::string_view text) -> core::commands::CommandOperationResult {
        const std::string trimmed_text = std::string(trim_ascii(text));
        if (trimmed_text.empty()) {
            return {.ok = false, .message = "Todo text cannot be empty."};
        }

        auto added = agent->add_todo(trimmed_text);
        if (!added.has_value()) return {.ok = false, .message = added.error()};
        save_session_snapshot();
        return {.ok = true, .message = std::format("Added todo {{{}}}.", added->id)};
    };

    auto set_todo_completed = [&](std::string_view selector, bool completed)
        -> core::commands::CommandOperationResult {
        auto updated = agent->set_todo_status(
            selector,
            completed ? core::session::TodoStatus::Completed
                      : core::session::TodoStatus::Pending);
        if (!updated.has_value()) return {.ok = false, .message = updated.error()};
        const auto todo_label = updated->id.empty()
            ? updated->text
            : std::format("{{{}}}", updated->id);
        save_session_snapshot();
        return {
            .ok = true,
            .message = completed
                ? std::format("Marked todo {} complete.", todo_label)
                : std::format("Reopened todo {}.", todo_label),
        };
    };

    auto remove_todo = [&](std::string_view selector) -> core::commands::CommandOperationResult {
        auto removed = agent->remove_todo(selector);
        if (!removed.has_value()) return {.ok = false, .message = removed.error()};
        const auto todo_label = removed->id.empty()
            ? removed->text
            : std::format("{{{}}}", removed->id);
        save_session_snapshot();
        return {.ok = true, .message = std::format("Removed todo {}.", todo_label)};
    };

    auto clear_completed_todos = [&]() -> core::commands::CommandOperationResult {
        const auto removed = agent->clear_completed_todos();
        save_session_snapshot();
        return {
            .ok = true,
            .message = removed == 0
                ? "No completed todos to clear."
                : std::format("Cleared {} completed todo(s).", removed),
        };
    };

    auto current_goal = [&]() -> std::optional<core::session::SessionGoal> {
        std::lock_guard lock(ui_mutex);
        return goal_manager.current();
    };

    auto set_goal = [&](std::string_view objective) -> core::commands::CommandOperationResult {
        core::session::SessionGoal updated;
        {
            std::lock_guard lock(ui_mutex);
            auto maybe_goal = goal_manager.set(objective);
            if (!maybe_goal.has_value()) {
                return {.ok = false, .message = "Goal objective cannot be empty."};
            }
            updated = *maybe_goal;
        }
        agent->set_session_goal(updated);
        save_session_snapshot();
        return {.ok = true, .message = "Goal set."};
    };

    auto set_goal_status = [&](std::string_view raw_status,
                               std::string_view note) -> core::commands::CommandOperationResult {
        const std::string status = to_lower_ascii(std::string(trim_ascii(raw_status)));
        if (status != "active" && status != "blocked" && status != "complete") {
            return {.ok = false, .message = "Unknown goal status."};
        }
        const auto goal_status = core::session::goal_status_from_string(status);

        core::session::SessionGoal updated;
        {
            std::lock_guard lock(ui_mutex);
            auto maybe_goal = goal_manager.set_status(goal_status, note);
            if (!maybe_goal.has_value()) {
                return {.ok = false, .message = "No goal is set."};
            }
            updated = *maybe_goal;
        }
        agent->set_session_goal(core::session::is_active(updated.status)
            ? std::optional<core::session::SessionGoal>{updated}
            : std::nullopt);
        save_session_snapshot();
        if (status == "complete") {
            return {.ok = true, .message = "Goal marked complete."};
        }
        if (status == "blocked") {
            return {.ok = true, .message = "Goal marked blocked."};
        }
        return {.ok = true, .message = "Goal resumed."};
    };

    auto clear_goal = [&]() -> core::commands::CommandOperationResult {
        {
            std::lock_guard lock(ui_mutex);
            if (!goal_manager.has_goal()) {
                return {.ok = true, .message = "No goal to clear."};
            }
            goal_manager.clear();
        }
        agent->set_session_goal(std::nullopt);
        if (goal_engine) {
            goal_engine->clear();
        }
        save_session_snapshot();
        return {.ok = true, .message = "Goal cleared."};
    };

    // Lazily builds (and rehydrates) the session's goal-graph engine. The
    // engine only needs the agent and a transcript sink, so it is constructed
    // from a minimal context rather than the full command context.
    auto goal_engine_accessor = [&]() -> std::shared_ptr<void> {
        std::shared_ptr<core::goal::GoalEngine> engine;
        bool created = false;
        {
            std::lock_guard lock(ui_mutex);
            if (!goal_engine) {
                core::commands::CommandContext engine_ctx;
                engine_ctx.agent = agent;
                engine_ctx.append_history_fn = append_history;
                goal_engine = core::commands::GoalExecutor::make_engine(engine_ctx);

                if (!pending_goal_graph_snapshot.empty()) {
                    if (const auto restored =
                            goal_engine->restore_snapshot(pending_goal_graph_snapshot);
                        !restored.has_value()) {
                        core::logging::warn("Failed to restore goal graph: {}", restored.error());
                    }
                    pending_goal_graph_snapshot.clear();
                }
                created = true;
            }
            engine = goal_engine;
        }

        // Publish the graph into the agent's system prompt so a work turn sees
        // the objective and frontier, not just its own directive. Done outside
        // ui_mutex: this takes the agent's history lock, and the agent takes
        // ui_mutex from its streaming callbacks. The capture is weak because
        // the engine's hooks already hold a strong reference to the agent.
        if (created) {
            agent->set_goal_graph_context_fn(
                [weak = std::weak_ptr<core::goal::GoalEngine>(engine)] {
                    const auto live = weak.lock();
                    return live ? live->prompt_context() : std::string{};
                });
        }
        return engine;
    };

    auto memory_state = [memory_system]() {
        return memory_system->semantic().load();
    };

    auto memory_thread_policy = [agent]() {
        return agent->memory_thread_policy();
    };

    auto set_memory_thread_policy =
        [agent](core::memory::MemoryThreadPolicy policy)
            -> core::commands::CommandOperationResult {
        agent->set_memory_thread_policy(policy);
        return {.ok = true, .message = "Thread memory policy updated."};
    };

    auto run_memory_review = [agent, append_history]() -> core::commands::CommandOperationResult {
        agent->run_memory_review_async([append_history](std::string message) {
            if (!message.empty()) {
                append_history("\n[" + message + "]\n");
            }
        });
        return {.ok = true, .message = "Memory background review queued."};
    };

    auto set_memory_settings =
        [memory_system](core::memory::MemorySettings settings)
            -> core::commands::CommandOperationResult {
        std::string error;
        if (!memory_system->semantic().save_settings(settings, &error)) {
            return {.ok = false, .message = error};
        }
        if (settings.auto_capture || settings.background_review) {
            return {.ok = true, .message = "Memory and automatic capture enabled."};
        }
        if (settings.enabled) {
            return {.ok = true, .message = "Memory context enabled."};
        }
        return {.ok = true, .message = "Memory disabled."};
    };

    auto add_memory = [memory_system](std::string_view content)
        -> core::commands::CommandOperationResult {
        auto result = memory_system->semantic().remember(content, "global", {}, "manual");
        return {.ok = result.ok, .message = result.message};
    };

    auto forget_memory = [memory_system](std::string_view selector)
        -> core::commands::CommandOperationResult {
        auto result = memory_system->semantic().forget(selector);
        return {.ok = result.ok, .message = result.message};
    };

    auto clean_memory = [memory_system]() -> core::commands::CommandOperationResult {
        auto result = memory_system->semantic().clean();
        return {.ok = result.ok, .message = result.message};
    };

    auto clear_memory = [memory_system]() -> core::commands::CommandOperationResult {
        auto result = memory_system->semantic().clear();
        return {.ok = result.ok, .message = result.message};
    };

    auto save_memory_markdown =
        [memory_system](std::filesystem::path path) -> core::commands::CommandOperationResult {
        auto result = memory_system->semantic().save_markdown(path);
        return {.ok = result.ok, .message = result.message};
    };

    auto load_memory_markdown =
        [memory_system](std::filesystem::path path) -> core::commands::CommandOperationResult {
        auto result = memory_system->semantic().load_markdown(path);
        return {.ok = result.ok, .message = result.message};
    };

    auto add_mcp_server = [&](const core::config::McpServerConfig& server,
                              core::config::SettingsScope scope)
        -> core::commands::CommandOperationResult {
        std::string error;
        if (!core::config::ConfigManager::get_instance().persist_mcp_server(
                server,
                scope,
                std::filesystem::current_path(),
                &error)) {
            return {.ok = false, .message = error};
        }
        std::string oauth_note;
        if (core::mcp::mcp_server_wants_oauth(server)
            && !core::mcp::has_mcp_oauth_session(
                   server.name,
                   core::config::ConfigManager::get_instance().get_config_dir())) {
            std::string login_error;
            std::string login_success;
            auto run_login = [&]() {
                try {
                    core::auth::ui::ConsoleAuthUI ui;
                    (void)core::mcp::login_mcp_oauth(
                        server.name,
                        server.url,
                        core::config::ConfigManager::get_instance().get_config_dir(),
                        core::mcp::McpOAuthLoginOptions{
                            .preferred_scopes = server.oauth_scopes,
                            .client_id = server.oauth_client_id,
                            .client_secret = server.oauth_client_secret,
                            .ui = &ui,
                        });
                    login_success = "OAuth login complete.";
                } catch (const std::exception& e) {
                    login_error = e.what();
                }
            };
            auto closure = screen.WithRestoredIO(run_login);
            closure();
            if (login_error.empty()) {
                const std::string reload_message = reload_mcp_live();
                return {
                    .ok = true,
                    .message = std::format(
                        "Saved MCP server '{}' to the {} overlay. {} {}",
                        server.name,
                        scope == core::config::SettingsScope::User ? "user" : "workspace",
                        login_success,
                        reload_message),
                };
            }
            oauth_note = std::format(
                " OAuth is required — run `/mcp login {}` ({}).",
                server.name,
                login_error);
        }

        const std::string reload_message = reload_mcp_live();
        return {
            .ok = true,
            .message = std::format(
                "Saved MCP server '{}' to the {} overlay. {}{}",
                server.name,
                scope == core::config::SettingsScope::User ? "user" : "workspace",
                reload_message,
                oauth_note),
        };
    };

    auto remove_mcp_server = [&](std::string_view server_name,
                                 core::config::SettingsScope scope)
        -> core::commands::CommandOperationResult {
        std::string error;
        if (!core::config::ConfigManager::get_instance().remove_mcp_server(
                server_name,
                scope,
                std::filesystem::current_path(),
                &error)) {
            return {.ok = false, .message = error};
        }
        const std::string reload_message = reload_mcp_live();
        return {
            .ok = true,
            .message = std::format(
                "Removed MCP server '{}' from the {} overlay. {}",
                std::string(server_name),
                scope == core::config::SettingsScope::User ? "user" : "workspace",
                reload_message),
        };
    };

    auto list_mcp_servers = [&]() {
        return core::config::ConfigManager::get_instance().get_config().mcp_servers;
    };

    auto login_mcp_server = [&](std::string_view server_name)
        -> core::commands::CommandOperationResult {
        const std::string name(server_name);
        const auto servers = core::config::ConfigManager::get_instance().get_config().mcp_servers;
        const auto it = std::find_if(
            servers.begin(), servers.end(),
            [&](const core::config::McpServerConfig& s) { return s.name == name; });
        if (it == servers.end()) {
            return {.ok = false, .message = "Unknown MCP server '" + name + "'."};
        }
        if (it->transport != "http" || it->url.empty()) {
            return {.ok = false,
                    .message = "MCP OAuth login requires an http server with a url."};
        }

        std::string error;
        std::string success;
        auto run_login = [&]() {
            try {
                core::auth::ui::ConsoleAuthUI ui;
                (void)core::mcp::login_mcp_oauth(
                    name,
                    it->url,
                    core::config::ConfigManager::get_instance().get_config_dir(),
                    core::mcp::McpOAuthLoginOptions{
                        .preferred_scopes = it->oauth_scopes,
                        .client_id = it->oauth_client_id,
                        .client_secret = it->oauth_client_secret,
                        .ui = &ui,
                    });
                success = "Authenticated with '" + name + "'.";
            } catch (const std::exception& e) {
                error = e.what();
            }
        };

        auto closure = screen.WithRestoredIO(run_login);
        closure();

        if (!error.empty()) {
            return {.ok = false, .message = error};
        }
        const std::string reload_message = reload_mcp_live();
        return {
            .ok = true,
            .message = success + " " + reload_message,
        };
    };

    auto logout_mcp_server = [&](std::string_view server_name)
        -> core::commands::CommandOperationResult {
        const std::string name(server_name);
        try {
            core::mcp::logout_mcp_oauth(
                name,
                core::config::ConfigManager::get_instance().get_config_dir());
            const std::string reload_message = reload_mcp_live();
            return {
                .ok = true,
                .message = "Cleared OAuth session for '" + name + "'. " + reload_message,
            };
        } catch (const std::exception& e) {
            return {.ok = false, .message = e.what()};
        }
    };

    auto make_turn_callbacks =
        [&](const ThreadRuntime::Ptr& runtime,
            std::shared_ptr<LiveAssistantTimeline> timeline) {
        core::agent::Agent::TurnCallbacks callbacks;
        callbacks.on_step_begin = [runtime, timeline,
                                   &ui_mutex,
                                   &animation_cv,
                                   &wake_ui]() {
            {
                std::lock_guard lock(ui_mutex);
                auto& timers = runtime->activity_timers();
                const std::string previous_id(timeline->current_message_id());
                std::string previous_elapsed;
                if (timeline->step_started()) {
                    previous_elapsed = format_elapsed_compact(
                        timers.elapsed(previous_id)
                            .value_or(std::chrono::seconds{0}));
                }
                UiMessage* message =
                    timeline->begin_step(*runtime->messages(), std::move(previous_elapsed));
                if (message == nullptr) {
                    timers.stop(previous_id);
                } else if (message->id != previous_id) {
                    timers.stop(previous_id);
                    timers.start(message->id);
                }
            }
            animation_cv.notify_one();
            wake_ui();
        };
        callbacks.on_reasoning =
            [runtime, timeline, &update_live_assistant_message](
                const std::string& delta) {
                update_live_assistant_message(runtime, timeline, [&](UiMessage& message) {
                    // Reasoning that arrives after finalization is dropped; a
                    // completed card must not reopen its live thinking box.
                    if (message.finalized) {
                        return;
                    }
                    // Streamed reasoning belongs to the Thinking phase.
                    message.reasoning_kind = UiMessage::ActivityKind::Thinking;
                    message.reasoning_active = true;
                    message.activity_recorded = true;
                    message.reasoning_text += delta;
                });
            };
        callbacks.on_tool_start =
            [runtime, timeline, &update_live_assistant_message](
                const core::llm::ToolCall& tool_call) {
                update_live_assistant_message(runtime, timeline, [&](UiMessage& message) {
                    if (message.finalized) {
                        return;
                    }
                    message.pending = true;
                    message.thinking = false;
                    message.show_activity_status = true;
                    message.tools.push_back(make_tool_activity(
                        tool_call.id,
                        tool_call.function.name,
                        tool_call.function.arguments,
                        summarize_tool_arguments(
                            tool_call.function.name,
                            tool_call.function.arguments)));
                    message.tools.back().status = ToolActivity::Status::Executing;
                });
            };
        callbacks.on_tool_finish =
            [runtime, timeline, &update_live_assistant_message](
                const core::llm::ToolCall& tool_call,
                const core::llm::Message& result) {
                update_live_assistant_message(runtime, timeline, [&](UiMessage& message) {
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
                        message.show_activity_status = true;
                    }
                });
            };
        callbacks.on_subagent_event =
            [runtime, timeline, &update_live_assistant_message](
                const core::agent::SubagentEvent& event) {
                if (event.parent_tool_call_id.empty() || event.task_id.empty()) {
                    return;
                }

                update_live_assistant_message(runtime, timeline, [&](UiMessage& message) {
                    auto* parent_tool = find_tool_activity(message, event.parent_tool_call_id);
                    if (parent_tool == nullptr) {
                        message.tools.push_back(make_tool_activity(
                            event.parent_tool_call_id,
                            std::string(core::agent::SubagentOrchestrator::kTaskToolName),
                            "{}",
                            event.description));
                        parent_tool = &message.tools.back();
                        parent_tool->status = ToolActivity::Status::Executing;
                    }

                    auto* subagent = find_subagent_activity(*parent_tool, event.task_id);
                    if (subagent == nullptr) {
                        ToolActivity::SubagentActivity created;
                        created.id = event.task_id;
                        created.worker_name = event.worker_name;
                        created.description = event.description;
                        created.provider = event.provider_name;
                        created.model = event.model_name;
                        parent_tool->subagents.push_back(std::move(created));
                        subagent = &parent_tool->subagents.back();
                    }

                    if (!event.worker_name.empty()) subagent->worker_name = event.worker_name;
                    if (!event.description.empty()) subagent->description = event.description;
                    if (!event.provider_name.empty()) subagent->provider = event.provider_name;
                    if (!event.model_name.empty()) subagent->model = event.model_name;
                    subagent->steps = std::max(subagent->steps, event.steps);
                    subagent->tool_calls = std::max(subagent->tool_calls, event.tool_calls);
                    subagent->failed_tool_calls = std::max(subagent->failed_tool_calls, event.failed_tool_calls);

                    auto upsert_child_tool = [&]() -> ToolActivity::ChildToolActivity* {
                        if (event.tool_call_id.empty()) {
                            return nullptr;
                        }
                        for (auto& child_tool : subagent->recent_tools) {
                            if (child_tool.id == event.tool_call_id) {
                                return &child_tool;
                            }
                        }
                        ToolActivity::ChildToolActivity child;
                        child.id = event.tool_call_id;
                        child.name = event.tool_name;
                        child.args = event.tool_arguments;
                        child.description = summarize_tool_arguments(event.tool_name, event.tool_arguments);
                        child.status = ToolActivity::Status::Executing;
                        subagent->recent_tools.push_back(std::move(child));
                        constexpr std::size_t kMaxSubagentTools = 12;
                        if (subagent->recent_tools.size() > kMaxSubagentTools) {
                            subagent->recent_tools.erase(subagent->recent_tools.begin());
                        }
                        return &subagent->recent_tools.back();
                    };

                    switch (event.kind) {
                        case core::agent::SubagentEvent::Kind::Started:
                        case core::agent::SubagentEvent::Kind::Progress:
                            subagent->status = ToolActivity::Status::Executing;
                            parent_tool->status = ToolActivity::Status::Executing;
                            break;
                        case core::agent::SubagentEvent::Kind::TextDelta:
                            if (!event.text_delta.empty()) {
                                subagent->latest_text += event.text_delta;
                                constexpr std::size_t kMaxLatestText = 1200;
                                if (subagent->latest_text.size() > kMaxLatestText) {
                                    subagent->latest_text.erase(
                                        0,
                                        subagent->latest_text.size() - kMaxLatestText);
                                }
                            }
                            subagent->status = ToolActivity::Status::Executing;
                            parent_tool->status = ToolActivity::Status::Executing;
                            break;
                        case core::agent::SubagentEvent::Kind::ToolStarted:
                            if (auto* child_tool = upsert_child_tool()) {
                                child_tool->status = ToolActivity::Status::Executing;
                            }
                            subagent->status = ToolActivity::Status::Executing;
                            parent_tool->status = ToolActivity::Status::Executing;
                            break;
                        case core::agent::SubagentEvent::Kind::ToolFinished:
                            if (auto* child_tool = upsert_child_tool()) {
                                // The probe exists only to derive a status from
                                // apply_tool_result and is then discarded, so
                                // skip the diff build (and its file read).
                                ToolActivity probe = make_tool_activity(
                                    child_tool->id,
                                    child_tool->name,
                                    child_tool->args,
                                    child_tool->description,
                                    /*build_diff_preview=*/false);
                                apply_tool_result(probe, event.tool_result);
                                child_tool->status = probe.status;
                            }
                            subagent->status = ToolActivity::Status::Executing;
                            parent_tool->status = ToolActivity::Status::Executing;
                            break;
                        case core::agent::SubagentEvent::Kind::Finished:
                            subagent->status = ToolActivity::Status::Succeeded;
                            subagent->summary = event.summary;
                            break;
                        case core::agent::SubagentEvent::Kind::Failed:
                            subagent->status = ToolActivity::Status::Failed;
                            subagent->summary = event.summary.empty()
                                ? "Subagent failed before producing a final summary."
                                : event.summary;
                            break;
                        case core::agent::SubagentEvent::Kind::Cancelled:
                            subagent->status = ToolActivity::Status::Cancelled;
                            subagent->summary = event.summary.empty()
                                ? "Subagent was cancelled."
                                : event.summary;
                            break;
                    }

                    if (!message.finalized) {
                        message.pending = true;
                        message.thinking = false;
                        message.show_activity_status = true;
                    }
                });
            };
        // Route out-of-band lifecycle status (auto-compaction progress) to the
        // standalone history log rather than the streaming assistant-message
        // chunk callback. Emitting status through the chunk callback would
        // pollute the assistant response body and re-mark the finalized message
        // as pending, which leaves the UI stuck on "Analyzing..." and breaks
        // /copy ("Nothing to copy yet").
        callbacks.on_status_log = [runtime, &append_runtime_history](const std::string& status) {
            append_runtime_history(runtime, status);
        };
        callbacks.allow_efficiency_rotation = true;
        callbacks.min_context_utilization_for_rotation = 0.75;
        return callbacks;
    };

    // ── Agent turn submission ─────────────────────────────────────────────────
    // Extracted so that SkillCommand can inject an expanded prompt as a full
    // agent turn (with user message card + tool cards) via send_user_message_fn.
    std::function<void(ThreadRuntime::Ptr,
                       std::string,
                       core::agent::Agent::TurnCallbacks)> submit_agent_turn;

    auto submit_or_queue_agent_turn =
        [&](std::string text, core::agent::Agent::TurnCallbacks turn_callbacks) {
            if (text.empty()) {
                return;
            }

            auto runtime = current_runtime;
            PendingAgentTurn pending{
                .text = std::move(text),
                .callbacks = std::move(turn_callbacks),
            };
            const bool should_submit_now = runtime->begin_or_queue(pending);

            if (!should_submit_now) {
                runtime->request_stop();
                append_runtime_history(
                    runtime,
                    "\n↪  Steering queued; stopping the current turn.\n");
                wake_ui();
                return;
            }
            submit_agent_turn(
                runtime,
                std::move(pending.text),
                std::move(pending.callbacks));
        };

    submit_agent_turn = [&](ThreadRuntime::Ptr runtime,
                            std::string text,
                            core::agent::Agent::TurnCallbacks turn_callbacks) {
        if (text.empty()) {
            return;
        }
        std::string timestamp = current_time_str();
        std::string assistant_message_id;
        std::string user_message_id;
        {
            std::lock_guard lock(ui_mutex);
            auto messages = runtime->messages();
            append_ui_message(*messages, make_user_message(text, timestamp));
            user_message_id = messages->back().id;
            append_ui_message(*messages, make_assistant_message("", "", true));
            assistant_message_id = messages->back().id;
        }
        auto live_timeline =
            std::make_shared<LiveAssistantTimeline>(assistant_message_id);
        if (thread_runtimes.current() == runtime) {
            reset_history_view();
        }
        runtime->activity_timers().start(assistant_message_id);
        runtime->activity_timers().start(user_message_id);
        animation_cv.notify_one();
        wake_ui();

        auto effective_callbacks = make_turn_callbacks(runtime, live_timeline);
        auto retry_callbacks = turn_callbacks;
        effective_callbacks.provider_override = std::move(turn_callbacks.provider_override);
        effective_callbacks.model_override = std::move(turn_callbacks.model_override);
        effective_callbacks.allowed_tools = std::move(turn_callbacks.allowed_tools);
        effective_callbacks.allow_efficiency_rotation = turn_callbacks.allow_efficiency_rotation;
        if (turn_callbacks.min_context_utilization_for_rotation > 0.0) {
            effective_callbacks.min_context_utilization_for_rotation =
                turn_callbacks.min_context_utilization_for_rotation;
        }
        effective_callbacks.on_authentication_required =
            [runtime,
             retry_text = text,
             retry_callbacks = std::move(retry_callbacks),
             &authentication_recovery_state,
             &authentication_manager,
             &ui_mutex,
             &wake_ui](
                const core::llm::AuthenticationRecoveryRequest& request) {
                if (request.provider_id.empty()) {
                    return;
                }
                const auto provider =
                    authentication_manager.describe_provider(request.provider_id);
                if (!provider.has_value()) {
                    return;
                }
                {
                    std::lock_guard lock(ui_mutex);
                    if (!authentication_recovery_state.providers
                             .insert(provider->credential_id)
                             .second) {
                        return;
                    }

                    PendingAuthenticationRecovery pending{
                        .runtime = runtime,
                        .request = request,
                        .provider = *provider,
                        .retry_text = retry_text,
                        .retry_callbacks = retry_callbacks,
                    };
                    if (!authentication_recovery_state.active.has_value()) {
                        authentication_recovery_state.active = std::move(pending);
                        authentication_recovery_state.selected = 0;
                    } else {
                        authentication_recovery_state.queued.push_back(
                            std::move(pending));
                    }
                }
                wake_ui();
            };

        runtime->begin_worker();
        try {
          std::thread([text = std::string(text),
                     base_dir = std::filesystem::current_path(),
                     runtime,
                     effective_callbacks = std::move(effective_callbacks),
                     live_timeline,
                     user_message_id,
                     &update_live_assistant_message,
                     &submit_agent_turn,
                     &ui_mutex,
                     &animation_cv,
                     &wake_ui,
                     &save_runtime_snapshot]() mutable {
            struct WorkerCompletion {
                ThreadRuntime::Ptr runtime;
                ~WorkerCompletion() { runtime->finish_worker(); }
            } worker_completion{runtime};
            auto agent = runtime->agent();
            const auto expanded_prompt = core::context::expand_prompt(text, base_dir);
            // An absolute @ mention is an explicit user selection. Finder
            // drag-and-drop arrives through bracketed paste in this form.
            agent->grant_workspace_paths(expanded_prompt.explicit_path_mentions);

            core::llm::Message user_message;
            user_message.role = "user";
            user_message.content = expanded_prompt.display_text;
            user_message.input_text = text;
            if (core::llm::message_has_media_input(expanded_prompt.content_parts)) {
                user_message.content_parts = expanded_prompt.content_parts;
            }

            agent->send_message(user_message,
                [runtime, live_timeline,
                 &update_live_assistant_message](const std::string& chunk) {
                    update_live_assistant_message(runtime, live_timeline, [&](UiMessage& message) {
                        // Never revert a finalized assistant message back to
                        // pending. Late/out-of-band callbacks (e.g. compaction
                        // status racing the done callback) must not resurrect
                        // the "Analyzing..." spinner or block /copy.
                        if (!message.finalized
                            && runtime->turn_active()) {
                            message.pending = true;
                        }
                        if (message.thinking) {
                            message.thinking = false;
                            message.show_activity_status = true;
                        }

                        // Keep provider bytes separate from display-only pause markers.
                        message.assistant_source_text += chunk;

                        if (message.text.empty()) {
                            message.text = chunk;
                        } else {
                            message.text += chunk;
                        }
                    });
                },
                [](const std::string&, const std::string&) {},
                [runtime, live_timeline, agent, user_message_id,
                 &submit_agent_turn,
                 &ui_mutex,
                 &animation_cv,
                 &wake_ui,
                 &save_runtime_snapshot]() {
                    const bool was_stopped = agent->is_stop_requested();
                    // A turn only counts as successfully completed when it was
                    // neither cancelled (ESC/Ctrl+C) nor ended with an error.
                    const bool turn_succeeded = !was_stopped && !agent->last_turn_failed();
                    {
                        std::lock_guard lock(ui_mutex);
                        const std::string current_id(
                            live_timeline->current_message_id());
                        const std::string reasoning_elapsed =
                            format_elapsed_compact(
                                runtime->activity_timers().elapsed(current_id)
                                    .value_or(std::chrono::seconds{0}));
                        runtime->activity_timers().stop(current_id);
                        std::string turn_elapsed;
                        if (const auto elapsed =
                                runtime->activity_timers().elapsed(user_message_id)) {
                            turn_elapsed = format_elapsed_compact(*elapsed);
                            stamp_user_turn_elapsed(
                                *runtime->messages(),
                                user_message_id,
                                turn_elapsed);
                        }
                        runtime->activity_timers().stop(user_message_id);
                        if (auto* assistant =
                                live_timeline->current(*runtime->messages())) {
                            assistant->timestamp = current_time_str();
                            if (!turn_elapsed.empty()) {
                                assistant->activity_elapsed = turn_elapsed;
                            }
                        }
                        live_timeline->finish(
                            *runtime->messages(),
                            reasoning_elapsed,
                            was_stopped);
                    }
                    animation_cv.notify_one();
                    wake_ui();
                    save_runtime_snapshot(runtime);
                    if (auto next_turn = runtime->finish_turn(turn_succeeded);
                        next_turn.has_value()) {
                        submit_agent_turn(runtime,
                                          std::move(next_turn->text),
                                          std::move(next_turn->callbacks));
                    }
                },
                std::move(effective_callbacks));
          }).detach();
        } catch (const std::exception& error) {
            runtime->finish_worker();
            {
                std::lock_guard lock(ui_mutex);
                std::string turn_elapsed;
                if (const auto elapsed =
                        runtime->activity_timers().elapsed(user_message_id)) {
                    turn_elapsed = format_elapsed_compact(*elapsed);
                    stamp_user_turn_elapsed(
                        *runtime->messages(),
                        user_message_id,
                        turn_elapsed);
                }
                runtime->activity_timers().stop(user_message_id);
                if (auto* assistant =
                        live_timeline->current(*runtime->messages())) {
                    assistant->timestamp = current_time_str();
                    if (!turn_elapsed.empty()) {
                        assistant->activity_elapsed = turn_elapsed;
                    }
                }
                live_timeline->finish(
                    *runtime->messages(),
                    {},
                    true);
                append_ui_message(
                    *runtime->messages(),
                    make_warning_message(std::format(
                        "Could not start the agent worker: {}",
                        error.what())));
            }
            if (auto next_turn = runtime->finish_turn(false); next_turn.has_value()) {
                submit_agent_turn(
                    runtime,
                    std::move(next_turn->text),
                    std::move(next_turn->callbacks));
            }
            wake_ui();
        }
    };

    auto submit_skill_turn = [&](const std::string& text,
                                 const std::string& model_hint,
                                 const std::vector<std::string>& allowed_tools) {
        const auto resolution = core::commands::detail::resolve_skill_turn(
            model_hint,
            allowed_tools,
            active_provider_name);
        if (!resolution.warning.empty()) {
            append_history(std::format("\n⚠  {}\n", resolution.warning));
        }
        submit_or_queue_agent_turn(text, std::move(resolution.callbacks));
    };

    auto submit_direct_shell_command = [&](std::string command) {
        if (command.empty()) {
            return;
        }
        if (agent->turn_in_progress()) {
            append_history(
                "\nℹ  Stop the active agent turn before starting a direct shell command.\n");
            return;
        }

        std::string message_id;
        std::string direct_shell_session_id;
        auto runtime = current_runtime;
        {
            std::lock_guard lock(ui_mutex);
            append_ui_message(
                *runtime->messages(),
                make_shell_command_message(command, current_time_str(), true));
            message_id = runtime->messages()->back().id;
            direct_shell_session_id = session_id;
        }
        direct_shell_animation_count.fetch_add(1, std::memory_order_release);
        reset_history_view();
        animation_cv.notify_one();
        wake_ui();

        const std::string working_dir = std::filesystem::current_path().string();
        direct_shell_state->begin_worker();
        try {
          std::thread([command = std::move(command),
                     working_dir,
                     direct_shell_session_id = std::move(direct_shell_session_id),
                     message_id = std::move(message_id),
                     runtime,
                     direct_shell_state,
                     &direct_shell_animation_count,
                     &save_runtime_snapshot,
                     &update_ui_message]() mutable {
            struct WorkerCompletion {
                std::shared_ptr<DirectShellState> state;
                ~WorkerCompletion() { state->finish_worker(); }
            } completion{direct_shell_state};
            core::tools::shell::IShellExecutor::Result result;
            if (direct_shell_state && direct_shell_state->executor) {
                result = direct_shell_state->run(
                    command,
                    working_dir,
                    std::move(direct_shell_session_id));
            } else {
                result.output = "Direct shell executor is unavailable.\n";
                result.exit_code = -1;
            }

            if (result.exit_code != 0) {
                if (!result.output.empty() && result.output.back() != '\n') {
                    result.output.push_back('\n');
                }
                result.output += std::format("Command exited with status {}.", result.exit_code);
            }

            const std::string shell_history_content =
                make_direct_shell_history_content(command, result.output);

            direct_shell_animation_count.fetch_sub(1, std::memory_order_acq_rel);
            update_ui_message(runtime, message_id, [&](UiMessage& message) {
                if (message.type != MessageType::ShellCommand) {
                    return;
                }
                message.secondary_text = std::move(result.output);
                message.pending = false;
                message.finalized = true;
                message.stopped = result.exit_code != 0;
            });

            if (auto agent = runtime->agent()) {
                agent->append_history_message(core::llm::Message{
                    .role = "user",
                    .content = shell_history_content,
                    .synthetic = true,
                });
                save_runtime_snapshot(runtime);
            }
          }).detach();
        } catch (const std::exception& error) {
            direct_shell_state->finish_worker();
            direct_shell_animation_count.fetch_sub(1, std::memory_order_acq_rel);
            update_ui_message(runtime, message_id, [&](UiMessage& message) {
                message.secondary_text = error.what();
                message.pending = false;
                message.finalized = true;
                message.stopped = true;
            });
        }
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
                || question_dialog.active()
                || model_picker_state.active
                || model_provider_picker_state.active
                || provider_model_picker_state.active
                || command_option_picker_state.active
                || provider_picker_state.active
                || review_picker_state.active
                || local_model_picker_state.active
                || rewind_picker_state.active
                || code_block_runner.active()
                || conversation_search_state.active
                || settings_panel_state.active
                || remote_activity_panel_state.active
                || prompts_picker_state.active
                || file_picker_state.active
                || session_picker_state.active) return;
        }
        if (input_text.empty()) return;
        std::string text = input_text;
        input_text.clear();
        prompt_history.save(text);

        // /remote takes no arguments; tolerate trailing whitespace/arguments
        // so it behaves like every other picker-advertised command.
        const std::string_view trimmed_command = trim_ascii(text);
        if (opts.remote_mcp_server_enabled
            && (trimmed_command == "/remote"
                || trimmed_command.starts_with("/remote "))) {
            {
                std::lock_guard lock(ui_mutex);
                remote_activity_panel_state.active = true;
                remote_activity_panel_state.selected = 0;
            }
            wake_ui();
            return;
        }

        core::commands::CommandContext ctx{
            .text             = text,
            .clear_input_fn   = []() {},
            .append_history_fn = append_history,
            .append_assistant_output_fn = append_assistant_output,
            .agent            = agent,
            .session_stats_registry = session_stats_registry,
            .clear_screen_fn  = clear_screen,
            .quit_fn          = screen.ExitLoopClosure(),
            .model_status_fn  = describe_models,
            .switch_model_fn  = switch_provider,
            .refresh_providers_fn = apply_active_profile_live,
            .profile_status_fn = profile_status,
            .switch_profile_fn = switch_profile,
            .effort_status_fn = describe_effort,
            .switch_effort_fn = switch_effort,
            .compression_status_fn = describe_compression,
            .switch_compression_fn = switch_compression,
            .open_model_picker_fn = open_model_picker,
            .open_command_option_picker_fn = open_command_option_picker,
            .open_directory_picker_fn = open_workspace_directory_picker,
            .open_settings_picker_fn = open_settings_picker,
            .open_threads_picker_fn = open_threads_picker,
            .open_sessions_picker_fn = open_sessions_picker,
            .start_new_thread_fn = [&]() {
                if (const auto err = start_new_thread(); err.has_value()) {
                    append_history(std::format("\n✗  {}\n", *err));
                }
            },
            .open_prompts_picker_fn = open_prompts_picker,
            .resume_session_fn = [&](std::string_view id_or_idx) {
                // Prefer an already-loaded runtime: its in-memory state is
                // authoritative, and a brand-new thread may not exist on
                // disk yet. Only fall back to the store for cold sessions.
                std::optional<core::session::SessionData> data_opt;
                if (!id_or_idx.empty()) {
                    if (auto runtime = thread_runtimes.find(id_or_idx)) {
                        data_opt = live_session_data(runtime);
                    }
                }
                if (!data_opt.has_value()) {
                    data_opt = id_or_idx.empty()
                        ? session_store->load_most_recent()
                        : session_store->load(id_or_idx);
                }

                if (!data_opt.has_value()) {
                    append_history(std::format(
                        "\n\xe2\x9c\x97  Session '{}' not found. Use /sessions to list available sessions.\n",
                        id_or_idx.empty() ? std::string("most recent") : std::string(id_or_idx)));
                    return;
                }
                if (const auto resume_error = resume_session(*data_opt);
                    resume_error.has_value()) {
                    append_history(std::format("\n✗  {}\n", *resume_error));
                }
            },
            .rename_session_fn = [&](std::string_view new_name)
                -> core::commands::CommandOperationResult {
                std::string previous;
                {
                    std::lock_guard lock(ui_mutex);
                    previous = session_name;
                    session_name = std::string(new_name);
                }
                save_session_snapshot();
                if (new_name.empty()) {
                    return {.ok = true, .message = previous.empty()
                        ? std::string("Session name cleared.")
                        : std::format("Session name '{}' cleared.", previous)};
                }
                return {.ok = true, .message = std::format(
                    "Session renamed to '{}'. Resume it later with /resume {} or filo -r {}.",
                    new_name, new_name, new_name)};
            },
            .session_name_fn = [&]() {
                std::lock_guard lock(ui_mutex);
                return session_name;
            },
            .open_provider_picker_fn = [&](std::vector<std::string> providers,
                                           std::function<void(std::optional<std::string>)> on_select) {
                std::lock_guard lock(ui_mutex);
                provider_picker_state.active   = true;
                provider_picker_state.selected  = 0;
                provider_picker_state.providers = std::move(providers);
                provider_picker_state.on_select = std::move(on_select);
                wake_ui();
            },
            .open_review_picker_fn = open_review_picker,
            .dispatch_async_fn = [](std::function<void()> task) {
                std::thread(std::move(task)).detach();
            },
            .settings_status_fn = settings_status,
            .yolo_mode_enabled_fn = is_yolo_mode_enabled,
            .set_yolo_mode_enabled_fn = set_yolo_mode_enabled,
            .tool_rules = {
                .list = list_tool_rules,
                .add = add_tool_rule,
                .remove = remove_tool_rule,
                .clear = clear_tool_rules,
            },
            .fork_session_fn = fork_session,
            .open_rewind_picker_fn = open_rewind_menu,
            .suspend_tui_fn   = [&](std::function<void()> task) {
                // Suspends the TUI loop and restores standard I/O for the duration of the task.
                auto closure = screen.WithRestoredIO(task);
                closure();
            },
            .latest_assistant_output_fn = latest_completed_assistant_output,
            .history_store_fn = history_store,
            .clear_history_fn = [&]() {
                prompt_history.clear();
            },
            // Skill commands (SkillCommand) use this to inject an expanded
            // prompt as a full agent turn with TUI message cards and tool
            // activity panels, identical to normal user input.
            .send_user_message_fn = [&](const std::string& message) {
                submit_or_queue_agent_turn(message, {});
            },
            .set_review_activity_fn = set_review_activity,
            .send_user_skill_message_fn = submit_skill_turn,
            .list_todos_fn = list_todos,
            .add_todo_fn = add_todo,
            .set_todo_completed_fn = set_todo_completed,
            .remove_todo_fn = remove_todo,
            .clear_completed_todos_fn = clear_completed_todos,
            .current_goal_fn = current_goal,
            .set_goal_fn = set_goal,
            .set_goal_status_fn = set_goal_status,
            .clear_goal_fn = clear_goal,
            .goal_engine_fn = goal_engine_accessor,
            .save_goal_graph_fn = save_session_snapshot,
            .memory_state_fn = memory_state,
            .set_memory_settings_fn = set_memory_settings,
            .memory_thread_policy_fn = memory_thread_policy,
            .set_memory_thread_policy_fn = set_memory_thread_policy,
            .run_memory_review_fn = run_memory_review,
            .add_memory_fn = add_memory,
            .forget_memory_fn = forget_memory,
            .clean_memory_fn = clean_memory,
            .clear_memory_fn = clear_memory,
            .save_memory_markdown_fn = save_memory_markdown,
            .load_memory_markdown_fn = load_memory_markdown,
            .list_mcp_servers_fn = list_mcp_servers,
            .add_mcp_server_fn = add_mcp_server,
            .remove_mcp_server_fn = remove_mcp_server,
            .login_mcp_server_fn = login_mcp_server,
            .logout_mcp_server_fn = logout_mcp_server,
            .list_active_terminals_fn = list_active_terminals,
            .stop_active_terminal_fn = stop_active_terminal,
            .direct_shell_command_fn = submit_direct_shell_command,
            .open_code_block_runner_fn = open_code_blocks,
            .change_workspace_root_fn = change_workspace_root,
            .steering_policy_fn = [&]() {
                return agent ? agent->session_context_snapshot().steering_policy : core::context::SteeringPolicy{};
            },
            .set_steering_policy_fn = [&](core::context::SteeringPolicy pol) -> core::commands::CommandOperationResult {
                if (!agent) {
                    return {.ok = false, .message = "No active agent session."};
                }
                agent->update_session_context([&](core::context::SessionContext& sctx) {
                    sctx.steering_policy = pol;
                });
                steering_context = core::context::load_project_steering_context(
                    agent_session_context.workspace_view().primary(), pol);
                context_sources_label = join_context_source_labels(steering_context.source_labels);
                return {.ok = true, .message = std::format("Steering policy set to {}.", pol.format())};
            },
            .open_steering_picker_fn = open_steering_picker,
        };

        if (cmd_executor.try_execute(text, ctx)) return;

        submit_or_queue_agent_turn(std::move(text), {});
    };

    // Ctrl+G / Ctrl+X — hand the draft to the configured prompt editor.
    auto apply_editor_outcome = [&](const editor::EditorOutcome& outcome) {
        if (outcome.text.has_value()) {
            input_text = normalize_newlines(*outcome.text);
            input_cursor_position = static_cast<int>(input_text.size());
        }
        if (outcome.notice.has_value()) {
            append_history(std::format(
                "\n{}  {}\n",
                outcome.notice->success ? "✓" : "✗",
                outcome.notice->message));
        }
        wake_ui();
    };

    auto open_external_editor = [&]() -> bool {
        {
            std::lock_guard lock(ui_mutex);
            if (perm_state.active
                || question_dialog.active()
                || model_picker_state.active
                || model_provider_picker_state.active
                || provider_model_picker_state.active
                || command_option_picker_state.active
                || provider_picker_state.active
                || review_picker_state.active
                || local_model_picker_state.active
                || rewind_picker_state.active
                || code_block_runner.active()
                || conversation_search_state.active
                || file_picker_state.active
                || settings_panel_state.active) {
                return true;
            }
        }

        // Terminal backends answer inline; detached ones report through
        // take_outcome() once the user is done in the other application.
        if (const auto outcome = external_editor.open(config.prompt_editor, input_text)) {
            apply_editor_outcome(*outcome);
        } else {
            wake_ui();  // Draw the overlay while the detached session runs.
        }
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

    auto quit_confirm_active = [&]() -> bool {
        if (quit_confirm_key.empty()) {
            return false;
        }
        if (std::chrono::steady_clock::now() > quit_confirm_deadline) {
            quit_confirm_key.clear();
            quit_confirm_deadline = std::chrono::steady_clock::time_point::min();
            return false;
        }
        return true;
    };

    auto reset_quit_confirm = [&]() {
        quit_confirm_key.clear();
        quit_confirm_deadline = std::chrono::steady_clock::time_point::min();
    };

    // Second consecutive press of the same key within the confirmation window
    // confirms the quit; any first (or mismatched) press arms it and shows the
    // "Press <key> again to quit" footer hint instead.
    auto confirm_quit_or_arm = [&](std::string_view key_label) -> bool {
        if (quit_confirm_active() && quit_confirm_key == key_label) {
            return true;
        }
        quit_confirm_key = key_label;
        quit_confirm_deadline = std::chrono::steady_clock::now() + kExitConfirmWindow;
        wake_ui();
        return false;
    };

    auto input_component = PromptInput(&input_text, "Ask anything", input_option);
    auto input_stack = Container::Stacked({
        input_component,
        question_dialog.editor_component(),
    });
    std::size_t history_snapshot_revision = 0;
    auto history_snapshot = std::make_shared<const std::vector<UiMessage>>();
    auto history_component = Make<HistoryComponent>(
        std::function<HistoryComponent::MessageSnapshot()>{[&]() {
            const auto revision = ui_state_revision.load(std::memory_order_acquire);
            if (history_snapshot_revision != revision) {
                std::lock_guard lock(ui_mutex);
                history_snapshot =
                    std::make_shared<const std::vector<UiMessage>>(*selected_messages);
                history_snapshot_revision = revision;
            }
            return history_snapshot;
        }},
        std::cref(animation_tick),
        [&]() {
            return ConversationRenderOptions{
                .show_timestamps = ui_show_timestamps,
                .show_spinner = ui_show_spinner.load(std::memory_order_relaxed),
                .expand_system_details = tool_output_expanded,
                .expand_tool_results = tool_output_expanded,
                .show_reasoning = ui_show_reasoning,
                .tool_result_preview_max_lines = kToolResultPreviewMaxLines,
                // scroll_pos set by component
                .activity_elapsed = [&current_runtime](std::string_view message_id) {
                    const auto elapsed =
                        current_runtime->activity_timers().elapsed(message_id);
                    return elapsed.has_value() ? format_elapsed_compact(*elapsed) : std::string{};
                },
            };
        }
    );
    reset_history_view = [history_component]() {
        history_component->ResetToBottom();
    };

    // ── Event handling ───────────────────────────────────────────────────────
    auto component = CatchEvent(input_stack, [&](Event event) {
        reset_double_escape_on_non_escape(double_escape_state, event);

        // A detached editor session owns the prompt: apply its result as soon
        // as it lands, and swallow every key except the cancel gesture.
        if (auto outcome = external_editor.take_outcome()) {
            apply_editor_outcome(*outcome);
            return true;
        }
        if (external_editor.busy()) {
            if (event == Event::Escape || is_ctrl_c_event(event)) {
                external_editor.cancel();
            }
            return true;
        }

        if (event.is_mouse()
            && event.mouse().button == Mouse::Left
            && event.mouse().motion == Mouse::Pressed) {
            const std::size_t target_count = std::min(
                thread_tab_hitboxes.size(),
                thread_tab_session_ids.size());
            for (std::size_t i = 0; i < target_count; ++i) {
                if (!thread_tab_hitboxes[i].Contain(event.mouse().x, event.mouse().y)) {
                    continue;
                }
                const std::string target_session_id = thread_tab_session_ids[i];
                if (target_session_id == session_id) {
                    return true;
                }
                if (auto runtime = thread_runtimes.find(target_session_id)) {
                    if (auto data = live_session_data(runtime)) {
                        {
                            std::lock_guard lock(ui_mutex);
                            session_picker_state = {};
                        }
                        if (const auto error = resume_session(*data); error.has_value()) {
                            append_history(std::format("\n✗  {}\n", *error));
                        }
                    }
                }
                return true;
            }
        }

        if (event.is_mouse()
            && event.mouse().button == Mouse::Left
            && event.mouse().motion == Mouse::Pressed
            && usage_status_box.Contain(event.mouse().x, event.mouse().y)) {
            {
                std::lock_guard lock(ui_mutex);
                usage_details_panel_active = !usage_details_panel_active;
            }
            wake_ui();
            return true;
        }

        bool usage_panel_was_active = false;
        {
            std::lock_guard lock(ui_mutex);
            if (usage_details_panel_active) {
                if (event == Event::Escape
                    || event == Event::Character('q')
                    || event == Event::Character('Q')) {
                    usage_details_panel_active = false;
                    usage_panel_was_active = true;
                }
            }
        }
        if (usage_panel_was_active) {
            wake_ui();
            return true;
        }

        if (event.is_mouse()
            && event.mouse().button == Mouse::Left
            && event.mouse().motion == Mouse::Pressed
            && agents_status_box.Contain(event.mouse().x, event.mouse().y)) {
            {
                std::lock_guard lock(ui_mutex);
                agents_visualizer_panel_active = !agents_visualizer_panel_active;
                if (agents_visualizer_panel_active) {
                    const auto pol = agent ? agent->session_context_snapshot().steering_policy : agent_session_context.steering_policy;
                    steering_context = core::context::load_project_steering_context(
                        agent_session_context.workspace_view().primary(),
                        pol);
                    context_sources_label =
                        join_context_source_labels(steering_context.source_labels);
                    agents_visualizer_scroll_offset = 0;
                }
            }
            wake_ui();
            return true;
        }

        if (event.is_mouse()
            && event.mouse().button == Mouse::Left
            && event.mouse().motion == Mouse::Pressed
            && agents_visualizer_panel_active) {
            for (std::size_t i = 0; i < agents_tab_hitboxes.size(); ++i) {
                if (agents_tab_hitboxes[i].Contain(event.mouse().x, event.mouse().y)) {
                    {
                        std::lock_guard lock(ui_mutex);
                        agents_visualizer_selected_file = i;
                        agents_visualizer_scroll_offset = 0;
                    }
                    wake_ui();
                    return true;
                }
            }
        }

        if (agents_visualizer_panel_active && event.is_mouse()) {
            if (event.mouse().button == Mouse::WheelUp) {
                {
                    std::lock_guard lock(ui_mutex);
                    agents_visualizer_scroll_offset = std::max(0, agents_visualizer_scroll_offset - 3);
                }
                wake_ui();
                return true;
            }
            if (event.mouse().button == Mouse::WheelDown) {
                {
                    std::lock_guard lock(ui_mutex);
                    agents_visualizer_scroll_offset += 3;
                }
                wake_ui();
                return true;
            }
        }

        bool agents_panel_was_active = false;
        {
            std::lock_guard lock(ui_mutex);
            if (agents_visualizer_panel_active) {
                if (event == Event::Escape
                    || event == Event::Character('q')
                    || event == Event::Character('Q')) {
                    agents_visualizer_panel_active = false;
                    agents_panel_was_active = true;
                } else if (event == Event::ArrowUp || event == Event::Character('k') || event == Event::Character('K')) {
                    agents_visualizer_scroll_offset = std::max(0, agents_visualizer_scroll_offset - 1);
                    agents_panel_was_active = true;
                } else if (event == Event::ArrowDown || event == Event::Character('j') || event == Event::Character('J')) {
                    agents_visualizer_scroll_offset += 1;
                    agents_panel_was_active = true;
                } else if (event == Event::PageUp) {
                    agents_visualizer_scroll_offset = std::max(0, agents_visualizer_scroll_offset - 10);
                    agents_panel_was_active = true;
                } else if (event == Event::PageDown) {
                    agents_visualizer_scroll_offset += 10;
                    agents_panel_was_active = true;
                } else if (event == Event::Home) {
                    agents_visualizer_scroll_offset = 0;
                    agents_panel_was_active = true;
                } else if (event == Event::ArrowLeft || event == Event::TabReverse) {
                    if (!steering_context.files.empty()) {
                        agents_visualizer_selected_file = (agents_visualizer_selected_file + steering_context.files.size() - 1) % steering_context.files.size();
                        agents_visualizer_scroll_offset = 0;
                    }
                    agents_panel_was_active = true;
                } else if (event == Event::ArrowRight || event == Event::Tab) {
                    if (!steering_context.files.empty()) {
                        agents_visualizer_selected_file = (agents_visualizer_selected_file + 1) % steering_context.files.size();
                        agents_visualizer_scroll_offset = 0;
                    }
                    agents_panel_was_active = true;
                } else if (event == Event::Character('u')
                           || event == Event::Character('U')
                           || event == Event::Character(' ')) {
                    if (!steering_context.files.empty() && agent) {
                        const auto& cur = steering_context.files[agents_visualizer_selected_file];
                        auto pol = agent->session_context_snapshot().steering_policy;
                        if (cur.enabled) {
                            pol.disable_source(cur.label);
                        } else {
                            pol.enable_source(cur.label);
                        }
                        agent->update_session_context([&](core::context::SessionContext& sctx) {
                            sctx.steering_policy = pol;
                        });
                        steering_context = core::context::load_project_steering_context(
                            agent_session_context.workspace_view().primary(), pol);
                        context_sources_label =
                            join_context_source_labels(steering_context.source_labels);
                    }
                    agents_panel_was_active = true;
                }
            }
        }
        if (agents_panel_was_active) {
            wake_ui();
            return true;
        }

        if (opts.remote_mcp_server_enabled
            && event.is_mouse()
            && event.mouse().button == Mouse::Left
            && event.mouse().motion == Mouse::Pressed
            && remote_activity_pill_box.Contain(event.mouse().x, event.mouse().y)) {
            bool closed = false;
            {
                std::lock_guard lock(ui_mutex);
                closed = remote_activity_panel_state.active;
                remote_activity_panel_state.active = !closed;
                if (!closed) {
                    remote_activity_panel_state.selected = 0;
                }
            }
            if (closed) {
                remote_activity_hub.acknowledge_errors();
            }
            wake_ui();
            return true;
        }

        bool remote_panel_was_active = false;
        bool remote_panel_closed = false;
        {
            std::lock_guard lock(ui_mutex);
            if (remote_activity_panel_state.active) {
                remote_panel_was_active = true;
                // Metadata-only: the row count is all this needs, and payloads
                // can run to tens of kilobytes per entry.
                const auto activity_count =
                    remote_activity_hub.snapshot(false).activities.size();
                if (event == Event::Escape) {
                    remote_activity_panel_state.active = false;
                    remote_panel_closed = true;
                } else if (event == Event::ArrowUp && activity_count > 0) {
                    remote_activity_panel_state.selected =
                        remote_activity_panel_state.selected == 0
                            ? activity_count - 1
                            : remote_activity_panel_state.selected - 1;
                } else if (event == Event::ArrowDown && activity_count > 0) {
                    remote_activity_panel_state.selected =
                        (remote_activity_panel_state.selected + 1) % activity_count;
                } else if (event == Event::Character('c')
                           || event == Event::Character('C')) {
                    remote_activity_hub.clear_completed();
                    remote_activity_panel_state.selected = 0;
                }
            }
        }
        if (remote_panel_closed) {
            // Acknowledged on close rather than on open: the badge exists to
            // pull the user into the panel, so clearing it on entry would hide
            // failures that arrive while the panel is on screen. Called outside
            // ui_mutex because it notifies, which re-enters the UI wake path.
            remote_activity_hub.acknowledge_errors();
        }
        if (remote_panel_was_active) {
            wake_ui();
            return true;
        }

        if (event == Event::Escape
            || event == Event::Character('c')
            || event == Event::Character('C')
            || event == Event::Character('x')
            || event == Event::Character('X')) {
            std::lock_guard lock(ui_mutex);
            if (stderr_panel_state.active) {
                stderr_panel_state.active = false;
                return true;
            }
        }

        const auto code_block_outcome = code_block_runner.handle(
            event,
            is_ctrl_c_event(event) || is_ctrl_r_event(event));
        if (code_block_outcome.handled) {
            if (code_block_outcome.attachment.has_value()) {
                insert_token_with_spacing(
                    input_text,
                    input_cursor_position,
                    std::format("@\"{}\"", code_block_outcome.attachment->string()));
            }
            if (code_block_outcome.notice.has_value()) {
                append_history(std::format(
                    "\n{}  {}\n",
                    code_block_outcome.notice->success ? "✓" : "✗",
                    code_block_outcome.notice->message));
            }
            wake_ui();
            return true;
        }

        RewindPickerEventResult rewind_result;
        {
            std::lock_guard lock(ui_mutex);
            rewind_result = handle_rewind_picker_event(
                rewind_picker_state,
                event,
                is_ctrl_c_event(event));
        }
        if (rewind_result.handled) {
            if (rewind_result.selection.has_value()) {
                switch (rewind_result.selection->action) {
                case RewindPickerOption::Action::RewindToMessage:
                    rewind_to_message(*rewind_result.selection);
                    break;
                case RewindPickerOption::Action::SummarizeAndCompact:
                    compact_history_from_rewind(
                        *agent,
                        current_runtime->turn_active(),
                        append_history,
                        save_session_snapshot);
                    break;
                case RewindPickerOption::Action::Cancel:
                    break;
                }
            }
            wake_ui();
            return true;
        }

        // ── OAuth reauthentication dialog ───────────────────────────────
        // Provider callbacks arrive on worker threads. The prompt state is
        // populated there, but browser/device login must run here on the TUI
        // event thread so WithRestoredIO can safely hand terminal ownership
        // back to the interactive authentication flow.
        std::optional<PendingAuthenticationRecovery> authentication_recovery;
        bool authentication_recovery_was_active = false;
        bool authentication_recovery_accepted = false;
        {
            std::lock_guard lock(ui_mutex);
            if (authentication_recovery_state.active.has_value()) {
                authentication_recovery_was_active = true;
                if (event == Event::ArrowUp || event == Event::ArrowDown) {
                    authentication_recovery_state.selected =
                        authentication_recovery_state.selected == 0 ? 1 : 0;
                } else if (event == Event::Character('1')
                           || event == Event::Character('y')
                           || event == Event::Character('Y')) {
                    authentication_recovery_accepted = true;
                    authentication_recovery =
                        std::move(authentication_recovery_state.active);
                    authentication_recovery_state.active.reset();
                } else if (event == Event::Return) {
                    authentication_recovery_accepted =
                        authentication_recovery_state.selected == 0;
                    authentication_recovery =
                        std::move(authentication_recovery_state.active);
                    authentication_recovery_state.active.reset();
                } else if (event == Event::Character('2')
                           || event == Event::Character('n')
                           || event == Event::Character('N')
                           || event == Event::Escape
                           || is_ctrl_c_event(event)) {
                    authentication_recovery =
                        std::move(authentication_recovery_state.active);
                    authentication_recovery_state.active.reset();
                }
            }
        }
        if (authentication_recovery_was_active) {
            if (!authentication_recovery.has_value()) {
                wake_ui();
                return true;
            }

            const auto& provider = authentication_recovery->provider;
            bool login_succeeded = false;
            std::string login_error;

            if (authentication_recovery_accepted) {
                auto authenticate = [&]() {
                    try {
                        authentication_manager.login(provider.login_provider);
                        login_succeeded = true;
                    } catch (const std::exception& error) {
                        login_error = error.what();
                    } catch (...) {
                        login_error = "Unknown authentication error.";
                    }
                };
                screen.WithRestoredIO(authenticate)();
            }

            if (!authentication_recovery_accepted) {
                append_history(std::format(
                    "\nℹ  {} remains signed out. Reconnect any time with `/auth {}`.\n",
                    provider.display_name,
                    provider.login_provider));
            } else if (!login_succeeded) {
                append_history(std::format(
                    "\n✗  Could not reconnect {}: {}\n"
                    "ℹ  Try again with `/auth {}`.\n",
                    provider.display_name,
                    login_error.empty() ? std::string("Unknown error.") : login_error,
                    provider.login_provider));
            } else if (authentication_recovery->request.retry_safe
                       && !authentication_recovery->retry_text.empty()) {
                append_history(std::format(
                    "\n✓  {} reconnected. Retrying your request…\n",
                    provider.display_name));
                // Remove the failed user/error pair from model history before
                // replaying. The transcript keeps the visible diagnostic, but
                // the provider receives one clean copy of the user request.
                auto retry_runtime = authentication_recovery->runtime;
                retry_runtime->agent()->undo_last();
                if (retry_runtime->begin_turn()) {
                    submit_agent_turn(
                        retry_runtime,
                        std::move(authentication_recovery->retry_text),
                        std::move(authentication_recovery->retry_callbacks));
                }
            } else {
                append_history(std::format(
                    "\n✓  {} reconnected. The interrupted request was not replayed "
                    "because output may already have started; use `/retry` when ready.\n",
                    provider.display_name));
            }

            {
                std::lock_guard lock(ui_mutex);
                authentication_recovery_state.providers.erase(
                    provider.credential_id);
                if (!authentication_recovery_state.active.has_value()
                    && !authentication_recovery_state.queued.empty()) {
                    authentication_recovery_state.active =
                        std::move(authentication_recovery_state.queued.front());
                    authentication_recovery_state.queued.pop_front();
                    authentication_recovery_state.selected = 0;
                }
            }
            input_component->TakeFocus();
            wake_ui();
            return true;
        }

        // ── Permission overlay keypresses ────────────────────────────────
        // Resolve the promise OUTSIDE the lock to avoid locking issues when
        // the worker thread wakes up and might try to re-acquire ui_mutex.
        std::shared_ptr<std::promise<bool>> perm_prom;
        std::optional<bool> perm_answer;
        bool enable_yolo_from_permission = false;
        bool enable_always_allow = false;
        std::string always_allow_rule;
        std::string always_allow_label;
        ThreadRuntime::Ptr permission_runtime;
        bool perm_was_active = false;
        {
            std::lock_guard lock(ui_mutex);
            if (perm_state.active) {
                perm_was_active = true;
                permission_runtime = perm_state.origin_runtime;
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
                    always_allow_rule  = perm_state.remember_rule;
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
                    always_allow_rule  = perm_state.remember_rule;
                    always_allow_label = perm_state.allow_label;
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = sel != 3;
                    enable_always_allow         = sel == 1;
                    enable_yolo_from_permission = sel == 2;
                    perm_state.active = false;
                } else if (is_ctrl_c_event(event)) {
                    perm_prom   = std::move(perm_state.promise);
                    perm_answer = false;
                    perm_state.active = false;
                    // Stop the thread that asked, which may be a hidden one;
                    // fall back to the visible agent if attribution is lost.
                    if (perm_state.origin_runtime) {
                        perm_state.origin_runtime->request_stop();
                    } else {
                        agent->request_stop();
                    }
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
            // permission check while we're still updating its runtime state.
            if (perm_prom && perm_answer.has_value()) {
                perm_prom->set_value(*perm_answer);
            }
            if (enable_always_allow && !always_allow_rule.empty()) {
                if (!permission_runtime) {
                    permission_runtime = current_runtime;
                }
                permission_runtime->mutate_metadata(
                    [&](ThreadRuntimeMetadata& metadata) {
                        metadata.permission_rules.insert(always_allow_rule);
                    });
                // Session allow-list updated - no status message needed
                (void)always_allow_label;
            }
            if (enable_yolo_from_permission) {
                if (!permission_runtime) {
                    permission_runtime = current_runtime;
                }
                permission_runtime->mutate_metadata(
                    [](ThreadRuntimeMetadata& metadata) {
                        metadata.yolo_enabled = true;
                    });
                append_history(
                    "\n\xe2\x9a\xa0  Approval mode set to YOLO: sensitive tools will auto-run.\n");
            }
            if (perm_prom && perm_answer.has_value() && !*perm_answer) {
                append_history(
                    "\n\xe2\x84\xb9  Tool call rejected. Share a suggestion and Filo will adapt.\n");
            }
            return true;
        }

        // ── Question dialog input handling ───────────────────────────────
        auto question_result = question_dialog.handle_event(
            event,
            is_ctrl_c_event(event));
        if (question_result.handled) {
            if (question_result.restore_main_input_focus) {
                input_component->TakeFocus();
            }
            if (question_result.stop_agent) {
                if (auto origin = thread_runtimes.find(
                        question_result.origin_session_id)) {
                    origin->request_stop();
                } else {
                    agent->request_stop();
                }
            }
            const bool resolved_question = question_result.has_resolution();
            question_result.resolve();
            if (resolved_question) {
                question_prompt_cv.notify_one();
            }
            return true;
        }

        bool review_picker_was_active = false;
        bool review_picker_cancelled = false;
        std::optional<std::string> review_picker_request;
        std::function<void(std::optional<std::string>)> review_picker_on_select;
        {
            std::lock_guard lock(ui_mutex);
            if (review_picker_state.active) {
                review_picker_was_active = true;

                if (review_picker_state.mode == ReviewPickerMode::SelectTarget) {
                    if (event == Event::ArrowUp) {
                        review_picker_state.selected = (review_picker_state.selected + 2) % 3;
                    } else if (event == Event::ArrowDown) {
                        review_picker_state.selected = (review_picker_state.selected + 1) % 3;
                    } else if (event == Event::Character('1')) {
                        review_picker_request = std::string{};
                        review_picker_on_select = std::move(review_picker_state.on_select);
                        review_picker_state.active = false;
                    } else if (event == Event::Character('2')) {
                        review_picker_state.selected = 1;
                        review_picker_state.mode = ReviewPickerMode::EnterBaseBranch;
                        review_picker_state.input_text.clear();
                    } else if (event == Event::Character('3')) {
                        review_picker_state.selected = 2;
                        review_picker_state.mode = ReviewPickerMode::EnterCustomPrompt;
                        review_picker_state.input_text.clear();
                    } else if (event == Event::Return) {
                        if (review_picker_state.selected == 0) {
                            review_picker_request = std::string{};
                            review_picker_on_select = std::move(review_picker_state.on_select);
                            review_picker_state.active = false;
                        } else if (review_picker_state.selected == 1) {
                            review_picker_state.mode = ReviewPickerMode::EnterBaseBranch;
                            review_picker_state.input_text.clear();
                        } else {
                            review_picker_state.mode = ReviewPickerMode::EnterCustomPrompt;
                            review_picker_state.input_text.clear();
                        }
                    } else if (event == Event::Escape) {
                        review_picker_cancelled = true;
                        review_picker_on_select = std::move(review_picker_state.on_select);
                        review_picker_state.active = false;
                    }
                } else if (event == Event::Escape) {
                    review_picker_state.mode = ReviewPickerMode::SelectTarget;
                    review_picker_state.input_text.clear();
                    review_picker_state.selected_base_ref = 0;
                    refresh_review_base_refs_locked();
                } else if (review_picker_state.mode == ReviewPickerMode::EnterBaseBranch
                           && event == Event::ArrowUp
                           && !review_picker_state.filtered_base_refs.empty()) {
                    review_picker_state.selected_base_ref =
                        (review_picker_state.selected_base_ref
                         + static_cast<int>(review_picker_state.filtered_base_refs.size()) - 1)
                        % static_cast<int>(review_picker_state.filtered_base_refs.size());
                } else if (review_picker_state.mode == ReviewPickerMode::EnterBaseBranch
                           && event == Event::ArrowDown
                           && !review_picker_state.filtered_base_refs.empty()) {
                    review_picker_state.selected_base_ref =
                        (review_picker_state.selected_base_ref + 1)
                        % static_cast<int>(review_picker_state.filtered_base_refs.size());
                } else if (event == Event::Backspace || event == Event::Delete) {
                    erase_last_utf8_codepoint(review_picker_state.input_text);
                    if (review_picker_state.mode == ReviewPickerMode::EnterBaseBranch) {
                        refresh_review_base_refs_locked();
                    }
                } else if (event == Event::Return) {
                    const std::string_view trimmed =
                        trim_ascii(review_picker_state.input_text);
                    if (review_picker_state.mode == ReviewPickerMode::EnterBaseBranch) {
                        if (!review_picker_state.filtered_base_refs.empty()) {
                            const int selected_ref = std::clamp(
                                review_picker_state.selected_base_ref,
                                0,
                                static_cast<int>(review_picker_state.filtered_base_refs.size()) - 1);
                            review_picker_request = std::format(
                                "--base {}",
                                review_picker_state.filtered_base_refs[static_cast<std::size_t>(selected_ref)].name);
                            review_picker_on_select = std::move(review_picker_state.on_select);
                            review_picker_state.active = false;
                        } else if (!trimmed.empty()) {
                            review_picker_request = std::format("--base {}", std::string(trimmed));
                            review_picker_on_select = std::move(review_picker_state.on_select);
                            review_picker_state.active = false;
                        }
                    } else if (!trimmed.empty()) {
                        review_picker_request = std::string(trimmed);
                        review_picker_on_select = std::move(review_picker_state.on_select);
                        review_picker_state.active = false;
                    }
                } else if (event.is_character()) {
                    const std::string input = event.character();
                    if (!input.empty()) {
                        const unsigned char first =
                            static_cast<unsigned char>(input.front());
                        const bool is_control = input.size() == 1 && std::iscntrl(first);
                        if (!is_control) {
                            review_picker_state.input_text += input;
                            if (review_picker_state.mode == ReviewPickerMode::EnterBaseBranch) {
                                review_picker_state.selected_base_ref = 0;
                                refresh_review_base_refs_locked();
                            }
                        }
                    }
                }
            }
        }
        if (review_picker_was_active) {
            if (review_picker_on_select) {
                if (review_picker_request.has_value()) {
                    review_picker_on_select(*review_picker_request);
                } else if (review_picker_cancelled) {
                    review_picker_on_select(std::nullopt);
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

        bool command_option_picker_was_active = false;
        std::optional<std::string> command_option_choice;
        std::function<std::string(std::string_view)> command_option_on_select;
        bool command_option_prefill = false;
        {
            std::lock_guard lock(ui_mutex);
            if (command_option_picker_state.active) {
                command_option_picker_was_active = true;
                const int option_count =
                    static_cast<int>(command_option_picker_state.options.size());
                if (event == Event::ArrowUp) {
                    if (option_count > 0) {
                        command_option_picker_state.selected =
                            (command_option_picker_state.selected + option_count - 1)
                            % option_count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (option_count > 0) {
                        command_option_picker_state.selected =
                            (command_option_picker_state.selected + 1) % option_count;
                    }
                } else if (event == Event::Return) {
                    if (option_count > 0) {
                        command_option_choice =
                            command_option_picker_state
                                .options[static_cast<std::size_t>(
                                    command_option_picker_state.selected)]
                                .value;
                        command_option_on_select = command_option_picker_state.on_select;
                        command_option_prefill = command_option_picker_state.prefill_input;
                        command_option_picker_state.active = false;
                    }
                } else if (event == Event::Escape) {
                    command_option_picker_state.active = false;
                } else {
                    for (int n = 1; n <= std::min(option_count, 9); ++n) {
                        if (event == Event::Character(static_cast<char>('0' + n))) {
                            command_option_choice =
                                command_option_picker_state
                                    .options[static_cast<std::size_t>(n - 1)]
                                    .value;
                            command_option_on_select = command_option_picker_state.on_select;
                            command_option_prefill = command_option_picker_state.prefill_input;
                            command_option_picker_state.active = false;
                            break;
                        }
                    }
                }
            }
        }
        if (command_option_picker_was_active) {
            if (command_option_choice.has_value() && command_option_on_select) {
                const std::string result = command_option_on_select(*command_option_choice);
                if (result.empty()) {
                    // The handler took over the interaction (e.g. it opened a
                    // follow-up picker) and will report its own outcome.
                } else if (command_option_prefill) {
                    std::lock_guard lock(ui_mutex);
                    input_text = result;
                    input_cursor_position = static_cast<int>(input_text.size());
                } else {
                    const bool success = result.starts_with("Set")
                        || result.starts_with("Switched")
                        || result.starts_with("Cleared")
                        || result.starts_with("Applied");
                    append_history(std::format(
                        "\n{}\n",
                        success ? "✓  " + result : "✗  " + result));
                }
            }
            return true;
        }

        SessionPickerEventResult session_picker_result;
        SessionPickerResource session_picker_resource =
            SessionPickerResource::SavedSessions;
        {
            std::lock_guard lock(ui_mutex);
            if (session_picker_state.active) {
                session_picker_resource = session_picker_state.resource;
                session_picker_result = handle_session_picker_event(session_picker_state, event);
            }
        }
        if (session_picker_result.handled) {
            if (session_picker_result.action == SessionPickerAction::NewThread) {
                if (const auto err = start_new_thread(); err.has_value()) {
                    append_history(std::format("\n✗  {}\n", *err));
                }
                return true;
            }
            if (session_picker_result.action == SessionPickerAction::CloseThread
                && session_picker_result.filtered_index.has_value()) {
                std::string sid;
                std::string query;
                int selected = 0;
                {
                    std::lock_guard lock(ui_mutex);
                    const auto idx = static_cast<std::size_t>(
                        *session_picker_result.filtered_index);
                    if (idx < session_picker_state.filtered.size()) {
                        sid = session_picker_state.filtered[idx].session_id;
                    }
                    query = session_picker_state.query;
                    selected = session_picker_state.selected;
                }
                if (sid.empty()) {
                    return true;
                }
                if (const auto error = close_thread(sid); error.has_value()) {
                    std::lock_guard lock(ui_mutex);
                    session_picker_state.status_message = *error;
                    return true;
                }

                auto catalogue = active_thread_catalogue();
                std::lock_guard lock(ui_mutex);
                open_thread_picker(session_picker_state, std::move(catalogue), session_id);
                session_picker_state.query = std::move(query);
                refresh_session_picker_filter(session_picker_state);
                session_picker_state.selected = std::min(
                    selected,
                    std::max(0,
                        static_cast<int>(session_picker_state.filtered.size()) - 1));
                session_picker_state.status_message =
                    std::format("Closed active thread {}", sid);
                return true;
            }
            if (session_picker_result.action == SessionPickerAction::Delete
                && session_picker_result.filtered_index.has_value()) {
                std::string error;
                std::string sid;
                std::string current_id;
                {
                    std::lock_guard lock(ui_mutex);
                    const auto idx = static_cast<std::size_t>(*session_picker_result.filtered_index);
                    if (idx < session_picker_state.filtered.size()) {
                        sid = session_picker_state.filtered[idx].session_id;
                    }
                    current_id = session_id;
                }
                if (sid.empty()) {
                    return true;
                }
                if (sid == current_id) {
                    append_history(
                        "\n✗  Cannot delete the session owned by the current active thread.\n");
                    return true;
                }
                if (thread_runtimes.find(sid)) {
                    append_history(
                        "\n✗  Cannot delete a saved session while it belongs to an active thread.\n");
                    return true;
                }
                if (session_store->remove(sid, &error)) {
                    session_stats_registry->reset(sid);
                    core::budget::BudgetTracker::get_instance().reset_session(sid);
                    auto catalogue = session_store->list();
                    std::lock_guard lock(ui_mutex);
                    const auto query = session_picker_state.query;
                    const auto selected = session_picker_state.selected;
                    open_session_picker(session_picker_state, std::move(catalogue), current_id);
                    session_picker_state.query = query;
                    refresh_session_picker_filter(session_picker_state);
                    if (session_picker_state.sessions.empty()) {
                        session_picker_state.active = false;
                    }
                    session_picker_state.selected = std::min(
                        selected,
                        std::max(0, static_cast<int>(session_picker_state.filtered.size()) - 1));
                    session_picker_state.status_message =
                        std::format("Deleted saved session {}", sid);
                } else {
                    append_history(std::format("\n✗  Failed to delete session: {}\n", error));
                }
                return true;
            }
            if (session_picker_result.action == SessionPickerAction::RenameCommit
                && session_picker_result.filtered_index.has_value()) {
                std::string sid;
                {
                    std::lock_guard lock(ui_mutex);
                    const auto idx = static_cast<std::size_t>(*session_picker_result.filtered_index);
                    if (idx < session_picker_state.filtered.size()) {
                        sid = session_picker_state.filtered[idx].session_id;
                    }
                }
                if (!sid.empty()) {
                    if (auto runtime = thread_runtimes.find(sid)) {
                        runtime->mutate_metadata([&](ThreadRuntimeMetadata& metadata) {
                            metadata.thread_name = session_picker_result.rename_name;
                            metadata.auto_thread_name = false;
                        });
                        static_cast<void>(open_threads_picker());
                        return true;
                    }
                    append_history(std::format(
                        "\n✗  Active thread {} is no longer available.\n", sid));
                }
                return true;
            }
            if (session_picker_result.action == SessionPickerAction::Open
                && session_picker_result.filtered_index.has_value()) {
                core::session::SessionInfo info;
                {
                    std::lock_guard lock(ui_mutex);
                    const auto idx = static_cast<std::size_t>(*session_picker_result.filtered_index);
                    if (idx < session_picker_state.filtered.size()) {
                        info = session_picker_state.filtered[idx];
                    }
                }
                if (!info.session_id.empty()) {
                    std::optional<core::session::SessionData> data_opt;
                    if (session_picker_resource
                        == SessionPickerResource::ActiveThreads) {
                        if (auto runtime = thread_runtimes.find(info.session_id)) {
                            data_opt = live_session_data(runtime);
                        }
                    } else {
                        // The catalogue remains persistence-backed. If its
                        // session already belongs to a live thread, however,
                        // that runtime is the authoritative in-memory owner.
                        if (auto runtime = thread_runtimes.find(info.session_id)) {
                            data_opt = live_session_data(runtime);
                        } else {
                            data_opt = session_store->load_by_id(info.session_id);
                        }
                    }
                    if (data_opt) {
                        if (const auto resume_error = resume_session(*data_opt);
                            resume_error.has_value()) {
                            append_history(std::format("\n✗  {}\n", *resume_error));
                        }
                    } else {
                        append_history(
                            session_picker_resource == SessionPickerResource::ActiveThreads
                                ? std::format(
                                      "\n✗  Active thread {} is no longer available.\n",
                                      info.session_id)
                                : std::format(
                                      "\n✗  Failed to load saved session {}.\n",
                                      info.session_id));
                    }
                }
            }
            return true;
        }

        bool prompts_picker_was_active = false;
        std::optional<int> prompts_choice;
        std::optional<int> prompts_delete_idx;
        bool prompts_copy_requested = false;
        {
            std::lock_guard lock(ui_mutex);
            if (prompts_picker_state.active) {
                prompts_picker_was_active = true;
                const int count = static_cast<int>(prompts_picker_state.prompts.size());
                if (event == Event::ArrowUp) {
                    if (count > 0) {
                        prompts_picker_state.selected = (prompts_picker_state.selected + count - 1) % count;
                        prompts_picker_state.status_message.clear();
                    }
                } else if (event == Event::ArrowDown) {
                    if (count > 0) {
                        prompts_picker_state.selected = (prompts_picker_state.selected + 1) % count;
                        prompts_picker_state.status_message.clear();
                    }
                } else if (event == Event::Return) {
                    if (count > 0) {
                        prompts_choice = prompts_picker_state.selected;
                        prompts_picker_state.active = false;
                    }
                } else if (tui::is_ctrl_enter_event(event)) {
                    if (count > 0) {
                        prompts_copy_requested = true;
                    }
                } else if (event == Event::Backspace || event == Event::Delete) {
                    if (count > 0) {
                        prompts_delete_idx = prompts_picker_state.selected;
                    }
                } else if (event == Event::Escape) {
                    prompts_picker_state.active = false;
                }
            }
        }
        if (prompts_picker_was_active) {
            if (prompts_copy_requested) {
                std::string prompt_text;
                {
                    std::lock_guard lock(ui_mutex);
                    prompt_text = prompts_picker_state.prompts[
                        static_cast<size_t>(prompts_picker_state.selected)];
                }
                if (const auto err = core::commands::copy_text_to_clipboard(prompt_text);
                    !err.has_value()) {
                    std::lock_guard lock(ui_mutex);
                    prompts_picker_state.status_message =
                        "\xe2\x9c\x93 Copied to clipboard";
                } else {
                    std::lock_guard lock(ui_mutex);
                    prompts_picker_state.status_message =
                        std::format("\xe2\x9c\x97 Copy failed: {}", *err);
                }
                return true;
            }
            if (prompts_delete_idx.has_value()) {
                std::string error;
                const int picker_idx = *prompts_delete_idx;
                // Reload the store to compute the correct store-side index.
                // The store uses its own cross-process lock, not ui_mutex, so
                // this is safe outside the render lock.  All append_history
                // calls must also happen outside the lock to avoid a deadlock
                // (append_history itself acquires ui_mutex).
                static_cast<void>(history_store->load());
                const std::size_t store_size = history_store->size();
                // Picker is newest-first (index 0 = newest); store is
                // oldest-first (index 0 = oldest).
                const std::size_t store_idx =
                    store_size > static_cast<std::size_t>(picker_idx)
                        ? store_size - 1 - static_cast<std::size_t>(picker_idx)
                        : 0;
                if (history_store->remove_at_and_save(store_idx, &error)) {
                    std::lock_guard lock(ui_mutex);
                    prompts_picker_state.prompts = history_store->entries_newest_first();
                    if (prompts_picker_state.prompts.empty()) {
                        prompts_picker_state.active = false;
                    } else {
                        prompts_picker_state.selected = std::min(
                            prompts_picker_state.selected,
                            static_cast<int>(prompts_picker_state.prompts.size()) - 1);
                    }
                    prompts_picker_state.status_message.clear();
                } else {
                    append_history(std::format(
                        "\n\xe2\x9c\x97  Failed to delete prompt: {}\n", error));
                }
                prompt_history.reload();
                wake_ui();
                return true;
            }
            if (prompts_choice.has_value()) {
                std::string prompt_text;
                {
                    std::lock_guard lock(ui_mutex);
                    prompt_text = prompts_picker_state.prompts[
                        static_cast<size_t>(*prompts_choice)];
                    input_text = std::move(prompt_text);
                    input_cursor_position = static_cast<int>(input_text.size());
                    prompts_picker_state.status_message.clear();
                }
                wake_ui();
            }
            return true;
        }

        bool model_picker_was_active = false;
        std::optional<int> model_choice;
        bool open_local_picker_from_model = false;

        bool provider_model_picker_was_active = false;
        std::optional<int> provider_model_choice;
        std::string provider_model_provider_name;
        std::vector<tui::ModelPickerRow> provider_model_rows_snapshot;
        bool return_to_model_providers = false;
        {
            std::lock_guard lock(ui_mutex);
            if (provider_model_picker_state.active) {
                provider_model_picker_was_active = true;
                const int count = static_cast<int>(provider_model_picker_state.models.size());
                if (event == Event::ArrowUp) {
                    if (count > 0) {
                        provider_model_picker_state.selected =
                            (provider_model_picker_state.selected + count - 1) % count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (count > 0) {
                        provider_model_picker_state.selected =
                            (provider_model_picker_state.selected + 1) % count;
                    }
                } else if (event == Event::Return) {
                    if (count > 0) {
                        provider_model_choice = provider_model_picker_state.selected;
                        provider_model_provider_name = provider_model_picker_state.provider_name;
                        provider_model_rows_snapshot = provider_model_picker_state.models;
                        provider_model_picker_state.active = false;
                    }
                } else if (event == Event::Escape) {
                    provider_model_picker_state.active = false;
                    return_to_model_providers = true;
                } else {
                    for (int n = 1; n <= std::min(count, 9); ++n) {
                        if (event == Event::Character(static_cast<char>('0' + n))) {
                            provider_model_choice = n - 1;
                            provider_model_provider_name = provider_model_picker_state.provider_name;
                            provider_model_rows_snapshot = provider_model_picker_state.models;
                            provider_model_picker_state.active = false;
                            break;
                        }
                    }
                }
            }
        }
        if (return_to_model_providers) {
            open_model_picker();
            return true;
        }
        if (provider_model_picker_was_active) {
            if (provider_model_choice.has_value()
                && static_cast<std::size_t>(*provider_model_choice) < provider_model_rows_snapshot.size()) {
                const auto& row = provider_model_rows_snapshot[static_cast<std::size_t>(*provider_model_choice)];
                const std::string effective_provider = row.source_provider.empty()
                    ? provider_model_provider_name
                    : row.source_provider;
                const std::string selector = row.selector.empty()
                    ? effective_provider
                    : effective_provider + " " + row.selector;
                const std::string result = apply_model_selector(selector, true);
                const bool success = result.starts_with("Switched");
                append_history(std::format(
                    "\n{}\n",
                    success ? "\xe2\x9c\x93  " + result
                            : "\xe2\x9c\x97  " + result));
            }
            return true;
        }

        bool model_provider_picker_was_active = false;
        std::optional<int> model_provider_choice;
        std::vector<tui::ModelProviderPickerRow> model_provider_rows_snapshot;
        {
            std::lock_guard lock(ui_mutex);
            if (model_provider_picker_state.active) {
                model_provider_picker_was_active = true;
                const int count = static_cast<int>(model_provider_picker_state.providers.size());
                if (event == Event::ArrowUp) {
                    if (count > 0) {
                        model_provider_picker_state.selected =
                            (model_provider_picker_state.selected + count - 1) % count;
                    }
                } else if (event == Event::ArrowDown) {
                    if (count > 0) {
                        model_provider_picker_state.selected =
                            (model_provider_picker_state.selected + 1) % count;
                    }
                } else if (event == Event::Return) {
                    if (count > 0) {
                        model_provider_choice = model_provider_picker_state.selected;
                        model_provider_rows_snapshot = model_provider_picker_state.providers;
                        model_provider_picker_state.active = false;
                    }
                } else if (event == Event::Escape) {
                    model_provider_picker_state.active = false;
                } else if (event == Event::Character('l') || event == Event::Character('L')) {
                    open_local_picker_from_model = true;
                    model_provider_picker_state.active = false;
                } else {
                    for (int n = 1; n <= std::min(count, 9); ++n) {
                        if (event == Event::Character(static_cast<char>('0' + n))) {
                            model_provider_choice = n - 1;
                            model_provider_rows_snapshot = model_provider_picker_state.providers;
                            model_provider_picker_state.active = false;
                            break;
                        }
                    }
                }
            }
        }
        if (open_local_picker_from_model) {
            open_local_model_picker();
            return true;
        }
        if (model_provider_picker_was_active) {
            if (model_provider_choice.has_value()
                && static_cast<std::size_t>(*model_provider_choice) < model_provider_rows_snapshot.size()) {
                open_provider_model_picker(
                    model_provider_rows_snapshot[static_cast<std::size_t>(*model_provider_choice)].name);
            }
            return true;
        }

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
                const ModelSelectionSnapshot before = current_model_selection_snapshot();
                std::string result =
                    (*model_choice == 0) ? activate_manual_mode() :
                    (*model_choice == 2) ? activate_auto_mode()   :
                                          activate_router_mode();
                const bool success = result.starts_with("Switched");
                if (success) {
                    remember_previous_model_selection(before);
                    result = with_persisted_model_preferences(std::move(result));
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

        // ── Filesystem browser (folder / file picker) ─────────────────────────
        // The component owns the whole keyboard contract; MainApp only decides
        // what a confirmed path means, via the callback registered on open.
        FileSystemPickerEventResult file_picker_result;
        std::function<void(const std::filesystem::path&)> file_picker_confirm;
        {
            std::lock_guard lock(ui_mutex);
            if (file_picker_state.active) {
                file_picker_result =
                    handle_file_system_picker_event(file_picker_state, event);
                if (file_picker_result.action != FileSystemPickerAction::None) {
                    file_picker_confirm = std::exchange(file_picker_on_confirm, {});
                }
            }
        }
        if (file_picker_result.handled) {
            // Invoked outside the lock: handlers may re-enter the UI state.
            if (file_picker_result.action == FileSystemPickerAction::Confirm
                && file_picker_confirm) {
                file_picker_confirm(file_picker_result.path);
            }
            wake_ui();
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
            wake_ui();
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
            wake_ui();
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
                    message_count = selected_messages->size();
                }
                history_component->JumpToMessage(
                    static_cast<std::size_t>(*search_jump_index),
                    message_count);
            }
            wake_ui();
            return true;
        }

        if (is_ctrl_f_event(event)) {
            {
                std::lock_guard lock(ui_mutex);
                conversation_search_state.active = true;
                conversation_search_state.selected = 0;
                refresh_conversation_search_locked();
            }
            wake_ui();
            return true;
        }

        if (is_ctrl_letter_event(event, 'b')) {  // Ctrl+B — browse for an attachment
            open_attachment_picker();
            return true;
        }

        if (is_ctrl_r_event(event)) {  // Ctrl+R — run code from the latest response
            const auto result = open_code_blocks(std::nullopt);
            if (!result.ok) {
                append_history(std::format("\n✗  {}\n", result.message));
            }
            return true;
        }

        if (event == Event::Escape
            && current_runtime->turn_active()) {
            [[maybe_unused]] const auto ignored = stop_active_terminal();
            return true;
        }

        auto command_snapshot = current_command_snapshot();
        sync_command_picker(command_snapshot);
        if (!command_snapshot.suggestions.empty() && !command_picker.suppressed) {
            if (event == Event::Return
                && command_snapshot.active.has_value()
                && trim_ascii(input_text) == command_snapshot.active->token) {
                // Submit a command-only input directly from the autocomplete
                // layer. Letting an exact command fall through to the focused
                // Input component required a second navigation event on some
                // terminals after the command opened an overlay.
                return execute_selected_command();
            }
            if (event == Event::Return && has_pending_command_completion()) {
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

        if (event == Event::Escape) {
            if (!record_escape_press(double_escape_state)) {
                return true;
            }

            command_picker.key.clear();
            command_picker.selected = 0;
            command_picker.suppress();
            mention_picker.key.clear();
            mention_picker.selected = 0;

            if (!input_text.empty()) {
                prompt_history.save(input_text);
                input_text.clear();
                input_cursor_position = 0;
                append_history("\n↶  Draft cleared. Press Up to restore it.\n");
                wake_ui();
                return true;
            }

            open_rewind_menu();
            return true;
        }

        if (event == Event::F2) {
            current_mode_idx = (current_mode_idx + 1) % static_cast<int>(modes.size());
            agent->set_mode(modes[current_mode_idx].first);
            return true;
        }
        if (is_ctrl_p_event(event)) {
            return open_model_picker();
        }
        if (is_ctrl_t_event(event)) {
            if (!open_prompts_picker()) {
                append_history("\n\xe2\x9a\xa0  No saved prompts are available yet.\n");
            }
            return true;
        }
        // Ctrl+N — start a new thread (agentty-compatible). Archives the current
        // conversation so it remains switchable via /threads.
        if (is_ctrl_n_event(event)) {
            if (const auto err = start_new_thread(); err.has_value()) {
                append_history(std::format("\n✗  {}\n", *err));
            }
            return true;
        }
        // Enhanced-keyboard Ctrl+J / Ctrl+H aliases open the thread browser.
        if (is_ctrl_j_event(event) || is_ctrl_h_event(event)) {
            open_threads_picker();
            return true;
        }
        if (is_ctrl_l_event(event)) {  // Ctrl+L — clear screen (same as /clear)
            clear_screen();
            return true;
        }
        // Ctrl+G matches Claude Code, Ctrl+X matches Gemini CLI.
        if (is_ctrl_g_event(event) || is_ctrl_x_event(event)) {
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
                wake_ui();
                return true;
            }

            if (const auto text = read_clipboard_text(); text.has_value() && !text->empty()) {
                insert_text_at_cursor(
                    input_text,
                    input_cursor_position,
                    normalize_newlines(*text));
                wake_ui();
                return true;
            }

            append_history(
                "\n\xe2\x9c\x97  Clipboard paste is unavailable (no compatible clipboard provider found).\n");
            wake_ui();
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
        if (is_ctrl_o_event(event)) {  // Ctrl+O — toggle verbose output expansion
            tool_output_expanded = !tool_output_expanded;
            append_history(tool_output_expanded
                ? "\n\xe2\x84\xb9  Verbose output view set to EXPANDED: full tool output and all disclosure details are visible.\n"
                : "\n\xe2\x84\xb9  Verbose output view set to COMPACT: long output is collapsed (you can still click disclosure rows).\n");
            return true;
        }
        if (is_ctrl_c_event(event)) {
            // Idle: quit immediately on a single press. Unlike Ctrl+D (which
            // doubles as delete-right and needs a confirmation guard), Ctrl+C
            // has no other meaning when nothing is running. The explicit
            // predicate — rather than stop_active_terminal()'s failure flag —
            // guarantees we can never quit while work is still active, even if
            // stop_active_terminal() grows new failure modes.
            if (!has_stoppable_activity()) {
                screen.ExitLoopClosure()();
                return true;
            }
            const auto result = stop_active_terminal();
            reset_quit_confirm();
            append_history(std::format("\n»  {}\n", result.message));
            return true;
        }

        // Up/Down arrows with no active autocomplete → navigate history
        if (event == Event::ArrowUp) {
            return navigate_history_prev();
        }
        if (event == Event::ArrowDown) {
            return navigate_history_next();
        }

        // ── Ctrl+D: delete-right; close a secondary tab or confirm app exit ─
        if (is_ctrl_d_event(event)) {
            if (!input_text.empty()) {
                // Normalize terminal-specific Ctrl+D variants through PromptInput's
                // canonical control-byte handler.
                reset_quit_confirm();
                input_component->OnEvent(Event::Special({4}));
                return true;
            }

            if (current_runtime != main_runtime) {
                reset_quit_confirm();
                if (const auto error = close_current_secondary_thread();
                    error.has_value()) {
                    append_history(std::format("\n✗  {}\n", *error));
                }
                return true;
            }

            if (confirm_quit_or_arm("Ctrl+D")) {
                screen.ExitLoopClosure()();
            }
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
        bool                   model_provider_picker_active = false;
        bool                   provider_model_picker_active = false;
        bool                   command_option_picker_active = false;
        bool                   provider_picker_active = false;
        bool                   review_picker_active = false;
        bool                   review_activity_active = false;
        bool                   settings_panel_active = false;
        bool                   authentication_recovery_active = false;
        std::string            perm_tool, perm_args, perm_allow_label, perm_origin_label;
        std::string            review_activity_hint;
        std::string            settings_panel_status;
        std::string            authentication_recovery_provider;
        std::string            authentication_recovery_reason;
        bool                   authentication_recovery_retry_safe = false;
        ToolDiffPreview        perm_diff;
        int                    perm_selected = 0;
        int                    model_picker_selected = 0;
        int                    model_provider_picker_selected = 0;
        int                    provider_model_picker_selected = 0;
        int                    command_option_picker_selected = 0;
        int                    provider_picker_selected = 0;
        int                    review_picker_selected = 0;
        int                    settings_panel_selected = 0;
        int                    authentication_recovery_selected = 0;
        std::vector<std::string> provider_picker_providers;
        std::vector<tui::ModelProviderPickerRow> model_provider_picker_providers;
        std::vector<tui::ModelPickerRow> provider_model_picker_models;
        std::string provider_model_picker_provider;
        std::string            command_option_picker_title;
        std::string            command_option_picker_current;
        std::string            command_option_picker_help;
        std::vector<tui::OptionPickerRow> command_option_picker_options;
        ReviewPickerMode       review_picker_mode = ReviewPickerMode::SelectTarget;
        std::string            review_picker_input;
        std::vector<ReviewBaseRef> review_picker_base_refs;
        int                    review_picker_base_ref_selected = 0;
        std::chrono::steady_clock::time_point review_activity_started_at =
            std::chrono::steady_clock::time_point::min();
        core::config::SettingsScope settings_panel_scope = core::config::SettingsScope::User;
        bool                            local_model_picker_active   = false;
        int                             local_model_picker_selected = 0;
        std::string                     local_model_picker_dir;
        std::vector<tui::LocalModelEntry> local_model_picker_entries;
        FileSystemPickerState           file_picker_snapshot;
        bool                            session_picker_active = false;
        int                             session_picker_selected = 0;
        SessionPickerResource           session_picker_resource =
            SessionPickerResource::SavedSessions;
        std::vector<core::session::SessionInfo> session_picker_filtered;
        std::string                     session_picker_current_id;
        std::string                     session_picker_query;
        bool                            session_picker_filter_active = false;
        bool                            session_picker_rename_active = false;
        std::string                     session_picker_rename_buffer;
        std::string                     session_picker_status;
        std::unordered_set<std::string> session_picker_running_ids;
        bool                            prompts_picker_active = false;
        int                             prompts_picker_selected = 0;
        std::vector<std::string>        prompts_picker_prompts;
        std::string                     prompts_picker_status;
        bool                            rewind_picker_active = false;
        int                             rewind_picker_selected = 0;
        std::vector<RewindPickerOption> rewind_picker_options;
        CodeBlockRunnerState            code_block_runner_snapshot;
        bool                            conversation_search_active = false;
        int                             conversation_search_selected = 0;
        std::string                     conversation_search_query;
        std::vector<tui::ConversationSearchHit> conversation_search_hits;
        bool                            stderr_panel_active = false;
        std::vector<std::string>        stderr_panel_lines;
        bool                            remote_activity_panel_active = false;
        std::size_t                     remote_activity_panel_selected = 0;
        bool                            usage_panel_visible = false;
        bool                            agents_visualizer_visible = false;
        std::size_t                     agents_visualizer_selected = 0;
        int                             agents_visualizer_scroll = 0;
        std::size_t                     queued_steering_count = 0;
        std::string                     external_editor_status;
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
            perm_origin_label = perm_state.origin_label;
            model_picker_active   = model_picker_state.active;
            model_picker_selected = model_picker_state.selected;
            model_provider_picker_active = model_provider_picker_state.active;
            model_provider_picker_selected = model_provider_picker_state.selected;
            model_provider_picker_providers = model_provider_picker_state.providers;
            provider_model_picker_active = provider_model_picker_state.active;
            provider_model_picker_selected = provider_model_picker_state.selected;
            provider_model_picker_provider = provider_model_picker_state.provider_name;
            provider_model_picker_models = provider_model_picker_state.models;
            command_option_picker_active = command_option_picker_state.active;
            command_option_picker_selected = command_option_picker_state.selected;
            command_option_picker_title = command_option_picker_state.title;
            command_option_picker_current = command_option_picker_state.current_value;
            command_option_picker_help = command_option_picker_state.help_text;
            command_option_picker_options = command_option_picker_state.options;
            provider_picker_active    = provider_picker_state.active;
            provider_picker_selected  = provider_picker_state.selected;
            provider_picker_providers = provider_picker_state.providers;
            review_picker_active = review_picker_state.active;
            review_picker_selected = review_picker_state.selected;
            review_picker_mode = review_picker_state.mode;
            review_picker_input = review_picker_state.input_text;
            review_picker_base_refs = review_picker_state.filtered_base_refs;
            review_picker_base_ref_selected = review_picker_state.selected_base_ref;
            review_activity_active = review_activity_state.active;
            review_activity_hint = review_activity_state.hint;
            review_activity_started_at = review_activity_state.started_at;
            settings_panel_active = settings_panel_state.active;
            settings_panel_selected = settings_panel_state.selected;
            settings_panel_scope = settings_panel_state.scope;
            settings_panel_status = settings_panel_state.status_message;
            authentication_recovery_active =
                authentication_recovery_state.active.has_value();
            if (authentication_recovery_state.active.has_value()) {
                authentication_recovery_provider =
                    authentication_recovery_state.active->provider.display_name;
                authentication_recovery_reason =
                    authentication_recovery_state.active->request.reason;
                authentication_recovery_retry_safe =
                    authentication_recovery_state.active->request.retry_safe;
                authentication_recovery_selected =
                    authentication_recovery_state.selected;
            }
            local_model_picker_active   = local_model_picker_state.active;
            local_model_picker_selected = local_model_picker_state.selected;
            local_model_picker_dir      = local_model_picker_state.current_dir.string();
            local_model_picker_entries  = local_model_picker_state.entries;
            file_picker_snapshot        = file_picker_state;
            session_picker_active = session_picker_state.active;
            session_picker_selected = session_picker_state.selected;
            session_picker_resource = session_picker_state.resource;
            session_picker_filtered = session_picker_state.filtered;
            session_picker_current_id = session_picker_state.current_session_id;
            session_picker_query = session_picker_state.query;
            session_picker_filter_active =
                session_picker_state.mode == SessionPickerMode::Filter;
            session_picker_rename_active =
                session_picker_state.mode == SessionPickerMode::Rename;
            session_picker_rename_buffer = session_picker_state.rename_buffer;
            session_picker_status = session_picker_state.status_message;
            prompts_picker_active = prompts_picker_state.active;
            prompts_picker_selected = prompts_picker_state.selected;
            prompts_picker_prompts = prompts_picker_state.prompts;
            prompts_picker_status = prompts_picker_state.status_message;
            rewind_picker_active = rewind_picker_state.active;
            rewind_picker_selected = rewind_picker_state.selected;
            rewind_picker_options = rewind_picker_state.options;
            code_block_runner_snapshot = code_block_runner.state();
            conversation_search_active = conversation_search_state.active;
            conversation_search_selected = conversation_search_state.selected;
            conversation_search_query = conversation_search_state.query;
            conversation_search_hits = conversation_search_state.hits;
            stderr_panel_active = stderr_panel_state.active
                && !stderr_panel_state.lines.empty();
            stderr_panel_lines = stderr_panel_state.lines;
            remote_activity_panel_active = remote_activity_panel_state.active;
            remote_activity_panel_selected = remote_activity_panel_state.selected;
            usage_panel_visible = usage_details_panel_active;
            agents_visualizer_visible = agents_visualizer_panel_active;
            agents_visualizer_selected = agents_visualizer_selected_file;
            agents_visualizer_scroll = agents_visualizer_scroll_offset;
            queued_steering_count = current_runtime->queued_turn_count();
        }
        auto remote_activity_snapshot = opts.remote_mcp_server_enabled
            ? remote_activity_hub.snapshot(remote_activity_panel_active)
            : core::mcp::RemoteActivitySnapshot{};
        if (opts.remote_mcp_server_enabled
            && remote_activity_snapshot.server_state
                == core::mcp::RemoteServerState::disabled) {
            // The daemon thread may not have published its first lifecycle
            // event yet, but the composition root has already enabled it.
            remote_activity_snapshot.server_state =
                core::mcp::RemoteServerState::starting;
        }
        external_editor_status = external_editor.status_label();
        session_picker_running_ids = thread_runtimes.running_session_ids();

        std::vector<ThreadTab> thread_tabs;
        thread_tab_session_ids.clear();
        if (auto runtimes = thread_runtimes.ordered_snapshot(main_runtime->session_id());
            runtimes.size() > 1) {
            struct RuntimeTabSnapshot {
                ThreadRuntime::Ptr runtime;
                ThreadRuntimeMetadata metadata;
            };
            std::vector<RuntimeTabSnapshot> snapshots;
            snapshots.reserve(runtimes.size());
            for (auto& runtime : runtimes) {
                snapshots.push_back({runtime, runtime->metadata()});
            }
            const auto selected_runtime = thread_runtimes.current();
            thread_tabs.reserve(snapshots.size());
            thread_tab_session_ids.reserve(snapshots.size());
            for (std::size_t i = 0; i < snapshots.size(); ++i) {
                const auto& item = snapshots[i];
                std::string label = item.metadata.thread_name;
                if (label.empty()) {
                    label = i == 0
                        ? std::string{"main"}
                        : std::format("{} {}", project_thread_base_name, i + 1);
                }
                thread_tabs.push_back(ThreadTab{
                    .label = std::move(label),
                    .active = item.runtime == selected_runtime,
                    .running = item.runtime->turn_active(),
                });
                thread_tab_session_ids.push_back(item.metadata.session_id);
            }
        }

        const bool question_dialog_active = question_dialog.active();
        
        auto history_el = history_component->Render() | flex;

        Element banner_el = emptyElement();
        if (ui_show_banner) {
            banner_el = render_startup_banner_panel(
                active_provider_name,
                active_model_name.empty() ? "<provider default>" : active_model_name,
                core::mcp::McpConnectionManager::get_instance().connected_count(),
                context_sources_label,
                provider_setup_hint(active_provider_name),
                current_time_str(),
                thread_tabs,
                &thread_tab_hitboxes,
                &agents_status_box);
        } else {
            agents_status_box = {0, -1, 0, -1};
            thread_tab_hitboxes.clear();
            thread_tab_session_ids.clear();
        }

        // ── Rate limit and context info for bottom panel & status bar ──
        const std::string visible_session_id = current_runtime->session_id();
        std::string budget_str =
            core::budget::BudgetTracker::get_instance().status_string(
                visible_session_id);
        const auto context_window = agent->context_window_snapshot();
        const int32_t ctx_pct = context_window.remaining_pct;

        Color ctx_color = Color::Green;
        if (ctx_pct >= 0 && ctx_pct < 25)       ctx_color = Color::Red;
        else if (ctx_pct >= 0 && ctx_pct < 50)  ctx_color = ColorWarn;
        
        // Get rate limit info from the current provider (for Anthropic/OAuth users)
        auto rate_limit_info = llm_provider->get_last_rate_limit_info();
        
        // Update our local rate limit state for quota notifications
        {
            std::lock_guard lock(ui_mutex);
            rate_limit_state.latest = rate_limit_info;
        }
        // Check for quota notifications
        check_and_notify_quota();

        // Billing behavior belongs to the active provider. Usage windows remain
        // a fallback for aggregate/router providers that expose subscription quota.
        const bool is_subscription = !llm_provider->should_estimate_cost()
            || !rate_limit_info.usage_windows.empty();

        // ── Permission overlay ───────────────────────────────────────────
        Element bottom_el;
        if (!external_editor_status.empty()) {
            bottom_el = render_default_prompt_panel(
                hbox({
                    text(external_editor_status) | color(Color::GrayLight) | xflex,
                    text("Esc to cancel") | color(Color::GrayDark),
                }),
                {});
        } else if (usage_panel_visible) {
            auto total = core::budget::BudgetTracker::get_instance().session_total(
                visible_session_id);
            double cost = core::budget::BudgetTracker::get_instance().session_cost_usd(
                visible_session_id);
            bottom_el = render_usage_details_panel(
                rate_limit_info,
                active_provider_name,
                active_model_name,
                session_effort_value,
                is_subscription,
                total,
                cost,
                ctx_pct);
        } else if (agents_visualizer_visible) {
            bottom_el = render_agents_visualizer_panel(
                steering_context.files,
                agents_visualizer_selected,
                agents_visualizer_scroll,
                &agents_tab_hitboxes);
        } else if (remote_activity_panel_active) {
            bottom_el = render_remote_activity_panel(
                remote_activity_snapshot,
                remote_activity_panel_selected);
        } else if (authentication_recovery_active) {
            bottom_el = render_authentication_recovery_panel(
                authentication_recovery_provider,
                authentication_recovery_reason,
                authentication_recovery_retry_safe,
                authentication_recovery_selected);
        } else if (settings_panel_active) {
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
        } else if (question_dialog_active) {
            bottom_el = question_dialog.render();
        } else if (code_block_runner_snapshot.active) {
            bottom_el = render_code_block_runner_panel(code_block_runner_snapshot);
        } else if (rewind_picker_active) {
            bottom_el = render_rewind_picker_panel(
                rewind_picker_options,
                rewind_picker_selected);
        } else if (conversation_search_active) {
            bottom_el = render_conversation_search_panel(
                conversation_search_query,
                conversation_search_hits,
                conversation_search_selected);
        } else if (session_picker_active) {
            bottom_el = render_session_picker_panel(
                session_picker_filtered,
                session_picker_selected,
                session_picker_resource,
                session_picker_current_id,
                session_picker_query,
                session_picker_filter_active,
                session_picker_rename_active,
                session_picker_rename_buffer,
                session_picker_status,
                session_picker_running_ids);
        } else if (prompts_picker_active) {
            bottom_el = render_prompts_picker_panel(
                prompts_picker_prompts,
                prompts_picker_selected,
                prompts_picker_status);
        } else if (provider_model_picker_active) {
            bottom_el = render_provider_model_picker_panel(
                provider_model_picker_provider,
                provider_model_picker_models,
                provider_model_picker_selected);
        } else if (model_provider_picker_active) {
            constexpr bool kLocalModelAvailable =
#ifdef FILO_ENABLE_LLAMACPP
                true;
#else
                false;
#endif
            bottom_el = render_model_provider_picker_panel(
                model_provider_picker_providers,
                model_provider_picker_selected,
                kLocalModelAvailable);
        } else if (provider_picker_active) {
            bottom_el = render_provider_selection_panel(provider_picker_providers,
                                                        provider_picker_selected);
        } else if (review_picker_active) {
            bottom_el = render_review_picker_panel(
                review_picker_mode,
                review_picker_selected,
                review_picker_input,
                review_picker_base_refs,
                review_picker_base_ref_selected);
        } else if (file_picker_snapshot.active) {
            bottom_el = render_file_system_picker_panel(file_picker_snapshot);
        } else if (local_model_picker_active) {
            bottom_el = render_local_model_picker_panel(
                local_model_picker_dir,
                local_model_picker_entries,
                local_model_picker_selected);
        } else if (command_option_picker_active) {
            bottom_el = render_option_selection_panel(
                command_option_picker_title,
                command_option_picker_options,
                command_option_picker_selected,
                command_option_picker_current,
                command_option_picker_help);
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
                perm_selected,
                perm_origin_label);
        } else {
            Element input_el =
                input_component->Render() | color(tui::ColorYellowBright) | xflex;

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
        const std::size_t tick = animation_tick.load(std::memory_order_relaxed);
        const bool response_in_progress = current_runtime->turn_active();

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
        if (quit_confirm_active()) {
            right_el = text(" Press " + quit_confirm_key + " again to quit ")
                | color(Color::GrayDark);
        } else {
            Elements right_items;
            // Keep the workspace protection affordance quiet and colocated with its scope.
            std::string cwd_str = format_cwd();
            if (!cwd_str.empty()) {
                right_items.push_back(
                    text(format_workspace_status_label(
                        cwd_str,
                        core::landrun::LandrunSettings::instance().enabled()))
                    | color(Color::GrayLight));
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
                auto total = core::budget::BudgetTracker::get_instance().session_total(
                    visible_session_id);
                const std::string token_usage = format_subscription_token_usage(
                    total,
                    !rate_limit_info.usage_windows.empty());
                if (!token_usage.empty()) {
                    budget_el = text("  " + token_usage) | color(Color::GrayLight);
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
                const std::string pct_str =
                    std::format("{:.0f}", w.utilization * 100.0f);
                if (w.label == "overage" && pct_str == "0") {
                    continue;
                }
                util_str += std::format(" {}:{}%", w.label, pct_str);
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
        } else if (rate_limit_info.is_rate_limited
                   || rate_limit_info.unified_status == "rate_limited") {
            // No usage windows or numeric limits are available, but the provider
            // has blocked the request (e.g. a QwenCloud Token Plan 429). Surface
            // the state honestly in red instead of leaving the slot blank.
            std::string label = " Rate limited";
            if (rate_limit_info.retry_after > 0) {
                label += " " + std::to_string(rate_limit_info.retry_after) + "s";
            }
            label += " ";
            rate_limit_el = text(label) | color(Color::Red);
        }

        if (rate_limit_info.has_data()) {
            rate_limit_el = std::move(rate_limit_el) | reflect(usage_status_box);
        } else {
            usage_status_box = {0, -1, 0, -1};
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

        Element review_activity_el = text("");
        if (review_activity_active) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = review_activity_started_at == std::chrono::steady_clock::time_point::min()
                ? std::chrono::seconds::zero()
                : std::chrono::duration_cast<std::chrono::seconds>(now - review_activity_started_at);
            const std::string hint = review_activity_hint.empty()
                ? std::string("current changes")
                : compact_single_line(review_activity_hint, 44);
            const std::string label = std::format(" reviewing {} ({})", hint, format_elapsed_compact(elapsed));
            const std::size_t spinner_frame =
                ui_show_spinner.load(std::memory_order_relaxed) ? (tick / 2) : 0;
            Element spinner_el = spinner(kReviewActivitySpinnerCharset, spinner_frame)
                               | color(ColorYellowBright)
                               | ftxui::bold
                               | size(WIDTH, EQUAL, 1);
            review_activity_el = hbox({
                text(" "),
                std::move(spinner_el),
                text(label + " ")
                    | color(Color::GrayLight),
            });
        }

        Element queued_steering_el = text("");
        if (queued_steering_count > 0) {
            queued_steering_el =
                text(std::format(" steer queued:{} ", queued_steering_count))
                | bgcolor(ColorWarn)
                | color(Color::Black);
        }
        
        Elements left_items;
        left_items.push_back(
            text(std::string(" ") + modes[current_mode_idx].first + " ")
                | ftxui::bold | bgcolor(modes[current_mode_idx].second) | color(Color::White));
        if (ui_show_model_info) {
            left_items.push_back(
                text(format_model_status_badge(
                    active_provider_name,
                    active_model_name,
                    session_effort_value))
                    | bgcolor(ColorYellowDark) | color(Color::Black));
        }
        left_items.push_back(budget_el);
        left_items.push_back(rate_limit_el);
        TurnActivityState turn_activity_state = TurnActivityState::Idle;
        if (response_in_progress) {
            turn_activity_state = TurnActivityState::Active;
        } else {
            switch (current_runtime->completion_status()) {
                case TurnCompletionStatus::Succeeded:
                    turn_activity_state = TurnActivityState::Completed;
                    break;
                case TurnCompletionStatus::Failed:
                    turn_activity_state = TurnActivityState::Failed;
                    break;
                case TurnCompletionStatus::None:
                    break;
            }
        }
        left_items.push_back(render_turn_activity_indicator(
            turn_activity_state,
            ui_show_spinner.load(std::memory_order_relaxed),
            tick));
        // The inbound MCP server status is a standalone affordance: instead of
        // stacking it after the turn indicators, it gets its own centered slot
        // in the status bar (composed below).
        bool remote_mcp_status_visible = false;
        Element remote_mcp_status_el = text("");
        if (opts.remote_mcp_server_enabled) {
            const auto remote_status = format_remote_footer_status(
                remote_activity_snapshot);
            if (!remote_status.label.empty()) {
                remote_mcp_status_visible = true;
                remote_mcp_status_el =
                    render_remote_footer_status(remote_status)
                    | reflect(remote_activity_pill_box);
            }
        }
        left_items.push_back(guardrail_el);
        left_items.push_back(queued_steering_el);
        left_items.push_back(review_activity_el);
        const bool show_status_footer =
            ui_show_footer
            || usage_panel_visible
            || agents_visualizer_visible
            || response_in_progress
            || review_activity_active
            || opts.remote_mcp_server_enabled
            || queued_steering_count > 0;
        auto left_el = show_status_footer
            ? hbox(std::move(left_items))
            : text("");

        auto status_el = text("");
        if (show_status_footer || quit_confirm_active()) {
            if (remote_mcp_status_visible) {
                // Symmetric fillers center the MCP status pill in the space
                // between the two clusters; it yields to them on narrow
                // terminals instead of overlapping either side.
                status_el = hbox({
                    left_el,
                    filler(),
                    std::move(remote_mcp_status_el),
                    filler(),
                    std::move(right_el),
                }) | xflex;
            } else {
                status_el = hbox({
                    left_el | xflex,
                    right_el,
                }) | xflex;
            }
        }

        Elements window_rows;
        window_rows.reserve(5);
        if (ui_show_banner) {
            window_rows.push_back(std::move(banner_el));
        }
        if (agents_visualizer_visible) {
            window_rows.push_back(std::move(bottom_el) | flex);
        } else {
            window_rows.push_back(std::move(history_el));
            if (stderr_panel_active) {
                window_rows.push_back(render_stderr_panel(stderr_panel_lines));
            }
            window_rows.push_back(std::move(bottom_el));
        }
        window_rows.push_back(std::move(status_el));

        return UiWindow(
            text(std::format(" {} ", kAppVersion)) | color(ColorYellowBright) | ftxui::bold,
            vbox(std::move(window_rows))
        ) | color(ColorYellowBright);
    });

    std::jthread animation_thread([&](std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
            std::unique_lock lock(animation_mutex);
            animation_cv.wait(lock, [&]() {
                return stop_token.stop_requested()
                    || animation_cadence().has_value();
            });
            if (stop_token.stop_requested()) {
                break;
            }

            const auto cadence = animation_cadence();
            if (!cadence.has_value()) {
                continue;
            }
            const bool state_changed = animation_cv.wait_for(
                lock,
                cadence->period,
                [&]() {
                    return stop_token.stop_requested()
                        || animation_cadence() != cadence;
                });
            if (state_changed) {
                continue;
            }
            lock.unlock();

            if (cadence->advance_frame) {
                animation_tick.fetch_add(1, std::memory_order_relaxed);
            }
            // No application event changed; request a render frame directly.
            // This avoids routing animation-only wakes through the event queue.
            screen.RequestAnimationFrame();
        }
    });

    screen.Loop(renderer);

    // Force-dismiss any interactive blockers before the idle barriers, or a
    // waiting worker would block shutdown forever (permission slot queue,
    // unanswered question dialog, queued steering turns).
    {
        std::lock_guard lock(ui_mutex);
        if (perm_state.active && perm_state.promise) {
            auto pending = std::move(perm_state.promise);
            perm_state.active = false;
            perm_state.origin_runtime.reset();
            pending->set_value(false);
        }
    }
    {
        std::lock_guard lock(permission_prompt_mutex);
        permission_prompt_shutdown = true;
        permission_prompt_cv.notify_all();
    }
    {
        std::lock_guard lock(question_prompt_mutex);
        question_prompt_shutdown = true;
        question_prompt_cv.notify_all();
    }
    {
        auto dismissed = question_dialog.force_interrupt();
        if (dismissed.has_resolution()) {
            dismissed.resolve();
            question_prompt_cv.notify_all();
        }
    }

    // Detached turn launchers only retain runtime-owned state, but their TUI
    // callbacks still reference this composition root. Stop and join logically
    // (via the runtime idle barrier) before any captured service is destroyed.
    thread_runtimes.request_stop_all();
    direct_shell_state->interrupt_active();
    thread_runtimes.wait_until_all_idle();
    direct_shell_state->wait_until_idle();
    thread_runtimes.wait_until_all_saved();

    if (opts.remote_mcp_server_enabled) {
        // Barrier: blocks until no daemon thread is inside the callback, so the
        // captured locals below can be destroyed safely as run() unwinds.
        remote_activity_hub.set_notify_callback({});
    }
    animation_thread.request_stop();
    animation_cv.notify_one();
    animation_thread.join();

    ui_accepting_events.store(false, std::memory_order_release);
    {
        std::lock_guard lock(ui_event_post_mutex);
    }
    mention_index_thread.request_stop();

    core::logging::Logger::get_instance().clear_callback_sink();
    core::agent::PermissionGate::get_instance().set_notify_fn({});
    ask_user_tool->setQuestionCallback({});
    for (const auto& runtime : thread_runtimes.snapshot()) {
        runtime->agent()->set_permission_fn({});
        runtime->agent()->set_loop_break_fn({});
        runtime->agent()->set_efficiency_decision_fn({});
    }

    if (mention_index_thread.joinable()) {
        mention_index_thread.join();
    }

    // Shut down MCP connections gracefully.
    core::mcp::McpConnectionManager::get_instance().shutdown_all();

    return RunResult{
        .session_id        = session_id,
        .session_file_path = session_file_path,
    };
}

} // namespace tui
