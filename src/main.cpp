#include <thread>
#include <CLI/CLI.hpp>
#include "core/config/ConfigManager.hpp"
#include "core/workspace/Workspace.hpp"
#include "core/auth/AuthenticationManager.hpp"
#include "core/logging/Logger.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/session/SessionReport.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "tui/MainApp.hpp"
#include "exec/Server.hpp"
#include "exec/Daemon.hpp"
#include "exec/Prompter.hpp"
#include <format>
#include <iostream>
#include <cstdio>
#include <signal.h>
#include <unistd.h>

namespace {

[[nodiscard]] bool stdin_has_data() {
#if defined(_WIN32)
    return _isatty(_fileno(stdin)) == 0;
#else
    return isatty(STDIN_FILENO) == 0;
#endif
}

} // namespace

/**
 * @brief  **Filo** the C++ AI coding assistant.
 *
 * @author Alessio Pollero
 */
int main(int argc, char** argv) {
    ::signal(SIGPIPE, SIG_IGN);
    CLI::App app{"Filo - AI Coding Assistant"};

    bool        mcp_mode    = false;
    bool        daemon_mode = false;
    bool        headless    = false;
    bool        prompter_mode = false;
    int         port        = 8080;
    std::string host        = "127.0.0.1";
    std::string login_provider;
    std::string prompter_prompt;
    std::string output_format = "text";
    std::string input_format = "text";
    bool        include_partial_messages = false;
    bool        continue_last = false;
    std::string resume_session;      // --resume [id|index]; empty means most recent
    bool        list_sessions = false;  // --list-sessions
    std::string work_dir;
    std::vector<std::string> add_dirs;

    app.add_flag("--mcp",     mcp_mode,    "Run as an MCP server");
    app.add_flag("--daemon",  daemon_mode, "Run as an HTTP daemon exposing the MCP endpoint over TCP");
    app.add_flag("--headless", headless,   "Suppress the Terminal UI (required for daemon/mcp-only usage)");
    app.add_flag("--prompter", prompter_mode, "Run a single non-interactive prompt (script/CI mode)");
    auto* prompt_opt = app.add_option("-p,--prompt", prompter_prompt,
                   "Prompt text for prompter mode. Can be combined with piped stdin input.");
    auto* output_format_opt = app.add_option("-o,--output-format", output_format,
                   "Prompter output format: text, json, stream-json")->capture_default_str();
    auto* input_format_opt = app.add_option("--input-format", input_format,
                   "Prompter input format: text, stream-json")->capture_default_str();
    app.add_flag("--include-partial-messages", include_partial_messages,
                 "Include content-delta events in stream-json prompter output");
    app.add_flag("--continue", continue_last,
                 "Prompter mode: continue the most recent session scoped to the current project");
    app.add_option("-w,--work-dir", work_dir,
                   "Working directory for the agent. Default: current directory.");
    app.add_option("--add-dir", add_dirs,
                   "Add an additional directory to the workspace scope. Can be specified multiple times.");
    app.add_option("--port",  port,        "TCP port for the HTTP daemon (default: 8080)")->capture_default_str();
    app.add_option("--host",  host,
                   "Listen address for the HTTP daemon (default: 127.0.0.1 — localhost only). "
                   "Use 0.0.0.0 to accept connections from any network interface."
    )->capture_default_str();
    app.add_option("--login", login_provider,
                   "Authenticate with a provider and exit (e.g. --login openai, --login claude)");
    auto* resume_opt = app.add_option(
        "--resume", resume_session,
        "Resume a session by ID or 1-based index (see --list-sessions). "
        "If no value is provided, resumes the most recent session.");
    resume_opt->expected(0, 1);
    app.add_flag("--list-sessions", list_sessions, "List available sessions and exit");

    CLI11_PARSE(app, argc, argv);

    core::logging::Logger::get_instance().configure_from_env();

    if (!work_dir.empty()) {
        std::error_code ec;
        std::filesystem::current_path(work_dir, ec);
        if (ec) {
            core::logging::error("Failed to change working directory to '{}': {}", work_dir, ec.message());
            return 1;
        }
    }

    std::vector<std::filesystem::path> parsed_add_dirs;
    for (const auto& dir : add_dirs) {
        parsed_add_dirs.emplace_back(dir);
    }
    const bool enforce_workspace = !work_dir.empty() || !add_dirs.empty();
    core::workspace::Workspace::get_instance().initialize(std::filesystem::current_path(), parsed_add_dirs, enforce_workspace);

    // --list-sessions: print available sessions and exit.
    if (list_sessions) {
        const auto store = core::session::SessionStore{
            core::session::SessionStore::default_sessions_dir()};
        const auto infos = store.list();
        if (infos.empty()) {
            std::cout << "No saved sessions found.\n";
        } else {
            std::cout << "Available sessions (most recent first):\n\n";
            for (std::size_t i = 0; i < infos.size(); ++i) {
                std::string ts = infos[i].last_active_at.empty()
                    ? infos[i].created_at : infos[i].last_active_at;
                if (ts.size() >= 16 && ts[10] == 'T') ts[10] = ' ';
                ts = ts.substr(0, 16);
                std::cout << std::format("  [{:2d}]  {}  {}  {}/{:20}  {:3d} turns  {}\n",
                    i + 1,
                    infos[i].session_id,
                    ts,
                    infos[i].provider,
                    infos[i].model,
                    infos[i].turn_count,
                    infos[i].mode);
            }
            std::cout << "\nResume with: filo --resume <id>  or  filo --resume\n";
        }
        return 0;
    }

    try {
        core::config::ConfigManager::get_instance().load();
    } catch (const std::exception& e) {
        core::logging::warn("Could not load configuration: {}", e.what());
    }

    // --login: run provider auth flow and exit immediately
    if (!login_provider.empty()) {
        auto config_dir = core::config::ConfigManager::get_instance().get_config_dir();
        auto auth_manager = core::auth::AuthenticationManager::create_with_defaults(config_dir);
        try {
            const auto result = auth_manager.login(login_provider);
            core::logging::info("Successfully authenticated with {}.", result.provider);
            std::string persist_error;
            if (!core::config::ConfigManager::get_instance().persist_login_profile(
                    login_provider, &persist_error)) {
                core::logging::warn(
                    "Authenticated, but could not persist login profile: {}",
                    persist_error);
            } else {
                const auto& config = core::config::ConfigManager::get_instance().get_config();
                core::logging::info(
                    "Updated default provider to '{}' with OAuth enabled.",
                    config.default_provider);
            }
            for (const auto& hint : result.hints) {
                core::logging::info("{}", hint);
            }
        } catch (const std::exception& e) {
            core::logging::error("Authentication failed: {}", e.what());
            return 1;
        }
        return 0;
    }

    const bool has_stdin_input = stdin_has_data();
    const bool run_prompter_mode = !mcp_mode
        && !daemon_mode
        && exec::prompter::should_run(
            prompter_mode,
            prompt_opt->count() > 0,
            has_stdin_input,
            output_format_opt->count() > 0,
            input_format_opt->count() > 0,
            include_partial_messages,
            continue_last);

    if (run_prompter_mode) {
        if (mcp_mode || daemon_mode) {
            core::logging::error(
                "Prompter mode cannot be combined with --mcp or --daemon.");
            return 2;
        }

        exec::prompter::RunOptions opts;
        if (prompt_opt->count() > 0) {
            opts.prompt = prompter_prompt;
            opts.prompt_was_provided = true;
        }
        opts.output_format = output_format;
        opts.output_format_was_provided = output_format_opt->count() > 0;
        opts.input_format = input_format;
        opts.input_format_was_provided = input_format_opt->count() > 0;
        opts.include_partial_messages = include_partial_messages;
        opts.continue_last = continue_last;
        if (resume_opt->count() > 0) {
            opts.resume_session = resume_session;
        }
        return exec::prompter::run(opts);
    }

    // --mcp uses stdin/stdout as the MCP transport.  Never print anything to
    // stdout when this mode is active: every byte on stdout is part of the
    // JSON-RPC stream and extra text will corrupt it.
    std::thread mcp_thread;
    if (mcp_mode) {
        mcp_thread = std::thread([]() {
            exec::mcp::run_server();
        });
    }

    std::thread daemon_thread;
    if (daemon_mode) {
        core::logging::info("Starting Filo daemon on {}:{}...", host, port);
        daemon_thread = std::thread([port, host]() {
            exec::daemon::run_server(port, host);
        });
    }

    if (!headless) {
        // Build resume options.
        tui::RunOptions run_opts;
        if (resume_opt->count() > 0) {
            // Empty value means "resume most recent".
            run_opts.resume_session_id = resume_session;
        }
        const auto run_result = tui::run(run_opts);

        if (mcp_mode)    exec::mcp::stop_server();
        if (daemon_mode) exec::daemon::stop_server();
        
        // Close stdin to unblock the MCP server's std::getline, then join
        if (mcp_thread.joinable()) {
            std::fclose(stdin);
            mcp_thread.join();
        }
        if (daemon_thread.joinable()) daemon_thread.join();

        // Print the end-of-session summary report.
        core::session::SessionReport::print(
            core::budget::BudgetTracker::get_instance(),
            core::session::SessionStats::get_instance().snapshot(),
            run_result.session_id,
            run_result.session_file_path);
    } else {
        // Headless: block until all servers stop.
        if (mcp_thread.joinable())    mcp_thread.join();
        if (daemon_thread.joinable()) daemon_thread.join();
    }

    return 0;
}
