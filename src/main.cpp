#include <thread>
#include <CLI/CLI.hpp>
#include "core/config/ConfigManager.hpp"
#include "core/workspace/Workspace.hpp"
#include "core/workspace/SessionWorkspace.hpp"
#include "core/auth/AuthLogin.hpp"
#include "core/logging/Logger.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/session/SessionReport.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/cli/TrustFlagResolver.hpp"
#include "core/landrun/LandrunHelper.hpp"
#include "core/landrun/LandrunDriverFactory.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"
#include "core/landrun/LandrunReadiness.hpp"
#include "core/landrun/LandrunRuntime.hpp"
#include "core/landrun/LandrunSettings.hpp"
#include "core/landrun/LandrunStatus.hpp"
#include "tui/MainApp.hpp"
#include "exec/Server.hpp"
#include "exec/Daemon.hpp"
#include "exec/Prompter.hpp"
#include <format>
#include <filesystem>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>
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
    if (core::landrun::is_landrun_helper_invocation(argc, argv)) {
        return core::landrun::run_landrun_helper(argc - 2, argv + 2);
    }
    ::signal(SIGPIPE, SIG_IGN);
    CLI::App app{"Filo - AI Coding Assistant"};

    std::string mcp_transport = "stdio";
    bool        daemon_mode_legacy = false;
    bool        headless    = false;
    bool        prompter_mode = false;
    int         port        = 8080;
    std::string host        = "127.0.0.1";
    std::string login_provider;
    std::vector<std::string> auth_args;
    std::string auth_action;
    std::string auth_provider;
    std::string prompter_prompt;
    std::string output_format = "text";
    std::string input_format = "text";
    bool        include_partial_messages = false;
    bool        continue_last = false;
    bool        yolo_mode = false;
    bool        enable_api_gateway = false;
    std::string resume_session;      // --resume [id|index]; empty means most recent
    bool        list_sessions = false;  // --list-sessions
    std::vector<std::string> work_dirs;
    std::vector<std::string> trusted_tools;
    std::string startup_model;
    std::string sandbox_mode = "off";
    std::optional<std::string> requested_sandbox_mode;
    bool sandbox_status = false;
    std::vector<std::filesystem::path> sandbox_excluded_paths;
    std::unique_ptr<core::landrun::LandrunRuntime> landrun_runtime;

    auto* mcp_opt = app.add_option(
        "--mcp",
        mcp_transport,
        "Run as an MCP server. Optional transport: stdio (default) or tcp.");
    mcp_opt->expected(0, 1);
    mcp_opt->capture_default_str();
    app.add_flag("--daemon",  daemon_mode_legacy,
                 "Deprecated alias for --mcp tcp");
    app.add_flag("--headless", headless,   "Suppress the Terminal UI (required for MCP-only usage)");
    app.add_flag("--prompter", prompter_mode, "Run a single non-interactive prompt (script/CI mode)");
    auto* prompt_opt = app.add_option("-p,--prompt", prompter_prompt,
                   "Prompt text for prompter mode. Can be combined with piped stdin input.");
    auto* output_format_opt = app.add_option("-o,--output-format", output_format,
                   "Prompter output format: text, json, stream-json")->capture_default_str();
    auto* input_format_opt = app.add_option("--input-format", input_format,
                   "Prompter input format: text, stream-json")->capture_default_str();
    app.add_flag("--include-partial-messages", include_partial_messages,
                 "Include content-delta events in stream-json prompter output");
    app.add_flag("-c,--continue", continue_last,
                 "Continue the most recent session scoped to the current project "
                 "(TUI + prompter mode)");
    app.add_flag(
        "-y,--yolo",
        yolo_mode,
        "Enable YOLO mode (auto-approve sensitive tools).");
    app.add_option(
        "--trust-tools",
        trusted_tools,
        "Comma-separated list of sensitive tools to auto-approve at startup "
        "(prompter + TUI; e.g. run_terminal_command,write_file). Use '*' to trust all.")
        ->delimiter(',');
    app.add_option(
        "-w,--work-dir",
        work_dirs,
        "Workspace directory for the agent. The first -w is the primary working directory; "
        "later -w entries are additional workspace directories.")
        ->expected(1)
        ->take_all();
    app.add_option(
        "--port",
        port,
        "TCP port for the HTTP daemon used by --mcp tcp/--daemon and --api (default: 8080)")
        ->capture_default_str();
    app.add_option("--host",  host,
                   "Listen address for the HTTP daemon used by --mcp tcp/--daemon and --api "
                   "(default: 127.0.0.1 — localhost only). "
                   "Use 0.0.0.0 to accept connections from any network interface."
    )->capture_default_str();
    app.add_flag(
        "--api",
        enable_api_gateway,
        "Enable optional OpenAI/Anthropic-compatible proxy endpoints "
        "(/v1/models, /v1/chat/completions, /v1/messages). "
        "Starts the HTTP daemon even without --mcp.");
    auto* login_opt = app.add_option(
        "--login", login_provider,
        "Deprecated alias for --auth <provider> (authenticate and exit)");
    auto* auth_opt = app.add_option(
        "--auth", auth_args,
        "Authenticate or sign out and exit: --auth <provider> [login|logout]. "
        "The action defaults to login and may also precede the provider.");
    auth_opt->expected(1, 2)->excludes(login_opt);
    auto* resume_opt = app.add_option(
        "-r,--resume", resume_session,
        "Resume a session by ID, 1-based index, or name (see --list-sessions and /rename). "
        "If no value is provided, resumes the most recent session.");
    resume_opt->expected(0, 1);
    app.add_flag("--list-sessions", list_sessions, "List available sessions and exit");
    auto* model_opt = app.add_option(
        "--model",
        startup_model,
        "Model for this process only: MODEL, PROVIDER, or PROVIDER/MODEL. "
        "Does not change saved defaults.");
    auto* sandbox_opt = app.add_option(
        "--sandbox",
        requested_sandbox_mode,
        "OS process sandbox: bare --sandbox enables workspace-write; "
        "otherwise off (default), read-only, or workspace-write.")
        ->expected(0, 1)
        ->check(CLI::IsMember(core::landrun::landrun_mode_names()));
    app.add_option(
        "--sandbox-exclude",
        sandbox_excluded_paths,
        "Path to exclude from sandboxed child access; repeat for additional paths.")
        ->expected(1);
    app.add_flag(
        "--sandbox-status",
        sandbox_status,
        "Verify and print the effective sandbox guarantees, then exit.");

    CLI11_PARSE(app, argc, argv);

    if (sandbox_opt->count() > 0) {
        sandbox_mode = requested_sandbox_mode.has_value()
                && !requested_sandbox_mode->empty()
            ? *requested_sandbox_mode
            : "workspace-write";
    }

    const auto parsed_sandbox_mode = core::landrun::parse_landrun_mode(sandbox_mode);
    if (!parsed_sandbox_mode.has_value()) {
        std::cerr << "Unsupported sandbox mode: " << sandbox_mode << '\n';
        return 2;
    }

    auto& landrun_settings = core::landrun::LandrunSettings::instance();
    landrun_settings.configure_startup({
        .mode = *parsed_sandbox_mode,
        .excluded_paths = std::move(sandbox_excluded_paths),
    });
    landrun_settings.freeze_startup_configuration();

    if (!auth_args.empty()) {
        auth_action = "login";
        if (auth_args.size() == 1) {
            if (auth_args.front() == "login" || auth_args.front() == "logout") {
                std::cerr << "--auth requires a provider.\n";
                return 2;
            }
            auth_provider = auth_args.front();
        } else {
            const bool first_is_action = auth_args.front() == "login"
                || auth_args.front() == "logout";
            const bool second_is_action = auth_args.back() == "login"
                || auth_args.back() == "logout";
            if (first_is_action == second_is_action) {
                std::cerr << "--auth expects one provider and, optionally, "
                             "one action (login or logout).\n";
                return 2;
            }
            auth_action = first_is_action ? auth_args.front() : auth_args.back();
            auth_provider = first_is_action ? auth_args.back() : auth_args.front();
        }
    }

    core::logging::Logger::get_instance().configure_from_env();
    if (core::landrun::LandrunSettings::instance().enabled()) {
        const auto driver = core::landrun::make_landrun_driver();
        const auto probe = driver->probe();
        if (!probe.available) {
            const auto message = std::format(
                "Requested sandbox backend '{}' is unavailable: {}. "
                "Use --sandbox off only if you explicitly accept unrestricted commands.",
                probe.backend,
                probe.detail);
            core::logging::error("{}", message);
            std::cerr << message << '\n';
            return 1;
        }
        core::logging::info("landrun enabled: {} ({})", probe.backend, probe.detail);
        try {
            landrun_runtime = std::make_unique<core::landrun::LandrunRuntime>();
        } catch (const std::exception& error) {
            const auto message = std::format(
                "Failed to initialize landrun: {}", error.what());
            core::logging::error("{}", message);
            std::cerr << message << '\n';
            return 1;
        }
    }
    const auto trust_resolution = core::cli::resolve_trust_flags(yolo_mode, trusted_tools);

    std::string normalized_mcp_transport = core::utils::str::to_lower_ascii_copy(mcp_transport);
    const bool mcp_option_provided = mcp_opt->count() > 0;
    if (mcp_option_provided && normalized_mcp_transport.empty()) {
        normalized_mcp_transport = "stdio";
    }
    if (mcp_option_provided
        && normalized_mcp_transport != "stdio"
        && normalized_mcp_transport != "tcp") {
        core::logging::error(
            "Invalid --mcp transport '{}'. Supported transports: stdio, tcp.",
            mcp_transport);
        return 2;
    }

    if (daemon_mode_legacy) {
        if (mcp_option_provided && normalized_mcp_transport != "tcp") {
            core::logging::error(
                "--daemon is equivalent to --mcp tcp and cannot be combined with --mcp {}.",
                normalized_mcp_transport);
            return 2;
        }
        normalized_mcp_transport = "tcp";
        core::logging::warn("--daemon is deprecated; use --mcp tcp instead.");
    }

    const bool mcp_mode = mcp_option_provided || daemon_mode_legacy;
    const bool mcp_stdio_mode = mcp_mode && normalized_mcp_transport == "stdio";
    const bool mcp_tcp_mode = mcp_mode && normalized_mcp_transport == "tcp";
    const bool http_daemon_mode = mcp_tcp_mode || enable_api_gateway;

    if (!work_dirs.empty()) {
        std::error_code ec;
        std::filesystem::current_path(work_dirs.front(), ec);
        if (ec) {
            core::logging::error(
                "Failed to change working directory to '{}': {}",
                work_dirs.front(),
                ec.message());
            return 1;
        }
    }

    std::vector<std::filesystem::path> additional_work_dirs;
    for (std::size_t i = 1; i < work_dirs.size(); ++i) {
        additional_work_dirs.emplace_back(work_dirs[i]);
    }
    core::workspace::Workspace::get_instance().initialize(
        std::filesystem::current_path(),
        additional_work_dirs,
        true);

    if (core::landrun::LandrunSettings::instance().enabled()) {
        const char* home_value = std::getenv("HOME");
        if (home_value && *home_value) {
            const auto home = core::workspace::SessionWorkspace::normalize_path(home_value);
            const auto workspace_root = core::workspace::SessionWorkspace::normalize_path(
                std::filesystem::current_path());
            bool exposes_home = core::landrun::is_landrun_path_within(
                workspace_root, home);
            for (const auto& additional : additional_work_dirs) {
                exposes_home = exposes_home
                    || core::landrun::is_landrun_path_within(
                        core::workspace::SessionWorkspace::normalize_path(additional),
                        home);
            }
            if (exposes_home) {
                constexpr std::string_view message =
                    "Refusing secure mode with the entire home directory as the workspace. "
                    "Start Filo inside a project directory or explicitly use --sandbox off.";
                core::logging::error("{}", message);
                std::cerr << message << '\n';
                return 1;
            }
        }
    }

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
                std::cout << std::format("  [{:2d}]  {}  {}  {}/{:20}  {:3d} turns  {}{}\n",
                    i + 1,
                    infos[i].session_id,
                    ts,
                    infos[i].provider,
                    infos[i].model,
                    infos[i].turn_count,
                    infos[i].mode,
                    infos[i].name.empty() ? std::string{} : std::format("  \"{}\"", infos[i].name));
            }
            std::cout << "\nResume with: filo --resume <id|name>  or  filo --resume\n";
        }
        return 0;
    }

    try {
        core::config::ConfigManager::get_instance().load();
    } catch (const std::exception& e) {
        core::logging::warn("Could not load configuration: {}", e.what());
    }

    if (auth_action == "logout") {
        try {
            const std::string config_dir =
                core::config::ConfigManager::get_instance().get_config_dir();
            auto manager = core::auth::AuthenticationManager::create_with_defaults(config_dir);
            std::cout << "Signed out of " << manager.logout(auth_provider) << ".\n";
            return 0;
        } catch (const std::exception& e) {
            core::logging::error("Logout failed: {}", e.what());
            return 1;
        }
    }

    // --auth <provider> (or the deprecated --login alias): authenticate and exit.
    if (!auth_provider.empty() || !login_provider.empty()) {
        auto config_dir = core::config::ConfigManager::get_instance().get_config_dir();
        try {
            std::string selected_provider = auth_provider.empty()
                ? login_provider
                : auth_provider;

            const auto outcome = core::auth::login_and_persist(
                selected_provider, config_dir);
            core::logging::info("Successfully authenticated with {}.", outcome.result.provider);
            if (!outcome.profile_persisted) {
                core::logging::warn(
                    "Authenticated, but could not persist login profile: {}",
                    outcome.profile_error);
            } else {
                core::logging::info(
                    "Updated default provider to '{}'.",
                    outcome.selected_provider);
            }
            for (const auto& hint : outcome.result.hints) {
                core::logging::info("{}", hint);
            }
        } catch (const std::exception& e) {
            core::logging::error("Authentication failed: {}", e.what());
            return 1;
        }
        return 0;
    }

    if (core::landrun::LandrunSettings::instance().enabled()) {
        const core::workspace::SessionWorkspace workspace_view(
            core::workspace::Workspace::get_instance().snapshot());
        core::landrun::LandrunResult readiness;
        try {
            const auto policy = core::landrun::LandrunPolicyCompiler::compile(
                workspace_view,
                core::landrun::LandrunSettings::instance().mode());
            readiness = core::landrun::verify_landrun_readiness(policy);
        } catch (const std::exception& error) {
            readiness = {
                .success = false,
                .detail = std::format("policy compilation failed: {}", error.what()),
            };
        }
        if (!readiness.success) {
            const auto message = std::format(
                "Sandbox readiness check failed: {}. Filo will not start with a sandbox "
                "it cannot enforce. If Filo is already inside another sandbox, launch it "
                "outside that sandbox; use --sandbox off only if you explicitly accept "
                "unrestricted commands.", readiness.detail);
            core::logging::error("{}", message);
            std::cerr << message << '\n';
            return 1;
        }
    }
    if (sandbox_status) {
        std::cout << core::landrun::landrun_status_label() << '\n'
                  << core::landrun::landrun_status_detail() << '\n';
        return 0;
    }

    const bool has_stdin_input = stdin_has_data();
    const bool run_prompter_mode = !mcp_mode
        && !enable_api_gateway
        && exec::prompter::should_run(
            prompter_mode,
            prompt_opt->count() > 0,
            has_stdin_input,
            output_format_opt->count() > 0,
            input_format_opt->count() > 0,
            include_partial_messages,
            continue_last);

    if (run_prompter_mode) {
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
        opts.yolo = trust_resolution.trust_all_tools;
        opts.trusted_tools = trust_resolution.trusted_tool_names;
        if (resume_opt->count() > 0) {
            opts.resume_session = resume_session;
        }
        return exec::prompter::run(opts);
    }

    // --mcp stdio uses stdin/stdout as the MCP transport. Never print anything
    // to stdout when this mode is active: every byte on stdout is part of the
    // JSON-RPC stream and extra text will corrupt it.
    std::thread mcp_stdio_thread;
    if (mcp_stdio_mode) {
        mcp_stdio_thread = std::thread([]() {
            exec::mcp::run_server();
        });
    }

    std::thread http_daemon_thread;
    if (http_daemon_mode) {
        if (mcp_tcp_mode && enable_api_gateway) {
            core::logging::info("Starting Filo HTTP daemon (MCP + API gateway) on {}:{}...", host, port);
        } else if (mcp_tcp_mode) {
            core::logging::info("Starting Filo MCP daemon on {}:{}...", host, port);
        } else {
            core::logging::info("Starting Filo API gateway daemon on {}:{}...", host, port);
        }

        http_daemon_thread = std::thread([port, host, enable_api_gateway, mcp_tcp_mode]() {
            exec::daemon::run_server(port, host, enable_api_gateway, mcp_tcp_mode);
        });
    }

    if (!headless) {
        // Build resume options.
        tui::RunOptions run_opts;
        if (resume_opt->count() > 0) {
            // Empty value means "resume most recent".
            run_opts.resume_session_id = resume_session;
        }
        run_opts.startup_trust = {
            .trust_all_tools = trust_resolution.trust_all_tools,
            .session_allow_rules = trust_resolution.session_allow_rules,
        };
        run_opts.continue_last = continue_last;
        run_opts.landrun_mode = *parsed_sandbox_mode;
        run_opts.landrun_environment = {
            .excluded_paths = landrun_settings.excluded_paths(),
            .runtime_root = landrun_settings.runtime_root(),
            .host_tmpdir = landrun_settings.host_tmpdir(),
        };
        if (model_opt->count() > 0) {
            run_opts.startup_model = startup_model;
        }
        const auto run_result = tui::run(run_opts);

        if (mcp_stdio_mode) exec::mcp::stop_server();
        if (http_daemon_mode) exec::daemon::stop_server();
        
        // Close stdin to unblock the MCP server's std::getline, then join
        if (mcp_stdio_thread.joinable()) {
            std::fclose(stdin);
            mcp_stdio_thread.join();
        }
        if (http_daemon_thread.joinable()) http_daemon_thread.join();

        // Print the end-of-session summary report.
        core::session::SessionReport::print(
            core::budget::BudgetTracker::get_instance(),
            core::session::SessionStats::get_instance().snapshot(),
            run_result.session_id,
            run_result.session_file_path);
    } else {
        // Headless: block until all servers stop.
        if (mcp_stdio_thread.joinable()) mcp_stdio_thread.join();
        if (http_daemon_thread.joinable())   http_daemon_thread.join();
    }

    return 0;
}
