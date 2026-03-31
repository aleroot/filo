#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/commands/CommandExecutor.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <mutex>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <format>
#include <cstdlib>
#include <fstream>

using namespace core::commands;

TEST_CASE("CommandExecutor - Basic Routing", "[commands]") {
    CommandExecutor executor;

    // Use shared_ptr so detached threads (spawned by the '!' shell command) can safely
    // call back into these objects even after the enclosing test scope exits.
    auto mock_history   = std::make_shared<std::string>();
    auto history_mutex  = std::make_shared<std::mutex>();
    auto screen_cleared = std::make_shared<bool>(false);
    auto input_cleared  = std::make_shared<bool>(false);
    auto quit_called    = std::make_shared<bool>(false);
    auto switch_target  = std::make_shared<std::string>();
    auto picker_opened  = std::make_shared<bool>(false);
    auto settings_picker_opened = std::make_shared<bool>(false);
    auto yolo_enabled   = std::make_shared<bool>(false);
    auto fork_result    = std::make_shared<std::string>("Forked session abc123 into def456.");

    CommandContext ctx{
        .text = "",
        .clear_input_fn = [input_cleared]() { *input_cleared = true; },
        .append_history_fn = [mock_history, history_mutex](const std::string& str) {
            std::lock_guard<std::mutex> lock(*history_mutex);
            *mock_history += str;
        },
        .agent = nullptr,
        .clear_screen_fn = [screen_cleared]() { *screen_cleared = true; },
        .quit_fn = [quit_called]() { *quit_called = true; },
        .model_status_fn = []() {
            return std::string("Active: grok (grok-code-fast-1)\n        Available: grok (grok-code-fast-1), grok-mini (grok-3-mini), openai (gpt-4o)");
        },
        .switch_model_fn = [switch_target](std::string_view name) {
            *switch_target = std::string(name);
            return std::string("Switched to ") + std::string(name);
        },
        .open_model_picker_fn = [picker_opened]() {
            *picker_opened = true;
            return false;
        },
        .settings_status_fn = []() {
            return std::string(
                "Effective settings\n"
                "Start mode: BUILD\n"
                "Approval mode: prompt\n"
                "Default router policy: smart-code");
        },
        .yolo_mode_enabled_fn = [yolo_enabled]() {
            return *yolo_enabled;
        },
        .set_yolo_mode_enabled_fn = [yolo_enabled](bool enabled) {
            *yolo_enabled = enabled;
        },
        .fork_session_fn = [fork_result]() {
            return *fork_result;
        },
        .latest_assistant_output_fn = []() { return std::string(); },
        .copy_to_clipboard_fn = [](std::string_view) {
            return std::optional<std::string>{};
        },
    };

    SECTION("Unknown commands fall through") {
        ctx.text = "Hello world";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == false);
        
        ctx.text = "/unknown_command";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == false);
    }

    SECTION("/clear command") {
        ctx.text = "/clear";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*screen_cleared == true);
        REQUIRE(*input_cleared == true);
    }

    SECTION("/help command") {
        ctx.text = "/help";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("[Filo Commands]"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("/auth [provider]"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("/settings"));

        // Test alias
        *mock_history = "";
        ctx.text = "/?";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("[Filo Commands]"));
    }

    SECTION("/auth command without provider shows usage") {
        *mock_history = "";
        ctx.text = "/auth";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Usage: /auth [provider]"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Available providers:"));
    }

    SECTION("/auth command without provider opens menu and supports cancel in interactive mode") {
        *mock_history = "";
        ctx.suspend_tui_fn = [](std::function<void()> task) { task(); };
        ctx.text = "/auth";

        std::istringstream fake_input("\n");
        std::ostringstream fake_output;
        auto* old_buf = std::cin.rdbuf(fake_input.rdbuf());
        auto* old_out = std::cout.rdbuf(fake_output.rdbuf());
        const bool handled = executor.try_execute(ctx.text, ctx);
        std::cin.rdbuf(old_buf);
        std::cout.rdbuf(old_out);

        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Authentication cancelled"));
    }

    SECTION("/auth without provider opens TUI picker when open_provider_picker_fn is set") {
        *mock_history = "";
        ctx.suspend_tui_fn = [](std::function<void()> task) { task(); };

        auto picker_providers = std::make_shared<std::vector<std::string>>();
        std::function<void(std::optional<std::string>)> captured_on_select;
        ctx.open_provider_picker_fn = [picker_providers, &captured_on_select](
            std::vector<std::string> providers,
            std::function<void(std::optional<std::string>)> on_select) {
            *picker_providers = providers;
            captured_on_select = on_select;
        };

        ctx.text = "/auth";
        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        // Picker should have been invoked (providers list populated)
        REQUIRE_FALSE(picker_providers->empty());
        // History untouched until user acts
        REQUIRE(mock_history->find("Authentication cancelled") == std::string::npos);

        // Simulate the user pressing Escape (nullopt = cancel)
        REQUIRE(captured_on_select);
        captured_on_select(std::nullopt);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Authentication cancelled"));
    }

    SECTION("/auth command with provider but no interactive TUI shows hint") {
        *mock_history = "";
        ctx.suspend_tui_fn = {};
        ctx.text = "/auth google";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Interactive authentication not supported"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("filo --login <provider>"));
    }

    SECTION("! shell command") {
        ctx.text = "!echo test";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);

        // The shell command execution is highly asynchronous (it spawns a detached thread).
        // Verify only the synchronous part: the original input is appended to history before
        // the thread spins up. Take a locked snapshot so TSAN doesn't flag the concurrent write
        // from the detached thread against our read here.
        std::string snapshot;
        {
            std::lock_guard<std::mutex> lock(*history_mutex);
            snapshot = *mock_history;
        }
        REQUIRE_THAT(snapshot, Catch::Matchers::ContainsSubstring("!echo test"));
    }

    SECTION("Command matching with aliases") {
        ctx.text = "/cls";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);

        *screen_cleared = false;
        // /exit is an alias for /quit; it will call exit(0)
        // We will skip /exit in tests since it calls exit(0) now instead of ox::Application::quit(0)

        ctx.text = "/h";
        *mock_history = "";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("[Filo Commands]"));
    }

    SECTION("Whitespace trimming in command matching") {
        *screen_cleared = false;
        ctx.text = "  /clear  ";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*screen_cleared == true);
    }

    SECTION("/model command") {
        *picker_opened = false;
        ctx.text = "/model";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*picker_opened == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Active: grok"));
    }

    SECTION("/model opens picker when callback handles it") {
        *picker_opened = false;
        ctx.open_model_picker_fn = [picker_opened]() {
            *picker_opened = true;
            return true;
        };

        ctx.text = "/model";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*picker_opened == true);
        REQUIRE(mock_history->find("Active: grok") == std::string::npos);
    }

    SECTION("/model switches provider presets") {
        ctx.text = "/model grok-mini";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*switch_target == "grok-mini");
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Switched to grok-mini"));
    }

    SECTION("/model forwards provider plus model selectors") {
        ctx.text = "/model claude opus";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*switch_target == "claude opus");
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Switched to claude opus"));
    }

    SECTION("/settings opens picker when callback handles it") {
        *settings_picker_opened = false;
        ctx.open_settings_picker_fn = [settings_picker_opened]() {
            *settings_picker_opened = true;
            return true;
        };
        *mock_history = "";

        ctx.text = "/settings";
        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*settings_picker_opened == true);
        REQUIRE(mock_history->empty());
    }

    SECTION("/settings falls back to status text when no picker is available") {
        *mock_history = "";
        ctx.open_settings_picker_fn = {};

        ctx.text = "/settings";
        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Effective settings"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Start mode: BUILD"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Default router policy: smart-code"));
    }

    SECTION("/yolo toggles approval mode") {
        *mock_history = "";
        *yolo_enabled = false;

        ctx.text = "/yolo";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*yolo_enabled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Approval mode set to YOLO"));

        *mock_history = "";
        ctx.text = "/yolo";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*yolo_enabled == false);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Approval mode set to PROMPT"));
    }

    SECTION("/yolo explicit on/off/status and invalid argument") {
        *mock_history = "";
        *yolo_enabled = false;

        ctx.text = "/yolo on";
        bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*yolo_enabled == true);

        *mock_history = "";
        ctx.text = "/yolo status";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Approval mode: YOLO"));

        *mock_history = "";
        ctx.text = "/yolo off";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(*yolo_enabled == false);

        *mock_history = "";
        ctx.text = "/yolo maybe";
        handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Unknown /yolo option"));
    }

    SECTION("/copy copies the latest assistant output") {
        std::string copied_text;
        ctx.latest_assistant_output_fn = []() {
            return std::string("Latest assistant output");
        };
        ctx.copy_to_clipboard_fn = [&copied_text](std::string_view text) {
            copied_text = std::string(text);
            return std::optional<std::string>{};
        };

        ctx.text = "/copy";
        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE(copied_text == "Latest assistant output");
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Copied the latest assistant response"));
    }

    SECTION("/copy reports when there is nothing to copy") {
        ctx.latest_assistant_output_fn = []() { return std::string(); };
        ctx.text = "/copy";

        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Nothing to copy yet"));
    }

    SECTION("/review requires an active agent") {
        *mock_history = "";
        ctx.agent = nullptr;
        ctx.text = "/review";

        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("requires an active agent"));
    }

    SECTION("/export requires an active agent") {
        *mock_history = "";
        ctx.agent = nullptr;
        ctx.text = "/export";

        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("unavailable without an active agent"));
    }

    SECTION("/fork delegates to session fork callback") {
        *mock_history = "";
        *fork_result = "Forked session old11111 into new22222.";
        ctx.text = "/fork";

        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Forked session old11111 into new22222"));
    }

    SECTION("/init scaffolds .filo/config.json in the working directory") {
        namespace fs = std::filesystem;
        struct CwdGuard {
            fs::path old;
            ~CwdGuard() {
                std::error_code ec;
                fs::current_path(old, ec);
            }
        } guard{fs::current_path()};

        const fs::path temp_dir =
            fs::temp_directory_path() / std::format("filo-test-init-{}", std::rand());
        std::error_code ec;
        fs::remove_all(temp_dir, ec);
        fs::create_directories(temp_dir, ec);
        REQUIRE_FALSE(ec);

        fs::current_path(temp_dir);
        *mock_history = "";
        ctx.text = "/init";
        const bool handled = executor.try_execute(ctx.text, ctx);

        REQUIRE(handled == true);
        REQUIRE(fs::exists(temp_dir / ".filo" / "config.json"));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Project scaffold ready"));
        std::ifstream in(temp_dir / ".filo" / "config.json");
        REQUIRE(in.good());
        std::stringstream cfg;
        cfg << in.rdbuf();
        REQUIRE_THAT(cfg.str(), Catch::Matchers::ContainsSubstring("\"default_provider\": \"grok\""));

        fs::remove_all(temp_dir, ec);
    }

    SECTION("/init accepts an optional provider argument") {
        namespace fs = std::filesystem;
        struct CwdGuard {
            fs::path old;
            ~CwdGuard() {
                std::error_code ec;
                fs::current_path(old, ec);
            }
        } guard{fs::current_path()};

        const fs::path temp_dir =
            fs::temp_directory_path() / std::format("filo-test-init-provider-{}", std::rand());
        std::error_code ec;
        fs::remove_all(temp_dir, ec);
        fs::create_directories(temp_dir, ec);
        REQUIRE_FALSE(ec);

        fs::current_path(temp_dir);
        *mock_history = "";
        ctx.text = "/init openAi";
        const bool handled = executor.try_execute(ctx.text, ctx);

        REQUIRE(handled == true);
        REQUIRE(fs::exists(temp_dir / ".filo" / "config.json"));
        std::ifstream in(temp_dir / ".filo" / "config.json");
        REQUIRE(in.good());
        std::stringstream cfg;
        cfg << in.rdbuf();
        REQUIRE_THAT(cfg.str(), Catch::Matchers::ContainsSubstring("\"default_provider\": \"openai\""));
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("default_provider: openai"));

        fs::remove_all(temp_dir, ec);
    }

    SECTION("/init rejects invalid provider tokens") {
        *mock_history = "";
        ctx.text = "/init openai!";
        const bool handled = executor.try_execute(ctx.text, ctx);
        REQUIRE(handled == true);
        REQUIRE_THAT(*mock_history, Catch::Matchers::ContainsSubstring("Invalid provider"));
    }

    SECTION("Command registry exposes slash commands for autocomplete") {
        const auto commands = executor.describe_commands();
        REQUIRE_FALSE(commands.empty());

        const auto help_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/help";
        });
        REQUIRE(help_it != commands.end());
        REQUIRE(help_it->aliases == std::vector<std::string>{"/?", "/h"});
        REQUIRE_FALSE(help_it->accepts_arguments);

        const auto model_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/model";
        });
        REQUIRE(model_it != commands.end());
        REQUIRE(model_it->accepts_arguments);

        const auto settings_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/settings";
        });
        REQUIRE(settings_it != commands.end());
        REQUIRE_FALSE(settings_it->accepts_arguments);

        const auto copy_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/copy";
        });
        REQUIRE(copy_it != commands.end());
        REQUIRE_FALSE(copy_it->accepts_arguments);

        const auto yolo_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/yolo";
        });
        REQUIRE(yolo_it != commands.end());
        REQUIRE(yolo_it->accepts_arguments);

        const auto auth_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/auth";
        });
        REQUIRE(auth_it != commands.end());
        REQUIRE(auth_it->accepts_arguments);

        const auto review_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/review";
        });
        REQUIRE(review_it != commands.end());
        REQUIRE(review_it->accepts_arguments);

        const auto export_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/export";
        });
        REQUIRE(export_it != commands.end());
        REQUIRE(export_it->accepts_arguments);

        const auto fork_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/fork";
        });
        REQUIRE(fork_it != commands.end());
        REQUIRE_FALSE(fork_it->accepts_arguments);

        const auto init_it = std::find_if(commands.begin(), commands.end(), [](const CommandDescriptor& cmd) {
            return cmd.name == "/init";
        });
        REQUIRE(init_it != commands.end());
        REQUIRE(init_it->accepts_arguments);
    }
}

TEST_CASE("Command completion helpers detect and replace the active slash command", "[commands]") {
    SECTION("Active command is detected while typing") {
        const auto active = find_active_command("/he", 3);
        REQUIRE(active.has_value());
        REQUIRE(active->token == "/he");
        REQUIRE(active->replace_begin == 0);
        REQUIRE(active->replace_end == 3);
    }

    SECTION("Leading whitespace is ignored for command completion") {
        const auto active = find_active_command("   /mo", 6);
        REQUIRE(active.has_value());
        REQUIRE(active->token == "/mo");
        REQUIRE(active->replace_begin == 3);
    }

    SECTION("Command completion stops once the cursor moves into arguments") {
        const auto active = find_active_command("/model grok", std::string_view{"/model grok"}.size());
        REQUIRE_FALSE(active.has_value());
    }

    SECTION("Command completion replaces only the slash token") {
        const auto active = find_active_command("/h", 2);
        REQUIRE(active.has_value());

        const auto completed = apply_command_completion("/h please", *active, "/help");
        REQUIRE(completed.text == "/help please");
        REQUIRE(completed.cursor == 5);
    }
}
