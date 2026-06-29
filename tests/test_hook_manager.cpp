#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/context/SessionContext.hpp"
#include "core/hooks/HookManager.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace fs = std::filesystem;

namespace {

struct ScopedEnvVar {
    std::string name;
    std::optional<std::string> old_value;

    ScopedEnvVar(std::string env_name, const std::string& value)
        : name(std::move(env_name)) {
        if (const char* existing = std::getenv(name.c_str())) {
            old_value = std::string(existing);
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (old_value.has_value()) {
            setenv(name.c_str(), old_value->c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

fs::path make_temp_dir(const std::string& label) {
    const auto path = fs::temp_directory_path()
        / (label + "_" + std::to_string(static_cast<long long>(std::rand())));
    fs::create_directories(path);
    return path;
}

void write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
}

core::context::SessionContext make_hook_test_context(const fs::path& project_dir) {
    return core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = project_dir,
            .additional = {},
            .enforce = false,
        },
        core::context::SessionTransport::cli,
        "hook-test");
}

void load_hook_config(const fs::path& project_dir, std::string hook_command) {
    write_text(
        project_dir / ".filo" / "config.json",
        R"({"hooks":{"pre_tool_use":[)"
        + std::move(hook_command)
        + R"(]}})");
    core::config::ConfigManager::get_instance().load(project_dir);
}

void reset_config(const fs::path& sandbox) {
    core::config::ConfigManager::get_instance().load(sandbox / "empty_project");
}

core::hooks::HookDecision run_pre_tool_use_with_config(
    const std::string& label,
    std::string hook_command,
    std::string payload_json) {
    const auto sandbox = make_temp_dir(label);
    const ScopedEnvVar xdg("XDG_CONFIG_HOME", (sandbox / "xdg").string());
    const auto project_dir = sandbox / "project";
    load_hook_config(project_dir, std::move(hook_command));

    const auto decision = core::hooks::run_pre_tool_use(
        std::move(payload_json),
        make_hook_test_context(project_dir));

    reset_config(sandbox);
    fs::remove_all(sandbox);
    return decision;
}

} // namespace

TEST_CASE("PreToolUse hook can deny via Claude-style JSON", "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_json_deny",
        R"("grep -q tool_name && printf '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"policy blocked\"}}'")",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE_FALSE(decision.allowed);
    REQUIRE_FALSE(decision.approved);
    REQUIRE(decision.reason == "policy blocked");
}

TEST_CASE("PreToolUse hook can approve via Claude-style JSON", "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_json_allow",
        R"("printf '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"allow\"}}'")",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE(decision.allowed);
    REQUIRE(decision.approved);
}

TEST_CASE("PreToolUse hook exit code 2 blocks tool execution", "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_exit_2",
        R"("printf blocked; exit 2")",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE_FALSE(decision.allowed);
    REQUIRE_THAT(decision.reason, Catch::Matchers::ContainsSubstring("blocked"));
}
