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

core::hooks::StopDecision run_stop_with_config(
    const std::string& label,
    std::string hook_config,
    std::string payload_json = R"({"mutation_observed":true})") {
    const auto sandbox = make_temp_dir(label);
    const ScopedEnvVar xdg("XDG_CONFIG_HOME", (sandbox / "xdg").string());
    const auto project_dir = sandbox / "project";
    write_text(
        project_dir / ".filo" / "config.json",
        R"({"hooks":{"stop":[)" + std::move(hook_config) + R"(]}})");
    core::config::ConfigManager::get_instance().load(project_dir);

    const auto decision = core::hooks::run_stop(
        std::move(payload_json), make_hook_test_context(project_dir));

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

// A hook command that never reads stdin leaves the payload writer without a
// reader. When the payload exceeds the pipe capacity the writer cannot buffer it
// and is guaranteed to hit EPIPE, at which point bash reports
// "printf: write error: Broken pipe" on stderr. Because the shell session merges
// stderr into stdout, that diagnostic used to be appended to the hook's output.
namespace {

// Comfortably above the 64 KiB pipe capacity on Linux and macOS, while keeping
// the base64 payload env var below the per-string execve limit.
[[nodiscard]] std::string oversized_payload(const std::string& fields) {
    return "{" + fields + R"("filler":")" + std::string(80 * 1024, 'x') + "\"}";
}

} // namespace

TEST_CASE("PreToolUse JSON decisions survive a payload the hook never reads",
          "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_json_allow_large",
        R"("printf '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"allow\"}}'")",
        oversized_payload(R"("tool_name":"run_terminal_command","arguments":"{}",)"));

    // Broken-pipe noise on stdout would make this JSON unparseable, silently
    // degrading an explicit "allow" into "no decision".
    REQUIRE(decision.allowed);
    REQUIRE(decision.approved);
}

TEST_CASE("Stop hook output stays clean when the hook never reads the payload",
          "[hooks]") {
    const auto decision = run_stop_with_config(
        "filo_hook_stop_large",
        R"("printf retry; exit 2")",
        oversized_payload(R"("mutation_observed":true,)"));

    REQUIRE_FALSE(decision.complete);
    REQUIRE(decision.reason == "retry");
    REQUIRE(decision.followup_message == "retry");
}

TEST_CASE("PreToolUse hook exit code 2 blocks tool execution", "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_exit_2",
        R"("printf blocked; exit 2")",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE_FALSE(decision.allowed);
    REQUIRE_THAT(decision.reason, Catch::Matchers::ContainsSubstring("blocked"));
}

TEST_CASE("A broken advisory PreToolUse hook stays permissive", "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_broken_advisory",
        R"("filo-hook-binary-that-does-not-exist")",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE(decision.allowed);
    REQUIRE_FALSE(decision.approved);
}

TEST_CASE("A fail-closed PreToolUse hook denies when it cannot run", "[hooks]") {
    // A policy hook whose interpreter is missing must not silently authorize
    // every tool call; that would turn a security control into a no-op.
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_failclosed_missing",
        R"({"name":"policy","command":"filo-hook-binary-that-does-not-exist","fail_closed":true})",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE_FALSE(decision.allowed);
    REQUIRE_FALSE(decision.approved);
    REQUIRE_THAT(decision.reason,
                 Catch::Matchers::ContainsSubstring("policy"));
}

TEST_CASE("A fail-closed PreToolUse hook denies on a non-zero exit", "[hooks]") {
    const auto decision = run_pre_tool_use_with_config(
        "filo_hook_failclosed_exit",
        R"({"name":"policy","command":"printf crashed; exit 9","fail_closed":true})",
        R"({"tool_name":"run_terminal_command","arguments":"{}"})");

    REQUIRE_FALSE(decision.allowed);
    REQUIRE_THAT(decision.reason, Catch::Matchers::ContainsSubstring("9"));
}

TEST_CASE("Stop hook can request another turn with structured output", "[hooks]") {
    const auto decision = run_stop_with_config(
        "filo_hook_stop_followup",
        R"("printf '{\"decision\":\"block\",\"reason\":\"run the integration suite\"}'")");

    REQUIRE_FALSE(decision.complete);
    REQUIRE(decision.followup_message == "run the integration suite");
    REQUIRE_FALSE(decision.quality_gate_configured);
}

TEST_CASE("Successful quality stop hook provides authoritative evidence", "[hooks]") {
    const auto decision = run_stop_with_config(
        "filo_hook_quality_pass",
        R"({"name":"project-tests","command":"exit 0","quality_gate":true})");

    REQUIRE(decision.complete);
    REQUIRE(decision.quality_gate_configured);
    REQUIRE(decision.quality_gate_passed);
}

TEST_CASE("Quality stop hooks fail closed", "[hooks]") {
    const auto decision = run_stop_with_config(
        "filo_hook_quality_failure",
        R"({"name":"project-tests","command":"printf failed; exit 7","quality_gate":true})");

    REQUIRE_FALSE(decision.complete);
    REQUIRE(decision.quality_gate_configured);
    REQUIRE_FALSE(decision.quality_gate_passed);
    REQUIRE_THAT(decision.reason, Catch::Matchers::ContainsSubstring("status 7"));
}

TEST_CASE("Hook matcher limits lifecycle execution", "[hooks]") {
    const auto decision = run_stop_with_config(
        "filo_hook_matcher",
        R"({"command":"exit 9","quality_gate":true,"matcher":"mutation_observed\\\":false"})");

    REQUIRE(decision.complete);
    REQUIRE_FALSE(decision.quality_gate_configured);
}

TEST_CASE("Completion hook gate bounds repeated feedback", "[hooks]") {
    const auto sandbox = make_temp_dir("filo_hook_completion_gate");
    const ScopedEnvVar xdg("XDG_CONFIG_HOME", (sandbox / "xdg").string());
    const auto project_dir = sandbox / "project";
    write_text(
        project_dir / ".filo" / "config.json",
        R"({"hooks":{"stop":["printf retry; exit 2"]}})");
    core::config::ConfigManager::get_instance().load(project_dir);
    const auto context = make_hook_test_context(project_dir);
    core::hooks::CompletionGateState state;

    for (int attempt = 0; attempt < 3; ++attempt) {
        const auto result = core::hooks::evaluate_completion("{}", context, state);
        REQUIRE(result.action ==
                core::session::TurnCompletionAction::Continue);
        REQUIRE(result.message == "retry");
    }
    const auto exhausted = core::hooks::evaluate_completion("{}", context, state);
    REQUIRE(exhausted.action ==
            core::session::TurnCompletionAction::Fail);
    REQUIRE_THAT(exhausted.message,
                 Catch::Matchers::ContainsSubstring("loop limit"));

    reset_config(sandbox);
    fs::remove_all(sandbox);
}
