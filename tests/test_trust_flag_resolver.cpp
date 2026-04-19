#include <catch2/catch_test_macros.hpp>

#include "core/cli/TrustFlagResolver.hpp"
#include "core/permissions/PermissionSystem.hpp"

TEST_CASE("resolve_trust_flags enables trust-all from yolo flag",
          "[cli][trust-flags]") {
    const auto resolved = core::cli::resolve_trust_flags(true, {});

    REQUIRE(resolved.trust_all_tools);
    REQUIRE(resolved.trusted_tool_names.empty());
    REQUIRE(resolved.session_allow_rules.empty());
}

TEST_CASE("resolve_trust_flags canonicalizes aliases for TUI startup trust rules",
          "[cli][trust-flags]") {
    const auto resolved = core::cli::resolve_trust_flags(
        false,
        {" shell ", "write", "RUN_TERMINAL_COMMAND", "write_file"});

    REQUIRE_FALSE(resolved.trust_all_tools);
    REQUIRE(resolved.trusted_tool_names
            == std::vector<std::string>{"run_terminal_command", "write_file"});
    REQUIRE(resolved.session_allow_rules
            == std::vector<std::string>{"tool:run_terminal_command", "tool:write_file"});

    REQUIRE(core::permissions::session_allow_rule_matches(
        "tool:run_terminal_command",
        "run_terminal_command",
        R"({"command":"git status"})"));
    REQUIRE(core::permissions::session_allow_rule_matches(
        "tool:write_file",
        "write_file",
        R"({"file_path":"README.md","content":"ok"})"));
}

TEST_CASE("resolve_trust_flags supports wildcard trust tools",
          "[cli][trust-flags]") {
    const auto resolved = core::cli::resolve_trust_flags(
        false,
        {"read", "*", "grep"});

    REQUIRE(resolved.trust_all_tools);
    REQUIRE(resolved.trusted_tool_names
            == std::vector<std::string>{"read_file", "grep_search"});
    REQUIRE(resolved.session_allow_rules
            == std::vector<std::string>{"tool:read_file", "tool:grep_search"});
}

