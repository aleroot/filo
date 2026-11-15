#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/CodexClientIdentity.hpp"
#include "core/utils/ClientPlatform.hpp"

TEST_CASE("The client platform names the system it runs on", "[utils][platform]") {
    const auto& platform = core::utils::client_platform();
    CHECK_FALSE(platform.os.empty());
    CHECK_FALSE(platform.version.empty());
    CHECK_FALSE(platform.architecture.empty());
    // The architecture is known at build time, so it is never "unknown".
    CHECK(platform.architecture != "unknown");
}

TEST_CASE("The Codex User-Agent names the platform, as the CLI does",
          "[utils][platform][codex]") {
    // codex-rs login/src/auth/default_client.rs get_codex_user_agent:
    // "{originator}/{version} ({os} {os_version}; {architecture}) {terminal}".
    const auto& agent = core::auth::codex_cli::user_agent();
    const auto& platform = core::utils::client_platform();
    CHECK_THAT(agent, Catch::Matchers::StartsWith(
                           std::string(core::auth::codex_cli::kOriginator) + "/"
                           + std::string(core::auth::codex_cli::kClientVersion) + " ("));
    CHECK_THAT(agent, Catch::Matchers::ContainsSubstring(platform.os));
    CHECK_THAT(agent, Catch::Matchers::ContainsSubstring(platform.version));
    CHECK_THAT(agent, Catch::Matchers::ContainsSubstring(platform.architecture));
}
