#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/CodexClientIdentity.hpp"
#include "core/utils/ClientPlatform.hpp"
#include "core/utils/HostTarget.hpp"

#include <format>

#include <cstdlib>
#include <optional>
#include <string>

#if !defined(_WIN32)
namespace {

/// One environment variable, restored when the guard goes away.
class ScopedEnv {
public:
    ScopedEnv(std::string name, const char* value) : name_(std::move(name)) {
        if (const char* previous = std::getenv(name_.c_str())) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value) {
            ::setenv(name_.c_str(), value, 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    ~ScopedEnv() {
        if (had_previous_) {
            ::setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

/// The terminal variables the detection reads, cleared for the duration.
class ClearedTerminalEnv {
public:
    ClearedTerminalEnv()
        : term_("TERM", nullptr)
        , program_("TERM_PROGRAM", nullptr)
        , program_version_("TERM_PROGRAM_VERSION", nullptr)
        , ghostty_("GHOSTTY_RESOURCES_DIR", nullptr)
        , wezterm_("WEZTERM_VERSION", nullptr)
        , iterm_("ITERM_SESSION_ID", nullptr)
        , iterm_profile_("ITERM_PROFILE", nullptr)
        , kitty_("KITTY_WINDOW_ID", nullptr)
        , alacritty_("ALACRITTY_SOCKET", nullptr)
        , konsole_("KONSOLE_VERSION", nullptr)
        , gnome_("GNOME_TERMINAL_SCREEN", nullptr)
        , vte_("VTE_VERSION", nullptr)
        , windows_("WT_SESSION", nullptr) {}

private:
    ScopedEnv term_;
    ScopedEnv program_;
    ScopedEnv program_version_;
    ScopedEnv ghostty_;
    ScopedEnv wezterm_;
    ScopedEnv iterm_;
    ScopedEnv iterm_profile_;
    ScopedEnv kitty_;
    ScopedEnv alacritty_;
    ScopedEnv konsole_;
    ScopedEnv gnome_;
    ScopedEnv vte_;
    ScopedEnv windows_;
};

} // namespace
#endif

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

#if !defined(_WIN32)
TEST_CASE("The terminal names itself, as the CLI names it",
          "[utils][platform][terminal]") {
    // codex-rs terminal-detection user_agent_token: the program the terminal
    // sets wins, with its version.
    const ClearedTerminalEnv cleared;
    {
        const ScopedEnv program("TERM_PROGRAM", "Apple_Terminal");
        const ScopedEnv version("TERM_PROGRAM_VERSION", "400");
        CHECK(core::utils::terminal_token() == "Apple_Terminal/400");
    }
    {
        const ScopedEnv program("TERM_PROGRAM", "iTerm.app");
        CHECK(core::utils::terminal_token() == "iTerm.app");
    }
}

TEST_CASE("A terminal that names itself wins over the TERM string",
          "[utils][platform][terminal]") {
    const ClearedTerminalEnv cleared;
    {
        const ScopedEnv term("TERM", "xterm-256color");
        const ScopedEnv wezterm("WEZTERM_VERSION", "20240101");
        CHECK(core::utils::terminal_token() == "WezTerm/20240101");
    }
    {
        const ScopedEnv term("TERM", "xterm-kitty");
        CHECK(core::utils::terminal_token() == "kitty");
    }
    {
        const ScopedEnv term("TERM", "screen-256color");
        const ScopedEnv windows("WT_SESSION", "a-guid");
        CHECK(core::utils::terminal_token() == "WindowsTerminal");
    }
}

TEST_CASE("The TERM string names the terminal when nothing else does",
          "[utils][platform][terminal]") {
    const ClearedTerminalEnv cleared;
    {
        const ScopedEnv term("TERM", "xterm-256color");
        CHECK(core::utils::terminal_token() == "xterm-256color");
    }
    // A multiplexer is not the terminal: the CLI names the terminal under it.
    {
        const ScopedEnv term("TERM", "screen");
        const ScopedEnv program("TERM_PROGRAM", "tmux");
        const ScopedEnv version("TERM_PROGRAM_VERSION", "3.4");
        CHECK(core::utils::terminal_token() == "screen");
    }
}

TEST_CASE("An environment that names no terminal says so",
          "[utils][platform][terminal]") {
    const ClearedTerminalEnv cleared;
    CHECK(core::utils::terminal_token() == "unknown");
}

TEST_CASE("The terminal token is safe to send as a header value",
          "[utils][platform][terminal]") {
    const ClearedTerminalEnv cleared;
    // codex-rs terminal-detection sanitize_header_value: a character a header
    // value may not carry becomes an underscore.
    const ScopedEnv term("TERM", "xterm 256color");
    CHECK(core::utils::terminal_token() == "xterm_256color");
}
#endif

TEST_CASE("A host target is spelled the way each toolchain spells it",
          "[utils][platform][host-target]") {
    using core::utils::Architecture;
    using core::utils::OperatingSystem;

    // Go: runtime.GOOS / runtime.GOARCH.
    CHECK(std::format("{:go}", OperatingSystem::Darwin) == "darwin");
    CHECK(std::format("{:go}", OperatingSystem::Linux) == "linux");
    CHECK(std::format("{:go}", OperatingSystem::Windows) == "windows");
    CHECK(std::format("{:go}", OperatingSystem::FreeBsd) == "freebsd");
    CHECK(std::format("{:go}", Architecture::Arm64) == "arm64");
    CHECK(std::format("{:go}", Architecture::X86_64) == "amd64");
    CHECK(std::format("{:go}", Architecture::Arm) == "arm");
    CHECK(std::format("{:go}", Architecture::X86) == "386");

    // Rust: std::env::consts::OS / ARCH.
    CHECK(std::format("{:rust}", OperatingSystem::Darwin) == "macos");
    CHECK(std::format("{:rust}", OperatingSystem::Linux) == "linux");
    CHECK(std::format("{:rust}", Architecture::Arm64) == "aarch64");
    CHECK(std::format("{:rust}", Architecture::X86_64) == "x86_64");
    CHECK(std::format("{:rust}", Architecture::X86) == "x86");

    CHECK(std::format("{:go}/{:rust}", OperatingSystem::Unknown, Architecture::Unknown)
          == "unknown/unknown");
}

TEST_CASE("A host target formatted without a naming is rejected",
          "[utils][platform][host-target]") {
    // A constant format string fails to compile; a runtime one throws.
    const auto os = core::utils::OperatingSystem::Linux;
    CHECK_THROWS_AS(std::vformat("{}", std::make_format_args(os)), std::format_error);
    CHECK_THROWS_AS(std::vformat("{:GO}", std::make_format_args(os)), std::format_error);
}

TEST_CASE("The build target is known and agrees with the client platform",
          "[utils][platform][host-target]") {
    constexpr auto host = core::utils::build_host_target();
    STATIC_REQUIRE(host.os != core::utils::OperatingSystem::Unknown);
    STATIC_REQUIRE(host.arch != core::utils::Architecture::Unknown);
    CHECK(core::utils::client_platform().architecture == std::format("{:rust}", host.arch));
}
