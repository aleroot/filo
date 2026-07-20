#include <catch2/catch_test_macros.hpp>

#include "core/landrun/LandrunDriverFactory.hpp"
#include "core/landrun/LandrunEnvironment.hpp"
#include "core/landrun/LandrunLaunch.hpp"
#include "core/landrun/LandrunPath.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"
#include "core/landrun/LandrunRuntime.hpp"
#include "core/landrun/LandrunSettings.hpp"
#include "core/workspace/SensitivePathPolicy.hpp"
#include "core/workspace/PathVisibility.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

TEST_CASE("landrun path containment is component-safe and resolves symlink prefixes",
          "[landrun]") {
    namespace landrun = core::landrun;
    const auto base = std::filesystem::temp_directory_path()
        / ("filo-path-policy-" + std::to_string(::getpid()));
    const auto root = base / "workspace";
    const auto outside = base / "workspace-neighbor";
    struct RemoveFixture {
        std::filesystem::path root;
        ~RemoveFixture() {
            std::error_code ignored;
            std::filesystem::remove_all(root, ignored);
        }
    } remove_fixture{base};

    std::error_code ec;
    std::filesystem::create_directories(root / "src", ec);
    REQUIRE_FALSE(ec);
    std::filesystem::create_directories(outside, ec);
    REQUIRE_FALSE(ec);
    std::filesystem::create_directory_symlink(outside, root / "escape", ec);
    REQUIRE_FALSE(ec);

    REQUIRE(landrun::is_landrun_path_within(root, root));
    REQUIRE(landrun::is_landrun_path_within(root, root / "src" / "future.cpp"));
    REQUIRE(landrun::is_landrun_path_within(
        root, root / "src" / ".." / "generated"));
    REQUIRE_FALSE(landrun::is_landrun_path_within(root, outside));
    REQUIRE_FALSE(landrun::is_landrun_path_within(
        root, root / "escape" / "secret"));
    REQUIRE_FALSE(landrun::is_landrun_path_within({}, root));
}

TEST_CASE("landrun startup configuration cannot change after it is frozen",
          "[landrun]") {
    // Exercise the irreversible production lifecycle in a child so the
    // randomized unit-test process remains configurable for other cases.
    (void)core::landrun::LandrunSettings::instance();
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        auto& settings = core::landrun::LandrunSettings::instance();
        settings.configure_startup({
            .mode = core::landrun::LandrunMode::workspace_write,
        });
        settings.freeze_startup_configuration();
        try {
            settings.configure_startup({.mode = core::landrun::LandrunMode::off});
        } catch (const std::logic_error&) {
            ::_exit(0);
        } catch (...) {
            ::_exit(2);
        }
        ::_exit(1);
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}

TEST_CASE("landrun policy compiler confines writes to workspace and scratch roots",
          "[landrun]") {
    namespace landrun = core::landrun;
    const auto primary = std::filesystem::current_path();
    const auto additional = primary / "tests";
    const core::workspace::SessionWorkspace workspace{
        core::workspace::WorkspaceSnapshot{
            .primary = primary,
            .additional = {additional},
            .enforce = true,
        }};

    const auto policy = landrun::LandrunPolicyCompiler::compile(
        workspace, landrun::LandrunMode::workspace_write);
    REQUIRE(policy.enabled());
    REQUIRE_FALSE(policy.allow_network);
    REQUIRE(std::ranges::find(policy.writable_roots,
                              landrun::normalize_landrun_path(primary))
            != policy.writable_roots.end());
    REQUIRE(std::ranges::find(policy.writable_roots,
                              landrun::normalize_landrun_path(additional))
            != policy.writable_roots.end());
    REQUIRE(std::ranges::find(policy.writable_roots,
                              landrun::normalize_landrun_path("/tmp"))
            != policy.writable_roots.end());
    const auto host_tmpdir = landrun::LandrunSettings::instance().host_tmpdir();
    if (!host_tmpdir.empty()) {
        REQUIRE(std::ranges::find(policy.writable_roots, host_tmpdir)
                != policy.writable_roots.end());
    }
}

TEST_CASE("secure landrun environment strips credentials and selects allowed temp",
          "[landrun]") {
    core::landrun::LandrunRuntime runtime;
    char path[] = "PATH=/usr/bin:/bin";
    char language[] = "LANG=C";
    char api_key[] = "OPENAI_API_KEY=must-not-leak";
    char ssh_socket[] = "SSH_AUTH_SOCK=/tmp/agent.sock";
    char* inherited[] = {path, language, api_key, ssh_socket, nullptr};
    const core::landrun::LandrunPolicy policy{
        .mode = core::landrun::LandrunMode::workspace_write,
    };
    const auto environment = core::landrun::build_landrun_environment(
        policy, inherited);

    REQUIRE(std::ranges::find(
                environment, std::string{"OPENAI_API_KEY=must-not-leak"})
            == environment.end());
    REQUIRE(std::ranges::find(
                environment, std::string{"SSH_AUTH_SOCK=/tmp/agent.sock"})
            == environment.end());
    REQUIRE(std::ranges::any_of(environment, [&](const std::string& entry) {
        return entry == "HOME=" + (runtime.root() / "home").string();
    }));
    REQUIRE(std::ranges::any_of(environment, [&](const std::string& entry) {
        return entry == "TMPDIR="
            + core::landrun::LandrunSettings::instance().effective_tmpdir().string();
    }));
}

TEST_CASE("landrun generic exclusions remove roots and protect nested paths", "[landrun]") {
    namespace landrun = core::landrun;
    auto& settings = landrun::LandrunSettings::instance();
    const auto previous = settings.startup_configuration();
    struct RestoreConfiguration {
        landrun::LandrunSettings& settings;
        landrun::LandrunStartupConfiguration configuration;
        ~RestoreConfiguration() {
            settings.configure_startup(std::move(configuration));
        }
    } restore{settings, previous};

    const auto root = std::filesystem::current_path();
    std::vector<std::filesystem::path> exclusions{"/tmp", root / "src"};
    const auto host_tmpdir = settings.host_tmpdir();
    if (!host_tmpdir.empty()) exclusions.push_back(host_tmpdir);
    settings.configure_startup({
        .mode = previous.mode,
        .excluded_paths = std::move(exclusions),
    });
    const core::workspace::SessionWorkspace workspace{
        core::workspace::WorkspaceSnapshot{.primary = root, .enforce = true}};
    const auto policy = landrun::LandrunPolicyCompiler::compile(
        workspace, landrun::LandrunMode::workspace_write);
    REQUIRE(std::ranges::find(policy.writable_roots,
                              landrun::normalize_landrun_path("/tmp"))
            == policy.writable_roots.end());
    if (!host_tmpdir.empty() && host_tmpdir != landrun::normalize_landrun_path("/tmp")) {
        REQUIRE(std::ranges::find(policy.writable_roots, host_tmpdir)
                == policy.writable_roots.end());
    }
    REQUIRE(std::ranges::find(
                policy.protected_read_paths,
                landrun::normalize_landrun_path(root / "src"))
            != policy.protected_read_paths.end());
    REQUIRE(std::ranges::find(
                policy.protected_write_paths,
                landrun::normalize_landrun_path(root / "src"))
            != policy.protected_write_paths.end());
}

TEST_CASE("automatic sensitive paths do not generate an unsupported Landlock subtraction",
          "[landrun]") {
    namespace landrun = core::landrun;
    const auto root = std::filesystem::temp_directory_path()
        / ("filo-policy-" + std::to_string(::getpid()));
    struct RemoveFixture {
        std::filesystem::path root;
        ~RemoveFixture() {
            std::error_code ignored;
            std::filesystem::remove_all(root, ignored);
        }
    } remove_fixture{root};
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    REQUIRE_FALSE(ec);
    {
        std::ofstream secret(root / ".env");
        secret << "TOKEN=fixture\n";
    }
    const core::workspace::SessionWorkspace workspace{
        core::workspace::WorkspaceSnapshot{.primary = root, .enforce = true}};
    const auto policy = landrun::LandrunPolicyCompiler::compile(
        workspace, landrun::LandrunMode::workspace_write);
    REQUIRE(std::ranges::find(
                policy.protected_read_paths,
                landrun::normalize_landrun_path(root / ".env"))
            == policy.protected_read_paths.end());
    REQUIRE(std::ranges::find(
                policy.protected_write_paths,
                landrun::normalize_landrun_path(root / ".git/hooks"))
            == policy.protected_write_paths.end());
}

TEST_CASE("secure data policy recognizes common credential paths", "[landrun]") {
    REQUIRE(core::workspace::is_sensitive_agent_path("project/.env.production"));
    REQUIRE(core::workspace::is_sensitive_agent_path("project/certs/client.pem"));
    REQUIRE(core::workspace::is_sensitive_agent_path("project/.ssh/id_ed25519"));
    REQUIRE_FALSE(core::workspace::is_sensitive_agent_path("project/src/main.cpp"));
}

TEST_CASE("secure path visibility cannot be weakened by workspace ignore rules",
          "[landrun]") {
    auto& settings = core::landrun::LandrunSettings::instance();
    const auto previous = settings.startup_configuration();
    struct RestoreConfiguration {
        core::landrun::LandrunSettings& settings;
        core::landrun::LandrunStartupConfiguration configuration;
        ~RestoreConfiguration() {
            settings.configure_startup(std::move(configuration));
        }
    } restore{settings, previous};
    settings.configure_startup({
        .mode = core::landrun::LandrunMode::workspace_write,
        .excluded_paths = previous.excluded_paths,
    });

    const core::workspace::AgentIgnorePathVisibilityFactory factory;
    const auto visibility = factory.for_root(std::filesystem::current_path());
    REQUIRE(visibility.is_hidden(std::filesystem::current_path() / ".env"));
    REQUIRE_FALSE(visibility.is_hidden(std::filesystem::current_path() / "README.md"));
}

TEST_CASE("disabled landrun policy prepares the shell directly", "[landrun]") {
    const auto launch = core::landrun::prepare_shell_launch({});
    REQUIRE((launch.executable == "/bin/bash" || launch.executable == "/bin/sh"));
    REQUIRE_FALSE(launch.arguments.empty());
    REQUIRE((launch.arguments.front() == "bash" || launch.arguments.front() == "sh"));
}

TEST_CASE("enabled landrun launch transports roots as exact argv entries", "[landrun]") {
    core::landrun::LandrunPolicy policy{
        .mode = core::landrun::LandrunMode::workspace_write,
        .writable_roots = {"/tmp/a root with spaces"},
    };
    const auto launch = core::landrun::prepare_shell_launch(policy);
    REQUIRE_FALSE(launch.executable.empty());
    REQUIRE(launch.arguments.size() >= 8);
    REQUIRE(launch.arguments[1] == "__landrun-exec");
    REQUIRE(launch.arguments[2] == "--rw");
    REQUIRE(launch.arguments[3] == "/tmp/a root with spaces");
    REQUIRE(std::ranges::find(launch.arguments, std::string{"--"})
            != launch.arguments.end());
}

TEST_CASE("platform landrun driver reports an explicit backend", "[landrun]") {
    const auto driver = core::landrun::make_landrun_driver();
    REQUIRE(driver);
    const auto probe = driver->probe();
    REQUIRE_FALSE(probe.backend.empty());
    REQUIRE_FALSE(probe.detail.empty());
}
