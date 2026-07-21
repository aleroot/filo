#include <catch2/catch_test_macros.hpp>

#include "core/landrun/LandrunDriverFactory.hpp"
#include "core/landrun/LandrunEnvironment.hpp"
#include "core/landrun/LandrunLaunch.hpp"
#include "core/landrun/LandrunPath.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"
#include "core/landrun/LandrunRuntime.hpp"
#include "core/landrun/LandrunSettings.hpp"
#include "core/landrun/LandrunStatus.hpp"
#include "core/tools/LandrunToolPolicy.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/workspace/SensitivePathPolicy.hpp"
#include "core/workspace/PathVisibility.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

class WorkspaceMutationProbeTool final : public core::tools::Tool {
public:
    core::tools::ToolDefinition get_definition() const override {
        return {.name = std::string(core::tools::names::kWriteFile)};
    }

    std::string execute(
        const std::string&,
        const core::context::SessionContext&) override
    {
        return R"({"executed":true})";
    }
};

} // namespace

TEST_CASE("landrun modes expose one authoritative capability contract",
          "[landrun]") {
    using core::landrun::LandrunMode;

    REQUIRE(core::landrun::parse_landrun_mode("off") == LandrunMode::off);
    REQUIRE(core::landrun::parse_landrun_mode("read-only") == LandrunMode::read_only);
    REQUIRE(core::landrun::parse_landrun_mode("workspace-write")
            == LandrunMode::workspace_write);
    REQUIRE_FALSE(core::landrun::parse_landrun_mode("workspace-read").has_value());

    REQUIRE_FALSE(core::landrun::landrun_enabled(LandrunMode::off));
    REQUIRE(core::landrun::landrun_enabled(LandrunMode::read_only));
    REQUIRE(core::landrun::landrun_workspace_writable(LandrunMode::off));
    REQUIRE_FALSE(core::landrun::landrun_workspace_writable(LandrunMode::read_only));
    REQUIRE(core::landrun::landrun_workspace_writable(LandrunMode::workspace_write));
}

TEST_CASE("read-only landrun mode blocks native mutation tools only",
          "[landrun]") {
    using core::landrun::LandrunMode;
    using core::tools::landrun_allows_tool;
    using namespace core::tools::names;

    REQUIRE_FALSE(landrun_allows_tool(LandrunMode::read_only, kWriteFile));
    REQUIRE_FALSE(landrun_allows_tool(LandrunMode::read_only, kApplyPatch));
    REQUIRE_FALSE(landrun_allows_tool(LandrunMode::read_only, kCreateDirectory));
    REQUIRE(landrun_allows_tool(LandrunMode::read_only, kReadFile));
    REQUIRE(landrun_allows_tool(LandrunMode::read_only, kRunTerminalCommand));
    REQUIRE(landrun_allows_tool(LandrunMode::workspace_write, kWriteFile));
    REQUIRE(landrun_allows_tool(LandrunMode::off, kWriteFile));
}

TEST_CASE("tool manager enforces read-only mode at advertise and execute boundaries",
          "[landrun]") {
    // Isolate the singleton registry and settings in a child so this test does
    // not alter tool registrations used by randomized sibling tests.
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        auto& settings = core::landrun::LandrunSettings::instance();
        settings.configure_startup({
            .mode = core::landrun::LandrunMode::read_only,
        });
        auto& tools = core::tools::ToolManager::get_instance();
        tools.register_tool(std::make_shared<WorkspaceMutationProbeTool>());

        const auto advertised = tools.get_all_tools();
        if (std::ranges::any_of(advertised, [](const core::llm::Tool& tool) {
                return tool.function.name == core::tools::names::kWriteFile;
            })) {
            ::_exit(1);
        }
        const core::context::SessionContext context{
            "landrun-tool-policy-test",
            core::workspace::SessionWorkspace{
                core::workspace::WorkspaceSnapshot{
                    .primary = std::filesystem::current_path(),
                    .enforce = true,
                }},
        };
        const auto denied = tools.execute_tool(
            std::string(core::tools::names::kWriteFile),
            "{}",
            context);
        if (denied.find("unavailable while sandbox mode is read-only")
            == std::string::npos) {
            ::_exit(2);
        }

        settings.configure_startup({
            .mode = core::landrun::LandrunMode::workspace_write,
        });
        const auto allowed = tools.execute_tool(
            std::string(core::tools::names::kWriteFile),
            "{}",
            context);
        ::_exit(allowed.find("\"executed\":true") != std::string::npos ? 0 : 3);
    }

    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
}

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

TEST_CASE("landrun policy compiler accepts explicit immutable dependencies",
          "[landrun]") {
    namespace landrun = core::landrun;
    const auto primary = std::filesystem::current_path();
    const auto excluded = primary / "tests";
    const auto process_temp = std::filesystem::temp_directory_path();
    const auto runtime = process_temp / ".filo-explicit-runtime";
    const core::workspace::SessionWorkspace workspace{
        core::workspace::WorkspaceSnapshot{
            .primary = primary,
            .additional = {excluded},
            .enforce = true,
        }};

    const landrun::LandrunPolicyCompiler compiler({
        .excluded_paths = {excluded, process_temp},
        .runtime_root = runtime,
        .host_tmpdir = process_temp,
    });
    const auto policy = compiler.build(
        workspace, landrun::LandrunMode::workspace_write);

    CHECK(std::ranges::find(policy.writable_roots,
                            landrun::normalize_landrun_path(primary))
          != policy.writable_roots.end());
    CHECK(std::ranges::find(policy.writable_roots,
                            landrun::normalize_landrun_path(excluded))
          == policy.writable_roots.end());
    CHECK(std::ranges::find(policy.writable_roots,
                            landrun::normalize_landrun_path(runtime))
          != policy.writable_roots.end());
    CHECK(std::ranges::find(policy.writable_roots,
                            landrun::normalize_landrun_path(process_temp))
          == policy.writable_roots.end());
}

TEST_CASE("read-only policy keeps workspaces readable and private scratch writable",
          "[landrun]") {
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
    settings.configure_startup({
        .mode = landrun::LandrunMode::read_only,
        .excluded_paths = previous.excluded_paths,
    });
    landrun::LandrunRuntime runtime;
    const auto primary = std::filesystem::current_path();
    const auto additional = primary / "tests";
    const core::workspace::SessionWorkspace workspace{
        core::workspace::WorkspaceSnapshot{
            .primary = primary,
            .additional = {additional},
            .enforce = true,
        }};

    const auto policy = landrun::LandrunPolicyCompiler::compile(
        workspace, landrun::LandrunMode::read_only);
    const auto normalized_primary = landrun::normalize_landrun_path(primary);
    const auto normalized_additional = landrun::normalize_landrun_path(additional);

    REQUIRE(policy.enabled());
    REQUIRE_FALSE(policy.allow_network);
    REQUIRE(std::ranges::find(policy.readable_roots, normalized_primary)
            != policy.readable_roots.end());
    REQUIRE(std::ranges::find(policy.readable_roots, normalized_additional)
            != policy.readable_roots.end());
    REQUIRE(std::ranges::find(policy.writable_roots, normalized_primary)
            == policy.writable_roots.end());
    REQUIRE(std::ranges::find(policy.writable_roots, normalized_additional)
            == policy.writable_roots.end());
    REQUIRE(std::ranges::find(
                policy.writable_roots, landrun::normalize_landrun_path("/tmp"))
            == policy.writable_roots.end());
    REQUIRE(std::ranges::find(
                policy.writable_roots,
                landrun::normalize_landrun_path(runtime.root()))
            != policy.writable_roots.end());
    REQUIRE(settings.effective_tmpdir(landrun::LandrunMode::read_only)
            == runtime.root() / "tmp");
}

TEST_CASE("landrun status distinguishes read-only from workspace-write",
          "[landrun]") {
    namespace landrun = core::landrun;
    REQUIRE(landrun::landrun_status_label(landrun::LandrunMode::off)
            == "sandbox: off");
    REQUIRE(landrun::landrun_status_label(landrun::LandrunMode::read_only)
            .find("workspace read-only") != std::string::npos);
    REQUIRE(landrun::landrun_status_label(landrun::LandrunMode::workspace_write)
            .find("workspace/temp write") != std::string::npos);
    REQUIRE(landrun::landrun_status_detail(landrun::LandrunMode::read_only)
            .find("Workspace writes are denied") != std::string::npos);
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
    REQUIRE(launch.arguments.size() >= 10);
    REQUIRE(launch.arguments[1] == "__landrun-exec");
    REQUIRE(launch.arguments[2] == "--mode");
    REQUIRE(launch.arguments[3] == "workspace-write");
    REQUIRE(launch.arguments[4] == "--rw");
    REQUIRE(launch.arguments[5] == "/tmp/a root with spaces");
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
