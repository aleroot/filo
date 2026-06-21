#include <catch2/catch_test_macros.hpp>

#include "core/context/SessionContext.hpp"
#include "core/scm/ScmFactory.hpp"
#include "core/workspace/SessionWorkspace.hpp"
#include "core/workspace/Workspace.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <format>
#include <string_view>
#include <system_error>

TEST_CASE("Workspace bounds logic", "[Workspace]") {
    using core::workspace::Workspace;

    auto& ws = Workspace::get_instance();
    std::error_code ec;
    const auto test_dir = std::filesystem::current_path(ec) / "test_workspace_dir";
    std::filesystem::create_directories(test_dir, ec);

    SECTION("Default behavior - unenforced") {
        ws.initialize(test_dir, {}, false);
        REQUIRE_FALSE(ws.is_enforced());
        REQUIRE(ws.is_path_allowed(test_dir / "some_file.txt"));
        REQUIRE(ws.is_path_allowed("/etc/passwd"));
        REQUIRE(ws.is_path_allowed("C:\\Windows\\System32\\config"));
    }

    SECTION("Enforced primary directory") {
        ws.initialize(test_dir, {}, true);
        REQUIRE(ws.is_enforced());
        REQUIRE(ws.is_path_allowed(test_dir));
        REQUIRE(ws.is_path_allowed(test_dir / "child_file.txt"));
        REQUIRE(ws.is_path_allowed(test_dir / "nested" / "file.txt"));

        REQUIRE_FALSE(ws.is_path_allowed(test_dir.parent_path()));
        REQUIRE_FALSE(ws.is_path_allowed("/tmp/outside_file.txt"));
        REQUIRE_FALSE(ws.is_path_allowed(test_dir / ".." / "outside.txt"));
    }

    SECTION("Shorter target path is denied safely") {
        const auto nested_root = test_dir / "nested" / "root";
        std::filesystem::create_directories(nested_root, ec);
        ws.initialize(nested_root, {}, true);

        REQUIRE_FALSE(ws.is_path_allowed(test_dir / "nested"));
        REQUIRE_FALSE(ws.is_path_allowed(test_dir));
    }

    SECTION("Enforced with additional directories") {
        const auto add_dir1 = std::filesystem::current_path(ec) / "test_workspace_add1";
        const auto add_dir2 = std::filesystem::current_path(ec) / "test_workspace_add2";
        std::filesystem::create_directories(add_dir1, ec);
        std::filesystem::create_directories(add_dir2, ec);

        ws.initialize(test_dir, {add_dir1, add_dir2}, true);

        REQUIRE(ws.is_path_allowed(test_dir / "file.txt"));
        REQUIRE(ws.is_path_allowed(add_dir1));
        REQUIRE(ws.is_path_allowed(add_dir1 / "file1.txt"));
        REQUIRE(ws.is_path_allowed(add_dir2 / "file2.txt"));

        REQUIRE_FALSE(ws.is_path_allowed("/tmp/outside_file.txt"));
        REQUIRE_FALSE(ws.is_path_allowed(test_dir.parent_path() / "another.txt"));

        std::filesystem::remove_all(add_dir1, ec);
        std::filesystem::remove_all(add_dir2, ec);
    }

    std::filesystem::remove_all(test_dir, ec);
    ws.initialize("", {}, false);
}

TEST_CASE("SessionWorkspace resolves relative paths against its primary root", "[Workspace]") {
    using core::workspace::SessionWorkspace;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto root = std::filesystem::current_path(ec) / "test_session_workspace_root";
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "nested", ec);

    const SessionWorkspace workspace{WorkspaceSnapshot{
        .primary = root,
        .additional = {},
        .enforce = true,
        .version = 7,
    }};

    REQUIRE(workspace.primary() == root.lexically_normal());
    REQUIRE(workspace.version() == 7);
    REQUIRE(workspace.resolve_path("nested/value.txt")
            == (root / "nested" / "value.txt").lexically_normal());
    REQUIRE(workspace.is_path_allowed("nested/value.txt"));
    REQUIRE_FALSE(workspace.is_path_allowed("../outside.txt"));

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("SessionContext delegates to its owned SessionWorkspace", "[Workspace][SessionContext]") {
    using core::context::SessionContext;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto primary = std::filesystem::current_path(ec) / "test_session_context_root";
    const auto extra = std::filesystem::current_path(ec) / "test_session_context_extra";
    std::filesystem::remove_all(primary, ec);
    std::filesystem::remove_all(extra, ec);
    std::filesystem::create_directories(primary / "nested", ec);
    std::filesystem::create_directories(extra, ec);

    const SessionContext context = core::context::make_session_context(
        WorkspaceSnapshot{
            .primary = primary,
            .additional = {extra},
            .enforce = true,
            .version = 23,
        },
        core::context::SessionTransport::mcp_http,
        "workspace-test-session");

    REQUIRE(context.workspace_view().primary() == primary.lexically_normal());
    REQUIRE(context.effective_workspace().version == 23);
    REQUIRE(context.resolve_path("nested/value.txt")
            == (primary / "nested" / "value.txt").lexically_normal());
    REQUIRE(context.is_path_allowed("nested/value.txt"));
    REQUIRE(context.is_path_allowed(extra / "allowed.txt"));
    REQUIRE_FALSE(context.is_path_allowed("../outside.txt"));

    std::filesystem::remove_all(primary, ec);
    std::filesystem::remove_all(extra, ec);
}

TEST_CASE("SourceControlProvider lists branch refs through abstraction", "[Workspace][SCM]") {
    if (std::system("git --version >/dev/null 2>&1") != 0) {
        SKIP("git is not available in this environment");
    }

    namespace fs = std::filesystem;
    struct CwdGuard {
        fs::path old;
        ~CwdGuard() {
            std::error_code ec;
            fs::current_path(old, ec);
        }
    } guard{fs::current_path()};

    std::error_code ec;
    const fs::path temp_dir =
        fs::temp_directory_path() / std::format("filo-test-scm-{}", std::rand());
    fs::remove_all(temp_dir, ec);
    fs::create_directories(temp_dir, ec);
    REQUIRE_FALSE(ec);
    fs::current_path(temp_dir);

    REQUIRE(std::system("git init -q") == 0);
    {
        std::ofstream readme("README.md");
        readme << "test\n";
    }
    REQUIRE(std::system("git add README.md") == 0);
    REQUIRE(std::system(
        "git -c user.name=Filo -c user.email=filo@example.invalid commit -qm initial")
        == 0);
    REQUIRE(std::system("git branch feature/ref-list") == 0);

    auto scm = core::scm::ScmFactory::create(temp_dir);
    const auto refs = scm->list_branch_refs();

    const auto has_ref = [&refs](std::string_view name) {
        return std::ranges::any_of(refs, [name](const core::scm::BranchRef& ref) {
            return ref.name == name;
        });
    };
    REQUIRE(has_ref("feature/ref-list"));

    fs::current_path(guard.old, ec);
    fs::remove_all(temp_dir, ec);
}
