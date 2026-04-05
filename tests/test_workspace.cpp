#include <catch2/catch_test_macros.hpp>

#include "core/context/SessionContext.hpp"
#include "core/workspace/SessionWorkspace.hpp"
#include "core/workspace/Workspace.hpp"

#include <filesystem>
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
