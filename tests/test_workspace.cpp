#include <catch2/catch_test_macros.hpp>
#include "core/workspace/Workspace.hpp"
#include <filesystem>
#include <system_error>

TEST_CASE("Workspace bounds logic", "[Workspace]") {
    using core::workspace::Workspace;

    // Reset workspace for each test
    auto& ws = Workspace::get_instance();
    std::error_code ec;
    auto test_dir = std::filesystem::current_path(ec) / "test_workspace_dir";
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
        auto nested_root = test_dir / "nested" / "root";
        std::filesystem::create_directories(nested_root, ec);
        ws.initialize(nested_root, {}, true);

        REQUIRE_FALSE(ws.is_path_allowed(test_dir / "nested"));
        REQUIRE_FALSE(ws.is_path_allowed(test_dir));
    }

    SECTION("Enforced with additional directories") {
        auto add_dir1 = std::filesystem::current_path(ec) / "test_workspace_add1";
        auto add_dir2 = std::filesystem::current_path(ec) / "test_workspace_add2";
        std::filesystem::create_directories(add_dir1, ec);
        std::filesystem::create_directories(add_dir2, ec);

        ws.initialize(test_dir, {add_dir1, add_dir2}, true);
        
        // Primary
        REQUIRE(ws.is_path_allowed(test_dir / "file.txt"));
        
        // Additional
        REQUIRE(ws.is_path_allowed(add_dir1));
        REQUIRE(ws.is_path_allowed(add_dir1 / "file1.txt"));
        REQUIRE(ws.is_path_allowed(add_dir2 / "file2.txt"));
        
        // Outside
        REQUIRE_FALSE(ws.is_path_allowed("/tmp/outside_file.txt"));
        REQUIRE_FALSE(ws.is_path_allowed(test_dir.parent_path() / "another.txt"));

        // Cleanup
        std::filesystem::remove_all(add_dir1, ec);
        std::filesystem::remove_all(add_dir2, ec);
    }

    // Cleanup
    std::filesystem::remove_all(test_dir, ec);
    ws.initialize("", {}, false);
}
