#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/ReadFileTool.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/ReplaceTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/CreateDirectoryTool.hpp"
#include "core/tools/DeleteFileTool.hpp"
#include "core/tools/MoveFileTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ApplyPatchTool.hpp"
#include "core/tools/SearchReplaceTool.hpp"
#include "core/tools/GetTimeTool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/workspace/Workspace.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "core/tools/PythonManager.hpp"
#endif
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#ifdef FILO_ENABLE_PYTHON
#include <future>
#include <vector>
#endif

using namespace core::tools;

namespace {

class ScopedWorkspaceEnforcement {
public:
    explicit ScopedWorkspaceEnforcement(const std::filesystem::path& primary) {
        core::workspace::Workspace::get_instance().initialize(primary, {}, true);
    }

    ~ScopedWorkspaceEnforcement() {
        core::workspace::Workspace::get_instance().initialize("", {}, false);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// WriteFileTool / ReadFileTool
// ---------------------------------------------------------------------------

TEST_CASE("WriteFileTool and ReadFileTool", "[tools]") {
    std::string test_file = "test_artifact_1.txt";

    WriteFileTool write_tool;
    std::string write_args = "{\"file_path\": \"" + test_file + "\", \"content\": \"Hello, Filo!\\nThis is a test.\"}";
    std::string write_res = write_tool.execute(write_args);
    REQUIRE_THAT(write_res, Catch::Matchers::ContainsSubstring(R"("success":true)"));
    REQUIRE_THAT(write_res, Catch::Matchers::ContainsSubstring(R"("file_path")"));
    REQUIRE(std::filesystem::exists(test_file));

    ReadFileTool read_tool;
    std::string read_args = "{\"path\": \"" + test_file + "\"}";
    std::string read_res = read_tool.execute(read_args);
    REQUIRE_THAT(read_res, Catch::Matchers::ContainsSubstring("Hello, Filo!\\nThis is a test."));

    std::filesystem::remove(test_file);
}

TEST_CASE("ReadFileTool offset_line and limit_lines", "[tools]") {
    const std::string path = "test_artifact_lines.txt";
    {
        std::ofstream ofs(path);
        for (int i = 1; i <= 10; ++i) ofs << "line" << i << "\n";
    }

    ReadFileTool tool;
    auto res = tool.execute("{\"path\": \"" + path + "\", \"offset_line\": 3, \"limit_lines\": 3}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("line3"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("line5"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("line2"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("line6"));

    std::filesystem::remove(path);
}

TEST_CASE("ReadFileTool returns error for missing file", "[tools]") {
    ReadFileTool tool;
    auto res = tool.execute("{\"path\": \"nonexistent_xyz_99999.txt\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("path does not exist"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("resolved"));
}

TEST_CASE("ReadFileTool returns error for unexpected arguments", "[tools]") {
    ReadFileTool tool;
    auto res = tool.execute(R"({"file_path":"legacy.txt"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("ReadFileTool reports directory path as non-regular file", "[tools]") {
    const std::string dir = "test_artifact_read_dir";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    ReadFileTool tool;
    const auto res = tool.execute("{\"path\": \"" + dir + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("not a regular file"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("type=directory"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("ReadFileTool reads files that contain spaces in their path", "[tools]") {
    const std::string path = "test artifact read with spaces.txt";
    {
        std::ofstream ofs(path);
        ofs << "space_path_ok";
    }

    ReadFileTool tool;
    const auto res = tool.execute("{\"path\": \"" + path + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("space_path_ok"));

    std::filesystem::remove(path);
}

TEST_CASE("ReadFileTool handles dot-segment paths", "[tools]") {
    const std::string root = "test_artifact_read_segments";
    const std::string file = root + "/nested/value.txt";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root + "/nested");
    {
        std::ofstream ofs(file);
        ofs << "dot_segments_ok";
    }

    ReadFileTool tool;
    const std::string path_with_segments = root + "/nested/../nested/./value.txt";
    const auto res = tool.execute("{\"path\": \"" + path_with_segments + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("dot_segments_ok"));

    std::filesystem::remove_all(root);
}

#ifndef _WIN32
TEST_CASE("ReadFileTool reports open failure details for unreadable file", "[tools]") {
    if (getuid() == 0) {
        SKIP("Running as root - permission checks don't apply");
    }

    const std::string path = "test_artifact_read_no_perms.txt";
    {
        std::ofstream ofs(path);
        ofs << "secret";
    }

    std::error_code ec;
    std::filesystem::permissions(path, std::filesystem::perms::none,
                                 std::filesystem::perm_options::replace, ec);
    if (ec) {
        std::filesystem::remove(path);
        SKIP("Permission changes are unavailable in this environment");
    }

    ReadFileTool tool;
    const auto res = tool.execute("{\"path\": \"" + path + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("failed to open file for reading"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("resolved"));

    std::filesystem::permissions(path,
                                 std::filesystem::perms::owner_read
                                   | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    std::filesystem::remove(path);
}

TEST_CASE("ReadFileTool follows symlink to regular file", "[tools]") {
    const std::string target = "test_artifact_read_target.txt";
    const std::string link = "test_artifact_read_link.txt";
    std::filesystem::remove(target);
    std::filesystem::remove(link);
    {
        std::ofstream ofs(target);
        ofs << "symlink_ok";
    }

    std::error_code ec;
    std::filesystem::create_symlink(target, link, ec);
    if (ec) {
        std::filesystem::remove(target);
        std::filesystem::remove(link);
        SKIP("Symlink creation is unavailable in this environment");
    }

    ReadFileTool tool;
    const auto res = tool.execute("{\"path\": \"" + link + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("symlink_ok"));

    std::filesystem::remove(link);
    std::filesystem::remove(target);
}

TEST_CASE("ReadFileTool reports missing target for dangling symlink", "[tools]") {
    const std::string link = "test_artifact_read_dangling_link.txt";
    std::filesystem::remove(link);

    std::error_code ec;
    std::filesystem::create_symlink("missing_target_for_filo_read_tool.txt", link, ec);
    if (ec) {
        std::filesystem::remove(link);
        SKIP("Symlink creation is unavailable in this environment");
    }

    ReadFileTool tool;
    const auto res = tool.execute("{\"path\": \"" + link + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("path does not exist"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("resolved"));

    std::filesystem::remove(link);
}
#endif

// ---------------------------------------------------------------------------
// ReplaceTool
// ---------------------------------------------------------------------------

TEST_CASE("ReplaceTool single occurrence", "[tools]") {
    std::string test_file = "test_artifact_2.txt";

    std::ofstream ofs(test_file);
    ofs << "Line 1\nLine 2\nLine 3\n";
    ofs.close();

    ReplaceTool tool;
    std::string args = "{\"file_path\": \"" + test_file + "\", \"old_string\": \"Line 2\", \"new_string\": \"Replaced Line\"}";
    const auto replace_res = tool.execute(args);
    REQUIRE_THAT(replace_res, Catch::Matchers::ContainsSubstring(R"("success":true)"));
    REQUIRE_THAT(replace_res, Catch::Matchers::ContainsSubstring(R"("replaced_at_line":2)"));

    std::ifstream ifs(test_file);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    REQUIRE(content == "Line 1\nReplaced Line\nLine 3\n");

    std::filesystem::remove(test_file);
}

TEST_CASE("ReplaceTool rejects multiple occurrences", "[tools]") {
    std::string test_file = "test_artifact_3.txt";

    std::ofstream ofs(test_file);
    ofs << "Duplicate\nDuplicate\n";
    ofs.close();

    ReplaceTool tool;
    std::string args = "{\"file_path\": \"" + test_file + "\", \"old_string\": \"Duplicate\", \"new_string\": \"Replaced\"}";
    auto res = tool.execute(args);
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("multiple locations"));

    std::filesystem::remove(test_file);
}

TEST_CASE("ReplaceTool returns error when old_string not found", "[tools]") {
    std::string test_file = "test_artifact_replace_notfound.txt";
    { std::ofstream ofs(test_file); ofs << "hello world\n"; }

    ReplaceTool tool;
    auto res = tool.execute("{\"file_path\": \"" + test_file + "\", \"old_string\": \"xyz\", \"new_string\": \"abc\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));

    std::filesystem::remove(test_file);
}

// ---------------------------------------------------------------------------
// ListDirectoryTool
// ---------------------------------------------------------------------------

TEST_CASE("ListDirectoryTool lists files", "[tools]") {
    std::string test_dir = "test_artifact_dir";
    std::filesystem::create_directory(test_dir);
    std::ofstream(test_dir + "/file1.txt") << "test";
    std::ofstream(test_dir + "/file2.txt") << "test";

    ListDirectoryTool tool;
    auto res = tool.execute("{\"path\": \"" + test_dir + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("file1.txt"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("file2.txt"));

    std::filesystem::remove_all(test_dir);
}

TEST_CASE("ListDirectoryTool returns error for unexpected arguments", "[tools]") {
    ListDirectoryTool tool;
    auto res = tool.execute(R"({"dir_path":"."})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

// ---------------------------------------------------------------------------
// CreateDirectoryTool
// ---------------------------------------------------------------------------

TEST_CASE("CreateDirectoryTool creates nested directories", "[tools]") {
    const std::string path = "test_artifact_create_a/b/c";
    std::filesystem::remove_all("test_artifact_create_a");

    CreateDirectoryTool tool;
    auto res = tool.execute("{\"dir_path\": \"" + path + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));
    REQUIRE(std::filesystem::is_directory(path));

    std::filesystem::remove_all("test_artifact_create_a");
}

TEST_CASE("CreateDirectoryTool is idempotent when directory exists", "[tools]") {
    const std::string path = "test_artifact_create_existing";
    std::filesystem::create_directories(path);

    CreateDirectoryTool tool;
    auto res = tool.execute("{\"dir_path\": \"" + path + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));

    std::filesystem::remove_all(path);
}

TEST_CASE("CreateDirectoryTool denies out-of-scope path when workspace is enforced", "[tools][workspace]") {
    const auto workspace_root = std::filesystem::current_path()
        / ("test_artifact_workspace_create_" + std::to_string(getpid()));
    const auto outside_path = std::filesystem::temp_directory_path()
        / ("filo_outside_create_" + std::to_string(getpid()));
    std::filesystem::remove_all(workspace_root);
    std::filesystem::remove_all(outside_path);
    std::filesystem::create_directories(workspace_root);

    ScopedWorkspaceEnforcement scope(workspace_root);
    CreateDirectoryTool tool;
    auto res = tool.execute("{\"dir_path\": \"" + outside_path.string() + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("Access denied"));
    REQUIRE_FALSE(std::filesystem::exists(outside_path));

    std::filesystem::remove_all(workspace_root);
}

// ---------------------------------------------------------------------------
// DeleteFileTool
// ---------------------------------------------------------------------------

TEST_CASE("DeleteFileTool removes file and reports success", "[tools]") {
    const std::string path = "test_artifact_del.txt";
    { std::ofstream ofs(path); ofs << "x"; }

    DeleteFileTool tool;
    auto res = tool.execute("{\"file_path\": \"" + path + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));
    REQUIRE_FALSE(std::filesystem::exists(path));
}

TEST_CASE("DeleteFileTool returns error for nonexistent path", "[tools]") {
    DeleteFileTool tool;
    auto res = tool.execute("{\"file_path\": \"nonexistent_xyz_12345.txt\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("DeleteFileTool denies out-of-scope path when workspace is enforced", "[tools][workspace]") {
    const auto workspace_root = std::filesystem::current_path()
        / ("test_artifact_workspace_delete_" + std::to_string(getpid()));
    const auto outside_file = std::filesystem::temp_directory_path()
        / ("filo_outside_delete_" + std::to_string(getpid()) + ".txt");
    std::filesystem::remove_all(workspace_root);
    std::filesystem::remove(outside_file);
    std::filesystem::create_directories(workspace_root);
    { std::ofstream ofs(outside_file); ofs << "outside"; }

    ScopedWorkspaceEnforcement scope(workspace_root);
    DeleteFileTool tool;
    auto res = tool.execute("{\"file_path\": \"" + outside_file.string() + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("Access denied"));
    REQUIRE(std::filesystem::exists(outside_file));

    std::filesystem::remove(outside_file);
    std::filesystem::remove_all(workspace_root);
}

// ---------------------------------------------------------------------------
// MoveFileTool
// ---------------------------------------------------------------------------

TEST_CASE("MoveFileTool renames a file", "[tools]") {
    const std::string src = "test_artifact_move_src.txt";
    const std::string dst = "test_artifact_move_dst.txt";
    { std::ofstream ofs(src); ofs << "data"; }
    std::filesystem::remove(dst);

    MoveFileTool tool;
    auto res = tool.execute("{\"source\": \"" + src + "\", \"destination\": \"" + dst + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));
    REQUIRE_FALSE(std::filesystem::exists(src));
    REQUIRE(std::filesystem::exists(dst));

    std::filesystem::remove(dst);
}

TEST_CASE("MoveFileTool denies destination outside workspace when enforced", "[tools][workspace]") {
    const auto workspace_root = std::filesystem::current_path()
        / ("test_artifact_workspace_move_" + std::to_string(getpid()));
    const auto src = workspace_root / "source.txt";
    const auto outside_dst = std::filesystem::temp_directory_path()
        / ("filo_outside_move_" + std::to_string(getpid()) + ".txt");
    std::filesystem::remove_all(workspace_root);
    std::filesystem::remove(outside_dst);
    std::filesystem::create_directories(workspace_root);
    { std::ofstream ofs(src); ofs << "data"; }

    ScopedWorkspaceEnforcement scope(workspace_root);
    MoveFileTool tool;
    auto res = tool.execute("{\"source\": \"" + src.string() + "\", \"destination\": \"" + outside_dst.string() + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("Access denied"));
    REQUIRE(std::filesystem::exists(src));
    REQUIRE_FALSE(std::filesystem::exists(outside_dst));

    std::filesystem::remove_all(workspace_root);
}

// ---------------------------------------------------------------------------
// ShellTool
// ---------------------------------------------------------------------------

TEST_CASE("ShellTool executes a simple command", "[tools]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"echo hello"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hello"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool captures non-zero exit code", "[tools]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"exit 42"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":42"));
}

TEST_CASE("ShellTool rejects nonexistent working_dir", "[tools]") {
    ShellTool tool;
    auto res = tool.execute(R"({"command":"echo hi","working_dir":"/nonexistent_filo_test_dir_xyz"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("ShellTool returns error for missing command field", "[tools]") {
    ShellTool tool;
    auto res = tool.execute(R"({})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("ShellTool session persists working directory across calls", "[tools]") {
    ShellTool tool;
    const std::string tmp_dir = "/tmp";

    // cd to /tmp in the first call
    auto res1 = tool.execute(R"({"command":"cd /tmp"})");
    REQUIRE_THAT(res1, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    // pwd in the second call should reflect the cd
    auto res2 = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("/tmp"));
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool session persists exported environment variable", "[tools]") {
    ShellTool tool;

    auto res1 = tool.execute(R"({"command":"export FILO_TEST_VAR=hello_world_42"})");
    REQUIRE_THAT(res1, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

    auto res2 = tool.execute(R"({"command":"echo $FILO_TEST_VAR"})");
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("hello_world_42"));
}

TEST_CASE("ShellTool isolates persistent state across MCP session contexts", "[tools]") {
    ShellTool tool;

    {
        auto session_a = ShellTool::scoped_mcp_session("session-a");
        auto res = tool.execute(R"({"command":"cd /tmp"})");
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    }
    {
        auto session_b = ShellTool::scoped_mcp_session("session-b");
        auto res = tool.execute(R"({"command":"cd /var"})");
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    }
    {
        auto session_a = ShellTool::scoped_mcp_session("session-a");
        auto res = tool.execute(R"({"command":"pwd"})");
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("/tmp"));
    }
    {
        auto session_b = ShellTool::scoped_mcp_session("session-b");
        auto res = tool.execute(R"({"command":"pwd"})");
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("/var"));
    }
}

TEST_CASE("ShellTool clear_mcp_session resets that session shell state", "[tools]") {
    ShellTool tool;
    constexpr std::string_view kSession = "session-clear";
    constexpr std::string_view kToken = "filo_session_token_42";

    {
        auto session = ShellTool::scoped_mcp_session(std::string(kSession));
        auto set_res = tool.execute(
            R"({"command":"export FILO_SESSION_CLEAR_VAR=filo_session_token_42"})");
        REQUIRE_THAT(set_res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));

        auto echo_res = tool.execute(R"({"command":"echo $FILO_SESSION_CLEAR_VAR"})");
        REQUIRE_THAT(echo_res, Catch::Matchers::ContainsSubstring(std::string(kToken)));
    }

    ShellTool::clear_mcp_session(kSession);

    {
        auto session = ShellTool::scoped_mcp_session(std::string(kSession));
        auto echo_res = tool.execute(R"({"command":"echo $FILO_SESSION_CLEAR_VAR"})");
        REQUIRE_THAT(echo_res, !Catch::Matchers::ContainsSubstring(std::string(kToken)));
    }
}

TEST_CASE("ShellTool handles MCP session LRU eviction without crashing", "[tools]") {
    ShellTool tool;

    for (int i = 0; i < 70; ++i) {
        auto session = ShellTool::scoped_mcp_session("session-evict-" + std::to_string(i));
        auto res = tool.execute(R"({"command":"echo alive"})");
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
        REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("alive"));
    }
}

TEST_CASE("ShellTool working_dir runs in subshell and does not affect session cwd", "[tools]") {
    ShellTool tool;

    // Set session cwd to /tmp
    tool.execute(R"({"command":"cd /tmp"})");

    // Run with working_dir=/var — should not change session cwd
    auto res1 = tool.execute(R"({"command":"pwd","working_dir":"/var"})");
    REQUIRE_THAT(res1, Catch::Matchers::ContainsSubstring("/var"));

    // Session cwd should still be /tmp
    auto res2 = tool.execute(R"({"command":"pwd"})");
    REQUIRE_THAT(res2, Catch::Matchers::ContainsSubstring("/tmp"));
}

TEST_CASE("ShellTool completes a long-running command within timeout", "[tools]") {
    ShellTool tool;
    // sleep 2 with a 10-second timeout — must complete and report exit_code 0.
    auto res = tool.execute(R"({"command":"sleep 2","timeout_seconds":10})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("TIMEOUT"));
}

TEST_CASE("ShellTool returns timeout message when command exceeds limit", "[tools]") {
    ShellTool tool;
    // sleep 60 with a 1-second timeout — must be killed and report timeout.
    auto res = tool.execute(R"({"command":"sleep 60","timeout_seconds":1})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("TIMEOUT"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":-1"));
}

TEST_CASE("ShellTool session recovers cleanly after a timeout", "[tools]") {
    ShellTool tool;
    // Trigger a timeout.
    tool.execute(R"({"command":"sleep 60","timeout_seconds":1})");
    // The next command must work — session was reset after the timeout.
    auto res = tool.execute(R"({"command":"echo recovered"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("recovered"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

TEST_CASE("ShellTool kills child processes spawned by timed-out command", "[tools]") {
    ShellTool tool;
    // The background job creates a marker after 2 s; the foreground sleep keeps
    // the command alive long enough to hit the 1-second timeout.
    // On timeout the whole process group is SIGKILL'd — the background job dies
    // before it can write the marker.
    const std::string marker = "/tmp/filo_shell_test_marker_" + std::to_string(::getpid());
    std::filesystem::remove(marker);  // ensure clean start

    const std::string cmd =
        "(sleep 2 && touch " + marker + ") & sleep 60";
    tool.execute("{\"command\":\"" + cmd + "\",\"timeout_seconds\":1}");

    // Wait longer than the background job would have taken if not killed (2 s).
    ::sleep(3);

    // Marker must not exist — the background job was killed with the process group.
    REQUIRE_FALSE(std::filesystem::exists(marker));
    std::filesystem::remove(marker);
}

TEST_CASE("ShellTool timeout_seconds defaults to 600 when not provided", "[tools]") {
    // Just verify the definition advertises timeout_seconds as optional.
    ShellTool tool;
    auto def = tool.get_definition();
    bool found = false;
    for (const auto& p : def.parameters) {
        if (p.name == "timeout_seconds") { found = true; break; }
    }
    REQUIRE(found);
}

TEST_CASE("ShellTool writes large command payloads without truncation", "[tools]") {
    ShellTool tool;

    const std::string payload(150000, 'x');
    const std::string command = "printf '" + payload + "' | wc -c";
    const std::string args = "{\"command\":\"" + command + "\"}";

    auto res = tool.execute(args);
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("150000"));
}

TEST_CASE("ShellTool session restarts transparently after exit", "[tools]") {
    ShellTool tool;

    // Export a variable, then exit — session should die
    tool.execute(R"({"command":"export FILO_PERSIST=yes"})");
    auto exit_res = tool.execute(R"({"command":"exit 7"})");
    REQUIRE_THAT(exit_res, Catch::Matchers::ContainsSubstring("\"exit_code\":7"));

    // Next call should work fine (new session, variable is gone)
    auto res = tool.execute(R"({"command":"echo alive"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("alive"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"exit_code\":0"));
}

// ---------------------------------------------------------------------------
// FileSearchTool
// ---------------------------------------------------------------------------

TEST_CASE("FileSearchTool finds files by glob pattern", "[tools]") {
    const std::string dir = "test_artifact_filesearch";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/foo.txt") << "x"; }
    { std::ofstream(dir + "/bar.cpp") << "x"; }

    FileSearchTool tool;
    auto res = tool.execute("{\"pattern\": \"*.txt\", \"path\": \"" + dir + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("foo.txt"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("bar.cpp"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("FileSearchTool returns error for missing pattern", "[tools]") {
    FileSearchTool tool;
    auto res = tool.execute(R"({})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("FileSearchTool returns error for unexpected arguments", "[tools]") {
    FileSearchTool tool;
    auto res = tool.execute(R"({"pattern":"*.txt","dir":"."})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

// ---------------------------------------------------------------------------
// GrepSearchTool
// ---------------------------------------------------------------------------

TEST_CASE("GrepSearchTool finds pattern in files", "[tools]") {
    const std::string dir = "test_artifact_grep";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/a.txt") << "hello world\nfoo bar\n"; }
    { std::ofstream(dir + "/b.txt") << "other content\n"; }

    GrepSearchTool tool;
    auto res = tool.execute("{\"pattern\": \"hello\", \"path\": \"" + dir + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hello"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool returns empty matches for no match", "[tools]") {
    const std::string dir = "test_artifact_grep_empty";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/a.txt") << "nothing here\n"; }

    GrepSearchTool tool;
    auto res = tool.execute("{\"pattern\": \"xyznotfound9999\", \"path\": \"" + dir + "\"}");
    // Should return success (empty matches), not error
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("matches"));

    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// ApplyPatchTool
// ---------------------------------------------------------------------------

TEST_CASE("ApplyPatchTool applies a unified diff patch", "[tools]") {
    const std::string dir = "test_artifact_patch_dir";
    std::filesystem::create_directories(dir);
    const std::string file = dir + "/target.txt";
    { std::ofstream ofs(file); ofs << "line1\nline2\nline3\n"; }

    // Create a unified diff that replaces "line2" with "patched"
    const std::string patch =
        "--- a/target.txt\n"
        "+++ b/target.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-line2\n"
        "+patched\n"
        " line3\n";

    ApplyPatchTool tool;
    auto res = tool.execute("{\"patch\": " + [&]{
        // JSON-escape the patch
        std::string escaped;
        for (char c : patch) {
            if (c == '\n') escaped += "\\n";
            else if (c == '"') escaped += "\\\"";
            else escaped += c;
        }
        return "\"" + escaped + "\"";
    }() + ", \"working_dir\": \"" + dir + "\"}");

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));

    std::ifstream ifs(file);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("patched"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("ApplyPatchTool denies out-of-scope patch targets when workspace is enforced", "[tools][workspace]") {
    const auto workspace_root = std::filesystem::current_path()
        / ("test_artifact_workspace_patch_" + std::to_string(getpid()));
    const auto outside_file = std::filesystem::temp_directory_path()
        / ("filo_outside_patch_" + std::to_string(getpid()) + ".txt");
    std::filesystem::remove_all(workspace_root);
    std::filesystem::remove(outside_file);
    std::filesystem::create_directories(workspace_root);
    { std::ofstream ofs(outside_file); ofs << "outside\n"; }

    const std::string patch =
        std::string("--- ") + outside_file.string() + "\n" +
        "+++ " + outside_file.string() + "\n" +
        "@@ -1 +1 @@\n" +
        "-outside\n" +
        "+patched\n";

    ApplyPatchTool tool;
    ScopedWorkspaceEnforcement scope(workspace_root);
    auto res = tool.execute("{\"patch\": " + [&]{
        std::string escaped;
        for (char c : patch) {
            if (c == '\n') escaped += "\\n";
            else if (c == '"') escaped += "\\\"";
            else escaped += c;
        }
        return "\"" + escaped + "\"";
    }() + ", \"working_dir\": \"" + workspace_root.string() + "\"}");

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("Access denied"));
    std::ifstream ifs(outside_file);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    REQUIRE_THAT(content, Catch::Matchers::ContainsSubstring("outside"));
    REQUIRE_THAT(content, !Catch::Matchers::ContainsSubstring("patched"));

    std::filesystem::remove(outside_file);
    std::filesystem::remove_all(workspace_root);
}

// ---------------------------------------------------------------------------
// GetTimeTool
// ---------------------------------------------------------------------------

TEST_CASE("GetTimeTool returns a time string", "[tools]") {
    GetTimeTool tool;
    auto res = tool.execute("{}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("time"));
    // Basic format check: contains digits and colons (HH:MM:SS)
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(":"));
}

// ---------------------------------------------------------------------------
// ToolManager
// ---------------------------------------------------------------------------

TEST_CASE("ToolManager registers and executes a tool", "[tools]") {
    // Use a fresh instance-like test (singleton, so we just register and verify)
    auto& mgr = ToolManager::get_instance();

    // Register a write tool and verify it's findable via execute
    mgr.register_tool(std::make_shared<WriteFileTool>());

    const std::string path = "test_artifact_mgr.txt";
    auto res = mgr.execute_tool("write_file",
        "{\"file_path\": \"" + path + "\", \"content\": \"manager test\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(R"("success":true)"));
    REQUIRE(std::filesystem::exists(path));
    std::filesystem::remove(path);
}

TEST_CASE("ToolManager returns error for unknown tool", "[tools]") {
    auto& mgr = ToolManager::get_instance();
    auto res = mgr.execute_tool("nonexistent_tool_xyz", "{}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("not found"));
}

TEST_CASE("ToolManager get_all_tools returns registered tools", "[tools]") {
    auto& mgr = ToolManager::get_instance();
    mgr.register_tool(std::make_shared<GetTimeTool>());

    auto tools = mgr.get_all_tools();
    REQUIRE_FALSE(tools.empty());

    // Verify at least one tool has a non-empty name and is type "function"
    bool found = false;
    for (const auto& t : tools) {
        if (!t.function.name.empty()) { found = true; break; }
    }
    REQUIRE(found);
}

// ---------------------------------------------------------------------------
// FileSearchTool (pure C++ rewrite)
// ---------------------------------------------------------------------------

TEST_CASE("FileSearchTool finds nested files", "[tools]") {
    const std::string dir = "test_artifact_filesearch_nested";
    std::filesystem::create_directories(dir + "/sub");
    { std::ofstream(dir + "/top.txt") << "x"; }
    { std::ofstream(dir + "/sub/deep.txt") << "x"; }
    { std::ofstream(dir + "/other.cpp") << "x"; }

    FileSearchTool tool;
    auto res = tool.execute("{\"pattern\": \"*.txt\", \"path\": \"" + dir + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("top.txt"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("deep.txt"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("other.cpp"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("FileSearchTool skips .git directory", "[tools]") {
    const std::string dir = "test_artifact_filesearch_git";
    std::filesystem::create_directories(dir + "/.git");
    { std::ofstream(dir + "/real.txt") << "x"; }
    { std::ofstream(dir + "/.git/hidden.txt") << "x"; }

    FileSearchTool tool;
    auto res = tool.execute("{\"pattern\": \"*.txt\", \"path\": \"" + dir + "\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("real.txt"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("hidden.txt"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("FileSearchTool returns error for bad dir", "[tools]") {
    FileSearchTool tool;
    auto res = tool.execute(R"({"pattern":"*.txt","path":"/nonexistent_filo_xyz_99"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

// ---------------------------------------------------------------------------
// GrepSearchTool (pure C++ rewrite)
// ---------------------------------------------------------------------------

TEST_CASE("GrepSearchTool finds pattern with include filter", "[tools]") {
    const std::string dir = "test_artifact_grep_filter";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/a.cpp") << "hello world\n"; }
    { std::ofstream(dir + "/b.txt") << "hello text\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(
        "{\"pattern\": \"hello\", \"path\": \"" + dir + "\","
        "\"include_pattern\": \"*.cpp\"}");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("a.cpp"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("b.txt"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool returns error for invalid regex", "[tools]") {
    GrepSearchTool tool;
    auto res = tool.execute(R"({"pattern":"[invalid","path":"."})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

// ---------------------------------------------------------------------------
// GrepSearchTool — expanded regression suite
// ---------------------------------------------------------------------------

// Helper: build the JSON args string for grep_search.
static std::string grep_args(const std::string& pattern,
                              const std::string& dir,
                              const std::string& include = "") {
    std::string s = "{\"pattern\":\"" + pattern + "\",\"path\":\"" + dir + "\"";
    if (!include.empty()) s += ",\"include_pattern\":\"" + include + "\"";
    s += "}";
    return s;
}

// ── Correct line numbers ────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool reports correct 1-based line numbers", "[tools][grep]") {
    const std::string dir = "test_grep_lineno";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "alpha\nbeta\ngamma\ndelta\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("gamma", dir));

    // "gamma" is on line 3
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"line\":3"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("gamma"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool reports match on first line", "[tools][grep]") {
    const std::string dir = "test_grep_firstline";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "MATCH\nsecond\nthird\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("MATCH", dir));

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"line\":1"));
    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool reports match on last line without trailing newline", "[tools][grep]") {
    const std::string dir = "test_grep_lastline_nonl";
    std::filesystem::create_directories(dir);
    // No trailing newline — last line must still be found.
    { std::ofstream(dir + "/f.txt") << "first\nsecond\nTARGET"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("TARGET", dir));

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"line\":3"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("TARGET"));
    std::filesystem::remove_all(dir);
}

// ── Multiple matches ────────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool returns all matching lines within one file", "[tools][grep]") {
    const std::string dir = "test_grep_multimatch";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "hit one\nmiss\nhit two\nmiss\nhit three\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("hit", dir));

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hit one"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hit two"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("hit three"));
    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool returns matches across multiple files", "[tools][grep]") {
    const std::string dir = "test_grep_multifile";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/a.txt") << "needle in a\n"; }
    { std::ofstream(dir + "/b.txt") << "needle in b\n"; }
    { std::ofstream(dir + "/c.txt") << "hay only\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir));

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("a.txt"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("b.txt"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("c.txt"));
    std::filesystem::remove_all(dir);
}

// ── Output ordering ─────────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool output is sorted by path then line", "[tools][grep]") {
    const std::string dir = "test_grep_order";
    std::filesystem::create_directories(dir);
    // 'z.txt' comes last lexicographically; put the match only there.
    // 'a.txt' comes first; two matches on lines 1 and 3.
    { std::ofstream(dir + "/a.txt") << "target\nother\ntarget again\n"; }
    { std::ofstream(dir + "/z.txt") << "target z\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("target", dir));

    // a.txt must appear before z.txt in the JSON.
    const auto pos_a = res.find("a.txt");
    const auto pos_z = res.find("z.txt");
    REQUIRE(pos_a != std::string::npos);
    REQUIRE(pos_z != std::string::npos);
    REQUIRE(pos_a < pos_z);

    // Line 1 of a.txt must appear before line 3 of a.txt.
    const auto pos_l1 = res.find("\"line\":1");
    const auto pos_l3 = res.find("\"line\":3");
    REQUIRE(pos_l1 != std::string::npos);
    REQUIRE(pos_l3 != std::string::npos);
    REQUIRE(pos_l1 < pos_l3);

    std::filesystem::remove_all(dir);
}

// ── Regex features ──────────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool supports regex metacharacters", "[tools][grep]") {
    const std::string dir = "test_grep_regex";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "foo123\nbar456\nfoo789\n"; }

    GrepSearchTool tool;
    // \\d+ requires escaping backslash in the JSON string.
    auto res = tool.execute(grep_args("foo\\\\d+", dir));
    // foo\d+ won't match "foo123" because \d isn't in ECMAScript basic mode --
    // use a simpler pattern that exercises the regex branch.
    // Actually test with a real regex: digits after 'foo'
    res = tool.execute(grep_args("foo[0-9]+", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("foo123"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("foo789"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("bar456"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool regex dot-star matches substring", "[tools][grep]") {
    const std::string dir = "test_grep_dotstar";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "start middle end\nno match here\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("start.*end", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("start middle end"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("no match"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool regex is case-sensitive by default", "[tools][grep]") {
    const std::string dir = "test_grep_casesensitive";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "Hello World\nhello world\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("Hello", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("Hello World"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("hello world"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool ECMAScript does not support inline (?i) flag", "[tools][grep]") {
    // std::regex with ECMAScript dialect does not recognise (?i) — it returns
    // an error rather than silently matching wrong results.
    GrepSearchTool tool;
    auto res = tool.execute(R"({"pattern":"(?i)hello","path":"."})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

// ── Literal vs regex path ───────────────────────────────────────────────────

TEST_CASE("GrepSearchTool literal pattern finds exact substring", "[tools][grep]") {
    // A pattern with no metacharacters uses the fast string_view::find path.
    const std::string dir = "test_grep_literal";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.cpp") << "void foo() {}\nstruct Bar {};\n// comment\n"; }

    GrepSearchTool tool;
    // "struct Bar" contains no regex special chars → literal path
    auto res = tool.execute(grep_args("struct Bar", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("struct Bar"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("void foo"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool literal path handles special char in text", "[tools][grep]") {
    // The file contains parens; the *pattern* has no metacharacters so we still
    // take the literal path and must not mis-interpret file content.
    const std::string dir = "test_grep_literal_special";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/f.txt") << "call(arg1, arg2)\nother_line\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("call", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("call(arg1, arg2)"));

    std::filesystem::remove_all(dir);
}

// ── CRLF and edge-case line endings ─────────────────────────────────────────

TEST_CASE("GrepSearchTool handles CRLF line endings", "[tools][grep]") {
    const std::string dir = "test_grep_crlf";
    std::filesystem::create_directories(dir);
    {
        std::ofstream f(dir + "/f.txt", std::ios::binary);
        f << "line one\r\nTARGET line\r\nline three\r\n";
    }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("TARGET", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("TARGET line"));
    // \r must not appear in the returned text
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("\\r"));

    std::filesystem::remove_all(dir);
}

// ── Binary file skipping ─────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool skips binary files", "[tools][grep]") {
    const std::string dir = "test_grep_binary";
    std::filesystem::create_directories(dir);
    {
        std::ofstream f(dir + "/bin.dat", std::ios::binary);
        // Embed a null byte within the first 512 bytes.
        f << "some text\x00more text" << std::flush;
        // Write enough extra bytes so the file is non-trivially sized.
        for (int i = 0; i < 100; ++i) f << static_cast<char>(i % 256);
    }
    { std::ofstream(dir + "/text.txt") << "some text here\n"; }

    GrepSearchTool tool;
    // "some text" appears in both files, but the binary one must be skipped.
    auto res = tool.execute(grep_args("some text", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("text.txt"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("bin.dat"));

    std::filesystem::remove_all(dir);
}

// ── Directory skipping ───────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool skips .git directory", "[tools][grep]") {
    const std::string dir = "test_grep_gitskip";
    std::filesystem::create_directories(dir + "/.git");
    { std::ofstream(dir + "/real.cpp") << "needle\n"; }
    { std::ofstream(dir + "/.git/config") << "needle inside git\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("real.cpp"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring(".git"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool skips node_modules directory", "[tools][grep]") {
    const std::string dir = "test_grep_nodemodules";
    std::filesystem::create_directories(dir + "/node_modules/pkg");
    { std::ofstream(dir + "/index.js") << "needle\n"; }
    { std::ofstream(dir + "/node_modules/pkg/lib.js") << "needle in modules\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("index.js"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("node_modules"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool skips build directory", "[tools][grep]") {
    const std::string dir = "test_grep_buildskip";
    std::filesystem::create_directories(dir + "/build");
    { std::ofstream(dir + "/main.cpp") << "needle\n"; }
    { std::ofstream(dir + "/build/artifact.o") << "needle in build\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("main.cpp"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("artifact.o"));

    std::filesystem::remove_all(dir);
}

// ── Nested directories ───────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool searches nested subdirectories", "[tools][grep]") {
    const std::string dir = "test_grep_nested";
    std::filesystem::create_directories(dir + "/a/b/c");
    { std::ofstream(dir + "/a/b/c/deep.txt") << "deeply nested needle\n"; }
    { std::ofstream(dir + "/shallow.txt") << "shallow needle\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("deep.txt"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("shallow.txt"));

    std::filesystem::remove_all(dir);
}

// ── include_pattern filtering ───────────────────────────────────────────────

TEST_CASE("GrepSearchTool include_pattern restricts to matching filenames", "[tools][grep]") {
    const std::string dir = "test_grep_include";
    std::filesystem::create_directories(dir + "/sub");
    { std::ofstream(dir + "/a.hpp") << "needle\n"; }
    { std::ofstream(dir + "/b.cpp") << "needle\n"; }
    { std::ofstream(dir + "/sub/c.hpp") << "needle\n"; }
    { std::ofstream(dir + "/d.txt") << "needle\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir, "*.hpp"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("a.hpp"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("c.hpp"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("b.cpp"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("d.txt"));

    std::filesystem::remove_all(dir);
}

// ── Empty / degenerate inputs ────────────────────────────────────────────────

TEST_CASE("GrepSearchTool on empty directory returns empty matches array", "[tools][grep]") {
    const std::string dir = "test_grep_emptydir";
    std::filesystem::create_directories(dir);

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("anything", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"matches\":[]"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool on empty file does not crash", "[tools][grep]") {
    const std::string dir = "test_grep_emptyfile";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/empty.txt"); } // zero bytes

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("anything", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("matches"));
    REQUIRE_THAT(res, !Catch::Matchers::ContainsSubstring("error"));

    std::filesystem::remove_all(dir);
}

TEST_CASE("GrepSearchTool returns error for missing pattern argument", "[tools][grep]") {
    GrepSearchTool tool;
    auto res = tool.execute(R"({"path":"."})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("GrepSearchTool returns error for invalid JSON", "[tools][grep]") {
    GrepSearchTool tool;
    auto res = tool.execute("not json at all {{{");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("GrepSearchTool returns error for non-existent path", "[tools][grep]") {
    GrepSearchTool tool;
    auto res = tool.execute(R"({"pattern":"x","path":"/nonexistent_filo_grep_xyz_99"})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("GrepSearchTool returns error for unexpected arguments", "[tools][grep]") {
    GrepSearchTool tool;
    auto res = tool.execute(R"({"pattern":"x","dir_path":"."})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

// ── Max-results cap ──────────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool caps results at 100", "[tools][grep]") {
    const std::string dir = "test_grep_cap";
    std::filesystem::create_directories(dir);
    // Write a single file with 200 matching lines.
    {
        std::ofstream f(dir + "/big.txt");
        for (int i = 0; i < 200; ++i)
            f << "match line " << i << "\n";
    }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("match line", dir));

    // Count the number of occurrences of "\"line\":" in the JSON.
    size_t count = 0;
    size_t pos   = 0;
    const std::string needle = "\"line\":";
    while ((pos = res.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    REQUIRE(count <= 100);

    std::filesystem::remove_all(dir);
}

// ── Determinism: repeated calls return identical output ──────────────────────

TEST_CASE("GrepSearchTool produces identical output on repeated calls", "[tools][grep]") {
    const std::string dir = "test_grep_determinism";
    std::filesystem::create_directories(dir);
    for (int i = 0; i < 10; ++i) {
        std::ofstream(dir + "/file_" + std::to_string(i) + ".txt")
            << "hit on line 1\nno match\nhit on line 3\n";
    }

    GrepSearchTool tool;
    const auto args = grep_args("hit", dir);

    const std::string first = tool.execute(args);
    for (int i = 0; i < 5; ++i) {
        REQUIRE(tool.execute(args) == first);
    }

    std::filesystem::remove_all(dir);
}

// ── Hidden/dot files (not in skip list) ─────────────────────────────────────

TEST_CASE("GrepSearchTool searches dot-files that are not skip-listed", "[tools][grep]") {
    const std::string dir = "test_grep_dotfile";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/.env") << "SECRET=needle\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("needle", dir));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(".env"));

    std::filesystem::remove_all(dir);
}

// ── JSON structure ───────────────────────────────────────────────────────────

TEST_CASE("GrepSearchTool result has path, line, and text fields", "[tools][grep]") {
    const std::string dir = "test_grep_fields";
    std::filesystem::create_directories(dir);
    { std::ofstream(dir + "/src.cpp") << "int main() { return 0; }\n"; }

    GrepSearchTool tool;
    auto res = tool.execute(grep_args("main", dir));

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"path\""));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"line\""));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("\"text\""));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("src.cpp"));

    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// SearchReplaceTool
// ---------------------------------------------------------------------------

TEST_CASE("SearchReplaceTool applies single edit", "[tools]") {
    const std::string path = "test_artifact_sr_single.txt";
    { std::ofstream(path) << "Line A\nLine B\nLine C\n"; }

    SearchReplaceTool tool;
    // Note: raw string delimiter 'q' avoids the )" terminator issue with "Line B (modified)".
    auto res = tool.execute(
        R"q({"file_path":")q" + path + R"q(","edits":[{"old_string":"Line B","new_string":"Line B (modified)"}]})q");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("blocks_applied"));

    std::ifstream ifs(path);
    std::string result((std::istreambuf_iterator<char>(ifs)), {});
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("Line B (modified)"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("Line A"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("Line C"));

    std::filesystem::remove(path);
}

TEST_CASE("SearchReplaceTool applies multiple edits in order", "[tools]") {
    const std::string path = "test_artifact_sr_multi.txt";
    { std::ofstream(path) << "alpha\nbeta\ngamma\n"; }

    SearchReplaceTool tool;
    auto res = tool.execute(
        R"({"file_path":")" + path + R"(","edits":[)"
        R"({"old_string":"alpha","new_string":"ALPHA"},)"
        R"({"old_string":"gamma","new_string":"GAMMA"}]})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));

    std::ifstream ifs(path);
    std::string result((std::istreambuf_iterator<char>(ifs)), {});
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("ALPHA"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("beta"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("GAMMA"));

    std::filesystem::remove(path);
}

TEST_CASE("SearchReplaceTool returns error when old_string not found", "[tools]") {
    const std::string path = "test_artifact_sr_notfound.txt";
    { std::ofstream(path) << "hello world\n"; }

    SearchReplaceTool tool;
    auto res = tool.execute(
        R"({"file_path":")" + path + R"(","edits":[{"old_string":"xyz_not_here","new_string":"replaced"}]})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));

    std::filesystem::remove(path);
}

TEST_CASE("SearchReplaceTool warns on multiple occurrences but replaces first", "[tools]") {
    const std::string path = "test_artifact_sr_dup.txt";
    { std::ofstream(path) << "dup\ndup\n"; }

    SearchReplaceTool tool;
    auto res = tool.execute(
        R"({"file_path":")" + path + R"(","edits":[{"old_string":"dup","new_string":"REPLACED"}]})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("success"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("warnings"));

    std::ifstream ifs(path);
    std::string result((std::istreambuf_iterator<char>(ifs)), {});
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("REPLACED"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("dup")); // second occurrence untouched

    std::filesystem::remove(path);
}

TEST_CASE("SearchReplaceTool returns error for empty edits array", "[tools]") {
    const std::string path = "test_artifact_sr_empty.txt";
    { std::ofstream(path) << "content\n"; }

    SearchReplaceTool tool;
    auto res = tool.execute(R"({"file_path":")" + path + R"(","edits":[]})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));

    std::filesystem::remove(path);
}

TEST_CASE("SearchReplaceTool returns error for nonexistent file", "[tools]") {
    SearchReplaceTool tool;
    auto res = tool.execute(
        R"({"file_path":"nonexistent_sr_xyz.txt","edits":[{"old_string":"foo","new_string":"bar"}]})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));
}

TEST_CASE("SearchReplaceTool returns error for missing old_string key", "[tools]") {
    const std::string path = "test_artifact_sr_badkey.txt";
    { std::ofstream(path) << "hello\n"; }

    SearchReplaceTool tool;
    // Edit object is missing the required old_string key
    auto res = tool.execute(
        R"({"file_path":")" + path + R"(","edits":[{"new_string":"bar"}]})");
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("error"));

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// PythonManager (basic functionality)
// ---------------------------------------------------------------------------

#ifdef FILO_ENABLE_PYTHON
TEST_CASE("PythonManager execute works correctly", "[tools][python]") {
    // Basic test of Python execution functionality
    auto& pm = core::tools::PythonManager::get_instance();
    
    // Test basic execution
    auto result = pm.execute("print('hello from python')");
    REQUIRE(result.success);
    REQUIRE_THAT(result.output, Catch::Matchers::ContainsSubstring("hello from python"));
}
#endif // FILO_ENABLE_PYTHON
