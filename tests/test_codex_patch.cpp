#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/ApplyPatchTool.hpp"
#include "core/tools/PatchTargets.hpp"
#include "TestSessionContext.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace core::tools;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string json_args(std::string_view patch, const std::filesystem::path& dir) {
    std::string escaped;
    escaped.reserve(patch.size() + 16);
    for (const char c : patch) {
        if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '"') escaped += "\\\"";
        else escaped += c;
    }
    return std::format(
        "{{\"patch\":\"{}\",\"working_dir\":\"{}\"}}", escaped, dir.string());
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}

struct TempDir {
    std::filesystem::path path;
    explicit TempDir(std::string_view name)
        : path(std::filesystem::current_path() / (std::string(name) + std::to_string(getpid()))) {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::filesystem::remove_all(path); }
};

std::string apply(ApplyPatchTool& tool, std::string_view patch, const std::filesystem::path& dir) {
    const auto context = test_support::make_session_context({.primary = dir, .enforce = true});
    return tool.execute(json_args(patch, dir), context);
}

} // namespace

TEST_CASE("Codex apply_patch updates a file by context", "[tools][patch]") {
    TempDir dir("test_artifact_codex_update_");
    const auto file = dir.path / "note.txt";
    { std::ofstream out(file); out << "alpha\nbeta\ngamma\n"; }

    const char* patch =
        "*** Begin Patch\n"
        "*** Update File: note.txt\n"
        "@@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "*** End Patch\n";

    ApplyPatchTool tool;
    const auto result = apply(tool, patch, dir.path);
    REQUIRE_THAT(result, ContainsSubstring("success"));
    CHECK(read_file(file) == "alpha\nBETA\ngamma\n");
}

TEST_CASE("Codex apply_patch matches context with trailing whitespace removed", "[tools][patch]") {
    TempDir dir("test_artifact_codex_fuzz_");
    const auto file = dir.path / "note.txt";
    { std::ofstream out(file); out << "alpha \nbeta\n"; }

    const char* patch =
        "*** Begin Patch\n"
        "*** Update File: note.txt\n"
        "@@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        "*** End Patch\n";

    ApplyPatchTool tool;
    REQUIRE_THAT(apply(tool, patch, dir.path), ContainsSubstring("success"));
    CHECK(read_file(file) == "alpha\nBETA\n");
}

TEST_CASE("Codex apply_patch reports a missing context and does not write", "[tools][patch]") {
    TempDir dir("test_artifact_codex_miss_");
    const auto file = dir.path / "note.txt";
    { std::ofstream out(file); out << "alpha\nbeta\n"; }

    const char* patch =
        "*** Begin Patch\n"
        "*** Update File: note.txt\n"
        "@@\n"
        " nowhere\n"
        "-beta\n"
        "+BETA\n"
        "*** End Patch\n";

    ApplyPatchTool tool;
    const auto result = apply(tool, patch, dir.path);
    REQUIRE_THAT(result, ContainsSubstring("Could not find this context"));
    CHECK(read_file(file) == "alpha\nbeta\n");
}

TEST_CASE("Codex apply_patch rejects a context that matches twice", "[tools][patch]") {
    TempDir dir("test_artifact_codex_ambiguous_");
    const auto file = dir.path / "note.txt";
    { std::ofstream out(file); out << "same\nsame\n"; }

    const char* patch =
        "*** Begin Patch\n"
        "*** Update File: note.txt\n"
        "@@\n"
        "-same\n"
        "+other\n"
        "*** End Patch\n";

    ApplyPatchTool tool;
    REQUIRE_THAT(apply(tool, patch, dir.path), ContainsSubstring("more than once"));
    CHECK(read_file(file) == "same\nsame\n");
}

TEST_CASE("Codex apply_patch adds, moves, and deletes", "[tools][patch]") {
    TempDir dir("test_artifact_codex_files_");
    { std::ofstream out(dir.path / "old.txt"); out << "hello\nworld\n"; }
    { std::ofstream out(dir.path / "gone.txt"); out << "bye\n"; }

    const char* patch =
        "*** Begin Patch\n"
        "*** Add File: created.txt\n"
        "+fresh\n"
        "*** Update File: old.txt\n"
        "*** Move to: nested/new.txt\n"
        "@@\n"
        " hello\n"
        "-world\n"
        "+there\n"
        "*** Delete File: gone.txt\n"
        "*** End Patch\n";

    ApplyPatchTool tool;
    const auto result = apply(tool, patch, dir.path);
    REQUIRE_THAT(result, ContainsSubstring("success"));
    CHECK(read_file(dir.path / "created.txt") == "fresh\n");
    CHECK(read_file(dir.path / "nested" / "new.txt") == "hello\nthere\n");
    CHECK_FALSE(std::filesystem::exists(dir.path / "old.txt"));
    CHECK_FALSE(std::filesystem::exists(dir.path / "gone.txt"));
}

TEST_CASE("A unified diff with a bad hunk count is refused before patch runs", "[tools][patch]") {
    TempDir dir("test_artifact_bad_hunk_");
    const auto file = dir.path / "note.txt";
    { std::ofstream out(file); out << "alpha\nbeta\ngamma\n"; }

    const char* patch =
        "--- a/note.txt\n"
        "+++ b/note.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " alpha\n"
        " beta\n"
        " gamma\n";

    ApplyPatchTool tool;
    const auto result = apply(tool, patch, dir.path);
    REQUIRE_THAT(result, ContainsSubstring("declares 2 old lines"));
    REQUIRE_THAT(result, ContainsSubstring("note.txt"));
    CHECK(read_file(file) == "alpha\nbeta\ngamma\n");
    CHECK_FALSE(std::filesystem::exists(dir.path / "Oops.rej"));
}

TEST_CASE("A fenced envelope and a git unified diff both apply", "[tools][patch]") {
    TempDir dir("test_artifact_codex_providers_");
    {
        std::ofstream out(dir.path / "note.txt");
        out << "alpha\nbeta\ngamma\n";
    }
    {
        std::ofstream out(dir.path / "other.txt");
        out << "one\ntwo\n";
    }

    const char* fenced =
        "```diff\n"
        "*** Begin Patch\n"
        "*** Update File: note.txt\n"
        "@@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "*** End Patch\n"
        "```\n";
    ApplyPatchTool tool;
    REQUIRE_THAT(apply(tool, fenced, dir.path), ContainsSubstring("success"));
    CHECK(read_file(dir.path / "note.txt") == "alpha\nBETA\ngamma\n");

    const char* git_diff =
        "diff --git a/other.txt b/other.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/other.txt\n"
        "+++ b/other.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " one\n"
        "-two\n"
        "+TWO\n";
    REQUIRE_THAT(apply(tool, git_diff, dir.path), ContainsSubstring("success"));
    CHECK(read_file(dir.path / "other.txt") == "one\nTWO\n");
}

TEST_CASE("patch_target_paths reads Codex file operations and not unified headers", "[tools][patch]") {
    const auto paths = patch_target_paths(
        "*** Begin Patch\n"
        "*** Add File: created.txt\n"
        "+x\n"
        "*** Update File: old.txt\n"
        "*** Move to: nested/new.txt\n"
        "*** Delete File: gone.txt\n"
        "--- not-a-header\n"
        "*** End Patch\n");
    CHECK(paths == std::vector<std::string>{
        "created.txt", "old.txt", "nested/new.txt", "gone.txt"});
}
