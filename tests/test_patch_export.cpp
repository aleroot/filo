#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/changes/PatchExport.hpp"

#include <algorithm>
#include <string>
#include <vector>

using core::changes::export_patch;
using core::changes::FileChange;
using core::changes::FileChangeContent;
using core::changes::FileChangeKind;
using core::changes::PatchExport;
using core::changes::TurnChanges;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

namespace {

[[nodiscard]] FileChange modified(
    std::string path,
    std::string diff,
    std::size_t added = 1,
    std::size_t deleted = 1) {
    return FileChange{
        .kind = FileChangeKind::Modified,
        .content = FileChangeContent::Text,
        .path = std::move(path),
        .diff = std::move(diff),
        .added = added,
        .deleted = deleted,
    };
}

/// The header pair `core::tools::detail::build_unified_diff` writes: both sides
/// name the destination, because the tracker diffs content and not paths.
[[nodiscard]] std::string stored_diff(std::string_view path, std::string_view hunk) {
    return std::string("--- a/") + std::string(path) + "\n+++ b/" + std::string(path)
        + "\n" + std::string(hunk);
}

constexpr std::string_view kHunk = "@@ -1,1 +1,1 @@\n-old\n+new\n";

} // namespace

TEST_CASE("export_patch shapes one change like git diff", "[changes][patch_export]") {
    const auto exported = export_patch(modified("src/app.cpp", stored_diff("src/app.cpp", kHunk)));

    REQUIRE(exported.files == 1);
    REQUIRE(exported.undiffed == 0);
    REQUIRE_FALSE(exported.empty());
    CHECK_THAT(
        exported.patch,
        StartsWith("diff --git a/src/app.cpp b/src/app.cpp\n"
                   "--- a/src/app.cpp\n"
                   "+++ b/src/app.cpp\n"
                   "@@ -1,1 +1,1 @@\n-old\n+new\n"));
    // Exactly one header pair: the stored pair is replaced, not repeated.
    CHECK(std::ranges::count(exported.patch, '\n') == 6);
}

TEST_CASE("export_patch keeps every change it can, in order", "[changes][patch_export]") {
    const std::vector<FileChange> changes{
        modified("a.cpp", stored_diff("a.cpp", kHunk)),
        modified("b.cpp", stored_diff("b.cpp", kHunk)),
    };
    const auto exported = export_patch(changes);

    REQUIRE(exported.files == 2);
    const auto a = exported.patch.find("diff --git a/a.cpp b/a.cpp");
    const auto b = exported.patch.find("diff --git a/b.cpp b/b.cpp");
    REQUIRE(a != std::string::npos);
    REQUIRE(b != std::string::npos);
    CHECK(a < b);
}

TEST_CASE("export_patch names both sides of a renamed file", "[changes][patch_export]") {
    FileChange change = modified("src/new.cpp", stored_diff("src/new.cpp", kHunk));
    change.kind = FileChangeKind::Renamed;
    change.previous_path = "src/old.cpp";

    const auto exported = export_patch(change);

    REQUIRE(exported.files == 1);
    CHECK_THAT(
        exported.patch,
        StartsWith("diff --git a/src/old.cpp b/src/new.cpp\n"
                   "rename from src/old.cpp\n"
                   "rename to src/new.cpp\n"
                   "--- a/src/old.cpp\n"
                   "+++ b/src/new.cpp\n"
                   "@@ -1,1 +1,1 @@\n-old\n+new\n"));
}

TEST_CASE("export_patch counts changes it cannot detail", "[changes][patch_export]") {
    FileChange pure_rename{
        .kind = FileChangeKind::Renamed,
        .path = "src/new.cpp",
        .previous_path = "src/old.cpp",
    };
    FileChange binary{
        .kind = FileChangeKind::Modified,
        .content = FileChangeContent::Binary,
        .path = "assets/logo.png",
        .added = 0,
        .deleted = 0,
    };
    FileChange budget_spent{
        .kind = FileChangeKind::Added,
        .content = FileChangeContent::BudgetSpent,
        .path = "generated/table.cpp",
        .added = 4000,
    };
    const std::vector<FileChange> changes{
        modified("src/app.cpp", stored_diff("src/app.cpp", kHunk)),
        pure_rename,
        binary,
        budget_spent,
    };

    const auto exported = export_patch(changes);

    CHECK(exported.files == 1);
    CHECK(exported.undiffed == 3);
    CHECK_THAT(exported.patch, ContainsSubstring("src/app.cpp"));
    CHECK_THAT(exported.patch, !ContainsSubstring("logo.png"));
    CHECK_THAT(exported.patch, !ContainsSubstring("table.cpp"));
}

TEST_CASE("export_patch of nothing exports nothing", "[changes][patch_export]") {
    const auto exported = export_patch(TurnChanges{});

    CHECK(exported.empty());
    CHECK(exported.files == 0);
    CHECK(exported.undiffed == 0);
    CHECK(exported.patch.empty());
}

TEST_CASE("export_patch reads a whole turn", "[changes][patch_export]") {
    TurnChanges turn;
    turn.files.push_back(modified("src/app.cpp", stored_diff("src/app.cpp", kHunk)));
    turn.partial_enumeration = true;

    const auto exported = export_patch(turn);

    CHECK(exported.files == 1);
    CHECK_THAT(exported.patch, ContainsSubstring("diff --git a/src/app.cpp b/src/app.cpp"));
}

TEST_CASE("export_patch passes through a diff with no header pair", "[changes][patch_export]") {
    // The exporter strips the stored pair because it writes its own. A body it
    // does not recognise is appended whole rather than truncated by a guess.
    const auto exported = export_patch(modified("src/app.cpp", "@@ -1,1 +1,1 @@\n-old\n+new\n"));

    REQUIRE(exported.files == 1);
    CHECK_THAT(
        exported.patch,
        StartsWith("diff --git a/src/app.cpp b/src/app.cpp\n"
                   "--- a/src/app.cpp\n"
                   "+++ b/src/app.cpp\n"
                   "@@ -1,1 +1,1 @@\n-old\n+new\n"));
}
