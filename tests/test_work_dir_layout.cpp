#include <catch2/catch_test_macros.hpp>

#include "core/cli/WorkDirLayout.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;

using core::cli::resolve_work_dirs;
using core::cli::WorkDirLayout;
using core::workspace::SessionWorkspace;

/// Scratch tree shared by a test case:
///   base/
///     proj-a/
///       nested/
///     proj-b/
struct Fixture {
    fs::path base;

    explicit Fixture(std::string_view tag) {
        std::error_code ec;
        base = fs::temp_directory_path(ec)
            / std::format("filo-workdir-{}-{}", tag, std::rand());
        fs::remove_all(base, ec);
        fs::create_directories(base / "proj-a" / "nested", ec);
        fs::create_directories(base / "proj-b", ec);
    }

    ~Fixture() {
        std::error_code ec;
        fs::remove_all(base, ec);
    }

    [[nodiscard]] fs::path norm(const fs::path& p) const {
        return SessionWorkspace::normalize_path(p);
    }
};

/// Runs @p body with the process cwd moved somewhere unrelated, proving the
/// resolver never consults it. Restores the original cwd afterwards so the rest
/// of the suite is unaffected.
template <typename Fn>
auto with_unrelated_cwd(const fs::path& unrelated, Fn&& body) {
    std::error_code ec;
    const auto original = fs::current_path(ec);
    fs::current_path(unrelated, ec);
    auto result = body();
    ec.clear();
    fs::current_path(original, ec);
    return result;
}

[[nodiscard]] std::vector<std::string> reject_reasons(const WorkDirLayout& layout) {
    std::vector<std::string> reasons;
    for (const auto& rejected : layout.rejected) {
        reasons.push_back(rejected.reason);
    }
    return reasons;
}

} // namespace

TEST_CASE("resolve_work_dirs returns an empty layout when no -w was given",
          "[cli][workspace]") {
    const auto layout = resolve_work_dirs({}, "/tmp");

    REQUIRE(layout.primary.empty());
    REQUIRE(layout.additional.empty());
    REQUIRE(layout.rejected.empty());
}

TEST_CASE("resolve_work_dirs absolutizes a single primary root",
          "[cli][workspace]") {
    const Fixture fixture("single");

    const auto layout = resolve_work_dirs({"proj-a"}, fixture.base);

    REQUIRE(layout.primary == fixture.norm(fixture.base / "proj-a"));
    REQUIRE(layout.additional.empty());
    REQUIRE(layout.rejected.empty());
}

// The regression this whole layer exists for: `filo -w proj -w .` used to chdir
// into `proj` first and then resolve `.` against it, silently collapsing the
// second root onto the primary and losing the enclosing workspace.
TEST_CASE("resolve_work_dirs resolves '.' against the launch directory, not the primary",
          "[cli][workspace]") {
    const Fixture fixture("dot");

    const auto layout = with_unrelated_cwd(fixture.base / "proj-b", [&] {
        return resolve_work_dirs({"proj-a", "."}, fixture.base);
    });

    REQUIRE(layout.primary == fixture.norm(fixture.base / "proj-a"));
    REQUIRE(layout.additional.size() == 1);
    REQUIRE(layout.additional.front() == fixture.norm(fixture.base));
    REQUIRE(layout.rejected.empty());

    // And the composed workspace actually grants the enclosing directory.
    const SessionWorkspace workspace{core::workspace::WorkspaceSnapshot{
        .primary = layout.primary,
        .additional = layout.additional,
        .enforce = true,
    }};
    REQUIRE(workspace.allows_read(fixture.base / "proj-b" / "file.txt"));
    REQUIRE(workspace.allows_read(fixture.base / "loose.txt"));
}

TEST_CASE("resolve_work_dirs keeps relative sibling roots out of the primary",
          "[cli][workspace]") {
    const Fixture fixture("sibling");

    const auto layout = with_unrelated_cwd(fixture.base / "proj-a", [&] {
        return resolve_work_dirs({"proj-a", "proj-b"}, fixture.base);
    });

    REQUIRE(layout.additional.size() == 1);
    REQUIRE(layout.additional.front() == fixture.norm(fixture.base / "proj-b"));
    REQUIRE(layout.rejected.empty());
}

TEST_CASE("resolve_work_dirs keeps a parent root that contains the primary",
          "[cli][workspace]") {
    const Fixture fixture("parent");

    // `-w proj-a/nested -w .` from base: the additional root is the primary's
    // own grandparent. Containment runs one way only — a root nested *under*
    // the primary is redundant, but a root above it is how a session reaches
    // the primary's siblings, so it must survive.
    const auto layout = resolve_work_dirs({"proj-a/nested", "."}, fixture.base);

    REQUIRE(layout.primary == fixture.norm(fixture.base / "proj-a" / "nested"));
    REQUIRE(layout.additional.size() == 1);
    REQUIRE(layout.additional.front() == fixture.norm(fixture.base));
    REQUIRE(layout.rejected.empty());
}

TEST_CASE("resolve_work_dirs accepts absolute additional roots unchanged",
          "[cli][workspace]") {
    const Fixture fixture("absolute");

    const auto layout = resolve_work_dirs(
        {"proj-a", (fixture.base / "proj-b").string()},
        fixture.base);

    REQUIRE(layout.additional.size() == 1);
    REQUIRE(layout.additional.front() == fixture.norm(fixture.base / "proj-b"));
    REQUIRE(layout.rejected.empty());
}

TEST_CASE("resolve_work_dirs drops redundant roots and says why",
          "[cli][workspace]") {
    const Fixture fixture("redundant");

    const auto layout = resolve_work_dirs(
        {"proj-a", "proj-a/nested", "proj-b", "proj-b", "missing", ""},
        fixture.base);

    REQUIRE(layout.primary == fixture.norm(fixture.base / "proj-a"));
    REQUIRE(layout.additional.size() == 1);
    REQUIRE(layout.additional.front() == fixture.norm(fixture.base / "proj-b"));

    REQUIRE(reject_reasons(layout) == std::vector<std::string>{
        "it is already inside the primary workspace",
        "it was already added",
        "it does not exist",
        "the path is empty",
    });
}

TEST_CASE("resolve_work_dirs drops an additional root that is the primary itself",
          "[cli][workspace]") {
    const Fixture fixture("same");

    const auto layout = resolve_work_dirs({"proj-a", "./proj-a"}, fixture.base);

    REQUIRE(layout.additional.empty());
    REQUIRE(reject_reasons(layout) == std::vector<std::string>{"it is the primary workspace"});
}

TEST_CASE("resolve_work_dirs leaves an empty primary for the caller to report",
          "[cli][workspace]") {
    const Fixture fixture("empty-primary");

    const auto layout = resolve_work_dirs({"", "proj-b"}, fixture.base);

    // main() still attempts the chdir and surfaces the errno text, exactly as
    // it did before the resolver existed.
    REQUIRE(layout.primary.empty());
    REQUIRE(layout.additional.size() == 1);
    REQUIRE(layout.additional.front() == fixture.norm(fixture.base / "proj-b"));
    REQUIRE(reject_reasons(layout).front() == "the path is empty");
}

TEST_CASE("SessionWorkspace drops an additional root identical to the primary",
          "[cli][workspace]") {
    const Fixture fixture("snapshot-dedup");
    const auto primary = fixture.norm(fixture.base / "proj-a");

    const SessionWorkspace workspace{core::workspace::WorkspaceSnapshot{
        .primary = primary,
        .additional = {primary, fixture.norm(fixture.base / "proj-b")},
        .enforce = true,
    }};

    REQUIRE(workspace.additional().size() == 1);
    REQUIRE(workspace.additional().front() == fixture.norm(fixture.base / "proj-b"));
}
