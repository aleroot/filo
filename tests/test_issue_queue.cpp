#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/goal/IssueHost.hpp"
#include "core/goal/IssueQueue.hpp"
#include "core/review/MergePolicy.hpp"
#include "core/review/RunReview.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

core::review::Finding finding(core::review::Severity severity,
                              core::review::Category category,
                              bool set_category,
                              std::string title) {
    core::review::Finding item;
    item.title = std::move(title);
    item.severity = severity;
    if (set_category) item.category = category;
    item.absolute_file_path = "src/a.cpp";
    item.line_start = 4;
    item.body = "details";
    return item;
}

core::goal::IssueHooks hooks_for(int& fixes, int& reviews, int& publishes,
                                 bool block_once, bool approve, bool changes) {
    core::goal::IssueHooks hooks;
    hooks.prepare = [](const core::goal::IssueRecord& issue) {
        return core::goal::PreparedWorkspace{
            .path = std::filesystem::path{"/tmp"} / ("issue-" + issue.number),
            .branch = "filo/issue-" + issue.number,
            .base_ref = "abc",
        };
    };
    hooks.release = [](const core::goal::PreparedWorkspace&) {};
    hooks.fix = [&fixes](const core::goal::IssueRecord&, const core::goal::PreparedWorkspace&,
                         std::string_view) -> std::expected<std::string, std::string> {
        ++fixes;
        return std::string("fixed");
    };
    hooks.review = [block_once, &reviews](const core::goal::PreparedWorkspace&)
        -> std::expected<core::goal::ReviewStep, std::string> {
        ++reviews;
        core::goal::ReviewStep step;
        if (block_once && reviews == 1) {
            step.findings.blocking.push_back(finding(
                core::review::Severity::High, core::review::Category::Bug, true, "crash"));
        }
        return step;
    };
    hooks.has_changes = [changes](const core::goal::PreparedWorkspace&) { return changes; };
    hooks.file_follow_ups = [](const core::goal::IssueRecord&,
                               const core::review::ClassifiedFindings&) {
        return std::vector<std::string>{"https://example/follow-up"};
    };
    hooks.approve_publish = [approve](const core::goal::IssueRecord&,
                                     const core::goal::PreparedWorkspace&) { return approve; };
    hooks.publish = [&publishes](const core::goal::IssueRecord& issue,
                                 const core::goal::PreparedWorkspace&,
                                 std::string_view) -> std::expected<std::string, std::string> {
        ++publishes;
        return "https://example/pr/" + issue.number;
    };
    return hooks;
}

const core::goal::IssueRecord kIssue{
    .number = "7", .title = "Leak", .url = "u", .body = "body", .priority = 1};

} // namespace

TEST_CASE("Merge blocking follows severity and category", "[review][issues]") {
    using core::review::Category;
    using core::review::Severity;
    CHECK(core::review::blocks_merge(finding(Severity::Critical, Category::Bug, true, "a")));
    CHECK(core::review::blocks_merge(finding(Severity::High, Category::Security, true, "b")));
    CHECK(core::review::blocks_merge(finding(Severity::High, Category::Bug, false, "unset")));
    CHECK_FALSE(core::review::blocks_merge(finding(Severity::High, Category::Style, true, "nit")));
    CHECK_FALSE(core::review::blocks_merge(finding(Severity::Medium, Category::Bug, true, "later")));

    core::review::Report report;
    report.findings.push_back(finding(Severity::High, Category::Bug, true, "block"));
    report.findings.push_back(finding(Severity::Low, Category::Style, true, "nit"));
    report.failed_groups.push_back({"src/a.cpp", "transport"});
    const auto classified = core::review::classify_findings(report);
    CHECK(classified.incomplete);
    REQUIRE(classified.blocking.size() == 1);
    REQUIRE(classified.follow_ups.size() == 1);

    core::review::WorkspaceReview review;
    review.ok = true;
    review.findings = classified;
    const std::string json = core::review::render_review_json(review);
    CHECK_THAT(json, Catch::Matchers::ContainsSubstring("\"blocking_count\":1"));
    CHECK_THAT(json, Catch::Matchers::ContainsSubstring("\"incomplete\":true"));

    // A review that stopped part-way still hands back what it found and which
    // units it never reached.
    review.ok = false;
    review.error = "review was interrupted";
    review.failed_groups = report.failed_groups;
    review.findings.blocking.front().body = std::string(2000, 'x') + " tail";
    const std::string partial = core::review::render_review_json(review);
    CHECK_THAT(partial, Catch::Matchers::ContainsSubstring("\"ok\":false,\"error\":\"review was interrupted\","));
    CHECK_THAT(partial, Catch::Matchers::ContainsSubstring("\"blocking_count\":1"));
    CHECK_THAT(partial, Catch::Matchers::ContainsSubstring(
        R"("unreviewed":[{"unit":"src/a.cpp","reason":"transport"}])"));
    CHECK(partial.find(" tail") == std::string::npos);
    CHECK(partial.back() == '}');

    core::review::WorkspaceReview failed;
    failed.error = "not a git repository";
    CHECK(core::review::render_review_json(failed)
          == R"({"ok":false,"error":"not a git repository"})");

    failed.error = "review was interrupted";
    failed.failed_groups = report.failed_groups;
    CHECK_THAT(core::review::render_review_json(failed), Catch::Matchers::ContainsSubstring(
        R"("unreviewed":[{"unit":"src/a.cpp","reason":"transport"}])"));
}

TEST_CASE("Open issues are ordered by label priority", "[issues]") {
    const char* json = R"([
        {"number": 12, "title": "Later", "url": "https://example/12", "labels": [{"name": "p2"}]},
        {"number": 3, "title": "Now", "url": "https://example/3", "labels": [{"name": "P0"}]},
        {"number": 4, "title": "Soon", "url": "https://example/4", "labels": [{"name": "high"}]}
    ])";
    const auto issues = core::goal::parse_open_issues(json);
    REQUIRE(issues.has_value());
    REQUIRE(issues->size() == 3);
    CHECK(issues->at(0).number == "3");
    CHECK(issues->at(0).priority == 0);
    CHECK(issues->at(1).number == "4");
    CHECK(issues->at(2).number == "12");
}

TEST_CASE("An issue with an open pull request is skipped", "[issues]") {
    int fixes = 0;
    int reviews = 0;
    int publishes = 0;
    auto hooks = hooks_for(fixes, reviews, publishes, false, true, true);
    hooks.existing_pr = [](const core::goal::IssueRecord&) {
        return std::string("https://example/pr/7");
    };
    const auto result = core::goal::run_issues(std::span{&kIssue, 1}, 2, hooks);
    CHECK(result.skipped == 1);
    CHECK(result.opened == 0);
    CHECK(fixes == 0);
    CHECK(publishes == 0);
}

TEST_CASE("A blocking review sends the fix back once", "[issues]") {
    int fixes = 0;
    int reviews = 0;
    int publishes = 0;
    auto hooks = hooks_for(fixes, reviews, publishes, true, true, true);
    const auto result = core::goal::run_issues(std::span{&kIssue, 1}, 2, hooks);
    CHECK(result.opened == 1);
    CHECK(fixes == 2);
    CHECK(reviews == 2);
    CHECK(publishes == 1);
}

TEST_CASE("Blocking findings past the budget do not publish", "[issues]") {
    int fixes = 0;
    int reviews = 0;
    int publishes = 0;
    auto hooks = hooks_for(fixes, reviews, publishes, true, true, true);
    const auto result = core::goal::run_issues(std::span{&kIssue, 1}, 1, hooks);
    CHECK(result.opened == 0);
    CHECK(result.blocked == 1);
    CHECK(publishes == 0);
}

TEST_CASE("A refused publish keeps the worktree", "[issues]") {
    int fixes = 0;
    int reviews = 0;
    int publishes = 0;
    auto hooks = hooks_for(fixes, reviews, publishes, false, false, true);
    int released = 0;
    hooks.release = [&](const core::goal::PreparedWorkspace&) { ++released; };
    const auto result = core::goal::run_issues(std::span{&kIssue, 1}, 2, hooks);
    CHECK(result.blocked == 1);
    CHECK(publishes == 0);
    CHECK(released == 0);
}

TEST_CASE("run_program captures a program without a shell", "[issues]") {
    const auto echoed = core::goal::run_program(
        {}, {"/bin/echo", "issue-queue"}, std::chrono::seconds(5));
    CHECK(echoed.exit_code == 0);
    CHECK_THAT(echoed.output, Catch::Matchers::ContainsSubstring("issue-queue"));
}

TEST_CASE("A fix worktree is created beside the checkout and removed", "[issues][git]") {
    const auto root = std::filesystem::temp_directory_path() / "filo_issue_worktree_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto init = core::goal::run_program(root, {"git", "init"}, std::chrono::seconds(15));
    if (init.exit_code != 0) {
        SKIP("git is not available");
    }
    REQUIRE(core::goal::run_program(
                root, {"git", "-C", root.string(), "config", "user.email", "filo@example.com"},
                std::chrono::seconds(10))
                .exit_code == 0);
    REQUIRE(core::goal::run_program(
                root, {"git", "-C", root.string(), "config", "user.name", "Filo"},
                std::chrono::seconds(10))
                .exit_code == 0);
    std::filesystem::create_directories(root / "src");
    {
        std::ofstream file(root / "src" / "a.txt");
        file << "one\n";
    }
    REQUIRE(core::goal::run_program(root, {"git", "-C", root.string(), "add", "src/a.txt"},
                                    std::chrono::seconds(10))
                .exit_code == 0);
    REQUIRE(core::goal::run_program(
                root, {"git", "-C", root.string(), "commit", "-m", "init"},
                std::chrono::seconds(15))
                .exit_code == 0);

    core::goal::IssueHost host(root);
    core::goal::IssueRecord issue{.number = "4", .title = "Add a line", .priority = 1};
    auto prepared = host.prepare(issue);
    REQUIRE(prepared.has_value());
    CHECK(prepared->path != root);
    CHECK_FALSE(host.has_changes(*prepared));
    {
        std::ofstream file(prepared->path / "src" / "a.txt", std::ios::app);
        file << "two\n";
    }
    CHECK(host.has_changes(*prepared));
    host.release(*prepared);
    CHECK_FALSE(std::filesystem::exists(prepared->path));
    std::filesystem::remove_all(root);
}
