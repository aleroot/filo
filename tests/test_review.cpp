#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/review/Engine.hpp"
#include "core/review/Findings.hpp"
#include "core/review/Plan.hpp"
#include "core/review/Prompt.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <format>
#include <memory>
#include <ranges>
#include <string>
#include <thread>
#include <vector>

using namespace core::review;
using Catch::Matchers::ContainsSubstring;

namespace {

class CapturingJsonProvider : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        requests.push_back(request);

        core::llm::StreamChunk content;
        if (request.messages.empty()) {
            content.content = "{}";
        } else if (request.messages.back().content.find("analysing risk") != std::string::npos) {
            content.content = R"({"risks":[{"title":"null deref","why":"pointer may be null","path":"big.cpp"}]})";
        } else {
            content.content =
                R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"No blocking issues found in the supplied patch.","overall_confidence_score":0.9})";
        }
        callback(content);

        core::llm::StreamChunk final_chunk;
        final_chunk.is_final = true;
        callback(final_chunk);
    }

    std::vector<core::llm::ChatRequest> requests;
};

class FailingProvider : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        core::llm::StreamChunk final_chunk;
        final_chunk.is_final = true;
        final_chunk.is_error = true;
        callback(final_chunk);
    }
};

class ScriptedRunner final : public TurnRunner {
public:
    std::vector<Request> requests;
    std::string review_json =
        R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"ok","overall_confidence_score":0.9})";

    Response run(const Request& request) override {
        requests.push_back(request);
        Response response;
        if (request.prompt.find("analysing risk") != std::string::npos) {
            response.text = R"({"risks":[]})";
        } else {
            response.text = review_json;
        }
        return response;
    }
};

// A read-only tool the review turn is allowed to call.
class ReviewProbeTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = "review_probe_tool",
            .title = "Review probe",
            .description = "Read-only no-op used by review turn-runner tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = true,
                .idempotent_hint = true,
            },
        };
    }

    [[nodiscard]] std::string execute(
        const std::string&,
        const core::context::SessionContext&) override {
        executed.store(true, std::memory_order_release);
        return R"({"ok":true})";
    }

    std::atomic_bool executed{false};
};

// First step asks for a tool, second step answers. The agent finishes such a
// turn on its own thread, so a runner that does not wait sees nothing at all.
class ToolThenJsonProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const int call = calls.fetch_add(1, std::memory_order_acq_rel);
        if (call == 0) {
            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = "review-probe-1";
            tool_call.type = "function";
            tool_call.function.name = "review_probe_tool";
            tool_call.function.arguments = "{}";

            core::llm::StreamChunk chunk;
            chunk.tools = {std::move(tool_call)};
            chunk.is_final = true;
            callback(chunk);
            return;
        }
        callback(core::llm::StreamChunk::make_content(
            R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"reviewed after tool use"})"));
        callback(core::llm::StreamChunk::make_final());
    }

    std::atomic<int> calls{0};
};

// Fails exactly one unit and reviews the rest, so a campaign can be observed
// surviving a single bad file instead of collapsing with it.
class OneBadGroupRunner final : public TurnRunner {
public:
    std::string failing_path = "alpha.cpp";
    std::vector<Request> requests;

    Response run(const Request& request) override {
        requests.push_back(request);
        Response response;
        if (request.prompt.find("analysing risk") != std::string::npos) {
            response.text = R"({"risks":[]})";
            return response;
        }
        if (request.prompt.find(failing_path) != std::string::npos) {
            response.error = "provider returned an error";
            return response;
        }
        response.text =
            R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"ok","overall_confidence_score":0.9})";
        return response;
    }
};

// Answers the tool-enabled turn with prose and only produces JSON once tools
// are withdrawn: the shape of a model that "explains" instead of answering.
class ProseThenJsonRunner final : public TurnRunner {
public:
    std::vector<Request> requests;

    Response run(const Request& request) override {
        requests.push_back(request);
        Response response;
        if (request.prompt.find("analysing risk") != std::string::npos) {
            response.text = R"({"risks":[]})";
            return response;
        }
        if (!request.allowed_tools.empty()) {
            response.text = "I looked at the diff and everything seems fine to me.";
            return response;
        }
        response.text =
            R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"retried without tools","overall_confidence_score":0.9})";
        return response;
    }
};

class ForkingRunner final : public TurnRunner {
public:
    struct State {
        std::atomic<int> active{0};
        std::atomic<int> max_active{0};
        std::atomic<int> runs{0};
    };

    explicit ForkingRunner(std::shared_ptr<State> state)
        : state_(std::move(state)) {}

    Response run(const Request&) override {
        const int active = state_->active.fetch_add(1, std::memory_order_acq_rel) + 1;
        int observed = state_->max_active.load(std::memory_order_acquire);
        while (active > observed
               && !state_->max_active.compare_exchange_weak(
                   observed,
                   active,
                   std::memory_order_acq_rel,
                   std::memory_order_acquire)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        state_->active.fetch_sub(1, std::memory_order_acq_rel);
        state_->runs.fetch_add(1, std::memory_order_release);

        Response response;
        response.text =
            R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"ok"})";
        return response;
    }

    [[nodiscard]] std::unique_ptr<TurnRunner>
    fork_for_parallel_request() const override {
        return std::make_unique<ForkingRunner>(state_);
    }

private:
    std::shared_ptr<State> state_;
};

[[nodiscard]] FileChange make_file(std::string path, int added, int deleted, std::size_t patch_chars) {
    FileChange file;
    file.path = std::move(path);
    file.added_lines = added;
    file.deleted_lines = deleted;
    file.patch = std::string(patch_chars, 'x');
    return file;
}

[[nodiscard]] std::string two_file_patch() {
    return
        "diff --git a/alpha.cpp b/alpha.cpp\n"
        "index 111..222 100644\n"
        "--- a/alpha.cpp\n"
        "+++ b/alpha.cpp\n"
        "@@ -1,2 +1,3 @@\n"
        " int a() {\n"
        "-  return 1;\n"
        "+  return 2;\n"
        " }\n"
        "diff --git a/beta.cpp b/beta.cpp\n"
        "index 333..444 100644\n"
        "--- a/beta.cpp\n"
        "+++ b/beta.cpp\n"
        "@@ -1,2 +1,3 @@\n"
        " int b() {\n"
        "-  return 3;\n"
        "+  return 4;\n"
        " }\n";
}

} // namespace

TEST_CASE("parse_unified_diff splits files and counts changed lines", "[review][plan]") {
    const auto files = parse_unified_diff(two_file_patch());
    REQUIRE(files.size() == 2);
    CHECK(files[0].path == "alpha.cpp");
    CHECK(files[1].path == "beta.cpp");
    CHECK(files[0].added_lines == 1);
    CHECK(files[0].deleted_lines == 1);
    CHECK(files[0].kind == FileChangeKind::Modified);
    CHECK(files[0].patch.find("diff --git a/alpha.cpp") != std::string::npos);
    CHECK(files[1].patch.find("return 4") != std::string::npos);
}

TEST_CASE("parse_unified_diff marks untracked files after the sentinel", "[review][plan]") {
    const std::string patch =
        "\n# Untracked file patches\n"
        "diff --git a/new_feature.cpp b/new_feature.cpp\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new_feature.cpp\n"
        "@@ -0,0 +1,1 @@\n"
        "+int answer() { return 42; }\n";
    const auto files = parse_unified_diff(patch);
    REQUIRE(files.size() == 1);
    CHECK(files[0].path == "new_feature.cpp");
    CHECK(files[0].untracked);
    CHECK(files[0].kind == FileChangeKind::Added);
    CHECK(files[0].added_lines == 1);
}

TEST_CASE("parse_unified_diff skips commit headers", "[review][plan]") {
    const std::string patch =
        "commit abcdef\n"
        "Author: Filo <filo@example.invalid>\n"
        "Date:   Mon Sep 1 00:00:00 2026 +0000\n"
        "\n"
        "    Initial\n"
        "\n"
        "diff --git a/initial.cpp b/initial.cpp\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/initial.cpp\n"
        "@@ -0,0 +1,1 @@\n"
        "+int boot() { return 1; }\n";
    const auto files = parse_unified_diff(patch);
    REQUIRE(files.size() == 1);
    CHECK(files[0].path == "initial.cpp");
    CHECK(files[0].kind == FileChangeKind::Added);
}

TEST_CASE("plan_review keeps one tiny file in a single group without a plan pass", "[review][plan]") {
    const auto files = parse_unified_diff(two_file_patch());
    auto one = files;
    one.resize(1);
    const auto plan = plan_review(one);
    REQUIRE(plan.groups.size() == 1);
    CHECK_FALSE(plan.groups.front().needs_plan);
    CHECK(plan.skipped_paths.empty());
}

TEST_CASE("plan_review packs tiny files and isolates large ones", "[review][plan]") {
    std::vector<FileChange> files;
    files.push_back(make_file("tiny_a.cpp", 2, 1, 200));
    files.push_back(make_file("tiny_b.cpp", 3, 0, 200));
    files.push_back(make_file("big.cpp", 80, 10, 4000));

    const auto plan = plan_review(files);
    REQUIRE(plan.groups.size() == 2);

    bool saw_big = false;
    bool saw_packed = false;
    for (const auto& group : plan.groups) {
        if (group.files.size() == 1 && group.files.front().path == "big.cpp") {
            saw_big = true;
            CHECK(group.needs_plan);
        }
        if (group.files.size() == 2) {
            saw_packed = true;
            CHECK_FALSE(group.needs_plan);
        }
    }
    CHECK(saw_big);
    CHECK(saw_packed);
}

TEST_CASE("plan_review skips files over the budget ratio", "[review][plan]") {
    GrouperConfig config;
    config.prompt_budget_chars = 1000;
    config.skip_budget_ratio = 0.80;
    std::vector<FileChange> files;
    files.push_back(make_file("huge.cpp", 1, 0, 900));
    files.push_back(make_file("ok.cpp", 1, 0, 100));

    const auto plan = plan_review(files, config);
    REQUIRE(plan.skipped_paths.size() == 1);
    CHECK(plan.skipped_paths.front() == "huge.cpp");
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().files.front().path == "ok.cpp");
    REQUIRE_FALSE(plan.warnings.empty());
}

TEST_CASE("parse_review_output keeps legacy schema and maps priority to severity", "[review][findings]") {
    const auto report = parse_review_output(
        R"({"findings":[{"title":"[P1] Null deref","body":"ptr may be null","priority":1,"confidence_score":0.8,"code_location":{"absolute_file_path":"src/a.cpp","line_range":{"start":10,"end":12}}}],"overall_correctness":"patch is incorrect","overall_explanation":"Blocking issue.","overall_confidence_score":0.7})");
    REQUIRE(report.findings.size() == 1);
    CHECK(report.findings.front().priority == 1);
    CHECK(effective_severity(report.findings.front()) == Severity::High);
    CHECK(report.overall_correctness == "patch is incorrect");
}

TEST_CASE("parse_review_output drops low-confidence findings", "[review][findings]") {
    const auto report = parse_review_output(
        R"({"findings":[{"title":"Maybe style","body":"nit","confidence_score":0.1,"priority":3,"code_location":{"absolute_file_path":"a.cpp","line_range":{"start":1,"end":1}}}],"overall_correctness":"patch is correct","overall_explanation":"ok"})");
    CHECK(report.findings.empty());
}

TEST_CASE("aggregate_reports dedupes and prefers the higher severity", "[review][findings]") {
    Report first;
    first.findings.push_back(Finding{
        .title = "Null deref",
        .body = "low",
        .priority = 3,
        .severity = Severity::Low,
        .absolute_file_path = "src/a.cpp",
        .line_start = 10,
        .line_end = 12,
    });
    first.overall_correctness = "patch is correct";
    first.overall_explanation = "Looks fine.";

    Report second;
    second.findings.push_back(Finding{
        .title = "Null deref",
        .body = "high",
        .priority = 1,
        .severity = Severity::High,
        .absolute_file_path = "src/a.cpp",
        .line_start = 10,
        .line_end = 12,
    });
    second.overall_correctness = "patch is incorrect";
    second.overall_explanation = "Bug.";

    const auto merged = aggregate_reports(std::array<Report, 2>{first, second});
    REQUIRE(merged.findings.size() == 1);
    CHECK(merged.findings.front().body == "high");
    CHECK(merged.overall_correctness == "patch is incorrect");
}

TEST_CASE("render_report hides low-severity findings by default", "[review][findings]") {
    Report report;
    report.overall_explanation = "Mostly fine.";
    report.findings.push_back(Finding{
        .title = "Urgent bug",
        .body = "fix me",
        .severity = Severity::High,
        .absolute_file_path = "a.cpp",
        .line_start = 1,
        .line_end = 2,
    });
    report.findings.push_back(Finding{
        .title = "Nit",
        .body = "style",
        .severity = Severity::Low,
        .absolute_file_path = "a.cpp",
        .line_start = 3,
        .line_end = 3,
    });

    const auto rendered = render_report(report);
    CHECK_THAT(rendered, ContainsSubstring("# Review"));
    CHECK_THAT(rendered, ContainsSubstring("## Summary"));
    CHECK_THAT(rendered, ContainsSubstring("## Comments"));
    CHECK_THAT(rendered, ContainsSubstring("Urgent bug"));
    CHECK_THAT(rendered, !ContainsSubstring("Nit"));
    CHECK_THAT(rendered, ContainsSubstring("1 low-severity note(s) omitted"));
}

[[nodiscard]] std::string two_isolated_file_patch() {
    std::string patch;
    auto append_file = [&](std::string_view name, int start) {
        patch += std::format(
            "diff --git a/{0} b/{0}\n"
            "--- a/{0}\n"
            "+++ b/{0}\n"
            "@@ -1,1 +1,26 @@\n",
            name);
        for (int i = 0; i < 26; ++i) {
            patch += std::format("+int {0}_{1} = {2};\n", name, i, start + i);
        }
    };
    append_file("alpha.cpp", 1);
    append_file("beta.cpp", 100);
    return patch;
}

TEST_CASE("engine reviews each file group as its own JSON turn", "[review][engine]") {
    auto provider = std::make_shared<CapturingJsonProvider>();
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(provider, "test-model"),
    });

    CampaignInput input;
    input.task = uncommitted_task();
    input.snapshot.worktree_root = "/tmp/repo";
    input.snapshot.status = " M alpha.cpp\n M beta.cpp";
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);
    REQUIRE(result.error.empty());
    REQUIRE_FALSE(result.interrupted);
    CHECK(provider->requests.size() == 2);
    CHECK(result.report.groups_reviewed == 2);
    CHECK(result.report.plan_passes == 0);

    for (const auto& request : provider->requests) {
        CHECK(request.tools.empty());
        CHECK(request.effort == "off");
        CHECK(request.max_tokens == kReviewMaxOutputTokens);
        CHECK(request.response_format.type == core::llm::ResponseFormat::Type::JsonObject);
        REQUIRE_FALSE(request.messages.empty());
        const auto& prompt = request.messages.back().content;
        CHECK_THAT(prompt, ContainsSubstring("The relevant git context is already included"));
        CHECK_THAT(prompt, ContainsSubstring("<git_context>"));
        CHECK_THAT(prompt, ContainsSubstring("## Patch"));
        CHECK_THAT(prompt, ContainsSubstring("Review only the git context supplied below"));
    }

    const auto& first_prompt = provider->requests.front().messages.back().content;
    const auto& second_prompt = provider->requests.back().messages.back().content;
    CHECK_THAT(first_prompt, ContainsSubstring("alpha.cpp"));
    CHECK_THAT(second_prompt, ContainsSubstring("beta.cpp"));
    CHECK(first_prompt.find("beta.cpp") == std::string::npos);
    CHECK(second_prompt.find("alpha.cpp") == std::string::npos);
}

TEST_CASE("engine runs a risk pass before reviewing a large file", "[review][engine]") {
    std::string large_patch =
        "diff --git a/big.cpp b/big.cpp\n"
        "--- a/big.cpp\n"
        "+++ b/big.cpp\n"
        "@@ -1,1 +1,60 @@\n";
    for (int i = 0; i < 60; ++i) {
        large_patch += std::format("+int v{} = {};\n", i, i);
    }

    auto provider = std::make_shared<CapturingJsonProvider>();
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(provider, "test-model"),
    });

    CampaignInput input;
    input.task = uncommitted_task();
    input.snapshot.worktree_root = "/tmp/repo";
    input.snapshot.patch = large_patch;

    const auto result = engine.run(input);
    REQUIRE(result.error.empty());
    REQUIRE(provider->requests.size() == 2);
    CHECK(result.report.plan_passes == 1);
    CHECK_THAT(provider->requests.front().messages.back().content,
               ContainsSubstring("analysing risk"));
    CHECK(provider->requests.front().max_tokens == kPlanMaxOutputTokens);
    CHECK_THAT(provider->requests.back().messages.back().content,
               ContainsSubstring("Prior risk analysis"));
    CHECK_THAT(provider->requests.back().messages.back().content,
               ContainsSubstring("null deref"));
}

TEST_CASE("engine reports progress for planning and each group", "[review][engine][progress]") {
    auto provider = std::make_shared<CapturingJsonProvider>();
    std::vector<Progress> events;
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(provider, "test-model"),
        .on_progress = [&events](const Progress& progress) { events.push_back(progress); },
    });

    CampaignInput input;
    input.task = uncommitted_task();
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);
    REQUIRE(result.error.empty());

    REQUIRE_FALSE(events.empty());
    CHECK(events.front().phase == ProgressPhase::Planned);
    CHECK(events.front().group_total == 2);
    CHECK(events.front().files == 2);
    CHECK(events.front().changed_lines == 52);
    CHECK(events.front().skipped_files == 0);

    const auto started = std::ranges::count_if(events, [](const Progress& p) {
        return p.phase == ProgressPhase::GroupStarted;
    });
    const auto finished = std::ranges::count_if(events, [](const Progress& p) {
        return p.phase == ProgressPhase::GroupFinished;
    });
    CHECK(started == 2);
    CHECK(finished == 2);

    const auto last_group = std::ranges::find_last_if(events, [](const Progress& p) {
        return p.phase == ProgressPhase::GroupFinished;
    });
    REQUIRE_FALSE(last_group.empty());
    CHECK(last_group.front().group_index == 2);
    CHECK(last_group.front().group_total == 2);
    CHECK_FALSE(last_group.front().label.empty());

    // Exactly one terminal event, always last, so a live view can settle.
    CHECK(events.back().phase == ProgressPhase::Finished);
    CHECK_FALSE(events.back().interrupted);
    CHECK(events.back().failure.empty());
    CHECK(std::ranges::count_if(events, [](const Progress& p) {
        return p.phase == ProgressPhase::Finished;
    }) == 1);
}

TEST_CASE("engine reports a terminal event when it fails", "[review][engine][progress]") {
    std::vector<Progress> events;
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(
            std::make_shared<FailingProvider>(), "test-model"),
        .on_progress = [&events](const Progress& progress) { events.push_back(progress); },
    });

    CampaignInput input;
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);
    REQUIRE_FALSE(events.empty());
    CHECK(events.back().phase == ProgressPhase::Finished);
    CHECK(events.back().failure == result.error);
}

TEST_CASE("engine reports a risk pass phase for large files", "[review][engine][progress]") {
    std::string large_patch =
        "diff --git a/big.cpp b/big.cpp\n"
        "--- a/big.cpp\n"
        "+++ b/big.cpp\n"
        "@@ -1,1 +1,60 @@\n";
    for (int i = 0; i < 60; ++i) {
        large_patch += std::format("+int v{} = {};\n", i, i);
    }

    auto provider = std::make_shared<CapturingJsonProvider>();
    std::vector<Progress> events;
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(provider, "test-model"),
        .on_progress = [&events](const Progress& progress) { events.push_back(progress); },
    });

    CampaignInput input;
    input.snapshot.patch = large_patch;

    const auto result = engine.run(input);
    REQUIRE(result.error.empty());
    CHECK(events.front().risk_passes == 1);
    CHECK(std::ranges::any_of(events, [](const Progress& p) {
        return p.phase == ProgressPhase::RiskPass;
    }));
}

TEST_CASE("engine surfaces provider failures instead of reporting a clean review",
          "[review][engine]") {
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(
            std::make_shared<FailingProvider>(), "test-model"),
    });

    CampaignInput input;
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);
    CHECK_FALSE(result.error.empty());
    CHECK(result.report.findings.empty());
    CHECK(result.report.overall_correctness.empty());
}

TEST_CASE("engine skips oversized diffs without calling the model", "[review][engine]") {
    GrouperConfig config;
    config.prompt_budget_chars = 200;
    config.skip_budget_ratio = 0.50;

    auto provider = std::make_shared<CapturingJsonProvider>();
    Engine engine({
        .runner = std::make_unique<ProviderTurnRunner>(provider, "test-model"),
        .grouper = config,
    });

    CampaignInput input;
    input.snapshot.patch =
        "diff --git a/huge.cpp b/huge.cpp\n"
        "--- a/huge.cpp\n"
        "+++ b/huge.cpp\n"
        "@@ -1,1 +1,2 @@\n"
        + std::string(400, 'x') + "\n";

    const auto result = engine.run(input);
    CHECK(provider->requests.empty());
    REQUIRE_FALSE(result.report.skipped_paths.empty());
    CHECK_THAT(render_report(result.report), ContainsSubstring("exceeds"));
}

TEST_CASE("engine keeps reviewing after one group fails", "[review][engine]") {
    auto runner = std::make_unique<OneBadGroupRunner>();
    auto* runner_view = runner.get();
    std::vector<Progress> events;
    Engine engine({
        .runner = std::move(runner),
        .on_progress = [&events](const Progress& progress) { events.push_back(progress); },
    });

    CampaignInput input;
    input.task = uncommitted_task();
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);

    // The healthy unit still produced a review.
    CHECK(result.error.empty());
    CHECK(result.report.groups_reviewed == 1);
    CHECK(runner_view->requests.size() == 2);

    REQUIRE(result.report.failed_groups.size() == 1);
    CHECK(result.report.failed_groups.front().label == "alpha.cpp");
    CHECK_THAT(render_report(result.report), ContainsSubstring("Could not review alpha.cpp"));
    CHECK_THAT(result.report.overall_explanation,
               ContainsSubstring("could not be reviewed"));

    const auto failures = std::ranges::count_if(events, [](const Progress& p) {
        return p.phase == ProgressPhase::GroupFailed;
    });
    CHECK(failures == 1);
    CHECK(events.back().phase == ProgressPhase::Finished);
    CHECK(events.back().failed_groups == 1);
    CHECK(events.back().failure.empty());
}

TEST_CASE("engine keeps findings from finished groups when interrupted",
          "[review][engine]") {
    class ReviewThenStopRunner final : public TurnRunner {
    public:
        std::atomic_bool* stop = nullptr;
        int reviews = 0;

        Response run(const Request& request) override {
            Response response;
            if (request.prompt.find("analysing risk") != std::string::npos) {
                response.text = R"({"risks":[]})";
                return response;
            }
            ++reviews;
            if (request.prompt.find("alpha.cpp") != std::string::npos) {
                response.text =
                    R"({"findings":[{"title":"Off-by-one on the new bound","body":"The added line uses an unchecked index.","severity":"high","category":"bug","confidence_score":0.9,"code_location":{"absolute_file_path":"alpha.cpp","line_range":{"start":2,"end":2}}}],"overall_correctness":"patch is incorrect","overall_explanation":"alpha.cpp introduces a bounds error.","overall_confidence_score":0.9})";
            } else {
                response.text =
                    R"({"findings":[],"overall_correctness":"patch is correct","overall_explanation":"ok","overall_confidence_score":0.9})";
            }
            if (reviews == 1 && stop != nullptr) {
                stop->store(true, std::memory_order_release);
            }
            return response;
        }
    };

    std::atomic_bool stop{false};
    auto runner = std::make_unique<ReviewThenStopRunner>();
    runner->stop = &stop;
    Engine engine({
        .runner = std::move(runner),
        .cancellation_requested = [&stop] {
            return stop.load(std::memory_order_acquire);
        },
    });

    CampaignInput input;
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);
    CHECK(result.interrupted);
    CHECK(result.error.empty());
    CHECK(result.report.groups_reviewed == 1);
    REQUIRE(result.report.findings.size() == 1);
    CHECK(result.report.findings.front().title == "Off-by-one on the new bound");
    CHECK_THAT(result.report.overall_explanation, ContainsSubstring("stopped after"));
    CHECK_THAT(render_report(result.report), ContainsSubstring("Off-by-one"));
}

TEST_CASE("engine fails only when no group could be reviewed", "[review][engine]") {
    auto runner = std::make_unique<OneBadGroupRunner>();
    runner->failing_path = ".cpp"; // matches every unit
    Engine engine({.runner = std::move(runner)});

    CampaignInput input;
    input.snapshot.patch = two_isolated_file_patch();

    const auto result = engine.run(input);
    CHECK(result.error == "provider returned an error");
    CHECK(result.report.findings.empty());
}

TEST_CASE("engine retries a tool-enabled group without tools when no JSON arrives",
          "[review][engine]") {
    std::string large_patch =
        "diff --git a/big.cpp b/big.cpp\n"
        "--- a/big.cpp\n"
        "+++ b/big.cpp\n"
        "@@ -1,1 +1,60 @@\n";
    for (int i = 0; i < 60; ++i) {
        large_patch += std::format("+int v{} = {};\n", i, i);
    }

    auto runner = std::make_unique<ProseThenJsonRunner>();
    auto* runner_view = runner.get();
    Engine engine({.runner = std::move(runner)});

    CampaignInput input;
    input.snapshot.patch = large_patch;

    const auto result = engine.run(input);
    REQUIRE(result.error.empty());
    CHECK(result.report.groups_reviewed == 1);
    CHECK(result.report.failed_groups.empty());
    CHECK_THAT(result.report.overall_explanation,
               ContainsSubstring("retried without tools"));

    // Risk pass, tool-enabled review, then the JSON-only retry.
    REQUIRE(runner_view->requests.size() == 3);
    CHECK_FALSE(runner_view->requests[1].allowed_tools.empty());
    CHECK(runner_view->requests[2].allowed_tools.empty());
    CHECK(runner_view->requests[2].json_object);
    // Prose is never reported as a review.
    CHECK(result.report.overall_explanation.find("seems fine to me") == std::string::npos);
}

TEST_CASE("engine reviews independent groups concurrently with forked runners",
          "[review][engine][parallel]") {
    std::string patch;
    for (int file = 0; file < 4; ++file) {
        patch += std::format(
            "diff --git a/parallel{}.cpp b/parallel{}.cpp\n"
            "--- a/parallel{}.cpp\n"
            "+++ b/parallel{}.cpp\n"
            "@@ -0,0 +1,30 @@\n",
            file,
            file,
            file,
            file);
        for (int line = 0; line < 30; ++line) {
            patch += std::format("+int value_{}_{} = {};\n", file, line, line);
        }
    }

    auto state = std::make_shared<ForkingRunner::State>();
    Engine engine({
        .runner = std::make_unique<ForkingRunner>(state),
        .max_parallel_groups = 4,
    });

    CampaignInput input;
    input.snapshot.patch = std::move(patch);
    const auto result = engine.run(input);

    CHECK(result.error.empty());
    CHECK(result.report.groups_reviewed == 4);
    CHECK(state->runs.load(std::memory_order_acquire) == 4);
    CHECK(state->max_active.load(std::memory_order_acquire) >= 2);
}

TEST_CASE("AgentTurnRunner waits for a turn that continues through tool calls",
          "[review][engine][regression]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto probe = std::make_shared<ReviewProbeTool>();
    tool_manager.register_tool(probe);

    auto provider = std::make_shared<ToolThenJsonProvider>();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    AgentTurnRunner runner(agent);
    const auto response = runner.run(TurnRunner::Request{
        .prompt = "review this",
        .allowed_tools = {"review_probe_tool"},
    });

    // Without waiting for the agent's done callback this returned empty text
    // and the error "the model returned an empty review response", while the
    // live turn kept writing into the caller's frame.
    CHECK(response.error.empty());
    CHECK_FALSE(response.interrupted);
    CHECK(probe->executed.load(std::memory_order_acquire));
    CHECK(provider->calls.load(std::memory_order_acquire) == 2);
    REQUIRE(parse_review_json(response.text).has_value());
    CHECK_FALSE(agent->turn_in_progress());
}

TEST_CASE("parse_review_json rejects prose and accepts embedded JSON", "[review][findings]") {
    CHECK_FALSE(parse_review_json("The patch looks fine to me.").has_value());
    const auto parsed = parse_review_json(
        R"(Here you go:
{"findings":[],"overall_correctness":"patch is correct"})");
    REQUIRE(parsed.has_value());
    CHECK(parsed->overall_correctness == "patch is correct");
}

TEST_CASE("file_family_key strips test prefixes and suffixes", "[review][plan]") {
    CHECK(file_family_key("src/foo.cpp") == "foo");
    CHECK(file_family_key("src/foo.hpp") == "foo");
    CHECK(file_family_key("include/Foo.h") == "foo");
    CHECK(file_family_key("tests/test_foo.cpp") == "foo");
    CHECK(file_family_key("tests/foo_test.cpp") == "foo");
    CHECK(file_family_key("tests/foo_spec.cpp") == "foo");
    CHECK(file_family_key("Lampo/MemoryRecallPipeline.swift") == "memoryrecallpipeline");
    CHECK(file_family_key("LampoTests/MemoryRecallPipelineTests.swift")
          == "memoryrecallpipeline");
    CHECK(file_family_key("src/FooTest.java") == "foo");
    // Case-sensitive *Tests / *Test only: these are not test files.
    CHECK(file_family_key("src/States.swift") == "states");
    CHECK(file_family_key("src/Latest.swift") == "latest");
}

TEST_CASE("plan_review bundles src, header and test instead of packing by size",
          "[review][plan]") {
    std::vector<FileChange> files;
    files.push_back(make_file("src/foo.cpp", 80, 10, 4000));
    files.push_back(make_file("src/foo.hpp", 30, 4, 800));
    files.push_back(make_file("tests/test_foo.cpp", 12, 0, 400));
    files.push_back(make_file("src/unrelated.cpp", 8, 1, 200));

    const auto plan = plan_review(files);
    REQUIRE(plan.groups.size() == 2);

    bool saw_family = false;
    bool saw_unrelated = false;
    for (const auto& group : plan.groups) {
        if (group.files.size() == 3) {
            saw_family = true;
            CHECK(group.needs_plan);
            std::vector<std::string> paths;
            for (const auto& file : group.files) paths.push_back(file.path);
            CHECK(std::ranges::find(paths, "src/foo.cpp") != paths.end());
            CHECK(std::ranges::find(paths, "src/foo.hpp") != paths.end());
            CHECK(std::ranges::find(paths, "tests/test_foo.cpp") != paths.end());
        }
        if (group.files.size() == 1 && group.files.front().path == "src/unrelated.cpp") {
            saw_unrelated = true;
            CHECK_FALSE(group.needs_plan);
        }
    }
    CHECK(saw_family);
    CHECK(saw_unrelated);
}

TEST_CASE("plan_review bundles Swift source with its Tests file", "[review][plan]") {
    std::vector<FileChange> files;
    files.push_back(make_file("Lampo/MemoryRecallPipeline.swift", 80, 10, 4000));
    files.push_back(make_file("LampoTests/MemoryRecallPipelineTests.swift", 40, 4, 1200));

    const auto plan = plan_review(files);
    REQUIRE(plan.groups.size() == 1);
    CHECK(plan.groups.front().files.size() == 2);
}

TEST_CASE("parse_unified_diff records new-file hunk ranges", "[review][plan]") {
    const auto files = parse_unified_diff(two_file_patch());
    REQUIRE(files.size() == 2);
    REQUIRE(files[0].hunks.size() == 1);
    CHECK(files[0].hunks[0].start == 1);
    CHECK(files[0].hunks[0].end == 3);
    REQUIRE(files[1].hunks.size() == 1);
    CHECK(files[1].hunks[0].start == 1);
    CHECK(files[1].hunks[0].end == 3);
}

TEST_CASE("localize_findings snaps nearby comments and drops the rest", "[review][plan]") {
    FileChange file;
    file.path = "src/a.cpp";
    file.hunks.push_back(LineRange{.start = 10, .end = 20});

    std::vector<Finding> findings;
    findings.push_back(Finding{
        .title = "keep",
        .absolute_file_path = "src/a.cpp",
        .line_start = 12,
        .line_end = 14,
    });
    findings.push_back(Finding{
        .title = "snap",
        .absolute_file_path = "/tmp/repo/src/a.cpp",
        .line_start = 22,
        .line_end = 22,
    });
    findings.push_back(Finding{
        .title = "drop far",
        .absolute_file_path = "src/a.cpp",
        .line_start = 100,
        .line_end = 100,
    });
    findings.push_back(Finding{
        .title = "drop path",
        .absolute_file_path = "src/other.cpp",
        .line_start = 12,
        .line_end = 12,
    });
    findings.push_back(Finding{
        .title = "no line",
        .absolute_file_path = "src/a.cpp",
    });

    const auto dropped = localize_findings(findings, std::array{file}, "/tmp/repo");
    CHECK(dropped == 2);
    REQUIRE(findings.size() == 3);
    CHECK(findings[0].title == "keep");
    CHECK(findings[0].line_start == 12);
    CHECK(findings[1].title == "snap");
    CHECK(findings[1].line_start == 10);
    CHECK(findings[1].line_end == 14);
    CHECK(findings[2].title == "no line");
    CHECK(findings[2].line_start == 10);
    CHECK(findings[2].line_end == 20);
}

TEST_CASE("aggregate_reports keeps localization warnings", "[review][findings]") {
    Report first;
    first.overall_explanation = "ok";
    first.warnings.push_back("Dropped 1 finding(s) whose location did not overlap the diff.");

    const auto merged = aggregate_reports(
        std::array<Report, 1>{first},
        std::vector<std::string>{},
        std::vector<std::string>{"plan warning"});
    REQUIRE(merged.warnings.size() == 2);
    CHECK_THAT(merged.warnings.front(), ContainsSubstring("plan warning"));
    CHECK_THAT(merged.warnings.back(), ContainsSubstring("Dropped 1"));
}

TEST_CASE("review prompt lists changed ranges and withholds tools by default",
          "[review][prompt]") {
    const auto files = parse_unified_diff(two_file_patch());
    REQUIRE_FALSE(files.empty());
    ReviewGroup group;
    group.files.push_back(files.front());

    CampaignInput input;
    input.task = uncommitted_task();
    input.snapshot.worktree_root = "/tmp/repo";

    const auto prompt = build_review_prompt(input, group, {}, false);
    CHECK_THAT(prompt, ContainsSubstring("Changed ranges"));
    CHECK_THAT(prompt, ContainsSubstring("alpha.cpp: 1-3"));
    CHECK_THAT(prompt, ContainsSubstring("Findings outside those ranges are discarded"));
    CHECK_THAT(prompt, ContainsSubstring("Return only valid JSON"));
    CHECK(prompt.find("Read-only tools") == std::string::npos);
}

TEST_CASE("engine grants read-only tools on large groups", "[review][engine]") {
    std::string large_patch =
        "diff --git a/big.cpp b/big.cpp\n"
        "--- a/big.cpp\n"
        "+++ b/big.cpp\n"
        "@@ -1,1 +1,60 @@\n";
    for (int i = 0; i < 60; ++i) {
        large_patch += std::format("+int v{} = {};\n", i, i);
    }

    auto runner = std::make_unique<ScriptedRunner>();
    auto* scripted = runner.get();
    Engine engine({.runner = std::move(runner)});

    CampaignInput input;
    input.snapshot.patch = large_patch;
    const auto result = engine.run(input);
    REQUIRE(result.error.empty());
    REQUIRE(scripted->requests.size() == 2);
    CHECK(scripted->requests.front().allowed_tools.empty());
    CHECK(scripted->requests.front().json_object);

    const auto& review = scripted->requests.back();
    CHECK_FALSE(review.json_object);
    REQUIRE(review.allowed_tools.size() == 3);
    CHECK(std::ranges::find(review.allowed_tools, "read") != review.allowed_tools.end());
    CHECK(std::ranges::find(review.allowed_tools, "grep_search") != review.allowed_tools.end());
    CHECK(std::ranges::find(review.allowed_tools, "file_search") != review.allowed_tools.end());
    CHECK_THAT(review.prompt, ContainsSubstring("After any tool use"));
    CHECK_THAT(review.prompt, ContainsSubstring("Changed ranges"));
    CHECK_THAT(review.prompt, ContainsSubstring("big.cpp: 1-60"));
}

TEST_CASE("engine drops findings that miss the diff hunks", "[review][engine]") {
    auto runner = std::make_unique<ScriptedRunner>();
    runner->review_json =
        R"({"findings":[{"title":"On hunk","body":"real","priority":1,"confidence_score":0.9,"code_location":{"absolute_file_path":"alpha.cpp","line_range":{"start":11,"end":12}}},{"title":"Drift","body":"noise","priority":0,"confidence_score":0.9,"code_location":{"absolute_file_path":"alpha.cpp","line_range":{"start":400,"end":400}}},{"title":"Wrong file","body":"nope","priority":1,"confidence_score":0.9,"code_location":{"absolute_file_path":"other.cpp","line_range":{"start":11,"end":12}}}],"overall_correctness":"patch is incorrect","overall_explanation":"bug","overall_confidence_score":0.8})";
    auto* scripted = runner.get();
    Engine engine({.runner = std::move(runner)});

    CampaignInput input;
    input.snapshot.worktree_root = "/tmp/repo";
    input.snapshot.patch =
        "diff --git a/alpha.cpp b/alpha.cpp\n"
        "--- a/alpha.cpp\n"
        "+++ b/alpha.cpp\n"
        "@@ -10,3 +10,4 @@\n"
        " int a() {\n"
        "-  return 1;\n"
        "+  return 2;\n"
        " }\n";

    const auto result = engine.run(input);
    REQUIRE(result.error.empty());
    CHECK(scripted->requests.size() == 1);
    REQUIRE(result.report.findings.size() == 1);
    CHECK(result.report.findings.front().title == "On hunk");
    CHECK(result.report.findings.front().line_start == 11);
    CHECK(result.report.findings.front().line_end == 12);
    REQUIRE_FALSE(result.report.warnings.empty());
    CHECK_THAT(result.report.warnings.back(), ContainsSubstring("Dropped 2"));
}
