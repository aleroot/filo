#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/context/CodexInstructions.hpp"
#include "core/context/CodexPromptFraming.hpp"
#include "core/context/ContextBuilder.hpp"
#include "core/context/ModelGuidance.hpp"
#include "core/context/WorkingRules.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <string_view>

namespace {

using core::context::ModelGuidance;
using core::context::WorkingRule;
using core::context::WorkingRuleOverride;

constexpr std::string_view kModes[] = {"BUILD", "AUTO", "PLAN", "RESEARCH", "EXECUTE", "DEBUG"};

class TempWorkspace {
public:
    TempWorkspace()
        : path_(std::filesystem::temp_directory_path()
                / std::format("filo_model_guidance_{}",
                              std::chrono::steady_clock::now().time_since_epoch().count())) {
        std::filesystem::create_directories(path_);
    }
    ~TempWorkspace() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

[[nodiscard]] std::string runtime_layer(const std::filesystem::path& workspace,
                                        std::string_view mode,
                                        const ModelGuidance* guidance) {
    const auto context = test_support::make_session_context(core::workspace::WorkspaceSnapshot{
        .primary = workspace, .enforce = true, .version = 1});
    const auto layers = core::context::ContextBuilder(context)
        .with_mode(mode)
        .with_model_guidance(guidance)
        .include_project_context(false)
        .build_layers();
    REQUIRE_FALSE(layers.empty());
    REQUIRE(layers[0].name == "runtime");
    return layers[0].content;
}

[[nodiscard]] std::string bullet(std::string_view text) {
    return "\n- " + std::string(text);
}

} // namespace

TEST_CASE("Model guidance recognizes GPT-5 and later under any vendor prefix",
          "[context][model_guidance]") {
    for (const std::string_view model :
         {"gpt-5", "gpt-5.6-terra", "gpt-6-luna", "gpt-6.2-sol", "gpt-5.2-codex",
          "openai/gpt-6.2", "GPT-6.1-Terra", "gpt-10"}) {
        CAPTURE(model);
        const auto* profile = core::context::find_model_guidance(model);
        REQUIRE(profile != nullptr);
        CHECK(profile->name == "openai_gpt");
    }
    for (const std::string_view model :
         {"", "gpt-", "gpt-4o", "gpt-4.1-mini", "gpt-oss-120b", "o3", "chatgpt-4o-latest",
          "router", "auto", "claude-opus-5-5", "gemini-4-hydrogen", "glm-5.3", "qwen4-max",
          "grok-4.7"}) {
        CAPTURE(model);
        CHECK(core::context::find_model_guidance(model) == nullptr);
    }
}

TEST_CASE("Working rules render overrides in the shared rule's place",
          "[context][model_guidance]") {
    const auto shared = core::context::render_working_rules();
    CHECK_THAT(shared, Catch::Matchers::StartsWith("\n\n[Working rules]\n- Bias to action."));

    constexpr std::array overrides{
        WorkingRuleOverride{.rule = WorkingRule::Assumptions, .text = "Second, restated."},
    };
    const auto restated = core::context::render_working_rules(overrides);

    const auto shared_text = core::context::working_rule_text(WorkingRule::Assumptions);
    CHECK(restated.find(shared_text) == std::string::npos);
    const auto first = restated.find(
        bullet(core::context::working_rule_text(WorkingRule::BiasToAction)));
    const auto second = restated.find(bullet("Second, restated."));
    const auto third = restated.find(
        bullet(core::context::working_rule_text(WorkingRule::ReportedProblems)));
    REQUIRE(first != std::string::npos);
    REQUIRE(second != std::string::npos);
    REQUIRE(third != std::string::npos);
    CHECK(first < second);
    CHECK(second < third);
}

TEST_CASE("Every profile renders in every mode in place of the shared rules",
          "[context][model_guidance]") {
    TempWorkspace workspace;
    for (const auto& profile : core::context::model_guidance_profiles()) {
        CAPTURE(profile.name);
        for (const auto mode : kModes) {
            CAPTURE(mode);
            const auto shared = runtime_layer(workspace.path(), mode, nullptr);
            const auto tailored = runtime_layer(workspace.path(), mode, &profile);

            for (const auto& override_rule : profile.rule_overrides) {
                const auto shared_bullet =
                    bullet(core::context::working_rule_text(override_rule.rule));
                CHECK_THAT(shared, Catch::Matchers::ContainsSubstring(shared_bullet));
                CHECK(tailored.find(shared_bullet) == std::string::npos);
                CHECK_THAT(tailored, Catch::Matchers::ContainsSubstring(
                                         bullet(override_rule.text)));
            }
            // The profile's instructions follow the working rules.
            const auto instructions = tailored.find(profile.instructions);
            REQUIRE(instructions != std::string::npos);
            CHECK(tailored.find("[Working rules]") < instructions);
            // Rules a profile does not restate keep the shared wording.
            CHECK_THAT(tailored, Catch::Matchers::ContainsSubstring(bullet(
                core::context::working_rule_text(WorkingRule::RepositorySafety))));
        }
    }
}

TEST_CASE("A code-mode model reads Codex's instructions instead of Filo's doctrine",
          "[context][model_guidance][codex]") {
    TempWorkspace workspace;
    const auto bundled = core::context::codex_prompt_for(
        "gpt-6.2-sol", "BUILD", {}, {}, {}, true);
    CHECK_THAT(bundled.instructions, Catch::Matchers::StartsWith(
        "You are Codex, an agent based on GPT-6"));
    CHECK_THAT(bundled.instructions, Catch::Matchers::ContainsSubstring(
        "# Autonomy and persistence"));
    CHECK_THAT(bundled.collaboration, Catch::Matchers::ContainsSubstring(
        "strongly prefer making reasonable assumptions"));

    const auto planned = core::context::codex_prompt_for(
        "gpt-6.2-sol", "PLAN", "Catalog instructions.", "Catalog default.", {}, true);
    CHECK(planned.instructions == "Catalog instructions.");
    CHECK_THAT(planned.collaboration, Catch::Matchers::ContainsSubstring("# Plan Mode"));

    CHECK(core::context::codex_prompt_for("gpt-4o", "BUILD", {}, {}, {}, false)
              .instructions.empty());

    const auto prompt = runtime_layer(workspace.path(), "BUILD", nullptr);
    const auto context = test_support::make_session_context(core::workspace::WorkspaceSnapshot{
        .primary = workspace.path(), .enforce = true, .version = 1});
    const auto layers = core::context::ContextBuilder(context)
        .with_mode("BUILD")
        .with_model_guidance(core::context::find_model_guidance("gpt-6.2-sol"))
        .with_framing(core::context::make_prompt_framing(bundled, nullptr))
        .include_project_context(false)
        .build_layers();
    REQUIRE_FALSE(layers.empty());
    CHECK_THAT(layers[0].content, Catch::Matchers::ContainsSubstring(
        "Autonomy and persistence"));
    CHECK(layers[0].content.find("[Working rules]") == std::string::npos);
    CHECK(layers[0].content.find("smallest correct change") == std::string::npos);
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("[Working rules]"));
}

TEST_CASE("A Codex-instructed prompt frames context the way Codex sends it",
          "[context][model_guidance][codex]") {
    TempWorkspace workspace;
    {
        std::ofstream(workspace.path() / "AGENTS.md") << "Run make test.\n";
        const auto skill = workspace.path() / ".filo" / "skills" / "release";
        std::filesystem::create_directories(skill);
        std::ofstream(skill / "SKILL.md")
            << "---\nname: release\ndescription: Cut a release.\n---\n\nTag it.\n";
    }
    const auto context = test_support::make_session_context(core::workspace::WorkspaceSnapshot{
        .primary = workspace.path(), .enforce = true, .version = 1});
    const auto render = [&](bool codex) {
        core::context::ContextBuilder builder(context);
        builder.with_mode("BUILD");
        if (codex) {
            builder.with_framing(core::context::make_prompt_framing(
                {.instructions = "Codex instructions.",
                 .collaboration = "<collaboration_mode>x</collaboration_mode>"},
                nullptr));
        }
        return builder.build();
    };

    const auto codex = render(true);
    CHECK_THAT(codex, Catch::Matchers::ContainsSubstring(
        "Codex instructions.\n\n<permissions instructions>\nFilesystem sandboxing"));
    CHECK_THAT(codex, Catch::Matchers::ContainsSubstring(
        "`approval_policy` is `unless-trusted`"));
    CHECK(codex.find("</permissions instructions>") < codex.find("<collaboration_mode>"));
    CHECK_THAT(codex, Catch::Matchers::ContainsSubstring(
        "# AGENTS.md instructions for " + std::filesystem::canonical(workspace.path()).string()
        + "\n\n<INSTRUCTIONS>\nRun make test.\n</INSTRUCTIONS>"));
    CHECK_THAT(codex, Catch::Matchers::ContainsSubstring(
        "<skills_instructions>\n## Skills\n"));
    CHECK_THAT(codex, Catch::Matchers::ContainsSubstring(
        "### Available skills\n- release: Cut a release. (tool: activate_skill)\n"));
    CHECK_THAT(codex, Catch::Matchers::ContainsSubstring("<environment_context>\n  <cwd>"));
    CHECK(codex.find("[Project Steering]") == std::string::npos);
    CHECK(codex.find("[Agent Skills]") == std::string::npos);

    const auto filo = render(false);
    CHECK_THAT(filo, Catch::Matchers::ContainsSubstring("[Project Steering]"));
    CHECK_THAT(filo, Catch::Matchers::ContainsSubstring("[Agent Skills]"));
    CHECK(filo.find("<permissions instructions>") == std::string::npos);
    CHECK(filo.find("<environment_context>") == std::string::npos);
}
