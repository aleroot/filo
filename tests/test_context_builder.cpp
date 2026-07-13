#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextBuilder.hpp"
#include "core/context/SessionContext.hpp"
#include "core/context/SteeringLoader.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/workspace/Workspace.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

class TempDir {
public:
    explicit TempDir(std::filesystem::path path)
        : path_(std::move(path)) {
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

[[nodiscard]] TempDir make_temp_workspace(std::string_view name) {
    const auto base = std::filesystem::temp_directory_path()
        / std::format("{}_{}", name, std::chrono::steady_clock::now().time_since_epoch().count());
    return TempDir(base);
}

void write_text(const std::filesystem::path& path, std::string_view content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    REQUIRE(out);
}

[[nodiscard]] core::context::SessionContext make_context(
    const std::filesystem::path& workspace,
    std::vector<std::filesystem::path> additional = {}) {
    return core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = workspace,
            .additional = std::move(additional),
            .enforce = true,
            .version = 1,
        },
        core::context::SessionTransport::cli,
        "context-builder-test");
}

[[nodiscard]] std::string build_prompt(const core::context::SessionContext& context,
                                       std::string_view mode = "BUILD") {
    return core::context::ContextBuilder(context)
        .with_mode(mode)
        .build();
}

} // namespace

TEST_CASE("ContextBuilder renders runtime prompt without project context",
          "[context][builder]") {
    auto workspace = make_temp_workspace("filo_context_builder_empty");
    const auto context = make_context(workspace.path());

    const auto layers = core::context::ContextBuilder(context).build_layers();
    const std::string prompt = build_prompt(context);

    REQUIRE(layers.size() == 2);
    CHECK(layers[0].kind == core::context::ContextLayerKind::RuntimeInstructions);
    CHECK(layers[0].stability == core::context::PromptStability::Stable);
    CHECK(layers[0].name == "runtime");
    CHECK(layers[1].kind == core::context::ContextLayerKind::WorkspaceFacts);
    CHECK(layers[1].stability == core::context::PromptStability::Workspace);
    CHECK(layers[1].name == "workspace");
    CHECK(layers[0].content + layers[1].content == prompt);
    CHECK(layers[0].content ==
          "You are Filo, an advanced AI coding assistant running in BUILD mode.\n\n"
          "Build software methodically. Search, read, edit, and run commands. "
          "Verify your changes where possible. Ask clarifying questions only when truly needed."
          "\n\nYou can delegate complex background work via the `task` tool."
          " Use the `subagent_type` values listed in the task tool schema/description."
          " Default profiles are `general` (broad multi-step work) and"
          " `explore` (fast read-only codebase search)."
          " If the user asks with `@general` or `@explore`, map that request to a `task` call.");
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("[Workspace]"));
    CHECK_THAT(
        prompt,
        Catch::Matchers::ContainsSubstring(
            "Primary: " + context.workspace_view().primary().string()));
}

TEST_CASE("ContextBuilder renders ordered additional workspace directories",
          "[context][builder][workspace]") {
    auto primary = make_temp_workspace("filo_context_builder_primary");
    auto secondary = make_temp_workspace("filo_context_builder_secondary");
    auto tertiary = make_temp_workspace("filo_context_builder_tertiary");

    const auto context = make_context(primary.path(), {secondary.path(), tertiary.path()});
    const std::string prompt = build_prompt(context);
    const auto& workspace = context.workspace_view();

    const auto primary_pos = prompt.find("Primary: " + workspace.primary().string());
    const auto secondary_pos = prompt.find("- " + workspace.additional()[0].string());
    const auto tertiary_pos = prompt.find("- " + workspace.additional()[1].string());

    REQUIRE(primary_pos != std::string::npos);
    REQUIRE(secondary_pos != std::string::npos);
    REQUIRE(tertiary_pos != std::string::npos);
    CHECK(primary_pos < secondary_pos);
    CHECK(secondary_pos < tertiary_pos);
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("Path enforcement: enabled"));
    CHECK_THAT(
        prompt,
        Catch::Matchers::ContainsSubstring(
            "Relative paths resolve against the primary workspace."));
}

TEST_CASE("ContextBuilder renders project steering before project facts",
          "[context][builder]") {
    auto workspace = make_temp_workspace("filo_context_builder_project");
    write_text(workspace.path() / "AGENTS.md", "Repository rules from AGENTS.md\n");
    write_text(workspace.path() / "FILO.md", "Project steering from FILO.md\n");
    write_text(workspace.path() / ".filo" / "steering" / "backend.md", "Backend steering document\n");
    write_text(workspace.path() / "src" / "main.cpp", "int main() { return 0; }\n");

    const auto context = make_context(workspace.path());
    const auto layers = core::context::ContextBuilder(context)
        .with_mode("BUILD")
        .build_layers();
    const std::string prompt = build_prompt(context);

    REQUIRE(layers.size() == 4);
    CHECK(layers[0].kind == core::context::ContextLayerKind::RuntimeInstructions);
    CHECK(layers[0].name == "runtime");
    CHECK(layers[1].kind == core::context::ContextLayerKind::WorkspaceFacts);
    CHECK(layers[1].name == "workspace");
    CHECK(layers[2].kind == core::context::ContextLayerKind::ProjectSteering);
    CHECK(layers[2].name == "project_steering");
    CHECK(layers[3].kind == core::context::ContextLayerKind::ProjectFacts);
    CHECK(layers[3].stability == core::context::PromptStability::Dynamic);
    CHECK(layers[3].name == "project_facts");
    CHECK(layers[0].content + layers[1].content + layers[2].content + layers[3].content
          == prompt);

    const auto runtime_pos = prompt.find("You are Filo");
    const auto workspace_pos = prompt.find("[Workspace]");
    const auto steering_pos = prompt.find("[Project Steering]");
    const auto agents_pos = prompt.find("Repository rules from AGENTS.md");
    const auto filo_pos = prompt.find("Project steering from FILO.md");
    const auto backend_pos = prompt.find("Backend steering document");
    const auto facts_pos = prompt.find("[Project Context]");

    REQUIRE(runtime_pos != std::string::npos);
    REQUIRE(workspace_pos != std::string::npos);
    REQUIRE(steering_pos != std::string::npos);
    REQUIRE(agents_pos != std::string::npos);
    REQUIRE(filo_pos != std::string::npos);
    REQUIRE(backend_pos != std::string::npos);
    REQUIRE(facts_pos != std::string::npos);

    CHECK(runtime_pos < workspace_pos);
    CHECK(workspace_pos < steering_pos);
    CHECK(steering_pos < agents_pos);
    CHECK(agents_pos < filo_pos);
    CHECK(filo_pos < backend_pos);
    CHECK(backend_pos < facts_pos);

    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("Source: AGENTS.md"));
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("Source: FILO.md"));
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("Source: .filo/steering/backend.md"));
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("Structure:\n"));
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("src/"));
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("main.cpp"));
}

TEST_CASE("PromptPlan exposes a cacheable prefix without mutable project facts",
          "[context][prompt-plan]") {
    auto workspace = make_temp_workspace("filo_prompt_plan_layers");
    write_text(workspace.path() / "AGENTS.md", "Stable repository guidance\n");
    write_text(workspace.path() / "src" / "main.cpp", "int main() {}\n");
    const auto context = make_context(workspace.path());

    const auto plan = core::context::ContextBuilder(context).build_plan();
    const std::string workspace_prefix =
        plan.render_through(core::context::PromptStability::Workspace);

    CHECK_THAT(workspace_prefix,
               Catch::Matchers::ContainsSubstring("Stable repository guidance"));
    CHECK(workspace_prefix.find("[Project Context]") == std::string::npos);
    CHECK_THAT(plan.render(), Catch::Matchers::ContainsSubstring("[Project Context]"));
}

TEST_CASE("SteeringLoader reports loaded project steering source labels",
          "[context][steering]") {
    auto workspace = make_temp_workspace("filo_steering_sources");
    write_text(workspace.path() / "AGENTS.md", "Root instructions\n");
    write_text(workspace.path() / "FILO.md", "Filo instructions\n");
    write_text(workspace.path() / ".filo" / "steering" / "backend.md", "Backend instructions\n");

    const auto result = core::context::load_project_steering_context(workspace.path());

    REQUIRE(result.source_labels.size() == 3);
    CHECK(result.source_labels[0] == "AGENTS.md");
    CHECK(result.source_labels[1] == "FILO.md");
    CHECK(result.source_labels[2] == ".filo/steering/backend.md");
    CHECK_THAT(result.block, Catch::Matchers::ContainsSubstring("Source: AGENTS.md"));
    CHECK_THAT(result.block, Catch::Matchers::ContainsSubstring("Source: FILO.md"));
    CHECK_THAT(result.block, Catch::Matchers::ContainsSubstring("Source: .filo/steering/backend.md"));
}

TEST_CASE("SteeringLoader finds GEMINI.MD case-insensitively",
          "[context][steering]") {
    auto workspace = make_temp_workspace("filo_steering_gemini");
    write_text(workspace.path() / "GEMINI.MD", "Gemini instructions\n");

    const auto result = core::context::load_project_steering_context(workspace.path());

    REQUIRE(result.source_labels.size() == 1);
    CHECK(result.source_labels[0] == "GEMINI.MD");
    CHECK_THAT(result.block, Catch::Matchers::ContainsSubstring("Source: GEMINI.MD"));
    CHECK_THAT(result.block, Catch::Matchers::ContainsSubstring("Gemini instructions"));
}

TEST_CASE("OpenAI Responses request renders ContextBuilder prompt as instructions",
          "[context][builder][openai-responses]") {
    auto workspace = make_temp_workspace("filo_context_builder_responses");
    write_text(workspace.path() / "AGENTS.md", "Repository rules from AGENTS.md\n");

    const auto context = make_context(workspace.path());
    const std::string prompt = build_prompt(context);

    core::llm::ChatRequest request;
    request.model = "gpt-5";
    request.stream = false;
    request.messages = {
        core::llm::Message{.role = "system", .content = prompt},
        core::llm::Message{.role = "user", .content = "Hello"},
    };

    const core::llm::protocols::OpenAIResponsesProtocol protocol;
    const std::string payload = protocol.serialize(request);

    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("instructions":)"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("You are Filo"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("[Project Steering]"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("Repository rules from AGENTS.md"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("Hello"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("input":[)"));
    CHECK(payload.find(R"("role":"system")") == std::string::npos);
}
