#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/agent/ToolCallPlanner.hpp"
#include "core/tools/OpenDocumentTool.hpp"
#include "core/tools/PathVisibilityToolDecorator.hpp"
#include "core/tools/StrictToolSchema.hpp"
#include "core/tools/ToolSchema.hpp"
#include "core/tools/ToolNames.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include <unistd.h>

using Catch::Matchers::ContainsSubstring;
using core::tools::DocumentDisplay;
using core::tools::DocumentRequest;
using core::tools::OpenDocumentTool;

namespace {

namespace fs = std::filesystem;

struct Workspace {
    fs::path root = fs::temp_directory_path()
        / std::format("filo_open_document_{}_{}", ::getpid(),
                      std::chrono::steady_clock::now().time_since_epoch().count());
    Workspace() { fs::create_directories(root / "docs"); }
    ~Workspace() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    [[nodiscard]] core::context::SessionContext context() const {
        return test_support::make_session_context(
            {.primary = root, .enforce = true}, core::context::SessionTransport::cli, "sess-1");
    }
};

} // namespace

TEST_CASE("open_document is Filo's own: never offered over MCP", "[tools][open_document]") {
    const auto definition = OpenDocumentTool{}.get_definition();
    CHECK(definition.name == core::tools::names::kOpenDocument);
    CHECK(definition.agent_only);
    CHECK(definition.annotations.read_only_hint);
    CHECK(core::tools::tool_allows_transport(definition, core::context::SessionTransport::cli));
    CHECK_FALSE(core::tools::tool_allows_transport(
        definition, core::context::SessionTransport::mcp_http));
    CHECK_FALSE(core::tools::tool_allows_transport(
        definition, core::context::SessionTransport::mcp_stdio));
}

TEST_CASE("open_document's contract holds under every provider's schema dialect",
          "[tools][open_document]") {
    const auto definition = OpenDocumentTool{}.get_definition();
    const auto canonical = core::tools::schema::canonical_input_schema(definition);
    CHECK_THAT(canonical, ContainsSubstring(R"("required":["path"])"));
    // Strict decoding (OpenAI, Anthropic) can carry it whole, so it is never
    // sent with a weakened claim.
    using core::tools::schema::StrictDialect;
    for (const auto dialect : {StrictDialect::OpenAI, StrictDialect::Anthropic}) {
        CAPTURE(core::tools::schema::strict_dialect_name(dialect));
        CHECK(core::tools::schema::strict_input_schema(definition, dialect).has_value());
    }
}

TEST_CASE("open_document hands the pane a checked, absolute file", "[tools][open_document]") {
    Workspace workspace;
    { std::ofstream(workspace.root / "docs" / "plan.md") << "# Plan\n"; }

    OpenDocumentTool tool;
    std::optional<DocumentRequest> seen;
    tool.set_opener([&seen](const DocumentRequest& request) {
        seen = request;
        return DocumentDisplay::shown;
    });

    const auto result = tool.execute(R"({"path":"docs/plan.md","line":4})", workspace.context());
    REQUIRE(seen.has_value());
    CHECK(seen->path == fs::weakly_canonical(workspace.root / "docs" / "plan.md"));
    CHECK(seen->line == 4);
    CHECK(seen->session_id == "sess-1");
    CHECK_THAT(result, ContainsSubstring(R"("display":"shown")"));
    CHECK_THAT(result, ContainsSubstring(R"("line":4)"));

    // A null line is what strict decoding sends for "no line".
    seen.reset();
    std::ignore = tool.execute(R"({"path":"docs/plan.md","line":null})", workspace.context());
    REQUIRE(seen.has_value());
    CHECK_FALSE(seen->line.has_value());
}

TEST_CASE("open_document tells the model when the user cannot see the file",
          "[tools][open_document]") {
    Workspace workspace;
    { std::ofstream(workspace.root / "a.md") << "a"; }
    OpenDocumentTool tool;
    tool.set_opener([](const DocumentRequest&) { return DocumentDisplay::elsewhere; });
    const auto result = tool.execute(R"({"path":"a.md"})", workspace.context());
    CHECK_THAT(result, ContainsSubstring(R"("display":"elsewhere")"));
    CHECK_THAT(result, ContainsSubstring("Link the file in your reply"));
    CHECK_THAT(result, !ContainsSubstring(R"("error")"));
}

TEST_CASE("open_document refuses what it cannot show", "[tools][open_document]") {
    Workspace workspace;
    { std::ofstream(workspace.root / "a.md") << "a"; }
    { std::ofstream(workspace.root / ".agentignore") << "secret.md\n"; }
    { std::ofstream(workspace.root / "secret.md") << "s"; }

    auto tool = std::make_shared<OpenDocumentTool>();
    int calls = 0;
    tool->set_opener([&calls](const DocumentRequest&) {
        ++calls;
        return DocumentDisplay::shown;
    });
    const auto guarded = core::tools::with_path_visibility(tool);
    const auto context = workspace.context();

    CHECK_THAT(guarded->execute(R"({"path":"missing.md"})", context),
               ContainsSubstring("does not exist"));
    CHECK_THAT(guarded->execute(R"({"path":"docs"})", context),
               ContainsSubstring("is not a file"));
    CHECK_THAT(guarded->execute(R"({"path":"a.md","line":0})", context),
               ContainsSubstring("positive integer"));
    CHECK_THAT(guarded->execute(R"({"path":"a.md","page":2})", context),
               ContainsSubstring("Unknown argument"));
    CHECK_THAT(guarded->execute(R"({"path":"/etc/hosts"})", context),
               ContainsSubstring("outside the allowed workspace"));
    CHECK_THAT(guarded->execute(R"({"path":"secret.md"})", context),
               ContainsSubstring(R"("error")"));
    CHECK(calls == 0);
    // The agent-ignore rule, not the file, kept it out: unguarded it opens.
    CHECK_THAT(tool->execute(R"({"path":"secret.md"})", context),
               ContainsSubstring(R"("display":"shown")"));
    CHECK(calls == 1);

    OpenDocumentTool paneless;
    CHECK_THAT(paneless.execute(R"({"path":"a.md"})", context),
               ContainsSubstring("no document pane"));
}

TEST_CASE("open_document runs beside reads of its file", "[tools][open_document]") {
    Workspace workspace;
    const auto context = workspace.context();
    core::llm::ToolCall call;
    call.function.name = std::string(core::tools::names::kOpenDocument);
    call.function.arguments = R"({"path":"a.md"})";
    const auto planned = core::agent::plan_tool_call(call, context);
    REQUIRE(planned.accesses.size() == 1);
    CHECK(planned.accesses.front().kind == core::agent::ToolAccess::Kind::File);
    CHECK(planned.accesses.front().file.operation == core::agent::ToolFileOperation::Read);
}
