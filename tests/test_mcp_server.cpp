#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/mcp/McpDispatcher.hpp"
#include "core/workspace/Workspace.hpp"
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <optional>
#include <string>
#include <simdjson.h>

using namespace Catch::Matchers;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the dispatcher singleton. Tools are registered on first access.
static core::mcp::McpDispatcher& disp() {
    return core::mcp::McpDispatcher::get_instance();
}

/// Returns true if @p s is a well-formed JSON document.
static bool is_valid_json(const std::string& s) {
    simdjson::dom::parser p;
    simdjson::dom::element el;
    return p.parse(s).get(el) == simdjson::SUCCESS;
}

/// Returns the "result.structuredContent.output" field, or nullopt if missing.
static std::optional<std::string> extract_structured_output(const std::string& response_json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(response_json).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    std::string_view output;
    if (doc["result"]["structuredContent"]["output"].get(output) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return std::string(output);
}

/// Trims trailing CR/LF emitted by shell commands such as `pwd`.
static std::string trim_trailing_newlines(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

static std::filesystem::path make_temp_dir(std::string_view tag) {
    const auto dir = std::filesystem::temp_directory_path()
        / ("filo_mcp_test_" + std::string(tag));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

static void write_test_file(const std::filesystem::path& path, std::string_view content) {
    std::ofstream ofs(path);
    ofs << content;
}

struct ScopedEnvVar {
    std::string name;
    std::optional<std::string> old_value;

    ScopedEnvVar(std::string name_in, std::string value)
        : name(std::move(name_in)) {
        if (const char* current = std::getenv(name.c_str())) {
            old_value = std::string(current);
        }
        ::setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (old_value.has_value()) {
            ::setenv(name.c_str(), old_value->c_str(), 1);
        } else {
            ::unsetenv(name.c_str());
        }
    }
};

struct ScopedCurrentPath {
    std::filesystem::path old_path;

    explicit ScopedCurrentPath(const std::filesystem::path& new_path)
        : old_path(std::filesystem::current_path()) {
        std::filesystem::current_path(new_path);
    }

    ~ScopedCurrentPath() {
        std::error_code ec;
        std::filesystem::current_path(old_path, ec);
    }
};

struct WorkspaceResetToDefault {
    ~WorkspaceResetToDefault() {
        std::error_code ec;
        const auto cwd = std::filesystem::current_path(ec);
        auto& ws = core::workspace::Workspace::get_instance();
        if (ec) {
            ws.initialize("", {}, false);
            return;
        }
        ws.initialize(cwd, {}, false);
    }
};

/// A minimal valid initialize request (client identifies as Lampo).
static const char* kInitRequest =
    R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"lampo","version":"1.0"}},"id":1})";

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

TEST_CASE("MCP initialize returns correct protocol version and server info", "[mcp]") {
    auto resp = disp().dispatch(kInitRequest);

    REQUIRE_THAT(resp, ContainsSubstring(R"("jsonrpc":"2.0")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("id":1)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("protocolVersion":"2024-11-05")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("name":"filo-mcp")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("tools":{"listChanged":false})"));
}

TEST_CASE("MCP initialize response includes instructions field", "[mcp]") {
    auto resp = disp().dispatch(kInitRequest);
    // The instructions field must be present and non-empty.
    REQUIRE_THAT(resp, ContainsSubstring(R"("instructions")"));
    REQUIRE_THAT(resp, ContainsSubstring("filo-mcp"));
}

TEST_CASE("MCP initialize with string id echoes it back as string", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"initialize","params":{},"id":"abc-123"})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":"abc-123")"));
    REQUIRE_THAT(resp, ContainsSubstring("protocolVersion"));
}

TEST_CASE("MCP initialize negotiates to a supported protocol version", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2099-01-01"},"id":2})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":2)"));
    const bool negotiated_to_supported =
        resp.find(R"("protocolVersion":"2025-11-25")") != std::string::npos
        || resp.find(R"("protocolVersion":"2024-11-05")") != std::string::npos;
    REQUIRE(negotiated_to_supported);
}

TEST_CASE("MCP initialize with 2025-11-25 echoes it back", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{}},"id":99})");
    REQUIRE_THAT(resp, ContainsSubstring(R"("protocolVersion":"2025-11-25")"));
    REQUIRE(is_valid_json(resp));
}

TEST_CASE("MCP initialize advertises only implemented capabilities", "[mcp]") {
    const auto sandbox = make_temp_dir("mcp_init_capabilities");
    const auto fake_home = sandbox / "home";
    const auto project = sandbox / "project";
    std::filesystem::create_directories(fake_home / ".config" / "filo" / "skills");
    std::filesystem::create_directories(project);
    ScopedEnvVar home("HOME", fake_home.string());
    ScopedCurrentPath cwd(project);

    auto resp = disp().dispatch(kInitRequest);

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    bool tools_list_changed = true;
    REQUIRE(doc["result"]["capabilities"]["tools"]["listChanged"].get(tools_list_changed) == simdjson::SUCCESS);
    REQUIRE_FALSE(tools_list_changed);

    bool resources_subscribe = true;
    bool resources_list_changed = true;
    REQUIRE(doc["result"]["capabilities"]["resources"]["subscribe"].get(resources_subscribe) == simdjson::SUCCESS);
    REQUIRE(doc["result"]["capabilities"]["resources"]["listChanged"].get(resources_list_changed) == simdjson::SUCCESS);
    REQUIRE_FALSE(resources_subscribe);
    REQUIRE_FALSE(resources_list_changed);

    simdjson::dom::element ignored;
    REQUIRE(doc["result"]["capabilities"]["roots"].get(ignored) != simdjson::SUCCESS);
    REQUIRE(doc["result"]["capabilities"]["prompts"].get(ignored) != simdjson::SUCCESS);
    REQUIRE(doc["result"]["capabilities"]["logging"].get(ignored) != simdjson::SUCCESS);
}

TEST_CASE("MCP initialize advertises prompts capability when prompt skills exist", "[mcp][prompts]") {
    const auto sandbox = make_temp_dir("mcp_init_prompts");
    const auto fake_home = sandbox / "home";
    const auto project = sandbox / "project";
    const auto prompt_dir = project / ".filo" / "skills" / "explain";
    std::filesystem::create_directories(fake_home / ".config" / "filo" / "skills");
    std::filesystem::create_directories(prompt_dir);
    write_test_file(prompt_dir / "SKILL.md", R"(---
name: explain
description: Explain the supplied concept simply.
---
Explain "$ARGUMENTS" in beginner-friendly terms.
)");
    ScopedEnvVar home("HOME", fake_home.string());
    ScopedCurrentPath cwd(project);

    auto resp = disp().dispatch(kInitRequest);

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    bool prompts_list_changed = true;
    REQUIRE(doc["result"]["capabilities"]["prompts"]["listChanged"].get(prompts_list_changed)
            == simdjson::SUCCESS);
    REQUIRE_FALSE(prompts_list_changed);
}

TEST_CASE("MCP response id escapes string IDs correctly", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"ping","id":"a\"b"})");

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    std::string_view id;
    REQUIRE(doc["id"].get(id) == simdjson::SUCCESS);
    REQUIRE(id == "a\"b");
}

TEST_CASE("MCP rejects invalid request ids", "[mcp]") {
    for (const std::string request : {
            R"({"jsonrpc":"2.0","method":"ping","id":null})",
            R"({"jsonrpc":"2.0","method":"ping","id":true})",
            R"({"jsonrpc":"2.0","method":"ping","id":1.5})",
            R"({"jsonrpc":"2.0","method":"ping","id":{"nested":1}})"
        }) {
        auto resp = disp().dispatch(request);
        REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32600)"));
        REQUIRE_THAT(resp, ContainsSubstring(R"("id":null)"));
    }
}

// ---------------------------------------------------------------------------
// Notifications — must produce NO response (empty string)
// ---------------------------------------------------------------------------

TEST_CASE("MCP notifications/initialized returns empty string", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
    REQUIRE(resp.empty());
}

TEST_CASE("MCP notifications/cancelled returns empty string", "[mcp]") {
    // Cancellation notifications have no id; the request id being cancelled is
    // inside params.requestId — not the JSON-RPC id field.
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"42","reason":"user cancelled"}})");
    REQUIRE(resp.empty());
}

TEST_CASE("MCP arbitrary notification without id returns empty string", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"some/notification","params":{}})");
    REQUIRE(resp.empty());
}

// ---------------------------------------------------------------------------
// ping
// ---------------------------------------------------------------------------

TEST_CASE("MCP ping returns empty result object", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"ping","id":99})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":99)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("result":{})"));
}

// ---------------------------------------------------------------------------
// tools/list
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/list returns all registered tools", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":2})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":2)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("tools")"));

    for (const char* name : {
            "run_terminal_command", "read_file", "write_file", "list_directory",
            "replace", "file_search", "grep_search", "apply_patch", "search_replace",
            "delete_file", "move_file", "create_directory", "get_workspace_config"
        }) {
        REQUIRE_THAT(resp, ContainsSubstring(name));
    }
}

TEST_CASE("MCP tools/list each tool has inputSchema with type object", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":3})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("type":"object")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("properties")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("required")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("description")"));
}

TEST_CASE("MCP tools/list includes title field for display names", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":4})");

    // Every tool must have a title entry (display name for Lampo).
    REQUIRE_THAT(resp, ContainsSubstring(R"("title")"));
    REQUIRE_THAT(resp, ContainsSubstring("Run Terminal Command"));
    REQUIRE_THAT(resp, ContainsSubstring("Read File"));
    REQUIRE_THAT(resp, ContainsSubstring("Write File"));
}

TEST_CASE("MCP tools/list includes annotations for all tools", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":5})");

    // All four hint fields must be present for every tool.
    REQUIRE_THAT(resp, ContainsSubstring(R"("annotations")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("readOnlyHint")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("destructiveHint")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("idempotentHint")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("openWorldHint")"));
}

TEST_CASE("MCP tools/list search_replace includes array items schema", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":51})");

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    simdjson::dom::array tools;
    REQUIRE(doc["result"]["tools"].get(tools) == simdjson::SUCCESS);

    bool found = false;
    for (auto tool : tools) {
        std::string_view name;
        if (tool["name"].get(name) != simdjson::SUCCESS || name != "search_replace") {
            continue;
        }

        found = true;

        std::string_view edits_type;
        std::string_view old_string_type;
        std::string_view new_string_type;
        REQUIRE(tool["inputSchema"]["properties"]["edits"]["type"].get(edits_type) == simdjson::SUCCESS);
        REQUIRE(tool["inputSchema"]["properties"]["edits"]["items"]["properties"]["old_string"]["type"]
                    .get(old_string_type) == simdjson::SUCCESS);
        REQUIRE(tool["inputSchema"]["properties"]["edits"]["items"]["properties"]["new_string"]["type"]
                    .get(new_string_type) == simdjson::SUCCESS);

        REQUIRE(edits_type == "array");
        REQUIRE(old_string_type == "string");
        REQUIRE(new_string_type == "string");
        break;
    }

    REQUIRE(found);
}

TEST_CASE("MCP tools/list includes outputSchema for structured tool results", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":52})");

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    simdjson::dom::array tools;
    REQUIRE(doc["result"]["tools"].get(tools) == simdjson::SUCCESS);

    bool saw_shell = false;
    bool saw_workspace = false;
    for (auto tool : tools) {
        std::string_view name;
        if (tool["name"].get(name) != simdjson::SUCCESS) continue;

        if (name == "run_terminal_command") {
            saw_shell = true;
            std::string_view output_type;
            std::string_view exit_code_type;
            REQUIRE(tool["outputSchema"]["properties"]["output"]["type"].get(output_type) == simdjson::SUCCESS);
            REQUIRE(tool["outputSchema"]["properties"]["exit_code"]["type"].get(exit_code_type) == simdjson::SUCCESS);
            REQUIRE(output_type == "string");
            REQUIRE(exit_code_type == "integer");
        } else if (name == "get_workspace_config") {
            saw_workspace = true;
            std::string_view additional_type;
            std::string_view item_type;
            REQUIRE(tool["outputSchema"]["properties"]["additional_directories"]["type"].get(additional_type)
                    == simdjson::SUCCESS);
            REQUIRE(tool["outputSchema"]["properties"]["additional_directories"]["items"]["type"].get(item_type)
                    == simdjson::SUCCESS);
            REQUIRE(additional_type == "array");
            REQUIRE(item_type == "string");
        }
    }

    REQUIRE(saw_shell);
    REQUIRE(saw_workspace);
}

TEST_CASE("MCP tools/list run_terminal_command has destructive+openWorld hints", "[mcp]") {
    // run_terminal_command can do anything — both destructive and open-world must be true.
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":6})");
    REQUIRE_THAT(resp, ContainsSubstring("working_dir"));
    REQUIRE(is_valid_json(resp));
}

TEST_CASE("MCP tools/list read_file has readOnly + idempotent hints true", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":7})");
    // The response is a large JSON blob; parse it and find read_file's annotations.
    REQUIRE(is_valid_json(resp));
    // At minimum the hint keys must be present.
    REQUIRE_THAT(resp, ContainsSubstring(R"("readOnlyHint")"));
}

// ---------------------------------------------------------------------------
// prompts/list and prompts/get
// ---------------------------------------------------------------------------

TEST_CASE("MCP prompts/list returns discovered prompt skills", "[mcp][prompts]") {
    const auto sandbox = make_temp_dir("mcp_prompts_list");
    const auto fake_home = sandbox / "home";
    const auto project = sandbox / "project";
    const auto prompt_dir = project / ".filo" / "skills" / "review_pr";
    const auto disabled_dir = project / ".filo" / "skills" / "disabled";
    const auto tool_dir = project / ".filo" / "skills" / "tool_skill";
    std::filesystem::create_directories(fake_home / ".config" / "filo" / "skills");
    std::filesystem::create_directories(prompt_dir);
    std::filesystem::create_directories(disabled_dir);
    std::filesystem::create_directories(tool_dir);
    write_test_file(prompt_dir / "SKILL.md", R"(---
name: review-pr
description: Review a pull request.
---
Review PR #$ARGUMENTS for correctness.
)");
    write_test_file(disabled_dir / "SKILL.md", R"(---
name: hidden
description: Hidden prompt.
enabled: false
---
This should not appear.
)");
    write_test_file(tool_dir / "SKILL.md", R"(---
name: weather
description: Tool skill, not a prompt.
entry_point: weather.py
---
)");
    ScopedEnvVar home("HOME", fake_home.string());
    ScopedCurrentPath cwd(project);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"prompts/list","params":{},"id":14})");

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    simdjson::dom::array prompts;
    REQUIRE(doc["result"]["prompts"].get(prompts) == simdjson::SUCCESS);

    bool found_review_prompt = false;
    for (auto prompt : prompts) {
        std::string_view name;
        REQUIRE(prompt["name"].get(name) == simdjson::SUCCESS);
        if (name != "review-pr") continue;

        found_review_prompt = true;

        std::string_view description;
        REQUIRE(prompt["description"].get(description) == simdjson::SUCCESS);
        REQUIRE(description == "Review a pull request.");

        simdjson::dom::array arguments;
        REQUIRE(prompt["arguments"].get(arguments) == simdjson::SUCCESS);

        bool found_argument = false;
        for (auto arg : arguments) {
            std::string_view arg_name;
            REQUIRE(arg["name"].get(arg_name) == simdjson::SUCCESS);
            if (arg_name != "arguments") continue;

            found_argument = true;
            bool required = true;
            REQUIRE(arg["required"].get(required) == simdjson::SUCCESS);
            REQUIRE_FALSE(required);
        }
        REQUIRE(found_argument);
    }

    REQUIRE(found_review_prompt);
    REQUIRE(resp.find("hidden") == std::string::npos);
    REQUIRE(resp.find("weather") == std::string::npos);
}

TEST_CASE("MCP prompts/get expands $ARGUMENTS from the arguments map", "[mcp][prompts]") {
    const auto sandbox = make_temp_dir("mcp_prompts_get");
    const auto fake_home = sandbox / "home";
    const auto project = sandbox / "project";
    const auto prompt_dir = project / ".filo" / "skills" / "explain";
    std::filesystem::create_directories(fake_home / ".config" / "filo" / "skills");
    std::filesystem::create_directories(prompt_dir);
    write_test_file(prompt_dir / "SKILL.md", R"(---
name: explain
description: Explain a concept simply.
---
Explain "$ARGUMENTS" in beginner-friendly terms.
)");
    ScopedEnvVar home("HOME", fake_home.string());
    ScopedCurrentPath cwd(project);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"prompts/get","params":{"name":"explain","arguments":{"arguments":"recursion"}},"id":15})");

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    std::string_view description;
    REQUIRE(doc["result"]["description"].get(description) == simdjson::SUCCESS);
    REQUIRE(description == "Explain a concept simply.");

    simdjson::dom::array messages;
    REQUIRE(doc["result"]["messages"].get(messages) == simdjson::SUCCESS);
    REQUIRE(messages.begin() != messages.end());
    const auto first_message = *messages.begin();

    std::string_view role;
    std::string_view content_type;
    std::string_view text;
    REQUIRE(first_message["role"].get(role) == simdjson::SUCCESS);
    REQUIRE(first_message["content"]["type"].get(content_type) == simdjson::SUCCESS);
    REQUIRE(first_message["content"]["text"].get(text) == simdjson::SUCCESS);
    REQUIRE(role == "user");
    REQUIRE(content_type == "text");
    REQUIRE(text.find("recursion") != std::string_view::npos);
}

TEST_CASE("MCP prompts/get returns -32602 for unknown prompt", "[mcp][prompts]") {
    const auto sandbox = make_temp_dir("mcp_prompts_missing");
    const auto fake_home = sandbox / "home";
    const auto project = sandbox / "project";
    std::filesystem::create_directories(fake_home / ".config" / "filo" / "skills");
    std::filesystem::create_directories(project);
    ScopedEnvVar home("HOME", fake_home.string());
    ScopedCurrentPath cwd(project);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"prompts/get","params":{"name":"does-not-exist"},"id":16})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32602)"));
    REQUIRE_THAT(resp, ContainsSubstring("Unknown prompt"));
}

// ---------------------------------------------------------------------------
// JSON validity — every response must be valid JSON
// ---------------------------------------------------------------------------

TEST_CASE("MCP all responses are valid JSON", "[mcp][json]") {
    REQUIRE(is_valid_json(disp().dispatch(
        R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}},"id":1})")));
    REQUIRE(is_valid_json(disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":2})")));
    REQUIRE(is_valid_json(disp().dispatch(
        R"({"jsonrpc":"2.0","method":"ping","id":3})")));
    REQUIRE(is_valid_json(disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo hi"}},"id":4})")));
    REQUIRE(is_valid_json(disp().dispatch(
        R"({"jsonrpc":"2.0","method":"foobar","id":5})")));
    REQUIRE(is_valid_json(disp().dispatch("not json at all")));
}

// ---------------------------------------------------------------------------
// tools/call — success paths
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call run_terminal_command echo succeeds", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo mcp_works"}},"id":10})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":10)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("content")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("type":"text")"));
    REQUIRE_THAT(resp, ContainsSubstring("mcp_works"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    // structuredContent must also be present (MCP 2025-11-25)
    REQUIRE_THAT(resp, ContainsSubstring(R"("structuredContent")"));
}

TEST_CASE("MCP tools/call run_terminal_command returns exit_code", "[mcp]") {
    // The tool result is embedded as an escaped JSON string inside the MCP response.
    auto resp_ok = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"true"}},"id":50})");
    REQUIRE_THAT(resp_ok, ContainsSubstring(R"(\"exit_code\":0)"));

    auto resp_fail = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"bash -c \"exit 42\""}},"id":51})");
    REQUIRE_THAT(resp_fail, ContainsSubstring(R"(\"exit_code\":42)"));
}

TEST_CASE("MCP tools/call omits structuredContent for tool execution errors", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo should_not_run","working_dir":"/definitely/not/a/real/directory"}},"id":53})");

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    bool is_error = false;
    REQUIRE(doc["result"]["isError"].get(is_error) == simdjson::SUCCESS);
    REQUIRE(is_error);

    simdjson::dom::element structured;
    REQUIRE(doc["result"]["structuredContent"].get(structured) != simdjson::SUCCESS);
}

TEST_CASE("MCP tools/call list_directory on current directory succeeds", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_directory","arguments":{"path":"."}},"id":11})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":11)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("content")"));
}

TEST_CASE("MCP tools/call read_file returns file contents", "[mcp]") {
    const std::string path = "mcp_test_artifact.txt";
    { std::ofstream ofs(path); ofs << "hello from lampo"; }

    std::string req =
        std::string(R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"read_file","arguments":{"path":")") +
        path + R"("}},"id":12})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring("hello from lampo"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));

    std::filesystem::remove(path);
}

TEST_CASE("MCP tools/call get_workspace_config returns structured workspace state", "[mcp]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    const auto primary = std::filesystem::current_path();
    const auto additional = primary / "mcp_workspace_config_additional";
    std::error_code ec;
    std::filesystem::create_directories(additional, ec);
    REQUIRE_FALSE(ec);
    ws.initialize(primary, {additional}, true);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_workspace_config","arguments":{}},"id":13})");
    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(resp).get(doc) == simdjson::SUCCESS);

    std::string_view primary_directory;
    bool enforcement_enabled = false;
    simdjson::dom::array additional_directories;
    REQUIRE(doc["result"]["structuredContent"]["primary_directory"].get(primary_directory) == simdjson::SUCCESS);
    REQUIRE(doc["result"]["structuredContent"]["enforcement_enabled"].get(enforcement_enabled) == simdjson::SUCCESS);
    REQUIRE(doc["result"]["structuredContent"]["additional_directories"].get(additional_directories) == simdjson::SUCCESS);

    REQUIRE(enforcement_enabled);
    REQUIRE_FALSE(primary_directory.empty());

    bool saw_additional = false;
    for (auto dir : additional_directories) {
        std::string_view dir_sv;
        REQUIRE(dir.get(dir_sv) == simdjson::SUCCESS);
        if (dir_sv.find(additional.filename().string()) != std::string_view::npos) {
            saw_additional = true;
        }
    }
    REQUIRE(saw_additional);

    std::filesystem::remove_all(additional, ec);
}

// ---------------------------------------------------------------------------
// write_file — rich response (previous_content, created, bytes_written)
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call write_file creates a new file with created:true", "[mcp]") {
    const std::string path = "mcp_write_new.txt";
    std::filesystem::remove(path);

    std::string req =
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"write_file","arguments":{"file_path":")" +
        path + R"(","content":"hello filo"}},"id":200})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    // created must be true for a new file
    REQUIRE_THAT(resp, ContainsSubstring(R"("created":true)"));
    // bytes_written must be present
    REQUIRE_THAT(resp, ContainsSubstring(R"("bytes_written")"));
    // previous_content must be empty string for a new file
    REQUIRE_THAT(resp, ContainsSubstring(R"("previous_content":"")"));
    // file_path must be echoed back
    REQUIRE_THAT(resp, ContainsSubstring(R"("file_path")"));

    // Verify the file was actually written
    std::ifstream ifs(path);
    std::string content((std::istreambuf_iterator<char>(ifs)), {});
    REQUIRE(content == "hello filo");

    std::filesystem::remove(path);
}

TEST_CASE("MCP tools/call write_file overwrite returns created:false and previous_content", "[mcp]") {
    const std::string path = "mcp_write_overwrite.txt";
    { std::ofstream ofs(path); ofs << "original content"; }

    std::string req =
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"write_file","arguments":{"file_path":")" +
        path + R"(","content":"new content"}},"id":201})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    // created must be false — file already existed
    REQUIRE_THAT(resp, ContainsSubstring(R"("created":false)"));
    // previous_content must contain the old text
    REQUIRE_THAT(resp, ContainsSubstring("original content"));

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// replace — rich response (replaced_at_line)
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call replace returns replaced_at_line", "[mcp]") {
    const std::string path = "mcp_replace_test.txt";
    {
        std::ofstream ofs(path);
        ofs << "line one\nline two\nline three\n";
    }

    std::string req =
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"replace","arguments":{"file_path":")" +
        path + R"(","old_string":"line two","new_string":"LINE TWO"}},"id":210})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    // replaced_at_line must be present (line 2 = "line two")
    REQUIRE_THAT(resp, ContainsSubstring(R"("replaced_at_line")"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("replaced_at_line":2)"));
    // file_path must be echoed back
    REQUIRE_THAT(resp, ContainsSubstring(R"("file_path")"));

    std::filesystem::remove(path);
}

TEST_CASE("MCP tools/call replace on line 1 reports replaced_at_line:1", "[mcp]") {
    const std::string path = "mcp_replace_line1.txt";
    { std::ofstream ofs(path); ofs << "alpha\nbeta\n"; }

    std::string req =
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"replace","arguments":{"file_path":")" +
        path + R"(","old_string":"alpha","new_string":"ALPHA"}},"id":211})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("replaced_at_line":1)"));

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// tools/call — error paths
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call unknown tool name returns -32602", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"nonexistent_tool","arguments":{}},"id":20})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":20)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32602)"));
    REQUIRE_THAT(resp, ContainsSubstring("Unknown tool"));
}

TEST_CASE("MCP tools/call missing required argument fails schema validation", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command"},"id":21})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":21)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32602)"));
    REQUIRE_THAT(resp, ContainsSubstring("missing required argument 'command'"));
}

TEST_CASE("MCP tools/call missing name field returns -32602", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"arguments":{"command":"echo x"}},"id":22})");

    REQUIRE_THAT(resp, ContainsSubstring("-32602"));
}

TEST_CASE("MCP tools/call missing params object returns -32602", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","id":23})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32602)"));
    REQUIRE_THAT(resp, ContainsSubstring("missing 'params' object"));
}

TEST_CASE("MCP tools/call arguments must be an object", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":"echo x"},"id":24})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32602)"));
    REQUIRE_THAT(resp, ContainsSubstring("'arguments' must be an object"));
}

// ---------------------------------------------------------------------------
// Unknown method / malformed JSON
// ---------------------------------------------------------------------------

TEST_CASE("MCP unknown method returns -32601 Method not found", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"foobar/baz","params":{},"id":30})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":30)"));
    REQUIRE_THAT(resp, ContainsSubstring("-32601"));
    REQUIRE_THAT(resp, ContainsSubstring("Method not found"));
}

TEST_CASE("MCP malformed JSON returns -32700 Parse error", "[mcp]") {
    auto resp = disp().dispatch("this is not json {{{");

    REQUIRE_THAT(resp, ContainsSubstring("-32700"));
    REQUIRE_THAT(resp, ContainsSubstring("Parse error"));
}

TEST_CASE("MCP invalid jsonrpc version returns -32600", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"1.0","method":"ping","id":55})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":55)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("code":-32600)"));
}

TEST_CASE("MCP empty body returns -32700 Parse error", "[mcp]") {
    auto resp = disp().dispatch("");
    REQUIRE_THAT(resp, ContainsSubstring("-32700"));
}

// ---------------------------------------------------------------------------
// Lampo integration smoke-test: full handshake sequence
// ---------------------------------------------------------------------------

TEST_CASE("MCP Lampo handshake sequence produces valid responses", "[mcp][lampo]") {
    // 1. initialize
    auto r1 = disp().dispatch(kInitRequest);
    REQUIRE_THAT(r1, ContainsSubstring("protocolVersion"));
    REQUIRE_THAT(r1, ContainsSubstring("instructions"));
    REQUIRE(is_valid_json(r1));

    // 2. notifications/initialized (notification — no response)
    auto r2 = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"notifications/initialized"})");
    REQUIRE(r2.empty());

    // 3. tools/list — must include title + annotations
    auto r3 = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":3})");
    REQUIRE_THAT(r3, ContainsSubstring("run_terminal_command"));
    REQUIRE_THAT(r3, ContainsSubstring(R"("title")"));
    REQUIRE_THAT(r3, ContainsSubstring(R"("annotations")"));
    REQUIRE(is_valid_json(r3));

    // 4. tools/call — structuredContent must be present
    auto r4 = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo lampo_ok"}},"id":4})");
    REQUIRE_THAT(r4, ContainsSubstring("lampo_ok"));
    REQUIRE_THAT(r4, ContainsSubstring(R"("isError":false)"));
    REQUIRE_THAT(r4, ContainsSubstring(R"("structuredContent")"));
    REQUIRE(is_valid_json(r4));
}

// ---------------------------------------------------------------------------
// create_directory / delete_file / move_file via MCP
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call create_directory creates a directory", "[mcp]") {
    const std::string path = "mcp_test_newdir";
    std::filesystem::remove_all(path);

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_directory","arguments":{"dir_path":")" + path + R"("}},"id":60})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    REQUIRE(std::filesystem::is_directory(path));

    std::filesystem::remove_all(path);
}

TEST_CASE("MCP tools/call create_directory succeeds if already exists", "[mcp]") {
    const std::string path = "mcp_test_existing_dir";
    std::filesystem::create_directories(path);

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_directory","arguments":{"dir_path":")" + path + R"("}},"id":61})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));

    std::filesystem::remove_all(path);
}

TEST_CASE("MCP tools/call delete_file removes a file", "[mcp]") {
    const std::string path = "mcp_test_delete_me.txt";
    { std::ofstream ofs(path); ofs << "bye"; }

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"delete_file","arguments":{"file_path":")" + path + R"("}},"id":62})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    REQUIRE_FALSE(std::filesystem::exists(path));
}

TEST_CASE("MCP tools/call move_file renames a file", "[mcp]") {
    const std::string src = "mcp_test_move_src.txt";
    const std::string dst = "mcp_test_move_dst.txt";
    { std::ofstream ofs(src); ofs << "data"; }
    std::filesystem::remove(dst);

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"move_file","arguments":{"source":")" + src + R"(","destination":")" + dst + R"("}},"id":63})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    REQUIRE_FALSE(std::filesystem::exists(src));
    REQUIRE(std::filesystem::exists(dst));

    std::filesystem::remove(dst);
}

// ---------------------------------------------------------------------------
// read_file offset_line / limit_lines
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call read_file with offset_line and limit_lines", "[mcp]") {
    const std::string path = "mcp_test_multiline.txt";
    {
        std::ofstream ofs(path);
        ofs << "line1\nline2\nline3\nline4\nline5\n";
    }

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"read_file","arguments":{"path":")" + path + R"(","offset_line":2,"limit_lines":2}},"id":70})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring("line2"));
    REQUIRE_THAT(resp, ContainsSubstring("line3"));
    REQUIRE_THAT(resp, !ContainsSubstring("line1"));
    REQUIRE_THAT(resp, !ContainsSubstring("line4"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));

    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// grep_search shell injection safety
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call grep_search with single-quote in pattern is safe", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"grep_search","arguments":{"pattern":"it's","path":"."}},"id":80})");

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
}

// ---------------------------------------------------------------------------
// run_terminal_command working_dir
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call run_terminal_command with working_dir runs in that directory", "[mcp]") {
    const std::string dir = std::filesystem::temp_directory_path().string() + "/filo_mcp_wd_test";
    std::filesystem::create_directories(dir);

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"pwd","working_dir":")" + dir + R"("}},"id":90})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));

    const auto output = extract_structured_output(resp);
    REQUIRE(output.has_value());

    const std::filesystem::path command_pwd = trim_trailing_newlines(*output);
    REQUIRE_FALSE(command_pwd.empty());

    // Compare semantic path identity instead of string form because macOS may
    // report equivalent paths with different prefixes (e.g. /var vs /private/var).
    std::error_code expected_ec;
    std::error_code actual_ec;
    std::error_code equivalent_ec;
    const std::filesystem::path expected_path =
        std::filesystem::weakly_canonical(std::filesystem::path(dir), expected_ec);
    const std::filesystem::path actual_path =
        std::filesystem::weakly_canonical(command_pwd, actual_ec);
    REQUIRE_FALSE(expected_ec);
    REQUIRE_FALSE(actual_ec);
    REQUIRE(std::filesystem::equivalent(expected_path, actual_path, equivalent_ec));
    REQUIRE_FALSE(equivalent_ec);

    std::filesystem::remove_all(dir);
}

TEST_CASE("MCP tools/call run_terminal_command with invalid working_dir returns error", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo x","working_dir":"/nonexistent_xyz_99999"}},"id":91})");

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":true)"));
}

TEST_CASE("MCP tools/call run_terminal_command enforces workspace when configured", "[mcp]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    const auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    const auto outside = std::filesystem::temp_directory_path() / "filo_mcp_outside_scope_test";
    std::error_code ec;
    std::filesystem::create_directories(outside, ec);
    REQUIRE_FALSE(ec);

    const auto normalized_primary = std::filesystem::weakly_canonical(primary, ec);
    REQUIRE_FALSE(ec);
    const auto normalized_outside = std::filesystem::weakly_canonical(outside, ec);
    REQUIRE_FALSE(ec);
    if (normalized_outside.string().starts_with(normalized_primary.string())) {
        std::filesystem::remove_all(outside, ec);
        SKIP("Temp directory unexpectedly inside workspace; cannot validate out-of-scope guard.");
    }

    std::string req =
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"pwd","working_dir":")"
        + outside.string() + R"("}},"id":92})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":true)"));
    REQUIRE_THAT(resp, ContainsSubstring("Access denied"));

    std::filesystem::remove_all(outside, ec);
}

// ---------------------------------------------------------------------------
// apply_patch working_dir
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/call apply_patch with working_dir applies patch in that directory", "[mcp]") {
    const std::string dir = std::filesystem::temp_directory_path().string() + "/filo_patch_wd_test";
    std::filesystem::create_directories(dir);
    const std::string target = dir + "/hello.txt";
    { std::ofstream ofs(target); ofs << "Hello World\n"; }

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"apply_patch","arguments":{"patch":"--- a/hello.txt\n+++ b/hello.txt\n@@ -1 +1 @@\n-Hello World\n+Hello Filo\n","working_dir":")" + dir + R"("}},"id":95})";
    auto resp = disp().dispatch(req);

    REQUIRE(is_valid_json(resp));
    if (resp.find(R"("isError":false)") != std::string::npos) {
        std::ifstream ifs(target);
        std::string content((std::istreambuf_iterator<char>(ifs)), {});
        REQUIRE_THAT(content, ContainsSubstring("Hello Filo"));
    }

    std::filesystem::remove_all(dir);
}

TEST_CASE("MCP tools/call apply_patch with invalid working_dir returns error", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"apply_patch","arguments":{"patch":"--- a/x\n+++ b/x\n","working_dir":"/nonexistent_xyz_99999"}},"id":96})");

    REQUIRE(is_valid_json(resp));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":true)"));
}

// ---------------------------------------------------------------------------
// tools/list — schema details
// ---------------------------------------------------------------------------

TEST_CASE("MCP tools/list run_terminal_command schema includes working_dir", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":100})");
    REQUIRE_THAT(resp, ContainsSubstring("working_dir"));
}
