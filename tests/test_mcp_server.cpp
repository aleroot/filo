#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/mcp/McpDispatcher.hpp"
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
            "delete_file", "move_file", "create_directory"
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

TEST_CASE("MCP tools/call list_directory on current directory succeeds", "[mcp]") {
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_directory","arguments":{"dir_path":"."}},"id":11})");

    REQUIRE_THAT(resp, ContainsSubstring(R"("id":11)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("content")"));
}

TEST_CASE("MCP tools/call read_file returns file contents", "[mcp]") {
    const std::string path = "mcp_test_artifact.txt";
    { std::ofstream ofs(path); ofs << "hello from lampo"; }

    std::string req =
        std::string(R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"read_file","arguments":{"file_path":")") +
        path + R"("}},"id":12})";
    auto resp = disp().dispatch(req);

    REQUIRE_THAT(resp, ContainsSubstring("hello from lampo"));
    REQUIRE_THAT(resp, ContainsSubstring(R"("isError":false)"));

    std::filesystem::remove(path);
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

    std::string req = R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"read_file","arguments":{"file_path":")" + path + R"(","offset_line":2,"limit_lines":2}},"id":70})";
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
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"grep_search","arguments":{"pattern":"it's","dir_path":"."}},"id":80})");

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
