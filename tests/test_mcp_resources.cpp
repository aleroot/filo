#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/mcp/McpDispatcher.hpp"
#include "core/workspace/Workspace.hpp"
#include <filesystem>
#include <fstream>
#include <simdjson.h>
#include <string>

using namespace Catch::Matchers;

static core::mcp::McpDispatcher& disp() {
    return core::mcp::McpDispatcher::get_instance();
}

struct ScopedPathCleanup {
    std::filesystem::path path;
    ~ScopedPathCleanup() {
        if (path.empty()) return;
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct WorkspaceResetToDefault {
    ~WorkspaceResetToDefault() {
        std::error_code ec;
        auto cwd = std::filesystem::current_path(ec);
        auto& ws = core::workspace::Workspace::get_instance();
        if (ec) {
            ws.initialize("", {}, false);
            return;
        }
        ws.initialize(cwd, {}, false);
    }
};

static simdjson::dom::element parse_json_or_require(const std::string& s,
                                                    simdjson::dom::parser& parser) {
    simdjson::dom::element doc;
    REQUIRE(parser.parse(s).get(doc) == simdjson::SUCCESS);
    return doc;
}

static std::string encode_spaces(std::string s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char ch : s) {
        if (ch == ' ') out += "%20";
        else out.push_back(ch);
    }
    return out;
}

// ---------------------------------------------------------------------------
// resources/list
// ---------------------------------------------------------------------------

TEST_CASE("MCP resources/list returns workspace roots", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    // Ensure workspace is initialized for the test
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/list","params":{},"id":1})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    int64_t id = 0;
    REQUIRE(root["id"].get(id) == simdjson::SUCCESS);
    REQUIRE(id == 1);

    simdjson::dom::array resources;
    REQUIRE(root["result"]["resources"].get(resources) == simdjson::SUCCESS);

    bool found_primary = false;
    for (auto item : resources) {
        std::string_view name;
        std::string_view uri;
        std::string_view mime;
        REQUIRE(item["name"].get(name) == simdjson::SUCCESS);
        REQUIRE(item["uri"].get(uri) == simdjson::SUCCESS);
        REQUIRE(item["mimeType"].get(mime) == simdjson::SUCCESS);

        if (name == "Primary Workspace") {
            found_primary = true;
            REQUIRE(uri.starts_with("file://"));
            REQUIRE(mime == "application/x-directory");
        }
    }
    REQUIRE(found_primary);
}

TEST_CASE("MCP resources/list includes additional workspaces", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    auto additional = primary / "mcp_test_additional";
    std::filesystem::create_directories(additional);
    ScopedPathCleanup cleanup{additional};

    ws.initialize(primary, {additional}, true);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/list","params":{},"id":2})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    simdjson::dom::array resources;
    REQUIRE(root["result"]["resources"].get(resources) == simdjson::SUCCESS);

    bool found_additional = false;
    for (auto item : resources) {
        std::string_view name;
        std::string_view uri;
        REQUIRE(item["name"].get(name) == simdjson::SUCCESS);
        REQUIRE(item["uri"].get(uri) == simdjson::SUCCESS);
        if (name == "Additional Workspace 1") {
            found_additional = true;
            REQUIRE(uri.find(additional.filename().string()) != std::string_view::npos);
        }
    }
    REQUIRE(found_additional);
}

TEST_CASE("MCP resources/list URI is percent-encoded", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    auto spaced = primary / "mcp resource root";
    std::filesystem::create_directories(spaced);
    ScopedPathCleanup cleanup{spaced};
    ws.initialize(spaced, {}, true);

    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/list","params":{},"id":9})");
    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    simdjson::dom::array resources;
    REQUIRE(root["result"]["resources"].get(resources) == simdjson::SUCCESS);
    bool found_encoded_uri = false;
    for (auto item : resources) {
        std::string_view uri;
        REQUIRE(item["uri"].get(uri) == simdjson::SUCCESS);
        if (uri.find("%20") != std::string_view::npos) {
            found_encoded_uri = true;
            break;
        }
    }
    REQUIRE(found_encoded_uri);
}

// ---------------------------------------------------------------------------
// resources/read
// ---------------------------------------------------------------------------

TEST_CASE("MCP resources/read on directory returns entry list", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto test_dir = primary / "mcp_res_dir_test";
    std::filesystem::create_directories(test_dir);
    ScopedPathCleanup cleanup{test_dir};
    { std::ofstream(test_dir / "file1.txt") << "content1"; }
    std::filesystem::create_directories(test_dir / "subdir");

    std::string uri = "file://" + test_dir.generic_string();
    std::string req = R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":3})";
    auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    simdjson::dom::array contents;
    REQUIRE(root["result"]["contents"].get(contents) == simdjson::SUCCESS);
    bool saw_entry = false;
    for (auto content : contents) {
        saw_entry = true;
        std::string_view mime;
        std::string_view text;
        REQUIRE(content["mimeType"].get(mime) == simdjson::SUCCESS);
        REQUIRE(content["text"].get(text) == simdjson::SUCCESS);
        REQUIRE(mime == "application/x-directory");
        REQUIRE_THAT(std::string(text), ContainsSubstring("file1.txt"));
        REQUIRE_THAT(std::string(text), ContainsSubstring("subdir"));
    }
    REQUIRE(saw_entry);
}

TEST_CASE("MCP resources/read on file returns content", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto test_file = primary / "mcp_res_file_test.cpp";
    ScopedPathCleanup cleanup{test_file};
    { std::ofstream ofs(test_file); ofs << "#include <iostream>\nint main() { return 0; }"; }

    std::string uri = "file://" + test_file.generic_string();
    std::string req = R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":4})";
    auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    simdjson::dom::array contents;
    REQUIRE(root["result"]["contents"].get(contents) == simdjson::SUCCESS);
    bool saw_file = false;
    for (auto content : contents) {
        saw_file = true;
        std::string_view mime;
        std::string_view text;
        REQUIRE(content["mimeType"].get(mime) == simdjson::SUCCESS);
        REQUIRE(content["text"].get(text) == simdjson::SUCCESS);
        REQUIRE(mime == "text/x-c++src");
        REQUIRE_THAT(std::string(text), ContainsSubstring("int main()"));
    }
    REQUIRE(saw_file);
}

TEST_CASE("MCP resources/read missing params object returns -32602", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    const auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","id":20})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32602);

    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("missing 'params' object"));
}

TEST_CASE("MCP resources/read missing uri returns -32602", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    const auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{},"id":21})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32602);

    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("missing 'uri'"));
}

TEST_CASE("MCP resources/read decodes percent-encoded URI paths", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto spaced_file = primary / "mcp resource spaced file.txt";
    ScopedPathCleanup cleanup{spaced_file};
    { std::ofstream ofs(spaced_file); ofs << "hello spaced resource"; }

    const std::string uri = "file://" + encode_spaces(spaced_file.generic_string());
    const std::string req =
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":10})";
    const auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    simdjson::dom::array contents;
    REQUIRE(root["result"]["contents"].get(contents) == simdjson::SUCCESS);

    bool saw_text = false;
    for (auto content : contents) {
        std::string_view text;
        if (content["text"].get(text) == simdjson::SUCCESS) {
            saw_text = true;
            REQUIRE_THAT(std::string(text), ContainsSubstring("hello spaced resource"));
        }
    }
    REQUIRE(saw_text);
}

TEST_CASE("MCP resources/read accepts localhost file URI authority", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto file = primary / "mcp_resource_localhost.txt";
    ScopedPathCleanup cleanup{file};
    { std::ofstream ofs(file); ofs << "localhost works"; }

    const std::string uri = "file://localhost" + file.generic_string();
    const std::string req =
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":11})";
    const auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    simdjson::dom::array contents;
    REQUIRE(root["result"]["contents"].get(contents) == simdjson::SUCCESS);
}

TEST_CASE("MCP resources/read accepts case-insensitive file URI scheme", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto file = primary / "mcp_resource_case_scheme.txt";
    ScopedPathCleanup cleanup{file};
    { std::ofstream ofs(file); ofs << "case-insensitive scheme works"; }

    const std::string uri = "FILE://LOCALHOST" + file.generic_string();
    const std::string req =
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":18})";
    const auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    simdjson::dom::array contents;
    REQUIRE(root["result"]["contents"].get(contents) == simdjson::SUCCESS);

    bool saw_text = false;
    for (auto content : contents) {
        std::string_view text;
        if (content["text"].get(text) == simdjson::SUCCESS) {
            saw_text = true;
            REQUIRE_THAT(std::string(text), ContainsSubstring("case-insensitive scheme works"));
        }
    }
    REQUIRE(saw_text);
}

TEST_CASE("MCP resources/read returns blob for binary file", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto bin = primary / "mcp_resource_blob.bin";
    ScopedPathCleanup cleanup{bin};
    {
        std::ofstream ofs(bin, std::ios::binary);
        const char bytes[] = { '\x01', '\x02', '\x03' };
        ofs.write(bytes, sizeof(bytes));
    }

    const std::string uri = "file://" + bin.generic_string();
    const std::string req =
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":12})";
    const auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    simdjson::dom::array contents;
    REQUIRE(root["result"]["contents"].get(contents) == simdjson::SUCCESS);
    bool saw_blob = false;
    for (auto content : contents) {
        std::string_view blob;
        if (content["blob"].get(blob) == simdjson::SUCCESS) {
            saw_blob = true;
            REQUIRE(blob == "AQID");
        }
    }
    REQUIRE(saw_blob);
}

TEST_CASE("MCP resources/read rejects oversized binary resource", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto big_bin = primary / "mcp_big_blob.bin";
    ScopedPathCleanup cleanup{big_bin};
    {
        std::ofstream ofs(big_bin, std::ios::binary);
        std::string payload(1024 * 1024 + 32, '\0');
        payload[0] = '\x1';
        payload.back() = '\x2';
        ofs.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }

    const std::string uri = "file://" + big_bin.generic_string();
    const std::string req =
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":13})";
    const auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32603);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("Binary resource too large"));
}

TEST_CASE("MCP resources/read rejects paths outside workspace", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    // Assuming /tmp is outside current path for this test
    std::string uri = "file:///tmp/some_secret_file";
    std::string req = R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":5})";
    auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32001);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("Access denied"));
}

TEST_CASE("MCP resources/read returns error for non-existent file", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto& ws = core::workspace::Workspace::get_instance();
    auto primary = std::filesystem::current_path();
    ws.initialize(primary, {}, true);

    auto test_file = primary / "non_existent_mcp_file.txt";
    std::string uri = "file://" + test_file.generic_string();
    std::string req = R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":")" + uri + R"("},"id":6})";
    auto resp = disp().dispatch(req);

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32002);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("Resource not found"));
}

TEST_CASE("MCP resources/read rejects non-file URIs", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":"http://google.com"},"id":7})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32602);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("file:// URIs are supported"));
}

TEST_CASE("MCP resources/read rejects non-local file URI authority", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":"file://example.com/tmp/file.txt"},"id":17})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32602);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("unsupported file URI authority"));
}

TEST_CASE("MCP resources/read rejects malformed percent encoding", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":"file:///tmp/bad%ZZuri"},"id":14})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32602);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("malformed percent-encoding"));
}

TEST_CASE("MCP resources/read rejects NUL bytes in URI path", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":"file:///tmp/bad%00uri"},"id":19})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);
    int64_t code = 0;
    REQUIRE(root["error"]["code"].get(code) == simdjson::SUCCESS);
    REQUIRE(code == -32602);
    std::string_view message;
    REQUIRE(root["error"]["message"].get(message) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(message), ContainsSubstring("NUL byte"));
}

TEST_CASE("MCP resources/read rejects URI query and fragment", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto resp_query = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":"file:///tmp/file.txt?x=1"},"id":15})");
    auto resp_fragment = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/read","params":{"uri":"file:///tmp/file.txt#frag"},"id":16})");

    simdjson::dom::parser p1;
    auto d1 = parse_json_or_require(resp_query, p1);
    int64_t c1 = 0;
    REQUIRE(d1["error"]["code"].get(c1) == simdjson::SUCCESS);
    REQUIRE(c1 == -32602);

    simdjson::dom::parser p2;
    auto d2 = parse_json_or_require(resp_fragment, p2);
    int64_t c2 = 0;
    REQUIRE(d2["error"]["code"].get(c2) == simdjson::SUCCESS);
    REQUIRE(c2 == -32602);
}

TEST_CASE("MCP resources/templates/list returns an empty array", "[mcp][resources]") {
    WorkspaceResetToDefault workspace_reset;
    auto resp = disp().dispatch(
        R"({"jsonrpc":"2.0","method":"resources/templates/list","params":{},"id":8})");

    simdjson::dom::parser parser;
    auto root = parse_json_or_require(resp, parser);

    simdjson::dom::array templates;
    REQUIRE(root["result"]["resourceTemplates"].get(templates) == simdjson::SUCCESS);
    REQUIRE(templates.begin() == templates.end());
}
