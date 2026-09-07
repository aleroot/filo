#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/tools/ReadTool.hpp"
#include "core/tools/PathVisibilityToolDecorator.hpp"
#include "core/tools/WebBackendAdapters.hpp"
#include "core/agent/ToolOutputHistory.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/session/SessionStats.hpp"
#include "TestSessionContext.hpp"
#include <simdjson.h>
#include <fstream>
#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <thread>

using namespace core::tools;
using Catch::Matchers::ContainsSubstring;
namespace {
struct Fixture {
    std::filesystem::path root;
    Fixture() {
        std::string pattern = (std::filesystem::temp_directory_path() / "filo-read-XXXXXX").string();
        const char* directory = ::mkdtemp(pattern.data());
        if (!directory) throw std::runtime_error("Cannot create read test directory.");
        root = directory;
    }
    ~Fixture() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    auto context(std::string id = "unified-read-test") const {
        return test_support::make_session_context({.primary = root, .enforce = true},
            core::context::SessionTransport::cli, std::move(id));
    }
    void write(std::string_view name, std::string_view text) const {
        std::ofstream file(root / name, std::ios::binary);
        file.write(text.data(), text.size());
    }
};
std::string field(std::string_view json, std::string_view key) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    simdjson::dom::element doc;
    REQUIRE(parser.parse(padded).get(doc) == simdjson::SUCCESS);
    std::string_view result;
    REQUIRE(doc[key].get(result) == simdjson::SUCCESS);
    return std::string(result);
}
class RecordingReader final : public core::llm::LLMProvider {
public:
    std::vector<core::llm::ChatRequest> requests;
    std::string response = R"({"answer":"The retry limit is three.","citations":[{"source":1,"first_line":2,"last_line":2}]})";
    bool block = false;
    std::atomic<bool> cancelled = false;
    void stream_response(const core::llm::ChatRequest& request, std::function<void(const core::llm::StreamChunk&)> callback) override {
        requests.push_back(request);
        if (block) while (!cancelled.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        set_last_usage(100, 20);
        callback({.content = response});
        callback(core::llm::StreamChunk::make_final());
    }
    void cancel() override { cancelled = true; }
    bool should_estimate_cost() const override { return false; }
};
read::ReaderWorker worker_for(const std::shared_ptr<RecordingReader>& provider,
                             std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
    return read::ReaderWorker([provider]() -> std::expected<read::WorkerProfile, std::string> {
        return read::WorkerProfile{provider, "local-reader", "test-reader"};
    }, timeout);
}

}

TEST_CASE("Unified read preserves exact text without consulting a reader model", "[read][resources]") {
    Fixture fixture;
    fixture.write("text.txt", "alpha\r\nbeta\nlast");
    auto provider = std::make_shared<RecordingReader>();
    ReadTool tool(read::ResourceReader{}, worker_for(provider));
    CHECK(field(tool.execute(R"({"path":"text.txt"})", fixture.context()), "content") == "alpha\r\nbeta\nlast");
    CHECK(field(tool.execute(R"({"path":"text.txt","offset_line":2,"limit_lines":1})", fixture.context()), "content") == "beta\n");
    CHECK(provider->requests.empty());
    fixture.write("schema.db", "plain text with an arbitrary extension");
    CHECK(field(tool.execute(R"({"path":"schema.db"})", fixture.context()), "content") == "plain text with an arbitrary extension");
    fixture.write("large.txt", std::string(2 * 1024 * 1024, 'x'));
    const auto content = field(tool.execute(R"({"path":"large.txt"})", fixture.context()), "content");
    CHECK(content.size() < 1024 * 1024 + 100);
    CHECK_THAT(content, ContainsSubstring("TRUNCATED"));
}
TEST_CASE("Unified read decodes notebooks and returns exact selected cell sources", "[read][resources]") {
    Fixture fixture;
    fixture.write("book.ipynb", R"({"cells":[{"id":"intro","cell_type":"markdown","source":["# Title\n"]},{"id":"code","cell_type":"code","source":"answer = 42\n","outputs":[{"text":"SECRET OUTPUT"}]}]})");
    ReadTool tool;
    const auto output = tool.execute(R"({"path":"book.ipynb","select":{"cell":"code"}})", fixture.context());
    CHECK(field(output, "content") == "[cell 2 id=code: code]\nanswer = 42\n");
    CHECK(output.find("SECRET OUTPUT") == std::string::npos);
    CHECK_THAT(field(tool.execute(R"({"path":"book.ipynb"})", fixture.context()), "content"),
        ContainsSubstring("SECRET OUTPUT"));
    CHECK_THAT(tool.execute(R"({"path":"book.ipynb","select":{"cell":"missing"}})", fixture.context()), ContainsSubstring("not found"));
    CHECK_THAT(tool.execute(R"({"path":"book.ipynb","select":{"page":1}})", fixture.context()), ContainsSubstring("error"));
}
TEST_CASE("Unified read evidence worker is isolated and sources are freshly validated", "[read][worker]") {
    Fixture fixture;
    fixture.write("retry.cpp", "// retry policy\nconst int limit = 3;\n");
    auto provider = std::make_shared<RecordingReader>();
    ReadTool tool(read::ResourceReader{}, worker_for(provider));
    auto stats = std::make_shared<core::session::SessionStatsRegistry>();
    const ToolInvocationContext context{.session_context = fixture.context(), .tool_call_id = "read-call", .session_stats = stats};
    auto output = tool.execute(R"({"path":"retry.cpp","question":"What is the retry limit?"})", context);
    REQUIRE(provider->requests.size() == 1);
    const auto& request = provider->requests.front();
    CHECK(request.messages.size() == 2);
    CHECK(request.tools.empty());
    CHECK(request.previous_response_id.empty());
    CHECK(request.model == "test-reader");
    CHECK_THAT(request.messages.back().content, ContainsSubstring("const int limit = 3;"));
    CHECK(field(output, "read_view") == "answer");
    CHECK_THAT(output, ContainsSubstring("const int limit = 3;"));
    CHECK_THAT(output, ContainsSubstring("Generated answer"));
    CHECK(core::budget::BudgetTracker::get_instance().snapshot({.session_id = "unified-read-test", .actor = "reader"}).total_tokens >= 120);
    simdjson::dom::parser parser;
    const auto doc = parser.parse(output).value();
    const std::string digest(doc["sources"].at(0)["digest"].get_string().value());
    fixture.write("retry.cpp", "// changed\nconst int limit = 4;\n");
    const auto recovery = tool.execute("{\"path\":\"retry.cpp\",\"view\":\"exact\",\"expected_digest\":\"" + digest + "\",\"offset_line\":2,\"limit_lines\":1}", context);
    CHECK_THAT(recovery, ContainsSubstring("error"));
    CHECK_THAT(recovery, ContainsSubstring("changed since"));
    CHECK(recovery.find("limit = 4") == std::string::npos);
    CHECK(provider->requests.size() == 1);
}
TEST_CASE("Unified read fails back to source when model evidence is invalid", "[read][worker]") {
    Fixture fixture;
    fixture.write("source.txt", "one\ntwo\n");
    auto provider = std::make_shared<RecordingReader>();
    provider->response = R"({"answer":"invented","citations":[{"source":1,"first_line":200,"last_line":201}]})";
    ReadTool tool(read::ResourceReader{}, worker_for(provider));
    const auto output = tool.execute(R"({"path":"source.txt","question":"Explain"})", fixture.context());
    CHECK_THAT(output, ContainsSubstring("fallback_reason"));
    CHECK(output.find("invented") == std::string::npos);
    CHECK(field(output, "content") == "one\ntwo\n");
    provider->response = "not JSON";
    CHECK_THAT(tool.execute(R"({"path":"source.txt","question":"Explain"})", fixture.context()), ContainsSubstring("invalid evidence"));
}
TEST_CASE("Unified read cancels an isolated worker at its deadline", "[read][worker]") {
    Fixture fixture;
    auto provider = std::make_shared<RecordingReader>();
    provider->block = true;
    auto worker = worker_for(provider, std::chrono::milliseconds(30));
    std::vector<read::Resource> sources{{.text = "one\ntwo\n"}};
    auto answer = worker.answer(sources, "Explain", {.session_context = fixture.context()});
    REQUIRE_FALSE(answer);
    CHECK(provider->cancelled);
    CHECK_THAT(answer.error(), ContainsSubstring("timed out"));
}
TEST_CASE("Unified read requires explicit worker configuration and never routes instruction files", "[read][worker]") {
    Fixture fixture;
    fixture.write("AGENTS.md", "Project instructions must remain verbatim.\n");
    auto provider = std::make_shared<RecordingReader>();
    ReadTool tool(read::ResourceReader{}, worker_for(provider));
    const auto instructions = tool.execute(R"({"path":"AGENTS.md","question":"Summarize"})", fixture.context());
    CHECK(field(instructions, "content") == "Project instructions must remain verbatim.\n");
    // The caller must learn that its question was never put to the worker.
    CHECK_THAT(field(instructions, "fallback_reason"),
        ContainsSubstring("never routed to the reader worker"));
    CHECK(provider->requests.empty());
    fixture.write("source.txt", "one\ntwo\n");
    ReadTool unavailable(read::ResourceReader{}, read::ReaderWorker([]() -> std::expected<read::WorkerProfile, std::string> {
        return std::unexpected("Reader not configured");
    }));
    CHECK_THAT(unavailable.execute(R"({"path":"source.txt","question":"Explain"})", fixture.context()), ContainsSubstring("Reader not configured"));
}
TEST_CASE("Unified read keeps exact slices above the compact 8 KiB budget", "[read][resources]") {
    Fixture fixture;
    std::string source;
    for (int i = 1; i <= 400; ++i) {
        source += "line-" + std::to_string(i) + " " + std::string(40, 'x') + "\n";
    }
    fixture.write("wide.txt", source);
    ReadTool tool;
    simdjson::dom::parser parser;
    const auto envelope = parser.parse(
        tool.execute(R"({"path":"wide.txt","view":"auto"})", fixture.context())).value();
    const std::string snapshot(envelope["sources"].at(0)["digest"].get_string().value());
    const auto recovered = tool.execute(
        "{\"path\":\"wide.txt\",\"view\":\"exact\",\"expected_digest\":\"" + snapshot
            + "\",\"offset_line\":1,\"limit_lines\":300}",
        fixture.context());
    bool partial = true;
    REQUIRE(parser.parse(recovered).value()["partial"].get(partial) == simdjson::SUCCESS);
    CHECK_FALSE(partial);
    const auto content = field(recovered, "content");
    CHECK(content.size() > 8192);
    CHECK_THAT(content, ContainsSubstring("line-1 "));
    CHECK_THAT(content, ContainsSubstring("line-300 "));
    CHECK(content.find("line-301 ") == std::string::npos);
}

TEST_CASE("Unified read recovers session-scoped stored tool results", "[read][resources]") {
    Fixture fixture;
    core::agent::ToolResultStore store(fixture.root / "results");
    auto stored = store.store("owner", "call", "alpha\nbeta\n");
    REQUIRE(stored);
    ReadTool tool(read::ResourceReader(store), read::ReaderWorker{});
    const auto args = "{\"path\":\"result://" + stored->reference + "\",\"offset_line\":2,\"limit_lines\":1}";
    CHECK(field(tool.execute(args, fixture.context("owner")), "content") == "beta\n");
    CHECK_THAT(tool.execute(args, fixture.context("other")), ContainsSubstring("error"));
}
TEST_CASE("Unified read rejects unsupported selectors and respects workspace boundaries", "[read][resources]") {
    Fixture fixture;
    fixture.write("source.txt", "data");
    ReadTool tool;
    for (const auto args : {R"({"path":[]})", R"({"path":"source.txt","select":{"page":1,"cell":"a"}})",
        R"({"path":"source.txt","view":"imaginary"})", R"({"path":["a","b"],"offset_line":1})",
        R"({"path":"source.txt","limit_lines":0})", R"({"path":"source.txt","select":{"member":"file"}})",
        R"({"path":"source.txt","question":"Explain","offset_line":2})"}) {
        CHECK_THAT(tool.execute(args, fixture.context()), ContainsSubstring("error"));
    }
    CHECK_THAT(tool.execute(R"({"path":"/etc/passwd","view":"auto"})", fixture.context()), ContainsSubstring("error"));
    CHECK_THAT(tool.execute(R"({"path":".","view":"auto"})", fixture.context()), ContainsSubstring("source.txt"));
    CHECK_THAT(tool.execute(R"({"path":"ssh://host/file"})", fixture.context()), ContainsSubstring("Unsupported resource scheme"));
}
TEST_CASE("Unified read compact views reduce parent payload without losing exact recovery", "[read][resources]") {
    Fixture fixture;
    std::string source = "class Service {\npublic:\n void run();\n};\n";
    for (int i = 0; i < 5000; ++i) source += "// implementation detail and explanatory text for fixture\n";
    fixture.write("large.cpp", source);
    ReadTool tool;
    const auto output = tool.execute(R"({"path":"large.cpp","view":"auto"})", fixture.context());
    CHECK(output.size() * 10 < source.size());
    CHECK(field(output, "read_view") == "outline");
    CHECK_THAT(output, ContainsSubstring("omitted bodies are not analyzed"));
    CHECK(field(tool.execute(R"({"path":"large.cpp","offset_line":3,"limit_lines":1})", fixture.context()), "content") == " void run();\n");
    CHECK(core::agent::tool_output_history::clamp_for_history("read", output) == output);
}

TEST_CASE("Unified read sends relevant late lines and rejects references to omitted gaps", "[read][worker]") {
    Fixture fixture;
    std::string text;
    for (int i = 1; i <= 5000; ++i) text += "// unrelated boilerplate for fixture line\n";
    text += "const int retry_limit = 9;\n";
    fixture.write("large.cpp", text);
    auto provider = std::make_shared<RecordingReader>();
    provider->response = R"({"answer":"Nine retries.","citations":[{"source":1,"first_line":5001,"last_line":5001}]})";
    ReadTool tool(read::ResourceReader{}, worker_for(provider));
    const auto result = tool.execute(R"({"path":"large.cpp","question":"What is retry_limit?"})", fixture.context());
    CHECK(field(result, "read_view") == "answer");
    REQUIRE(provider->requests.size() == 1);
    CHECK_THAT(provider->requests[0].messages.back().content, ContainsSubstring("5001: const int retry_limit = 9;"));
    CHECK(provider->requests[0].messages.back().content.size() < 30000);
    CHECK(result.size() * 10 < text.size());
    provider->response = R"({"answer":"Claims omitted code was reviewed.","citations":[{"source":1,"first_line":4500,"last_line":4500}]})";
    const auto invalid = tool.execute(R"({"path":"large.cpp","question":"What is retry_limit?"})", fixture.context());
    CHECK_THAT(invalid, ContainsSubstring("outside its supplied evidence"));
    CHECK(invalid.find("Claims omitted code") == std::string::npos);
}
TEST_CASE("Unified read validates every batched source before invoking the worker", "[read][worker]") {
    Fixture fixture;
    fixture.write("first.txt", "first\n");
    fixture.write("second.txt", "second\n");
    auto provider = std::make_shared<RecordingReader>();
    provider->response = R"({"answer":"Both sources are supplied.","citations":[{"source":1,"first_line":1,"last_line":1},{"source":2,"first_line":1,"last_line":1}]})";
    ReadTool tool(read::ResourceReader{}, worker_for(provider));
    CHECK(field(tool.execute(R"({"path":["first.txt","second.txt"],"question":"Compare"})", fixture.context()), "read_view") == "answer");
    REQUIRE(provider->requests.size() == 1);
    CHECK_THAT(provider->requests[0].messages.back().content, ContainsSubstring("Source 2"));
    CHECK_THAT(tool.execute(R"({"path":["first.txt","/etc/passwd"],"question":"Compare"})", fixture.context()), ContainsSubstring("error"));
    CHECK(provider->requests.size() == 1);
}
TEST_CASE("Unified read directory listings survive dangling symlinks", "[read][resources]") {
    Fixture fixture;
    fixture.write("visible.txt", "visible\n");
#ifndef _WIN32
    std::error_code ec;
    std::filesystem::create_symlink(fixture.root / "missing-target", fixture.root / "dangling", ec);
    if (ec) SKIP("Symlink creation is unavailable in this environment");
#endif
    ReadTool tool;
    const auto listing = tool.execute(R"({"path":".","view":"auto"})", fixture.context());
    CHECK_THAT(listing, ContainsSubstring("visible.txt"));
#ifndef _WIN32
    CHECK_THAT(listing, ContainsSubstring("dangling"));
#endif
}
TEST_CASE("Unified read directories and enhanced reads honor agent ignore", "[read][resources]") {
    Fixture fixture;
    fixture.write(".agentignore", "hidden.txt\n");
    fixture.write("hidden.txt", "do not disclose");
    fixture.write("visible.txt", "visible");
    auto tool = with_path_visibility(std::make_shared<ReadTool>());
    const auto listing = tool->execute(R"({"path":".","view":"auto"})", fixture.context());
    CHECK_THAT(listing, ContainsSubstring("visible.txt"));
    CHECK(listing.find("hidden.txt") == std::string::npos);
    CHECK_THAT(tool->execute(R"({"path":"hidden.txt","view":"auto"})", fixture.context()), ContainsSubstring("error"));
}
TEST_CASE("Unified read preserves URL text and propagates its policy to source-capable transports", "[read][resources]") {
    struct Backend final : web::IWebFetchBackend {
        mutable std::vector<web::FetchRequest> requests;
        bool preserves_bytes() const noexcept override { return true; }
        std::string_view name() const noexcept override { return "test-source-fetch"; }
        bool supports(const ToolInvocationContext&) const override { return true; }
        std::expected<web::FetchResponse, std::string> fetch(const web::FetchRequest& request, const ToolInvocationContext&) const override {
            requests.push_back(request);
            return web::FetchResponse{.final_url = request.url, .content_type = "text/plain", .text = "alpha\r\n  beta\n", .status_code = 200};
        }
    };
    Fixture fixture;
    auto backend = std::make_shared<Backend>();
    web::WebAccess web({}, {backend});
    ReadTool tool(read::ResourceReader(core::agent::ToolResultStore{}, &web), read::ReaderWorker{});
    const auto result = tool.execute(R"({"path":"https://example.com/source.txt","offset_line":2,"limit_lines":1})", fixture.context());
    CHECK(field(result, "content") == "  beta\n");
    REQUIRE(backend->requests.size() == 1);
    CHECK(backend->requests[0].policy_tool == "read");
    CHECK(backend->requests[0].preserve_bytes);
}

TEST_CASE("Unified read bounds remote slices at the fetch presentation cap",
          "[read][resources][web]") {
    struct Backend final : web::IWebFetchBackend {
        bool preserves_bytes() const noexcept override { return true; }
        std::string_view name() const noexcept override { return "test-bulk-fetch"; }
        bool supports(const ToolInvocationContext&) const override { return true; }
        std::expected<web::FetchResponse, std::string> fetch(const web::FetchRequest& request, const ToolInvocationContext&) const override {
            std::string body;
            for (int i = 0; i < 8000; ++i) body += "remote line " + std::to_string(i) + " with filler text\n";
            return web::FetchResponse{.final_url = request.url, .content_type = "text/plain", .text = std::move(body), .status_code = 200};
        }
    };
    Fixture fixture;
    auto backend = std::make_shared<Backend>();
    web::WebAccess web({}, {backend});
    ReadTool tool(read::ResourceReader(core::agent::ToolResultStore{}, &web), read::ReaderWorker{});
    simdjson::dom::parser parser;

    // A line slice must not import more remote bytes than fetch_url returns.
    const auto remote = tool.execute(
        R"({"path":"https://example.com/bulk.txt","offset_line":1,"limit_lines":8000})",
        fixture.context());
    CHECK(field(remote, "content").size() <= read::kMaxRemoteSliceChars);
    bool remote_partial = false;
    REQUIRE(parser.parse(remote).value()["partial"].get(remote_partial) == simdjson::SUCCESS);
    CHECK(remote_partial);

    // The same slice of a local file keeps the exact-recovery ceiling.
    std::string source;
    for (int i = 0; i < 8000; ++i) source += "local line " + std::to_string(i) + " with filler text\n";
    fixture.write("bulk.txt", source);
    const auto local = tool.execute(
        R"({"path":"bulk.txt","view":"auto","offset_line":1,"limit_lines":8000})",
        fixture.context());
    CHECK(field(local, "content").size() > read::kMaxRemoteSliceChars);
    uint64_t reported_lines = 0;
    REQUIRE(parser.parse(local).value()["sources"].at(0)["lines"].get(reported_lines) == simdjson::SUCCESS);
    CHECK(reported_lines == 8000);
}

TEST_CASE("Unified read stops waiting on a reader that ignores cancellation",
          "[read][worker]") {
    struct StubbornReader final : core::llm::LLMProvider {
        void stream_response(const core::llm::ChatRequest&,
                             std::function<void(const core::llm::StreamChunk&)>) override {
            // Deliberately unresponsive: cancel() is a no-op, as it is for a
            // provider blocked in a socket read.
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        void cancel() override { }
        bool should_estimate_cost() const override { return false; }
    };
    Fixture fixture;
    auto provider = std::make_shared<StubbornReader>();
    read::ReaderWorker worker([provider]() -> std::expected<read::WorkerProfile, std::string> {
        return read::WorkerProfile{provider, "local-reader", "test-reader"};
    }, std::chrono::milliseconds(30));
    const std::vector<read::Resource> sources{{.text = "one\ntwo\n"}};

    const auto started_at = std::chrono::steady_clock::now();
    auto answer = worker.answer(sources, "Explain", {.session_context = fixture.context()});
    const auto elapsed = std::chrono::steady_clock::now() - started_at;

    REQUIRE_FALSE(answer);
    CHECK_THAT(answer.error(), ContainsSubstring("timed out"));
    // The turn is released at the deadline plus its bounded grace, never when
    // the orphaned request finally returns.
    CHECK(elapsed < std::chrono::seconds(1));
}
