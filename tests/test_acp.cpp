#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "core/agent/Agent.hpp"
#include "core/jsonrpc/StdioLineReader.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/session/SessionStore.hpp"
#include "core/tools/AskUserQuestionTool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/utils/JsonValue.hpp"
#include "exec/acp/AcpAgentSession.hpp"
#include "exec/acp/AcpProtocol.hpp"
#include "exec/acp/AcpServer.hpp"
#include "exec/acp/AcpSession.hpp"
#include "TestSessionContext.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace exec::acp;

namespace {

class MemoryTransport final : public AcpTransport {
public:
    [[nodiscard]] std::optional<std::string> read_line() override {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return closed_ || !inbound_.empty(); });
        if (inbound_.empty()) return std::nullopt;
        auto line = std::move(inbound_.front());
        inbound_.pop_front();
        return line;
    }

    void write_line(std::string_view line) override {
        std::lock_guard lock(mutex_);
        outbound_.emplace_back(line);
        cv_.notify_all();
    }

    void push(std::string line) {
        std::lock_guard lock(mutex_);
        inbound_.push_back(std::move(line));
        cv_.notify_all();
    }

    void close() {
        std::lock_guard lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

    [[nodiscard]] std::size_t size() {
        std::lock_guard lock(mutex_);
        return outbound_.size();
    }

    [[nodiscard]] std::size_t count_of(std::string_view needle) {
        std::lock_guard lock(mutex_);
        return static_cast<std::size_t>(std::ranges::count_if(outbound_, [&](const std::string& line) {
            return line.find(needle) != std::string::npos;
        }));
    }

    /// Position of the first line holding `needle`, or npos.
    [[nodiscard]] std::size_t index_of(std::string_view needle) {
        std::lock_guard lock(mutex_);
        for (std::size_t i = 0; i < outbound_.size(); ++i) {
            if (outbound_[i].find(needle) != std::string::npos) return i;
        }
        return std::string::npos;
    }

    [[nodiscard]] std::string wait_for(std::string_view needle, std::size_t after = 0) {
        std::unique_lock lock(mutex_);
        const auto ok = cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            for (std::size_t i = after; i < outbound_.size(); ++i) {
                if (outbound_[i].find(needle) != std::string::npos) return true;
            }
            return false;
        });
        REQUIRE(ok);
        for (std::size_t i = outbound_.size(); i > after; --i) {
            if (outbound_[i - 1].find(needle) != std::string::npos) return outbound_[i - 1];
        }
        FAIL("matched line disappeared");
        return {};
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> inbound_;
    std::vector<std::string> outbound_;
    bool closed_ = false;
};

/// What the scripted session saw, for assertions after the server ran.
struct ScriptedLog {
    std::mutex mutex;
    bool discarded = false;
    std::vector<std::string> config;
};

ScriptedLog& scripted_log() {
    static ScriptedLog log;
    return log;
}

class ScriptedSession final : public Session {
public:
    [[nodiscard]] std::string id() const override { return "sess1"; }
    [[nodiscard]] core::utils::JsonValue opened_result() const override {
        auto result = core::utils::JsonValue::object();
        result.set("sessionId", "sess1");
        return result;
    }
    void announce(ClientChannel& client) override {
        auto params = core::utils::JsonValue::parse(
            R"({"sessionId":"sess1","update":{"sessionUpdate":"available_commands_update","availableCommands":[{"name":"announced","description":"x"}]}})");
        client.notify("session/update", *params);
    }
    void replay(ClientChannel& client) override {
        auto params = core::utils::JsonValue::parse(
            R"({"sessionId":"sess1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"replayed"}}})");
        client.notify("session/update", *params);
    }
    [[nodiscard]] std::string prompt(const core::utils::JsonValue& blocks, ClientChannel& client) override {
        if (blocks.dump(0).find("needs-auth") != std::string::npos) throw AuthRequired("sign in");
        auto update = core::utils::JsonValue::parse(
            R"({"sessionId":"sess1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"hello"}}})");
        client.notify("session/update", *update);
        auto notices = core::utils::JsonValue::parse(
            client.capabilities().notices
                ? R"({"sessionId":"sess1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"notices:yes"}}})"
                : R"({"sessionId":"sess1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"notices:no"}}})");
        client.notify("session/update", *notices);
        if (cancel_.load()) return "cancelled";
        auto permission = core::utils::JsonValue::parse(
            R"({"sessionId":"sess1","toolCall":{"toolCallId":"t1","title":"Edit"},"options":[{"optionId":"allow-once","name":"Allow once","kind":"allow_once"}]})");
        const auto result = client.request("session/request_permission", *permission);
        if (cancel_.load() || interpret_permission(result, {}).cancelled) return "cancelled";
        return "end_turn";
    }
    void cancel() override { cancel_.store(true); }
    void discard() override {
        std::lock_guard lock(scripted_log().mutex);
        scripted_log().discarded = true;
    }
    [[nodiscard]] std::optional<std::string> set_mode(std::string_view, ClientChannel&) override {
        return std::nullopt;
    }
    [[nodiscard]] std::optional<std::string> set_config(
        std::string_view id, const core::utils::JsonValue& value, ClientChannel&) override {
        std::lock_guard lock(scripted_log().mutex);
        scripted_log().config.push_back(std::string(id) + "=" + std::string(value.as_string()));
        return std::nullopt;
    }
    [[nodiscard]] core::utils::JsonValue config_state() const override {
        return core::utils::JsonValue::array();
    }

private:
    std::atomic<bool> cancel_{false};
};

class FakeCatalog final : public SessionCatalog {
public:
    std::vector<std::string> removed;
    std::optional<std::filesystem::path> last_cwd;

    [[nodiscard]] std::expected<SessionPage, std::string> list(
        const std::optional<std::filesystem::path>& cwd,
        std::string_view cursor) const override {
        if (cursor == "bad") return std::unexpected(std::string("Invalid cursor"));
        const_cast<FakeCatalog*>(this)->last_cwd = cwd;
        SessionPage page;
        page.sessions.push_back(ListedSession{
            .session_id = "old1", .cwd = "/tmp", .title = "Earlier work", .updated_at = "2026-01-01T00:00:00Z"});
        if (cursor.empty()) page.next_cursor = "1";
        return page;
    }

    void remove(std::string_view session_id) override { removed.emplace_back(session_id); }
};

class TextProvider final : public core::llm::LLMProvider {
    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        core::llm::StreamChunk chunk;
        chunk.content = "hello from filo";
        callback(chunk);
        callback(core::llm::StreamChunk::make_final());
    }
};

/// Asks for one tool call, then answers in text once it has the result.
class ToolThenTextProvider final : public core::llm::LLMProvider {
public:
    ToolThenTextProvider(std::string tool, std::string arguments)
        : tool_(std::move(tool)), arguments_(std::move(arguments)) {}

    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        if (calls_.fetch_add(1) == 0) {
            core::llm::ToolCall call;
            call.index = 0;
            call.id = "call-1";
            call.type = "function";
            call.function.name = tool_;
            call.function.arguments = arguments_;
            core::llm::StreamChunk chunk;
            chunk.tools = {std::move(call)};
            chunk.is_final = true;
            callback(chunk);
            return;
        }
        core::llm::StreamChunk chunk;
        chunk.content = "done";
        callback(chunk);
        callback(core::llm::StreamChunk::make_final());
    }

private:
    std::string tool_;
    std::string arguments_;
    std::atomic<int> calls_{0};
};

class RecordingChannel final : public ClientChannel {
public:
    std::vector<std::string> notifications;

    void notify(std::string_view, const core::utils::JsonValue& params) override {
        notifications.push_back(params.dump(0));
    }

    core::utils::JsonValue request(std::string_view, const core::utils::JsonValue&) override {
        FAIL("this turn should not ask for permission");
        return {};
    }
};

[[nodiscard]] std::filesystem::path scratch_dir() {
    auto dir = std::filesystem::temp_directory_path() / "filo-acp-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

} // namespace

TEST_CASE("ACP initialize advertises the editor session", "[acp]") {
    const auto result = initialize_result();
    CHECK(result.find("protocolVersion")->as_integer() == kProtocolVersion);
    const auto* capabilities = result.find("agentCapabilities");
    REQUIRE(capabilities);
    CHECK(capabilities->find("loadSession")->as_bool());
    CHECK(capabilities->find("promptCapabilities")->find("image")->as_bool());
    CHECK_FALSE(capabilities->find("promptCapabilities")->find("audio")->as_bool());
    CHECK(capabilities->find("sessionCapabilities")->find("additionalDirectories")->is_object());
    for (const auto* lifecycle : {"resume", "close", "list", "delete"}) {
        CAPTURE(lifecycle);
        const auto* capability = capabilities->find("sessionCapabilities")->find(lifecycle);
        REQUIRE(capability);
        CHECK(capability->is_object());
    }
    CHECK(result.find("agentInfo")->find("name")->as_string() == "filo");
    CHECK(result.find("authMethods")->is_array());

    std::string error;
    const auto newer = core::utils::JsonValue::parse(R"({"protocolVersion":2})");
    REQUIRE(newer);
    CHECK(negotiated_protocol_version(*newer, &error) == kProtocolVersion);
    const auto missing = core::utils::JsonValue::object();
    CHECK_FALSE(negotiated_protocol_version(missing, &error));
    CHECK(error.find("protocolVersion") != std::string::npos);
}

TEST_CASE("ACP prompt blocks keep text, files and images apart", "[acp]") {
    const auto dir = scratch_dir();
    const auto prompt = core::utils::JsonValue::parse(R"([
        {"type":"text","text":"Look at this."},
        {"type":"resource","resource":{"uri":"file:///tmp/note.txt","text":"note body"}},
        {"type":"resource_link","uri":"file:///tmp/main.cpp","name":"main.cpp"},
        {"type":"audio","data":"aaaa"},
        {"type":"image","mimeType":"image/png","data":"aGk="}
    ])");
    REQUIRE(prompt);
    std::string error;
    const auto blocks = parse_prompt(*prompt, dir, &error);
    REQUIRE(blocks);
    CHECK(blocks->text.find("Look at this.") != std::string::npos);
    CHECK(blocks->text.find("note body") != std::string::npos);
    CHECK(blocks->text.find("/tmp/main.cpp") != std::string::npos);
    CHECK(blocks->text.find("audio omitted") != std::string::npos);
    REQUIRE(blocks->grant_paths.size() == 1);
    CHECK(blocks->grant_paths.front() == "/tmp/main.cpp");
    REQUIRE(blocks->parts.size() == 1);
    CHECK(blocks->parts.front().type == core::llm::ContentPartType::Image);
    CHECK(std::filesystem::exists(blocks->parts.front().path));
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP tool presentation and permission outcomes", "[acp]") {
    CHECK(tool_kind("write_file") == "edit");
    CHECK(tool_kind("shell") == "execute");
    CHECK(tool_title("run_terminal_command", R"({"command":"git status"})") == "Run git status");

    CHECK(tool_display_text("run_terminal_command", R"({"output":"built\n","exit_code":2})")
          == "built\n[exit code 2]");
    CHECK(tool_display_text("run_terminal_command", R"({"output":"ok","exit_code":0})") == "ok");
    CHECK(tool_display_text("read", R"({"error":"no such file"})") == "no such file");
    CHECK(tool_display_text("read", R"({"content":"x"})") == R"({"content":"x"})");

    ToolCallView call{.tool_call_id = "c1", .name = "write_file", .title = "Edit a", .kind = "edit"};
    const auto update = tool_update("s", call, true).dump(0);
    CHECK(update.find(R"("toolName":"write_file")") != std::string::npos);
    CHECK(update.find(R"("name":"write_file")") != std::string::npos);
    CHECK(tool_result_failed(R"({"error":"nope"})"));
    CHECK_FALSE(tool_result_failed(R"({"ok":true})"));

    auto selected = core::utils::JsonValue::object();
    auto outcome = core::utils::JsonValue::object();
    outcome.set("outcome", "selected");
    outcome.set("optionId", "allow-always");
    selected.set("outcome", std::move(outcome));
    const auto allowed = interpret_permission(selected, "allow-always");
    CHECK(allowed.allowed);
    CHECK(allowed.remember);
    CHECK_FALSE(allowed.cancelled);

    auto unknown = core::utils::JsonValue::object();
    auto other = core::utils::JsonValue::object();
    other.set("outcome", "selected");
    other.set("optionId", "something-else");
    unknown.set("outcome", std::move(other));
    const auto denied = interpret_permission(unknown, "allow-always");
    CHECK_FALSE(denied.allowed);
    CHECK_FALSE(denied.cancelled);
}

TEST_CASE("ACP diffs show what a tool did to each file", "[acp]") {
    const auto dir = scratch_dir();
    const auto file = dir / "a.txt";
    const FileSide absent = read_file_side(file);
    CHECK(absent.readable);
    CHECK_FALSE(absent.text);

    {
        std::ofstream out(file);
        out << "old\n";
    }
    const FileSide old_side = read_file_side(file);
    REQUIRE(old_side.text);
    CHECK(*old_side.text == "old\n");

    SECTION("a created file has no old text") {
        const auto diff = file_diff(file, absent, old_side);
        REQUIRE(diff);
        CHECK(diff->find("oldText")->is_null());
        CHECK(diff->find("newText")->as_string() == "old\n");
        CHECK(diff->find("path")->as_string() == file.string());
    }
    SECTION("a modified file has both sides") {
        const auto diff = file_diff(file, old_side, FileSide{.readable = true, .text = "new\n"});
        REQUIRE(diff);
        CHECK(diff->find("oldText")->as_string() == "old\n");
        CHECK(diff->find("newText")->as_string() == "new\n");
    }
    SECTION("a deleted file has empty new text") {
        const auto diff = file_diff(file, old_side, absent);
        REQUIRE(diff);
        CHECK(diff->find("newText")->as_string().empty());
    }
    SECTION("nothing is shown when nothing changed or a side is unknown") {
        CHECK_FALSE(file_diff(file, old_side, old_side));
        CHECK_FALSE(file_diff(file, absent, absent));
        CHECK_FALSE(file_diff(file, FileSide{}, old_side));
        CHECK_FALSE(file_diff("relative.txt", absent, old_side));
    }
    SECTION("binary content is not shown as text") {
        {
            std::ofstream out(file, std::ios::binary);
            out.write("a\0b", 3);
        }
        CHECK_FALSE(read_file_side(file).readable);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP session titles stay on one line", "[acp]") {
    CHECK(session_title("  Fix the\n  build  ") == "Fix the build");
    CHECK(session_title("") .empty());
    const auto long_title = session_title(std::string(100, 'x'), 10);
    CHECK(long_title == std::string(10, 'x') + "…");
    // Never cut inside a multi-byte character.
    const auto accented = session_title("ééééé", 3);
    CHECK(accented == "é…");
}

TEST_CASE("ACP reads what Xcode says it can do", "[acp]") {
    const auto xcode = core::utils::JsonValue::parse(
        R"({"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":true,"writeTextFile":true},"terminal":true,"auth":{"terminal":true},"elicitation":{"form":{}}}})");
    REQUIRE(xcode);
    const auto caps = parse_client_capabilities(*xcode);
    CHECK(caps.terminal_auth);
    CHECK(caps.elicitation_form);
    CHECK_FALSE(caps.notices);

    const auto bare = parse_client_capabilities(core::utils::JsonValue::object());
    CHECK_FALSE(bare.terminal_auth);
    CHECK_FALSE(bare.elicitation_form);
    const auto nulled = core::utils::JsonValue::parse(
        R"({"clientCapabilities":{"session":{"notices":null},"elicitation":{"form":null},"auth":{"terminal":false}}})");
    REQUIRE(nulled);
    const auto off = parse_client_capabilities(*nulled);
    CHECK_FALSE(off.notices);
    CHECK_FALSE(off.elicitation_form);
    CHECK_FALSE(off.terminal_auth);

    const auto with_login = initialize_result(kProtocolVersion, "claude").dump(0);
    CHECK(with_login.find(R"("type":"terminal")") != std::string::npos);
    CHECK(with_login.find(R"("args":["--auth","claude"])") != std::string::npos);
    CHECK(initialize_result().find("authMethods")->items().empty());
}

TEST_CASE("ACP questions become an editor form and come back as answers", "[acp]") {
    std::vector<core::tools::QuestionItem> questions(2);
    questions[0].question = "Which database?";
    questions[0].header = "Database";
    questions[0].options = {{.label = "SQLite", .description = "embedded"}, {.label = "Postgres"},
                            {.accepts_free_text = true}};
    questions[1].question = "Which platforms?";
    questions[1].multi_select = true;
    questions[1].options = {{.label = "iOS"}, {.label = "macOS"}, {.accepts_free_text = true}};

    const auto form = question_form("s1", "call-9", questions);
    CHECK(form.find("mode")->as_string() == "form");
    CHECK(form.find("toolCallId")->as_string() == "call-9");
    const auto* properties = form.find("requestedSchema")->find("properties");
    REQUIRE(properties);
    CHECK(properties->find("q1")->find("enum")->items().size() == 2);
    CHECK(properties->find("q1")->find("description")->as_string().find("SQLite: embedded") != std::string::npos);
    CHECK(properties->find("q2")->find("type")->as_string() == "array");
    CHECK(properties->find("q2_other"));

    const auto accepted = core::utils::JsonValue::parse(
        R"({"action":"accept","content":{"q1":"Postgres","q2":["iOS","macOS"],"q2_other":"  "}})");
    REQUIRE(accepted);
    const auto answers = question_answers(*accepted, questions);
    REQUIRE(answers);
    CHECK((*answers)[0] == std::pair<std::string, std::string>{"Which database?", "Postgres"});
    CHECK((*answers)[1].second == "iOS, macOS");

    const auto other = core::utils::JsonValue::parse(R"({"action":"accept","content":{"q1":"SQLite","q1_other":"DuckDB"}})");
    REQUIRE(other);
    CHECK(question_answers(*other, questions)->front().second == "DuckDB");

    for (const auto* refused : {R"({"action":"decline"})", R"({"action":"cancel"})", R"({"outcome":{"outcome":"cancelled"}})"}) {
        const auto parsed = core::utils::JsonValue::parse(refused);
        REQUIRE(parsed);
        CHECK_FALSE(question_answers(*parsed, questions));
    }
}

TEST_CASE("ACP model state mirrors the model option", "[acp]") {
    ConfigSelect model{.id = "model", .current = "a", .options = {{"a", "Model A", {}}, {"p/b", "Model B · p", {}}}};
    const auto state = model_state(model);
    CHECK(state.find("currentModelId")->as_string() == "a");
    REQUIRE(state.find("availableModels")->items().size() == 2);
    CHECK(state.find("availableModels")->items()[1].find("modelId")->as_string() == "p/b");
}

TEST_CASE("ACP MCP servers parse without keeping secrets in the error", "[acp]") {
    const auto servers = core::utils::JsonValue::parse(R"([
        {"name":"xcode","command":"/usr/bin/xcrun","args":["mcp"],"env":[{"name":"TOKEN","value":"secret"}]},
        {"type":"http","name":"remote","url":"https://example.test/mcp","headers":[{"name":"Authorization","value":"Bearer secret"}]}
    ])");
    REQUIRE(servers);
    std::string error;
    const auto parsed = parse_mcp_servers(&*servers, &error);
    REQUIRE(parsed);
    REQUIRE(parsed->size() == 2);
    CHECK(parsed->front().command == "/usr/bin/xcrun");
    CHECK(parsed->front().env.front() == "TOKEN=secret");
    CHECK(parsed->back().transport == "http");
    CHECK(error.empty());

    const auto mapped = core::utils::JsonValue::parse(
        R"([{"name":"map","command":"/bin/true","env":{"TOKEN":"secret"}}])");
    REQUIRE(mapped);
    std::string map_error;
    const auto mapped_servers = parse_mcp_servers(&*mapped, &map_error);
    REQUIRE(mapped_servers);
    CHECK(mapped_servers->front().env.front() == "TOKEN=secret");
    CHECK(map_error.empty());

    const auto sse = core::utils::JsonValue::parse(
        R"([{"type":"sse","name":"old","url":"https://example.test"}])");
    std::string sse_error;
    CHECK_FALSE(parse_mcp_servers(&*sse, &sse_error));
    CHECK(sse_error.find("secret") == std::string::npos);
    CHECK(sse_error.find("old") != std::string::npos);
}

TEST_CASE("ACP line reader frames CRLF messages and times out", "[acp]") {
    int fds[2] = {-1, -1};
    REQUIRE(::pipe(fds) == 0);
    const std::string bytes = "alpha\r\nbeta\n";
    REQUIRE(::write(fds[1], bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()));

    core::jsonrpc::StdioLineReader reader(fds[0]);
    const auto first = reader.read_until(std::nullopt);
    const auto second = reader.read_until(std::nullopt);
    const auto third = reader.read_until(
        std::chrono::steady_clock::now() + std::chrono::milliseconds(30));
    CHECK(first.status == core::jsonrpc::StdioLineReader::Status::line);
    CHECK(first.line == "alpha");
    CHECK(second.line == "beta");
    CHECK(third.status == core::jsonrpc::StdioLineReader::Status::timeout);
    ::close(fds[0]);
    ::close(fds[1]);
}

TEST_CASE("ACP server completes a prompt and a cancellation", "[acp]") {
    MemoryTransport transport;
    std::thread server([&] {
        serve(transport, [](const OpenRequest&) {
            return std::unique_ptr<Session>(std::make_unique<ScriptedSession>());
        });
    });

    transport.push(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}})");
    CHECK(transport.wait_for("protocolVersion").find("\"filo\"") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":7,"method":"authenticate","params":{"methodId":"none"}})");
    const auto authenticated = transport.wait_for("\"id\":7");
    CHECK(authenticated.find("\"error\"") == std::string::npos);
    CHECK(authenticated.find("\"result\"") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":2,"method":"nope","params":{}})");
    CHECK(transport.wait_for("-32601").find("Method not found") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":3,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}})");
    CHECK(transport.wait_for("sess1").find("sessionId") != std::string::npos);

    transport.push(
        R"({"jsonrpc":"2.0","id":4,"method":"session/prompt","params":{"sessionId":"sess1","prompt":[{"type":"text","text":"hi"}]}})");
    CHECK(transport.wait_for("request_permission").find("t1") != std::string::npos);
    transport.push(
        R"({"jsonrpc":"2.0","id":1,"result":{"outcome":{"outcome":"selected","optionId":"allow-once"}}})");
    CHECK(transport.wait_for("end_turn").find("stopReason") != std::string::npos);

    const auto before_second = transport.size();
    transport.push(
        R"({"jsonrpc":"2.0","id":5,"method":"session/prompt","params":{"sessionId":"sess1","prompt":[{"type":"text","text":"again"}]}})");
    REQUIRE(transport.wait_for("request_permission", before_second).find("toolCallId") != std::string::npos);
    transport.push(R"({"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"sess1"}})");
    CHECK(transport.wait_for("cancelled", before_second).find("stopReason") != std::string::npos);

    transport.close();
    server.join();
}

TEST_CASE("ACP session lifecycle follows the protocol order", "[acp]") {
    MemoryTransport transport;
    FakeCatalog catalog;
    std::thread server([&] {
        serve(
            transport,
            [](const OpenRequest&) { return std::unique_ptr<Session>(std::make_unique<ScriptedSession>()); },
            ServeOptions{.catalog = &catalog, .login_provider = "claude"});
    });

    transport.push(
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{"session":{"notices":{}}}}})");
    static_cast<void>(transport.wait_for("protocolVersion"));

    SECTION("commands are announced only after the editor knows the session") {
        transport.push(R"({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}})");
        static_cast<void>(transport.wait_for("announced"));
        const auto response = transport.index_of(R"("id":2)");
        const auto announced = transport.index_of("announced");
        REQUIRE(response != std::string::npos);
        CHECK(response < announced);
    }

    SECTION("a loaded session replays before its answer, a resumed one never") {
        transport.push(R"({"jsonrpc":"2.0","id":2,"method":"session/load","params":{"sessionId":"sess1","cwd":"/tmp","mcpServers":[]}})");
        static_cast<void>(transport.wait_for("announced"));
        CHECK(transport.index_of("replayed") < transport.index_of(R"("id":2)"));

        transport.push(R"({"jsonrpc":"2.0","id":3,"method":"session/close","params":{"sessionId":"sess1"}})");
        static_cast<void>(transport.wait_for(R"("id":3)"));
        const auto before_resume = transport.size();
        transport.push(R"({"jsonrpc":"2.0","id":4,"method":"session/resume","params":{"sessionId":"sess1","cwd":"/tmp","mcpServers":[]}})");
        const auto resumed = transport.wait_for(R"("id":4)", before_resume);
        CHECK(resumed.find(R"("result")") != std::string::npos);
        CHECK(resumed.find("sessionId") == std::string::npos);
        CHECK(transport.count_of("replayed") == 1);
    }

    SECTION("closing cancels the running prompt, frees the session, and is idempotent") {
        transport.push(R"({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}})");
        static_cast<void>(transport.wait_for(R"("id":2)"));
        transport.push(
            R"({"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"sess1","prompt":[{"type":"text","text":"hi"}]}})");
        static_cast<void>(transport.wait_for("request_permission"));
        CHECK(transport.index_of("notices:yes") != std::string::npos);

        transport.push(R"({"jsonrpc":"2.0","id":4,"method":"session/close","params":{"sessionId":"sess1"}})");
        CHECK(transport.wait_for(R"("id":4)").find(R"("result":{})") != std::string::npos);
        CHECK(transport.wait_for(R"("stopReason":"cancelled")").find(R"("id":3)") != std::string::npos);

        transport.push(R"({"jsonrpc":"2.0","id":5,"method":"session/close","params":{"sessionId":"sess1"}})");
        CHECK(transport.wait_for(R"("id":5)").find(R"("result":{})") != std::string::npos);

        // The id is free again: the closed session no longer occupies it.
        transport.push(R"({"jsonrpc":"2.0","id":6,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}})");
        CHECK(transport.wait_for(R"("id":6)").find(R"("sessionId":"sess1")") != std::string::npos);
    }

    SECTION("the older model picker, sign-in failures and deletion reach the session") {
        {
            std::lock_guard lock(scripted_log().mutex);
            scripted_log().discarded = false;
            scripted_log().config.clear();
        }
        transport.push(R"({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}})");
        static_cast<void>(transport.wait_for(R"("id":2)"));
        transport.push(
            R"({"jsonrpc":"2.0","id":3,"method":"session/set_model","params":{"sessionId":"sess1","modelId":"gemini/gemini-2.5-pro"}})");
        CHECK(transport.wait_for(R"("id":3)").find(R"("result":{})") != std::string::npos);

        transport.push(
            R"({"jsonrpc":"2.0","id":4,"method":"session/prompt","params":{"sessionId":"sess1","prompt":[{"type":"text","text":"needs-auth"}]}})");
        const auto failed = transport.wait_for(R"("id":4)");
        CHECK(failed.find(R"("code":-32000)") != std::string::npos);

        transport.push(R"({"jsonrpc":"2.0","id":5,"method":"session/delete","params":{"sessionId":"sess1"}})");
        static_cast<void>(transport.wait_for(R"("id":5)"));
        std::lock_guard lock(scripted_log().mutex);
        CHECK(scripted_log().config == std::vector<std::string>{"model=gemini/gemini-2.5-pro"});
        CHECK(scripted_log().discarded);
    }

    SECTION("saved sessions can be listed and deleted") {
        transport.push(R"({"jsonrpc":"2.0","id":2,"method":"session/list","params":{"cwd":"/tmp"}})");
        const auto listed = transport.wait_for(R"("id":2)");
        CHECK(listed.find(R"("sessionId":"old1")") != std::string::npos);
        CHECK(listed.find(R"("title":"Earlier work")") != std::string::npos);
        CHECK(listed.find(R"("nextCursor":"1")") != std::string::npos);

        transport.push(R"({"jsonrpc":"2.0","id":3,"method":"session/list","params":{"cursor":"bad"}})");
        CHECK(transport.wait_for(R"("id":3)").find("-32602") != std::string::npos);
        transport.push(R"({"jsonrpc":"2.0","id":4,"method":"session/list","params":{"cwd":"relative"}})");
        CHECK(transport.wait_for(R"("id":4)").find("-32602") != std::string::npos);

        transport.push(R"({"jsonrpc":"2.0","id":5,"method":"session/delete","params":{"sessionId":"old1"}})");
        CHECK(transport.wait_for(R"("id":5)").find(R"("result":{})") != std::string::npos);
    }

    transport.close();
    server.join();
    if (!catalog.removed.empty()) {
        CHECK((catalog.removed == std::vector<std::string>{"old1"}
               || catalog.removed == std::vector<std::string>{"sess1"}));
    }
}

TEST_CASE("ACP session streams a model reply", "[acp]") {
    auto provider = std::make_shared<TextProvider>();
    auto& tools = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tools,
        test_support::make_workspace_session_context(core::context::SessionTransport::acp, "acp"));
    const auto dir = scratch_dir();
    AgentSession session(
        std::move(agent),
        "acp",
        std::filesystem::current_path(),
        "test",
        AgentSession::Options{},
        {},
        dir);

    RecordingChannel channel;
    const auto prompt = core::utils::JsonValue::parse(R"([{"type":"text","text":"Say hello"}])");
    REQUIRE(prompt);
    const auto reason = session.prompt(*prompt, channel);
    CHECK(reason == "end_turn");
    REQUIRE_FALSE(channel.notifications.empty());
    CHECK(std::ranges::any_of(channel.notifications, [](const std::string& note) {
        return note.find("hello from filo") != std::string::npos;
    }));
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP passes a prompt that starts with a path to the model", "[acp]") {
    auto agent = std::make_shared<core::agent::Agent>(
        std::make_shared<TextProvider>(),
        core::tools::ToolManager::get_instance(),
        test_support::make_workspace_session_context(core::context::SessionTransport::acp, "acp-path"));
    const auto dir = scratch_dir();
    AgentSession session(std::move(agent), "acp-path", std::filesystem::current_path(), "test",
                         AgentSession::Options{}, {}, dir);

    RecordingChannel channel;
    const auto path_prompt = core::utils::JsonValue::parse(
        R"([{"type":"text","text":"/Users/me/App/ContentView.swift why does this crash?"}])");
    REQUIRE(path_prompt);
    CHECK(session.prompt(*path_prompt, channel) == "end_turn");
    const auto said = [&](std::string_view needle) {
        return std::ranges::any_of(channel.notifications, [&](const std::string& note) {
            return note.find(needle) != std::string::npos;
        });
    };
    CHECK(said("hello from filo"));
    CHECK_FALSE(said("Unknown command"));
    // The first exchange names the conversation for the editor's history.
    CHECK(said(R"("sessionUpdate":"session_info_update")"));
    CHECK(said("/Users/me/App/ContentView.swift why does this crash?"));

    channel.notifications.clear();
    const auto command = core::utils::JsonValue::parse(R"([{"type":"text","text":"/mode"}])");
    REQUIRE(command);
    CHECK(session.prompt(*command, channel) == "end_turn");
    CHECK(said("Mode is"));
    CHECK_FALSE(said("hello from filo"));
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP tool cards carry the diff of the file a tool changed", "[acp]") {
    const auto root = std::filesystem::weakly_canonical(scratch_dir());
    const auto sessions = root / "sessions";
    const auto target = root / "notes.txt";
    const bool existed = GENERATE(false, true);
    if (existed) {
        std::ofstream out(target);
        out << "before\n";
    }

    auto& tools = core::tools::ToolManager::get_instance();
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    auto args = core::utils::JsonValue::object();
    args.set("file_path", target.string());
    args.set("content", "after\n");
    auto agent = std::make_shared<core::agent::Agent>(
        std::make_shared<ToolThenTextProvider>("write_file", args.dump(0)),
        tools,
        test_support::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = root, .enforce = true, .version = 1},
            core::context::SessionTransport::acp,
            "acp-diff"));
    AgentSession session(std::move(agent), "acp-diff", root, "test",
                         AgentSession::Options{.yolo = true}, {}, sessions);

    RecordingChannel channel;
    const auto prompt = core::utils::JsonValue::parse(R"([{"type":"text","text":"write the notes"}])");
    REQUIRE(prompt);
    CHECK(session.prompt(*prompt, channel) == "end_turn");

    const auto finished = std::ranges::find_if(channel.notifications, [](const std::string& note) {
        return note.find(R"("status":"completed")") != std::string::npos
            && note.find("call-1") != std::string::npos;
    });
    REQUIRE(finished != channel.notifications.end());
    CAPTURE(*finished);
    CHECK(finished->find(R"("type":"diff")") != std::string::npos);
    CHECK(finished->find(R"("newText":"after\n")") != std::string::npos);
    if (existed) {
        CHECK(finished->find(R"("oldText":"before\n")") != std::string::npos);
    } else {
        CHECK(finished->find(R"("oldText":null)") != std::string::npos);
    }
    // The diff replaces the tool's JSON receipt as the card's text.
    CHECK(finished->find(R"("type":"content")") == std::string::npos);
    std::filesystem::remove_all(root);
}

TEST_CASE("ACP store catalog lists conversations by project and deletes them", "[acp]") {
    const auto dir = scratch_dir();
    core::session::SessionStore store(dir);
    const auto save = [&](std::string id, std::string cwd, bool with_turn) {
        core::session::SessionData data;
        data.session_id = std::move(id);
        data.created_at = core::session::SessionStore::now_iso8601();
        data.last_active_at = data.created_at;
        data.working_dir = std::move(cwd);
        if (with_turn) {
            data.messages.push_back(core::llm::Message{.role = "user", .content = "Fix the build"});
            data.messages.push_back(core::llm::Message{.role = "assistant", .content = "Done"});
        }
        REQUIRE(store.save(data));
    };
    save("aaaa0001", "/tmp/project-a", true);
    save("aaaa0002", "/tmp/project-b", true);
    save("aaaa0003", "/tmp/project-a", false);

    const auto catalog = make_store_catalog(dir);
    const auto page = catalog->list(std::filesystem::path("/tmp/project-a"), "");
    REQUIRE(page);
    REQUIRE(page->sessions.size() == 1);
    CHECK(page->sessions.front().session_id == "aaaa0001");
    CHECK(page->sessions.front().title == "Fix the build");
    CHECK_FALSE(page->next_cursor);

    const auto all = catalog->list(std::nullopt, "");
    REQUIRE(all);
    CHECK(all->sessions.size() == 2);
    CHECK_FALSE(catalog->list(std::nullopt, "x1"));

    catalog->remove("aaaa0001");
    catalog->remove("never-existed");
    const auto after = catalog->list(std::filesystem::path("/tmp/project-a"), "");
    REQUIRE(after);
    CHECK(after->sessions.empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP resumed session keeps writing the same session file", "[acp]") {
    const auto dir = scratch_dir();
    core::session::SessionData data;
    data.session_id = "acpkeep1";
    data.created_at = "2026-01-02T03:04:05Z";
    data.last_active_at = data.created_at;
    data.working_dir = std::filesystem::current_path().string();
    data.messages.push_back(core::llm::Message{.role = "user", .content = "earlier"});
    REQUIRE(core::session::SessionStore(dir).save(data));

    auto agent = std::make_shared<core::agent::Agent>(
        std::make_shared<TextProvider>(),
        core::tools::ToolManager::get_instance(),
        test_support::make_workspace_session_context(core::context::SessionTransport::acp, "acpkeep1"));
    AgentSession session(std::move(agent), "acpkeep1", std::filesystem::current_path(), "test",
                         AgentSession::Options{.created_at = data.created_at}, data.messages, dir);

    std::size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.find("acpkeep1") != std::string::npos && name.ends_with(".json")) ++files;
    }
    CHECK(files == 1);
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP offers a plan mode that Xcode's plan control can leave", "[acp]") {
    auto agent = std::make_shared<core::agent::Agent>(
        std::make_shared<TextProvider>(),
        core::tools::ToolManager::get_instance(),
        test_support::make_workspace_session_context(core::context::SessionTransport::acp, "acp-plan"));
    const auto dir = scratch_dir();
    AgentSession session(std::move(agent), "acp-plan", std::filesystem::current_path(), "test",
                         AgentSession::Options{}, {}, dir);

    const auto opened = session.opened_result();
    const auto* modes = opened.find("modes");
    REQUIRE(modes);
    CHECK(modes->find("currentModeId")->as_string() == "build");
    CHECK(std::ranges::any_of(modes->find("availableModes")->items(), [](const core::utils::JsonValue& mode) {
        return mode.find("id")->as_string() == "plan" && mode.find("name")->as_string() == "Plan";
    }));
    REQUIRE(opened.find("models"));
    CHECK(opened.find("models")->find("currentModelId"));

    RecordingChannel channel;
    REQUIRE_FALSE(session.set_mode("plan", channel));
    CHECK(session.opened_result().find("modes")->find("currentModeId")->as_string() == "plan");

    const auto implement = core::utils::JsonValue::parse(
        R"([{"type":"text","text":"<system-reminder>The user exited plan mode via the \"Implement this Plan\" control.</system-reminder> go"}])");
    REQUIRE(implement);
    CHECK(session.prompt(*implement, channel) == "end_turn");
    CHECK(session.opened_result().find("modes")->find("currentModeId")->as_string() == "build");
    std::filesystem::remove_all(dir);
}

/// Answers the form the session sends, as the editor would.
class FormChannel final : public ClientChannel {
public:
    std::vector<std::string> requests;
    std::vector<std::string> notes;
    /// Runs while the question is open, before the editor answers.
    std::function<void()> while_asking;

    void notify(std::string_view, const core::utils::JsonValue& params) override { notes.push_back(params.dump(0)); }
    core::utils::JsonValue request(std::string_view method, const core::utils::JsonValue& params) override {
        requests.push_back(std::string(method) + " " + params.dump(0));
        if (while_asking) while_asking();
        return *core::utils::JsonValue::parse(R"({"action":"accept","content":{"q1":"Yes"}})");
    }
    [[nodiscard]] ClientCapabilities capabilities() const override {
        return ClientCapabilities{.elicitation_form = true};
    }
};

TEST_CASE("ACP asks the model's questions through the editor", "[acp]") {
    auto args = core::utils::JsonValue::parse(
        R"({"questions":[{"question":"Proceed?","header":"Go","options":[{"label":"Yes"},{"label":"No"}]}]})");
    REQUIRE(args);
    auto& tools = core::tools::ToolManager::get_instance();
    auto ask = std::make_shared<core::tools::AskUserQuestionTool>();
    ask->setQuestionCallback(route_question);
    tools.register_tool(ask);

    auto agent = std::make_shared<core::agent::Agent>(
        std::make_shared<ToolThenTextProvider>("AskUserQuestion", args->dump(0)),
        tools,
        test_support::make_workspace_session_context(core::context::SessionTransport::acp, "acp-ask"));
    const auto dir = scratch_dir();
    AgentSession session(std::move(agent), "acp-ask", std::filesystem::current_path(), "test",
                         AgentSession::Options{.yolo = true}, {}, dir);

    FormChannel channel;
    // Another conversation opens while this one waits for its answer, as when
    // the user starts a second chat. That must not wait for the answer.
    bool opened_meanwhile = false;
    channel.while_asking = [&] {
        auto other = std::async(std::launch::async, [&] {
            auto other_agent = std::make_shared<core::agent::Agent>(
                std::make_shared<TextProvider>(),
                core::tools::ToolManager::get_instance(),
                test_support::make_workspace_session_context(core::context::SessionTransport::acp, "acp-other"));
            AgentSession second(std::move(other_agent), "acp-other", std::filesystem::current_path(), "test",
                                AgentSession::Options{}, {}, dir);
        });
        opened_meanwhile = other.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    };
    const auto prompt = core::utils::JsonValue::parse(R"([{"type":"text","text":"ask me"}])");
    REQUIRE(prompt);
    CHECK(session.prompt(*prompt, channel) == "end_turn");
    REQUIRE(channel.requests.size() == 1);
    CHECK(opened_meanwhile);
    CHECK(channel.requests.front().starts_with("elicitation/create "));
    CHECK(channel.requests.front().find(R"("toolCallId":"call-1")") != std::string::npos);
    // The answer is what the tool hands back to the model.
    CHECK(std::ranges::any_of(channel.notes, [](const std::string& note) {
        return note.find(R"("status":"completed")") != std::string::npos
            && note.find(R"("rawOutput":{"answers":{"Proceed?":"Yes"}})") != std::string::npos
            && note.find(R"("title":"Ask the user")") != std::string::npos;
    }));
    static_cast<void>(tools.unregister_tool("AskUserQuestion"));
    std::filesystem::remove_all(dir);
}

TEST_CASE("ACP refuses a session the protocol cannot open", "[acp]") {
    HostOptions host;
    OpenRequest relative;
    relative.cwd = "relative";
    const auto rejected = open_session(relative, host);
    REQUIRE_FALSE(rejected);
    CHECK(rejected.error().kind == OpenError::Kind::invalid_params);
    CHECK(rejected.error().message.find("absolute") != std::string::npos);

    OpenRequest missing;
    missing.cwd = std::filesystem::temp_directory_path();
    missing.resume_id = "acp-missing-session";
    const auto absent = open_session(missing, host);
    REQUIRE_FALSE(absent);
    CHECK(absent.error().kind == OpenError::Kind::not_found);
}

TEST_CASE("ACP reports the errors editors look up by code", "[acp]") {
    MemoryTransport transport;
    std::thread server([&] {
        serve(transport, [](const OpenRequest& request)
                             -> std::expected<std::unique_ptr<Session>, OpenError> {
            if (request.resume_id && *request.resume_id == "missing") {
                return std::unexpected(OpenError::not_found("session not found"));
            }
            if (request.resume_id && *request.resume_id == "broken") {
                return std::unexpected(OpenError::internal("session store failed"));
            }
            return std::unique_ptr<Session>(std::make_unique<ScriptedSession>());
        });
    });

    transport.push(R"({"jsonrpc":"2.0","id":1,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[]}})");
    CHECK(transport.wait_for("-32600").find("Invalid request") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})");
    CHECK(transport.wait_for("-32602").find("protocolVersion") != std::string::npos);

    const auto before_version = transport.size();
    transport.push(R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":2}})");
    CHECK(transport.wait_for("\"protocolVersion\":1", before_version).find("agentCapabilities") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":2,"method":"session/prompt","params":{"sessionId":"absent","prompt":[]}})");
    const auto missing_prompt = transport.wait_for("-32002");
    CHECK(missing_prompt.find("Resource not found") != std::string::npos);
    CHECK(missing_prompt.find("absent") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":3,"method":"session/set_mode","params":{"sessionId":"absent","modeId":"BUILD"}})");
    CHECK(transport.wait_for("\"id\":3").find("\"sessionId\":\"absent\"") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":4,"method":"session/new","params":{"cwd":"/tmp","mcpServers":[],"systemPrompt":{"mode":"append","content":[]}}})");
    CHECK(transport.wait_for("\"id\":4").find("systemPrompt is not supported") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":6,"method":"session/load","params":{"sessionId":"missing","cwd":"/tmp","mcpServers":[]}})");
    const auto missing_load = transport.wait_for("\"id\":6");
    CHECK(missing_load.find("-32002") != std::string::npos);
    CHECK(missing_load.find("\"sessionId\":\"missing\"") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":7,"method":"session/load","params":{"sessionId":"broken","cwd":"/tmp","mcpServers":[]}})");
    CHECK(transport.wait_for("\"id\":7").find("-32603") != std::string::npos);

    transport.push(R"({"jsonrpc":"2.0","id":9,"method":"session/load","params":{"sessionId":"sess1","cwd":"/tmp","mcpServers":[]}})");
    CHECK(transport.wait_for("replayed").find("agent_message_chunk") != std::string::npos);
    const auto loaded = transport.wait_for("\"id\":9");
    CHECK(loaded.find("sessionId") == std::string::npos);
    CHECK(loaded.find("\"result\"") != std::string::npos);

    const auto before_prompt = transport.size();
    transport.push(
        R"({"jsonrpc":"2.0","id":10,"method":"session/prompt","params":{"sessionId":"sess1","prompt":[{"type":"text","text":"hi"}]}})");
    REQUIRE(transport.wait_for("request_permission", before_prompt).find("toolCallId") != std::string::npos);
    transport.push(R"({"jsonrpc":"2.0","method":"$/cancel_request","params":{"requestId":10}})");
    CHECK(transport.wait_for("cancelled", before_prompt).find("stopReason") != std::string::npos);

    transport.close();
    server.join();
}
