#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>

#include "exec/Daemon.hpp"

#include <chrono>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <format>
#include <future>
#include <string>
#include <thread>

namespace {

[[nodiscard]] int next_test_port() {
    static std::atomic<int> port{19080};
    return port.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] bool wait_for_ping(int port,
                                 std::chrono::milliseconds timeout = std::chrono::seconds{5}) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(std::chrono::milliseconds{200});
        client.set_read_timeout(std::chrono::milliseconds{200});
        if (auto res = client.Get("/ping"); res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{25});
    }
    return false;
}

[[nodiscard]] httplib::Headers default_mcp_headers(std::string_view session_id = "test-session") {
    return httplib::Headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
        {"MCP-Protocol-Version", "2025-11-25"},
        {"MCP-Session-Id", std::string(session_id)},
    };
}

[[nodiscard]] httplib::Result post_mcp_request(int port,
                                               const std::string& payload,
                                               httplib::Headers headers) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds{1});
    client.set_read_timeout(std::chrono::seconds{10});
    client.set_write_timeout(std::chrono::seconds{10});

    if (headers.find("Content-Type") == headers.end()) {
        headers.emplace("Content-Type", "application/json");
    }
    return client.Post("/mcp", headers, payload, "application/json");
}

[[nodiscard]] httplib::Result delete_mcp_request(int port, httplib::Headers headers) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds{1});
    client.set_read_timeout(std::chrono::seconds{10});
    client.set_write_timeout(std::chrono::seconds{10});
    return client.Delete("/mcp", headers);
}

[[nodiscard]] std::size_t count_lines(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::size_t lines = 0;
    std::string line;
    while (std::getline(in, line)) ++lines;
    return lines;
}

[[nodiscard]] std::filesystem::path temp_file_path(std::string_view prefix) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
         / std::format("{}_{}.txt", prefix, now);
}

class DaemonRunner {
public:
    explicit DaemonRunner(int port)
        : thread_([port]() { exec::daemon::run_server(port, "127.0.0.1"); }) {}

    ~DaemonRunner() {
        exec::daemon::stop_server();
        if (thread_.joinable()) thread_.join();
    }

    DaemonRunner(const DaemonRunner&) = delete;
    DaemonRunner& operator=(const DaemonRunner&) = delete;

private:
    std::thread thread_;
};

} // namespace

TEST_CASE("Daemon replays duplicate completed requests instead of re-running tools",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto out_file = temp_file_path("filo_daemon_replay_completed");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_completed >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto first = post_mcp_request(port, payload, default_mcp_headers("test-session-a"));
    REQUIRE(first);
    REQUIRE(first->status == 200);

    auto second = post_mcp_request(port, payload, default_mcp_headers("test-session-a"));
    REQUIRE(second);
    REQUIRE(second->status == 200);

    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 1);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon collapses in-flight duplicate requests to a single execution",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto out_file = temp_file_path("filo_daemon_replay_inflight");
    std::filesystem::remove(out_file);

    const std::string command =
        "sleep 1; echo replay_inflight >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto f1 = std::async(std::launch::async, [&]() {
        return post_mcp_request(port, payload, default_mcp_headers("test-session-b"));
    });
    auto f2 = std::async(std::launch::async, [&]() {
        return post_mcp_request(port, payload, default_mcp_headers("test-session-b"));
    });

    auto r1 = f1.get();
    auto r2 = f2.get();

    REQUIRE(r1);
    REQUIRE(r2);
    REQUIRE(r1->status == 200);
    REQUIRE(r2->status == 200);

    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 1);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon does not replay across different MCP sessions",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto out_file = temp_file_path("filo_daemon_replay_sessions");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_session_isolation >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto s1 = post_mcp_request(port, payload, default_mcp_headers("session-one"));
    auto s2 = post_mcp_request(port, payload, default_mcp_headers("session-two"));

    REQUIRE(s1);
    REQUIRE(s2);
    REQUIRE(s1->status == 200);
    REQUIRE(s2->status == 200);

    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 2);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon does not replay duplicate requests when MCP session header is missing",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto out_file = temp_file_path("filo_daemon_replay_no_session");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_without_session_header >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto headers_without_session = default_mcp_headers("unused-session");
    headers_without_session.erase("MCP-Session-Id");

    auto first = post_mcp_request(port, payload, headers_without_session);
    auto second = post_mcp_request(port, payload, headers_without_session);

    REQUIRE(first);
    REQUIRE(second);
    REQUIRE(first->status == 200);
    REQUIRE(second->status == 200);

    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 2);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon rejects missing Accept header on /mcp",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto headers = default_mcp_headers("accept-missing");
    headers.erase("Accept");

    auto res = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":1,"method":"ping"})",
        std::move(headers));
    REQUIRE(res);
    REQUIRE(res->status == 406);
    REQUIRE_THAT(res->body, Catch::Matchers::ContainsSubstring("Invalid Accept header"));
}

TEST_CASE("Daemon rejects unsupported MCP protocol version",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto headers = default_mcp_headers("bad-proto");
    headers.erase("MCP-Protocol-Version");
    headers.emplace("MCP-Protocol-Version", "2099-01-01");

    auto res = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":2,"method":"ping"})",
        std::move(headers));
    REQUIRE(res);
    REQUIRE(res->status == 400);
    REQUIRE_THAT(res->body, Catch::Matchers::ContainsSubstring("Unsupported MCP-Protocol-Version"));
}

TEST_CASE("Daemon enforces same-host or loopback origin policy",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto blocked_headers = default_mcp_headers("origin-blocked");
    blocked_headers.emplace("Origin", "https://example.com");
    auto blocked = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":3,"method":"ping"})",
        std::move(blocked_headers));
    REQUIRE(blocked);
    REQUIRE(blocked->status == 403);

    auto allowed_headers = default_mcp_headers("origin-allowed");
    allowed_headers.emplace("Origin", "http://localhost:3000");
    auto allowed = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":4,"method":"ping"})",
        std::move(allowed_headers));
    REQUIRE(allowed);
    REQUIRE(allowed->status == 200);
    REQUIRE_THAT(allowed->body, Catch::Matchers::ContainsSubstring(R"("jsonrpc":"2.0")"));
}

TEST_CASE("Daemon DELETE /mcp requires MCP-Session-Id header",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    httplib::Headers headers{};
    auto res = delete_mcp_request(port, std::move(headers));
    REQUIRE(res);
    REQUIRE(res->status == 400);
    REQUIRE_THAT(res->body, Catch::Matchers::ContainsSubstring("MCP-Session-Id"));
}

TEST_CASE("Daemon DELETE /mcp clears replay cache for that session",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto out_file = temp_file_path("filo_daemon_delete_session");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_after_delete >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto first = post_mcp_request(port, payload, default_mcp_headers("delete-session"));
    auto replayed = post_mcp_request(port, payload, default_mcp_headers("delete-session"));
    REQUIRE(first);
    REQUIRE(replayed);
    REQUIRE(first->status == 200);
    REQUIRE(replayed->status == 200);
    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 1);

    auto deleted = delete_mcp_request(
        port,
        httplib::Headers{{"MCP-Session-Id", "delete-session"}});
    REQUIRE(deleted);
    REQUIRE(deleted->status == 204);

    auto after_delete = post_mcp_request(port, payload, default_mcp_headers("delete-session"));
    REQUIRE(after_delete);
    REQUIRE(after_delete->status == 200);
    REQUIRE(count_lines(out_file) == 2);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon DELETE /mcp resets run_terminal_command shell state for that session",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    constexpr std::string_view token = "daemon_session_token_97";

    const std::string set_payload =
        R"({"jsonrpc":"2.0","id":"set-env","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"export FILO_DAEMON_VAR=daemon_session_token_97"}}})";
    const std::string get_payload_before =
        R"({"jsonrpc":"2.0","id":"get-env-before","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo $FILO_DAEMON_VAR"}}})";
    const std::string get_payload_after =
        R"({"jsonrpc":"2.0","id":"get-env-after","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo $FILO_DAEMON_VAR"}}})";

    auto set_res = post_mcp_request(port, set_payload, default_mcp_headers("session-reset"));
    auto get_res_before = post_mcp_request(port, get_payload_before, default_mcp_headers("session-reset"));
    REQUIRE(set_res);
    REQUIRE(get_res_before);
    REQUIRE(set_res->status == 200);
    REQUIRE(get_res_before->status == 200);
    REQUIRE_THAT(get_res_before->body, Catch::Matchers::ContainsSubstring(std::string(token)));

    auto deleted = delete_mcp_request(
        port,
        httplib::Headers{{"MCP-Session-Id", "session-reset"}});
    REQUIRE(deleted);
    REQUIRE(deleted->status == 204);

    auto get_res_after = post_mcp_request(port, get_payload_after, default_mcp_headers("session-reset"));
    REQUIRE(get_res_after);
    REQUIRE(get_res_after->status == 200);
    REQUIRE_THAT(get_res_after->body, !Catch::Matchers::ContainsSubstring(std::string(token)));
}
