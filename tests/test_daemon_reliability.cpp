#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>

#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderManager.hpp"
#include "exec/ApiGateway.hpp"
#include "exec/Daemon.hpp"

#include <chrono>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <format>
#include <future>
#include <optional>
#include <simdjson.h>
#include <string>
#include <thread>
#include <unordered_set>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

[[nodiscard]] int current_process_id() {
#if defined(_WIN32)
    return static_cast<int>(_getpid());
#else
    return static_cast<int>(getpid());
#endif
}

[[nodiscard]] int next_test_port() {
    // Avoid bind-probing with cpp-httplib: bind_to_port() without listen can
    // leave a bound socket around on some Linux/libstdc++ combos.
    // Use a deterministic per-process sequence instead and let wait_for_ping()
    // skip when the selected port is unavailable.
    constexpr int kMinPort = 20000;
    constexpr int kMaxPort = 60999;
    constexpr int kSpan = kMaxPort - kMinPort + 1; // 41,000 ports

    static std::atomic<int> next{
        kMinPort + ((current_process_id() * 37) % kSpan)
    };

    const int offset = next.fetch_add(1, std::memory_order_relaxed);
    return kMinPort + (offset % kSpan);
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

struct InitializedSession {
    std::string session_id;
    std::string protocol_version;
};

[[nodiscard]] httplib::Result post_mcp_request(int port,
                                               const std::string& payload,
                                               httplib::Headers headers);

[[nodiscard]] httplib::Headers session_headers(const InitializedSession& session) {
    return httplib::Headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
        {"MCP-Protocol-Version", session.protocol_version},
        {"MCP-Session-Id", session.session_id},
    };
}

[[nodiscard]] std::optional<std::string>
extract_negotiated_protocol_version(std::string_view response_body) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(response_body).get(doc) != simdjson::SUCCESS) return std::nullopt;

    std::string_view protocol_version;
    if (doc["result"]["protocolVersion"].get(protocol_version) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return std::string(protocol_version);
}

[[nodiscard]] std::optional<InitializedSession> initialize_mcp_session(
    int port,
    std::string_view requested_protocol = "2025-11-25")
{
    httplib::Headers init_headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
    };
    const std::string init_payload = std::format(
        R"({{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{}","capabilities":{{}},"clientInfo":{{"name":"daemon-test","version":"1.0"}}}}}})",
        requested_protocol);

    auto init_res = post_mcp_request(port, init_payload, std::move(init_headers));
    if (!init_res || init_res->status != 200) return std::nullopt;

    auto session_header_it = init_res->headers.find("MCP-Session-Id");
    if (session_header_it == init_res->headers.end() || session_header_it->second.empty()) {
        return std::nullopt;
    }

    auto negotiated_version = extract_negotiated_protocol_version(init_res->body);
    if (!negotiated_version.has_value()) return std::nullopt;

    InitializedSession session{
        .session_id = session_header_it->second,
        .protocol_version = *negotiated_version,
    };

    const std::string initialized_payload =
        R"({"jsonrpc":"2.0","method":"notifications/initialized"})";
    auto initialized_res = post_mcp_request(port, initialized_payload, session_headers(session));
    if (!initialized_res || initialized_res->status != 202) return std::nullopt;

    return session;
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

[[nodiscard]] httplib::Result get_request(int port, std::string_view path) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds{1});
    client.set_read_timeout(std::chrono::seconds{10});
    client.set_write_timeout(std::chrono::seconds{10});
    return client.Get(std::string(path));
}

[[nodiscard]] httplib::Result post_json_request(int port,
                                                std::string_view path,
                                                std::string_view payload) {
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds{1});
    client.set_read_timeout(std::chrono::seconds{10});
    client.set_write_timeout(std::chrono::seconds{10});
    return client.Post(std::string(path), std::string(payload), "application/json");
}

[[nodiscard]] std::unordered_set<std::string> extract_model_ids(std::string_view response_body) {
    std::unordered_set<std::string> ids;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(response_body).get(doc) != simdjson::SUCCESS) {
        return ids;
    }

    simdjson::dom::array models;
    if (doc["data"].get(models) != simdjson::SUCCESS) {
        return ids;
    }

    for (simdjson::dom::element model_el : models) {
        std::string_view id;
        if (model_el["id"].get(id) == simdjson::SUCCESS) {
            ids.insert(std::string(id));
        }
    }
    return ids;
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
    explicit DaemonRunner(int port,
                          bool enable_api_gateway = false,
                          bool enable_mcp_http = true)
        : thread_([port, enable_api_gateway, enable_mcp_http]() {
            exec::daemon::run_server(
                port,
                "127.0.0.1",
                enable_api_gateway,
                enable_mcp_http);
        }) {}

    ~DaemonRunner() {
        exec::daemon::stop_server();
        if (thread_.joinable()) thread_.join();
    }

    DaemonRunner(const DaemonRunner&) = delete;
    DaemonRunner& operator=(const DaemonRunner&) = delete;

private:
    std::thread thread_;
};

class GatewayServerRunner {
public:
    GatewayServerRunner(int port,
                        const core::config::AppConfig& config,
                        exec::gateway::ProviderCatalog provider_catalog)
        : port_(port),
          gateway_(config,
                   core::llm::ProviderManager::get_instance(),
                   std::move(provider_catalog)) {
        gateway_.register_routes(server_);
        thread_ = std::thread([this]() {
            server_.listen("127.0.0.1", port_);
        });
    }

    ~GatewayServerRunner() {
        server_.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    int port_;
    httplib::Server server_;
    exec::gateway::ApiGateway gateway_;
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

    const auto session = initialize_mcp_session(port);
    REQUIRE(session.has_value());

    const auto out_file = temp_file_path("filo_daemon_replay_completed");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_completed >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":42,"method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto first = post_mcp_request(port, payload, session_headers(*session));
    REQUIRE(first);
    REQUIRE(first->status == 200);

    auto second = post_mcp_request(port, payload, session_headers(*session));
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

    const auto session = initialize_mcp_session(port);
    REQUIRE(session.has_value());

    const auto out_file = temp_file_path("filo_daemon_replay_inflight");
    std::filesystem::remove(out_file);

    const std::string command =
        "sleep 1; echo replay_inflight >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto f1 = std::async(std::launch::async, [&]() {
        return post_mcp_request(port, payload, session_headers(*session));
    });
    auto f2 = std::async(std::launch::async, [&]() {
        return post_mcp_request(port, payload, session_headers(*session));
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

    const auto session_one = initialize_mcp_session(port);
    const auto session_two = initialize_mcp_session(port);
    REQUIRE(session_one.has_value());
    REQUIRE(session_two.has_value());

    const auto out_file = temp_file_path("filo_daemon_replay_sessions");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_session_isolation >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto s1 = post_mcp_request(port, payload, session_headers(*session_one));
    auto s2 = post_mcp_request(port, payload, session_headers(*session_two));

    REQUIRE(s1);
    REQUIRE(s2);
    REQUIRE(s1->status == 200);
    REQUIRE(s2->status == 200);

    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 2);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon rejects requests without MCP session header after initialization",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto session = initialize_mcp_session(port);
    REQUIRE(session.has_value());

    httplib::Headers headers_without_session{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
        {"MCP-Protocol-Version", session->protocol_version},
    };

    auto res = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo missing_session"}}})",
        std::move(headers_without_session));

    REQUIRE(res);
    REQUIRE(res->status == 400);
    REQUIRE_THAT(res->body, Catch::Matchers::ContainsSubstring("MCP-Session-Id"));
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

    const auto session = initialize_mcp_session(port);
    REQUIRE(session.has_value());

    auto headers = session_headers(*session);
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

    auto blocked_headers = httplib::Headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
    };
    blocked_headers.emplace("Origin", "https://example.com");
    auto blocked = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":3,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"origin-test","version":"1.0"}}})",
        std::move(blocked_headers));
    REQUIRE(blocked);
    REQUIRE(blocked->status == 403);

    auto allowed_headers = httplib::Headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
    };
    allowed_headers.emplace("Origin", "http://localhost:3000");
    auto allowed = post_mcp_request(
        port,
        R"({"jsonrpc":"2.0","id":4,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"origin-test","version":"1.0"}}})",
        std::move(allowed_headers));
    REQUIRE(allowed);
    REQUIRE(allowed->status == 200);
    REQUIRE_THAT(allowed->body, Catch::Matchers::ContainsSubstring(R"("jsonrpc":"2.0")"));
    REQUIRE(allowed->has_header("MCP-Session-Id"));
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

    const auto session = initialize_mcp_session(port);
    REQUIRE(session.has_value());

    const auto out_file = temp_file_path("filo_daemon_delete_session");
    std::filesystem::remove(out_file);

    const std::string command =
        "echo replay_after_delete >> '" + out_file.string() + "'";
    const std::string payload =
        std::string(R"({"jsonrpc":"2.0","id":"same-id","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":")")
        + command + R"("}}})";

    auto first = post_mcp_request(port, payload, session_headers(*session));
    auto replayed = post_mcp_request(port, payload, session_headers(*session));
    REQUIRE(first);
    REQUIRE(replayed);
    REQUIRE(first->status == 200);
    REQUIRE(replayed->status == 200);
    REQUIRE(std::filesystem::exists(out_file));
    REQUIRE(count_lines(out_file) == 1);

    auto deleted = delete_mcp_request(
        port,
        httplib::Headers{{"MCP-Session-Id", session->session_id}});
    REQUIRE(deleted);
    REQUIRE(deleted->status == 204);

    auto after_delete = post_mcp_request(port, payload, session_headers(*session));
    REQUIRE(after_delete);
    REQUIRE(after_delete->status == 404);
    REQUIRE(count_lines(out_file) == 1);
    std::filesystem::remove(out_file);
}

TEST_CASE("Daemon DELETE /mcp resets run_terminal_command shell state for that session",
          "[daemon][reliability]") {
    const int port = next_test_port();
    DaemonRunner daemon(port);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    const auto session = initialize_mcp_session(port);
    REQUIRE(session.has_value());

    constexpr std::string_view token = "daemon_session_token_97";

    const std::string set_payload =
        R"({"jsonrpc":"2.0","id":"set-env","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"export FILO_DAEMON_VAR=daemon_session_token_97"}}})";
    const std::string get_payload_before =
        R"({"jsonrpc":"2.0","id":"get-env-before","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo $FILO_DAEMON_VAR"}}})";
    const std::string get_payload_after =
        R"({"jsonrpc":"2.0","id":"get-env-after","method":"tools/call","params":{"name":"run_terminal_command","arguments":{"command":"echo $FILO_DAEMON_VAR"}}})";

    auto set_res = post_mcp_request(port, set_payload, session_headers(*session));
    auto get_res_before = post_mcp_request(port, get_payload_before, session_headers(*session));
    REQUIRE(set_res);
    REQUIRE(get_res_before);
    REQUIRE(set_res->status == 200);
    REQUIRE(get_res_before->status == 200);
    REQUIRE_THAT(get_res_before->body, Catch::Matchers::ContainsSubstring(std::string(token)));

    auto deleted = delete_mcp_request(
        port,
        httplib::Headers{{"MCP-Session-Id", session->session_id}});
    REQUIRE(deleted);
    REQUIRE(deleted->status == 204);

    auto get_res_after = post_mcp_request(port, get_payload_after, session_headers(*session));
    REQUIRE(get_res_after);
    REQUIRE(get_res_after->status == 404);
}

TEST_CASE("API gateway routes are disabled unless explicitly enabled",
          "[daemon][api_gateway]") {
    const int port = next_test_port();
    DaemonRunner daemon(port, false);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto models = get_request(port, "/v1/models");
    auto chat = post_json_request(
        port,
        "/v1/chat/completions",
        R"({"model":"openai","messages":[{"role":"user","content":"hi"}]})");
    auto messages = post_json_request(
        port,
        "/v1/messages",
        R"({"model":"openai","messages":[{"role":"user","content":"hi"}]})");

    REQUIRE(models);
    REQUIRE(chat);
    REQUIRE(messages);
    REQUIRE(models->status == 404);
    REQUIRE(chat->status == 404);
    REQUIRE(messages->status == 404);
}

TEST_CASE("API gateway exposes /v1/models when enabled",
          "[daemon][api_gateway]") {
    constexpr int kStartupAttempts = 6;
    bool saw_reachable_server = false;
    std::optional<int> last_status;
    std::string last_body;
    for (int startup_attempt = 0; startup_attempt < kStartupAttempts; ++startup_attempt) {
        const int port = next_test_port();
        DaemonRunner daemon(port, true);
        if (!wait_for_ping(port, std::chrono::seconds{1})) {
            continue;
        }
        saw_reachable_server = true;

        auto models = get_request(port, "/v1/models");
        if (!models) {
            continue;
        }
        last_status = models->status;
        last_body = models->body;
        if (models->status != 200) {
            continue;
        }
        REQUIRE_THAT(models->body, Catch::Matchers::ContainsSubstring(R"("object":"list")"));
        return;
    }

    if (!saw_reachable_server) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    INFO("Last /v1/models status: "
         << (last_status.has_value() ? std::to_string(*last_status) : std::string("<none>")));
    INFO("Last /v1/models body: " << last_body);
    FAIL("API gateway was reachable but /v1/models did not become ready after retries.");
}

TEST_CASE("API-only daemon mode does not expose MCP HTTP endpoints",
          "[daemon][api_gateway]") {
    const int port = next_test_port();
    DaemonRunner daemon(port, true, false);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto models = get_request(port, "/v1/models");
    auto mcp_get = get_request(port, "/mcp");
    auto mcp_post = post_json_request(
        port,
        "/mcp",
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})");

    REQUIRE(models);
    REQUIRE(mcp_get);
    REQUIRE(mcp_post);
    REQUIRE(models->status == 200);
    REQUIRE(mcp_get->status == 404);
    REQUIRE(mcp_post->status == 404);
}

TEST_CASE("API gateway /v1/models advertises only initialized providers",
          "[daemon][api_gateway]") {
    core::config::AppConfig config;
    config.default_provider = "openai";
    config.default_model_selection = "manual";

    config.providers["openai"].model = "gpt-5.4";
    config.providers["broken-custom"].api_type = core::config::ApiType::OpenAI;
    config.providers["broken-custom"].model = "broken-model";

    exec::gateway::ProviderCatalog provider_catalog;
    provider_catalog.providers.insert({"openai", false});
    provider_catalog.provider_default_models["openai"] = "gpt-5.4";

    const int port = next_test_port();
    GatewayServerRunner gateway_server(port, config, std::move(provider_catalog));

    bool ready = false;
    for (int attempt = 0; attempt < 120; ++attempt) {
        if (auto probe = get_request(port, "/v1/models"); probe) {
            ready = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{25});
    }
    if (!ready) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto models = get_request(port, "/v1/models");
    REQUIRE(models);
    REQUIRE(models->status == 200);

    const auto model_ids = extract_model_ids(models->body);
    REQUIRE(model_ids.contains("openai"));
    REQUIRE(model_ids.contains("gpt-5.4"));
    REQUIRE(model_ids.contains("openai/gpt-5.4"));

    REQUIRE_FALSE(model_ids.contains("broken-custom"));
    REQUIRE_FALSE(model_ids.contains("broken-model"));
    REQUIRE_FALSE(model_ids.contains("broken-custom/broken-model"));
}

TEST_CASE("API gateway validates OpenAI-compatible payloads locally",
          "[daemon][api_gateway]") {
    const int port = next_test_port();
    DaemonRunner daemon(port, true);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto invalid_json = post_json_request(
        port,
        "/v1/chat/completions",
        R"(not-json)");
    REQUIRE(invalid_json);
    REQUIRE(invalid_json->status == 400);
    REQUIRE_THAT(invalid_json->body, Catch::Matchers::ContainsSubstring(R"("error")"));
    REQUIRE_THAT(invalid_json->body, Catch::Matchers::ContainsSubstring("Invalid JSON"));

    auto stream_unsupported = post_json_request(
        port,
        "/v1/chat/completions",
        R"({"model":"openai","stream":true,"messages":[{"role":"user","content":"hi"}]})");
    REQUIRE(stream_unsupported);
    REQUIRE(stream_unsupported->status == 400);
    REQUIRE_THAT(stream_unsupported->body, Catch::Matchers::ContainsSubstring("stream=true"));
}

TEST_CASE("API gateway validates Anthropic-compatible payloads locally",
          "[daemon][api_gateway]") {
    const int port = next_test_port();
    DaemonRunner daemon(port, true);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto invalid_json = post_json_request(
        port,
        "/v1/messages",
        R"(not-json)");
    REQUIRE(invalid_json);
    REQUIRE(invalid_json->status == 400);
    REQUIRE_THAT(invalid_json->body, Catch::Matchers::ContainsSubstring(R"("type":"error")"));
    REQUIRE_THAT(invalid_json->body, Catch::Matchers::ContainsSubstring("Invalid JSON"));

    auto stream_unsupported = post_json_request(
        port,
        "/v1/messages",
        R"({"model":"openai","stream":true,"messages":[{"role":"user","content":"hi"}]})");
    REQUIRE(stream_unsupported);
    REQUIRE(stream_unsupported->status == 400);
    REQUIRE_THAT(stream_unsupported->body, Catch::Matchers::ContainsSubstring("stream=true"));
}

TEST_CASE("API gateway surfaces policy routing validation errors",
          "[daemon][api_gateway]") {
    const int port = next_test_port();
    DaemonRunner daemon(port, true);
    if (!wait_for_ping(port)) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }

    auto unknown_policy = post_json_request(
        port,
        "/v1/chat/completions",
        R"({"model":"policy/does-not-exist","messages":[{"role":"user","content":"hi"}]})");

    REQUIRE(unknown_policy);
    REQUIRE(unknown_policy->status == 400);

    const bool has_policy_error =
        unknown_policy->body.find("Unknown router policy") != std::string::npos
        || unknown_policy->body.find("Router policies are unavailable") != std::string::npos;
    REQUIRE(has_policy_error);
}
