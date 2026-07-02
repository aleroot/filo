#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <httplib.h>

#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/ClaudeOAuthFlow.hpp"
#include "core/auth/KimiOAuthFlow.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

class ScopedServerStop {
public:
    explicit ScopedServerStop(httplib::Server& server)
        : server_(server) {}

    ~ScopedServerStop() {
        server_.stop();
    }

private:
    httplib::Server& server_;
};

void wait_until_running(httplib::Server& server) {
    for (int i = 0; i < 50 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(server.is_running());
}

std::filesystem::path make_temp_dir(const std::string& label) {
    const auto path = std::filesystem::temp_directory_path()
        / (label + "_" + core::auth::KimiOAuthFlow::generateDeviceId());
    std::filesystem::create_directories(path);
    return path;
}

ChatRequest make_claude_request(std::string model = "claude-sonnet-4-6",
                                std::string user_text = "Hello") {
    ChatRequest req;
    req.model = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

int bind_to_first_available_port(httplib::Server& server, int begin, int end) {
    for (int candidate = begin; candidate < end; ++candidate) {
        if (server.bind_to_port("127.0.0.1", candidate)) {
            return candidate;
        }
    }
    return -1;
}

} // namespace

TEST_CASE("KimiProtocol uploads local videos as files and serializes ms references",
          "[integration][kimi][video][upload]") {
    const auto dir = make_temp_dir("filo_kimi_video_upload");
    const auto video = dir / "recording.mp4";
    {
        std::ofstream out(video, std::ios::binary | std::ios::trunc);
        out << "fake-video";
    }

    httplib::Server server;
    std::atomic<int> uploads{0};
    std::string authorization;
    std::string content_type;
    server.Post("/v1/files", [&](const httplib::Request& req, httplib::Response& res) {
        ++uploads;
        authorization = req.get_header_value("Authorization");
        content_type = req.get_header_value("Content-Type");
        res.set_content(R"({"id":"file_kimi_video_123"})", "application/json");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    core::auth::AuthInfo auth;
    auth.headers["Authorization"] = "Bearer test-token";

    ChatRequest req;
    req.model = "kimi-k2.6";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_video_attachment(video.string()),
        .content_parts = {
            ContentPart::make_text("Summarize this recording"),
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });

    KimiProtocol protocol;
    const std::string base_url = std::format("http://127.0.0.1:{}/v1", port);
    REQUIRE_NOTHROW(protocol.prepare_media_uploads(req, base_url, auth));

    REQUIRE(uploads.load() == 1);
    CHECK(authorization == "Bearer test-token");
    CHECK_THAT(content_type, Catch::Matchers::ContainsSubstring("multipart/form-data"));
    REQUIRE(req.messages[0].content_parts[1].url == "ms://file_kimi_video_123");
    REQUIRE(req.messages[0].content_parts[1].media_id == "file_kimi_video_123");

    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("video_url":{"url":"ms://file_kimi_video_123","id":"file_kimi_video_123"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("data:video/"));

    ChatRequest cached_req;
    cached_req.model = "kimi-k2.6";
    cached_req.messages.push_back(Message{
        .role = "user",
        .content_parts = {
            ContentPart::make_video(video.string(), "video/mp4"),
        },
    });
    REQUIRE_NOTHROW(protocol.prepare_media_uploads(cached_req, base_url, auth));
    CHECK(uploads.load() == 1);
    CHECK(cached_req.messages[0].content_parts[0].url == "ms://file_kimi_video_123");
}

TEST_CASE("Z.ai protocol enriches usage windows from dashboard quota endpoint",
          "[integration][zai][rate_limit][enrich]") {
    httplib::Server server;
    std::atomic<int> requests{0};
    std::string authorization;
    server.Get("/api/monitor/usage/quota/limit",
               [&](const httplib::Request& req, httplib::Response& res) {
                   ++requests;
                   authorization = req.get_header_value("Authorization");
                   res.set_content(
                       R"({"code":200,"msg":"Operation successful","data":{"limits":[{"type":"TIME_LIMIT","unit":5,"number":1,"usage":100,"currentValue":0,"remaining":100,"percentage":0,"nextResetTime":1785332618972,"usageDetails":[{"modelCode":"search-prime","usage":0},{"modelCode":"web-reader","usage":0},{"modelCode":"zread","usage":0}]},{"type":"TOKENS_LIMIT","unit":3,"number":5,"percentage":11,"nextResetTime":1782759322207},{"type":"TOKENS_LIMIT","unit":6,"number":1,"percentage":2,"nextResetTime":1783345418971}],"level":"lite"},"success":true})",
                       "application/json");
               });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    ZaiProtocol protocol;
    protocol.on_response(HttpResponse{200, "{}", {}});
    protocol.enrich_rate_limit(
        std::format("http://127.0.0.1:{}/api/coding/paas/v4", port),
        cpr::Header{{"Authorization", "Bearer test-token"}},
        HttpResponse{200, "{}", {}});

    const auto info = protocol.last_rate_limit();
    REQUIRE(requests.load() == 1);
    REQUIRE(authorization == "Bearer test-token");
    REQUIRE(info.usage_windows.size() == 3);
    REQUIRE(info.usage_windows[0].label == "5h");
    REQUIRE(info.usage_windows[0].utilization == 0.11f);
    REQUIRE(info.usage_windows[1].label == "7d");
    REQUIRE(info.usage_windows[1].utilization == 0.02f);
    REQUIRE(info.usage_windows[2].label == "web");
    REQUIRE(info.usage_windows[2].utilization == 0.0f);
}

TEST_CASE("HttpLLMProvider - Codex Responses transport headers are scoped to Codex backend",
          "[integration][openai][responses][codex][transport]") {
    httplib::Server server;

    std::atomic<int> codex_requests{0};
    std::string first_installation_id;
    std::string first_client_request_id;
    std::string first_window_id;
    std::string first_timing_header;
    std::string first_turn_state;
    std::string second_turn_state;
    std::string regular_installation_id;

    auto completed_sse = [](std::string_view id) {
        return std::format(
            "event: response.completed\n"
            "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"{}\",\"usage\":{{\"input_tokens\":1,\"output_tokens\":1}}}}}}\n\n",
            id);
    };

    server.Post("/backend-api/codex/responses", [&](const httplib::Request& req,
                                                    httplib::Response& res) {
        const int n = ++codex_requests;
        if (n == 1) {
            first_installation_id = req.get_header_value("x-codex-installation-id");
            first_client_request_id = req.get_header_value("x-client-request-id");
            first_window_id = req.get_header_value("x-codex-window-id");
            first_timing_header = req.get_header_value("x-responsesapi-include-timing-metrics");
            first_turn_state = req.get_header_value("x-codex-turn-state");
            res.set_header("x-codex-turn-state", "turn-state-one");
            res.set_content(completed_sse("resp_codex_1"), "text/event-stream");
            return;
        }

        second_turn_state = req.get_header_value("x-codex-turn-state");
        res.set_header("x-codex-turn-state", "turn-state-two");
        res.set_content(completed_sse("resp_codex_2"), "text/event-stream");
    });

    server.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
        regular_installation_id = req.get_header_value("x-codex-installation-id");
        res.set_content(completed_sse("resp_regular_1"), "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto codex_provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/backend-api/codex", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        "gpt-5",
        std::make_unique<CodexResponsesProtocol>(),
        core::config::ApiType::OpenAI,
        "openai");

    ChatRequest req;
    req.model = "gpt-5";
    req.session_id = "session-transport";
    req.transport_turn_id = "session-transport:1";
    req.messages.push_back(Message{.role = "user", .content = "first"});

    auto stream_once = [](const std::shared_ptr<HttpLLMProvider>& provider,
                          const ChatRequest& request) {
        auto done = std::make_shared<std::promise<void>>();
        auto completed = std::make_shared<std::atomic<bool>>(false);
        auto future = done->get_future();

        provider->stream_response(request, [done, completed](const StreamChunk& chunk) {
            if (chunk.is_final && !completed->exchange(true)) {
                done->set_value();
            }
        });

        REQUIRE(future.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    };

    stream_once(codex_provider, req);

    ChatRequest req2 = req;
    req2.messages[0].content = "second";
    stream_once(codex_provider, req2);

    CHECK_FALSE(first_installation_id.empty());
    CHECK(first_client_request_id == "session-transport");
    CHECK(first_window_id == "session-transport:0");
    CHECK(first_timing_header == "true");
    CHECK(first_turn_state.empty());
    CHECK(second_turn_state == "turn-state-one");

    auto regular_provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        "gpt-5",
        std::make_unique<OpenAIResponsesProtocol>(),
        core::config::ApiType::OpenAI,
        "openai-compatible");

    ChatRequest regular_req = req;
    regular_req.transport_turn_id = "session-transport:regular";
    stream_once(regular_provider, regular_req);
    CHECK(regular_installation_id.empty());
}

TEST_CASE("HttpLLMProvider - retries Anthropic stream error before output",
          "[integration][claude][sse][error][http]") {
    httplib::Server server;
    std::atomic<int> requests{0};
    server.Post("/v1/messages", [&](const httplib::Request&, httplib::Response& res) {
        const int attempt = ++requests;
        res.set_header("Content-Type", "text/event-stream");
        if (attempt == 1) {
            res.set_content(
                "event: error\n"
                "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
                "text/event-stream");
            return;
        }

        res.set_content(
            "event: message_start\n"
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n"
            "event: content_block_start\n"
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
            "event: content_block_delta\n"
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Recovered\"}}\n\n"
            "event: content_block_stop\n"
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
            "event: message_delta\n"
            "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n\n"
            "event: message_stop\n"
            "data: {\"type\":\"message_stop\"}\n\n",
            "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}", port),
        core::auth::ApiKeyCredentialSource::as_custom_header("test-key", "x-api-key"),
        "claude-sonnet-4-6",
        std::make_unique<AnthropicProtocol>());

    std::vector<StreamChunk> chunks;
    provider->stream_response(make_claude_request(), [&](const StreamChunk& chunk) {
        chunks.push_back(chunk);
    });

    REQUIRE(requests.load() == 2);
    REQUIRE_FALSE(chunks.empty());
    REQUIRE(chunks.back().is_final);
    REQUIRE_FALSE(chunks.back().is_error);

    std::string text;
    for (const auto& chunk : chunks) {
        text += chunk.content;
        REQUIRE_FALSE(chunk.is_error);
    }
    REQUIRE(text == "Recovered");
}

TEST_CASE("HttpLLMProvider - does not retry Anthropic stream error after output",
          "[integration][claude][sse][error][http]") {
    httplib::Server server;
    std::atomic<int> requests{0};
    server.Post("/v1/messages", [&](const httplib::Request&, httplib::Response& res) {
        ++requests;
        res.set_header("Content-Type", "text/event-stream");
        res.set_content(
            "event: message_start\n"
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3}}}\n\n"
            "event: content_block_start\n"
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
            "event: content_block_delta\n"
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Partial\"}}\n\n"
            "event: error\n"
            "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n",
            "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}", port),
        core::auth::ApiKeyCredentialSource::as_custom_header("test-key", "x-api-key"),
        "claude-sonnet-4-6",
        std::make_unique<AnthropicProtocol>());

    std::vector<StreamChunk> chunks;
    provider->stream_response(make_claude_request(), [&](const StreamChunk& chunk) {
        chunks.push_back(chunk);
    });

    REQUIRE(requests.load() == 1);
    REQUIRE(chunks.size() >= 2);
    CHECK(chunks.front().content == "Partial");
    REQUIRE(chunks.back().is_final);
    REQUIRE(chunks.back().is_error);
    CHECK_THAT(chunks.back().content,
               Catch::Matchers::ContainsSubstring("anthropic stream error: overloaded_error"));
}

TEST_CASE("ClaudeOAuthFlow::refresh retries without scope after invalid_scope",
          "[integration][ClaudeOAuthFlow]") {
    httplib::Server server;
    std::atomic<int> call_count{0};
    std::vector<std::string> request_bodies;
    std::mutex request_mutex;

    server.Post("/oauth/token", [&](const httplib::Request& req, httplib::Response& res) {
        const int call = ++call_count;
        {
            std::lock_guard lock(request_mutex);
            request_bodies.push_back(req.body);
        }

        if (call == 1) {
            res.status = 400;
            res.set_content(
                R"({"error":"invalid_scope","error_description":"The requested scope is invalid"})",
                "application/json");
            return;
        }

        res.status = 200;
        res.set_content(
            R"({"access_token":"new-access-token","token_type":"Bearer","expires_in":3600})",
            "application/json");
    });

    const int port = bind_to_first_available_port(server, 42000, 42100);
    if (port <= 0) {
        SKIP("Loopback port binding unavailable in this test environment");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    core::auth::ClaudeOAuthFlow flow(
        "test-client-id",
        base + "/oauth/authorize",
        base + "/oauth/token",
        {"unknown:scope", "user:inference"},
        42000,
        42001,
        nullptr);

    const auto token = flow.refresh("existing-refresh-token");

    REQUIRE(token.access_token == "new-access-token");
    REQUIRE(token.refresh_token == "existing-refresh-token");
    REQUIRE(call_count.load() == 2);
    REQUIRE(request_bodies.size() == 2);
    REQUIRE(request_bodies[0].find("\"scope\"") != std::string::npos);
    REQUIRE(request_bodies[1].find("\"scope\"") == std::string::npos);
}

TEST_CASE("ClaudeOAuthFlow::refresh parses top-level account/org identifiers",
          "[integration][ClaudeOAuthFlow]") {
    httplib::Server server;
    server.Post("/oauth/token", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({
                "access_token":"new-access-token",
                "token_type":"Bearer",
                "expires_in":3600,
                "scope":"user:profile user:inference",
                "account_id":"acct-top-level",
                "organization_uuid":"org-top-level"
            })",
            "application/json");
    });

    const int port = bind_to_first_available_port(server, 42100, 42200);
    if (port <= 0) {
        SKIP("Loopback port binding unavailable in this test environment");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    core::auth::ClaudeOAuthFlow flow(
        "test-client-id",
        base + "/oauth/authorize",
        base + "/oauth/token",
        {"user:profile", "user:inference"},
        42100,
        42101,
        nullptr);

    const auto token = flow.refresh("existing-refresh-token");

    REQUIRE(token.access_token == "new-access-token");
    REQUIRE(token.refresh_token == "existing-refresh-token");
    REQUIRE(token.account_id == "acct-top-level");
    REQUIRE(token.organization_id == "org-top-level");
    REQUIRE(token.scopes == std::vector<std::string>{"user:profile", "user:inference"});
}

TEST_CASE("ClaudeOAuthFlow::refresh parses nested account/organization uuid fields",
          "[integration][ClaudeOAuthFlow]") {
    httplib::Server server;
    server.Post("/oauth/token", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({
                "access_token":"new-access-token",
                "token_type":"Bearer",
                "expires_in":3600,
                "account":{"uuid":"acct-nested-uuid"},
                "organization":{"uuid":"org-nested-uuid"}
            })",
            "application/json");
    });

    const int port = bind_to_first_available_port(server, 42200, 42300);
    if (port <= 0) {
        SKIP("Loopback port binding unavailable in this test environment");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    core::auth::ClaudeOAuthFlow flow(
        "test-client-id",
        base + "/oauth/authorize",
        base + "/oauth/token",
        {"user:inference"},
        42200,
        42201,
        nullptr);

    const auto token = flow.refresh("existing-refresh-token");

    REQUIRE(token.account_id == "acct-nested-uuid");
    REQUIRE(token.organization_id == "org-nested-uuid");
}

TEST_CASE("ClaudeOAuthFlow::refresh parses nested account/organization id fallback",
          "[integration][ClaudeOAuthFlow]") {
    httplib::Server server;
    server.Post("/oauth/token", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 200;
        res.set_content(
            R"({
                "access_token":"new-access-token",
                "token_type":"Bearer",
                "expires_in":3600,
                "account":{"id":"acct-nested-id"},
                "organization":{"id":"org-nested-id"}
            })",
            "application/json");
    });

    const int port = bind_to_first_available_port(server, 42300, 42400);
    if (port <= 0) {
        SKIP("Loopback port binding unavailable in this test environment");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    core::auth::ClaudeOAuthFlow flow(
        "test-client-id",
        base + "/oauth/authorize",
        base + "/oauth/token",
        {"user:inference"},
        42300,
        42301,
        nullptr);

    const auto token = flow.refresh("existing-refresh-token");

    REQUIRE(token.account_id == "acct-nested-id");
    REQUIRE(token.organization_id == "org-nested-id");
}
