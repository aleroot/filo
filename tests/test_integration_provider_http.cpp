#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <httplib.h>
#include <simdjson.h>

#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/ClaudeOAuthFlow.hpp"
#include "core/auth/KimiOAuthFlow.hpp"
#include "core/auth/OAuthErrors.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/GrokProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/protocols/ZaiProtocol.hpp"

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

class ReauthenticationCredentialSource final
    : public core::auth::ICredentialSource {
public:
    core::auth::AuthInfo get_auth() override {
        throw core::auth::ReauthenticationRequired(
            "claude",
            "Claude rejected the saved refresh token. Sign in again to continue.");
    }

};

class RetryableServerErrorProtocol final : public OpenAIProtocol {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "retryable_test";
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<RetryableServerErrorProtocol>(*this);
    }

    [[nodiscard]] bool is_retryable(
        const HttpResponse& response) const noexcept override {
        return response.status_code == 500;
    }
};

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

TEST_CASE("Qwen Token Plan completes a streamed HTTP tool round trip",
          "[integration][http][qwen][tools]") {
    httplib::Server server;
    std::vector<httplib::Request> received;
    std::mutex received_mutex;
    server.Post("/compatible-mode/v1/chat/completions",
                [&](const httplib::Request& request, httplib::Response& response) {
        std::lock_guard lock(received_mutex);
        received.push_back(request);
        const std::string body = received.size() == 1
            ? "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Inspect the file.\"}}]}\r\n\r\n"
              "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_read\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\r\n\r\n"
              "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"README.md\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\r\n\r\n"
              "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":24,\"prompt_tokens_details\":{\"cached_tokens\":100},\"completion_tokens_details\":{\"reasoning_tokens\":8}}}\r\n\r\n"
              "data: [DONE]\r\n\r\n"
            : "data: {\"choices\":[{\"delta\":{\"content\":\"Verified.\"},\"finish_reason\":\"stop\"}]}\n\n"
              "data: [DONE]\n\n";
        // Fragment SSE envelopes and tool argument JSON across network writes.
        response.set_chunked_content_provider("text/event-stream",
            [body](std::size_t offset, httplib::DataSink& sink) {
                const auto size = std::min(std::size_t{17}, body.size() - offset);
                if (!sink.write(body.data() + offset, size)) return false;
                if (offset + size == body.size()) sink.done();
                return true;
            });
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    core::config::ProviderConfig config;
    config.api_key = "test-token-plan-key";
    config.base_url = std::format("http://127.0.0.1:{}/compatible-mode/v1", port);
    config.model = "qwen3.8-max";
    auto provider = ProviderFactory::create_provider("qwen-token-plan", config);
    REQUIRE(provider);
    CHECK_FALSE(provider->should_estimate_cost());
    ChatRequest request;
    request.model = config.model;
    request.stream = true;
    request.effort = "max";
    request.session_id = "qwen-parity-session";
    request.transport_turn_id = "qwen-parity-turn";
    request.messages = {Message{.role = "system", .content = "Inspect requested files."},
                        Message{.role = "user", .content = "Read README.md"}};
    Tool tool;
    tool.function.name = "read_file";
    tool.function.description = "Read a file";
    tool.function.input_schema = R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})";
    request.tools.push_back(tool);

    Message assistant;
    assistant.role = "assistant";
    ToolCall call;
    bool had_error = false;
    provider->stream_response(request, [&](const StreamChunk& chunk) {
        had_error |= chunk.is_error;
        assistant.reasoning_content += chunk.reasoning_content;
        if (!chunk.reasoning_protocol.empty()) assistant.reasoning_protocol = chunk.reasoning_protocol;
        for (const auto& delta : chunk.tools) {
            if (!delta.id.empty()) call.id = delta.id;
            call.function.name += delta.function.name;
            call.function.arguments += delta.function.arguments;
        }
    });
    REQUIRE_FALSE(had_error);
    REQUIRE(call.id == "call_read");
    REQUIRE(call.function.name == "read_file");
    REQUIRE(call.function.arguments == R"({"path":"README.md"})");
    REQUIRE(assistant.reasoning_content == "Inspect the file.");
    CHECK(assistant.reasoning_protocol == "dashscope");
    const auto usage = provider->get_last_usage();
    CHECK(usage.prompt_tokens == 120);
    CHECK(usage.completion_tokens == 24);
    CHECK(usage.cached_prompt_tokens == 100);
    CHECK(usage.reasoning_tokens == 8);
    assistant.tool_calls.push_back(call);
    request.messages.push_back(assistant);
    request.messages.push_back(Message{
        .role = "tool", .content = "# Filo", .tool_call_id = call.id});
    std::string answer;
    provider->stream_response(request, [&](const StreamChunk& chunk) {
        had_error |= chunk.is_error;
        answer += chunk.content;
    });
    CHECK_FALSE(had_error);
    CHECK(answer == "Verified.");

    std::lock_guard lock(received_mutex);
    REQUIRE(received.size() == 2);
    for (const auto& wire : received) {
        CHECK(wire.get_header_value("Authorization") == "Bearer test-token-plan-key");
        CHECK(wire.get_header_value("X-DashScope-CacheControl") == "enable");
        CHECK(wire.get_header_value("X-DashScope-AuthType") == "openai");
        simdjson::dom::parser parser;
        const auto body = parser.parse(wire.body);
        CHECK(std::string_view(body["reasoning_effort"]) == "xhigh");
        CHECK(bool(body["stream_options"]["include_usage"]));
        CHECK(std::string_view(body["metadata"]["sessionId"]) == request.session_id);
        CHECK(std::string_view(body["tools"].at(0)["cache_control"]["type"]) == "ephemeral");
    }
    CHECK_THAT(received.back().body, Catch::Matchers::ContainsSubstring(
        R"("reasoning_content":"Inspect the file.")"));
    CHECK_THAT(received.back().body, Catch::Matchers::ContainsSubstring(
        R"("tool_call_id":"call_read")"));
}

TEST_CASE("Qwen Token Plan stops HTTP retries when the weekly quota is exhausted",
          "[integration][http][qwen][retry]") {
    bool stream_error = false;
    SECTION("HTTP 429") {}
    SECTION("Quota error inside HTTP 200 SSE") { stream_error = true; }
    httplib::Server server;
    std::atomic<int> attempts{0};
    server.Post("/v1/chat/completions",
                [&](const httplib::Request&, httplib::Response& response) {
        ++attempts;
        const std::string body = R"({"error":{"code":"429","message":"Your token-plan 1-week quota has been exhausted. The quota will reset at 09-09 09:25:00 UTC."}})";
        response.status = stream_error ? 200 : 429;
        response.set_content(stream_error ? "data: " + body + "\n\n" : body,
                             stream_error ? "text/event-stream" : "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);
    auto provider = std::make_shared<HttpLLMProvider>(std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-token-plan-key"),
        "qwen3.8-max", std::make_unique<DashScopeTokenPlanProtocol>());
    ChatRequest request;
    request.messages.push_back(Message{.role = "user", .content = "Hello"});
    std::string error;
    provider->stream_response(request, [&](const StreamChunk& chunk) {
        if (chunk.is_error) error += chunk.content;
    });
    CHECK(attempts.load() == 1);
    CHECK_THAT(error, Catch::Matchers::ContainsSubstring("09-09 09:25:00 UTC"));
    CHECK(provider->get_last_rate_limit_info().unified_overage_status == "rejected");
    CHECK(provider->get_last_rate_limit_info().subscription_ends_at == 0);
}

TEST_CASE("HttpLLMProvider preserves streamed non-2xx JSON error bodies",
          "[integration][http][errors]") {
    httplib::Server server;
    server.Post("/v1/responses", [](const httplib::Request&, httplib::Response& res) {
        res.status = 418;
        res.set_content(
            R"({"error":"Upstream diagnostic details."})",
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

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-token"),
        "gpt-5",
        std::make_unique<OpenAIResponsesProtocol>());

    ChatRequest request;
    request.model = "gpt-5";
    request.messages.push_back(Message{.role = "user", .content = "Hello"});

    std::vector<StreamChunk> chunks;
    provider->stream_response(
        request,
        [&](const StreamChunk& chunk) { chunks.push_back(chunk); });

    const auto error = std::ranges::find_if(chunks, [](const StreamChunk& chunk) {
        return chunk.is_error;
    });
    REQUIRE(error != chunks.end());
    CHECK_THAT(error->content,
               Catch::Matchers::ContainsSubstring("Upstream diagnostic details"));
    CHECK_THAT(error->content,
               !Catch::Matchers::ContainsSubstring("[HTTP Error: 418 - ]"));
}

TEST_CASE("HttpLLMProvider honors protocol retry policy for server errors",
          "[integration][http][retry]") {
    httplib::Server server;
    std::atomic<int> attempts{0};
    server.Post("/v1/chat/completions",
                [&](const httplib::Request&, httplib::Response& res) {
        if (++attempts == 1) {
            res.status = 500;
            res.set_content(R"({"error":{"message":"temporary"}})",
                            "application/json");
            return;
        }
        res.set_content(
            "data: {\"choices\":[{\"delta\":{\"content\":\"recovered\"}}]}\n\n"
            "data: [DONE]\n\n",
            "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-token"),
        "test-model",
        std::make_unique<RetryableServerErrorProtocol>());

    ChatRequest request;
    request.model = "test-model";
    request.messages.push_back(Message{.role = "user", .content = "Hello"});
    std::vector<StreamChunk> chunks;
    provider->stream_response(
        request, [&](const StreamChunk& chunk) { chunks.push_back(chunk); });

    CHECK(attempts.load() == 2);
    CHECK(std::ranges::any_of(chunks, [](const StreamChunk& chunk) {
        return chunk.content == "recovered";
    }));
}

TEST_CASE("HttpLLMProvider retries a Grok Responses generation failure before output",
          "[integration][http][retry][grok][responses]") {
    httplib::Server server;
    std::atomic<int> attempts{0};
    server.Post("/v1/responses",
                [&](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream");
        if (++attempts == 1) {
            res.set_content(
                "event: response.failed\n"
                "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"server_error\",\"message\":\"Internal error during token generation\"}}}\n\n",
                "text/event-stream");
            return;
        }
        res.set_content(
            "event: response.output_text.delta\n"
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"recovered\"}\n\n"
            "event: response.completed\n"
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_recovered\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
            "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-token"),
        "grok-4.6",
        std::make_unique<GrokResponsesProtocol>());

    ChatRequest request;
    request.model = "grok-4.6";
    request.messages.push_back(Message{.role = "user", .content = "Hello"});

    std::vector<StreamChunk> chunks;
    provider->stream_response(
        request, [&](const StreamChunk& chunk) { chunks.push_back(chunk); });

    CHECK(attempts.load() == 2);
    CHECK(std::ranges::any_of(chunks, [](const StreamChunk& chunk) {
        return chunk.content == "recovered";
    }));
    CHECK(std::ranges::none_of(chunks, [](const StreamChunk& chunk) {
        return chunk.is_error;
    }));
}

TEST_CASE("HttpLLMProvider aborts a request whose response never starts",
          "[integration][http][timeout]") {
    httplib::Server server;
    server.Post("/v1/chat/completions",
                [](const httplib::Request&, httplib::Response& res) {
        std::this_thread::sleep_for(std::chrono::milliseconds(750));
        res.set_content("data: [DONE]\n\n", "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-token"),
        "test-model",
        std::make_unique<OpenAIProtocol>(),
        core::config::ApiType::Unknown,
        std::string{},
        nullptr,
        nullptr,
        std::string{},
        HttpStreamTransportOptions{
            .timeouts = {
                .response_start = std::chrono::milliseconds(100),
                .inactivity = std::chrono::seconds(1),
            },
            .retries = {.max_retries = 0},
        });

    ChatRequest request;
    request.model = "test-model";
    request.messages.push_back(Message{.role = "user", .content = "Hello"});
    std::vector<StreamChunk> chunks;
    provider->stream_response(
        request, [&](const StreamChunk& chunk) { chunks.push_back(chunk); });

    const auto error = std::ranges::find_if(chunks, [](const StreamChunk& chunk) {
        return chunk.is_error;
    });
    REQUIRE(error != chunks.end());
    CHECK_THAT(error->content,
               Catch::Matchers::ContainsSubstring("request timeout"));
}

TEST_CASE("HttpLLMProvider aborts an inactive response stream",
          "[integration][http][timeout]") {
    httplib::Server server;
    server.Post("/v1/chat/completions",
                [](const httplib::Request&, httplib::Response& res) {
        res.set_chunked_content_provider(
            "text/event-stream",
            [phase = 0](std::size_t, httplib::DataSink& sink) mutable {
                if (phase++ == 0) {
                    return sink.write(": connected\n\n", 13);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(750));
                sink.write("data: [DONE]\n\n", 14);
                sink.done();
                return false;
            });
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-token"),
        "test-model",
        std::make_unique<OpenAIProtocol>(),
        core::config::ApiType::Unknown,
        std::string{},
        nullptr,
        nullptr,
        std::string{},
        HttpStreamTransportOptions{
            .timeouts = {
                .response_start = std::chrono::seconds(1),
                .inactivity = std::chrono::milliseconds(100),
            },
            .retries = {.max_retries = 0},
        });

    ChatRequest request;
    request.model = "test-model";
    request.messages.push_back(Message{.role = "user", .content = "Hello"});
    std::vector<StreamChunk> chunks;
    provider->stream_response(
        request, [&](const StreamChunk& chunk) { chunks.push_back(chunk); });

    const auto error = std::ranges::find_if(chunks, [](const StreamChunk& chunk) {
        return chunk.is_error;
    });
    REQUIRE(error != chunks.end());
    CHECK_THAT(error->content,
               Catch::Matchers::ContainsSubstring("idle timeout"));
}

TEST_CASE("HttpLLMProvider reports an actionable, safely retryable OAuth recovery",
          "[integration][authentication][reauthentication]") {
    auto provider = std::make_shared<HttpLLMProvider>(
        "https://example.invalid",
        std::make_shared<ReauthenticationCredentialSource>(),
        "claude-sonnet-4-6",
        std::make_unique<AnthropicProtocol>());

    std::vector<StreamChunk> chunks;
    provider->stream_response(
        make_claude_request(),
        [&](const StreamChunk& chunk) { chunks.push_back(chunk); });

    REQUIRE(chunks.size() == 1);
    const auto& failure = chunks.front();
    CHECK(failure.is_error);
    CHECK(failure.is_final);
    REQUIRE(failure.authentication_recovery.has_value());
    CHECK(failure.authentication_recovery->provider_id == "claude");
    CHECK(failure.authentication_recovery->retry_safe);
    CHECK_THAT(failure.content,
               Catch::Matchers::ContainsSubstring("Authentication required"));
    CHECK_THAT(failure.content,
               !Catch::Matchers::ContainsSubstring("invalid_grant"));
}

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

TEST_CASE("Kimi protocol enriches a header-only 5h window with weekly usage",
          "[integration][kimi][rate_limit][enrich]") {
    httplib::Server server;
    std::atomic<int> requests{0};
    std::string authorization;
    server.Get("/coding/v1/usages",
               [&](const httplib::Request& req, httplib::Response& res) {
                   ++requests;
                   authorization = req.get_header_value("Authorization");
                   res.set_content(
                       R"({"usage":{"limit":"100","used":"31","remaining":"69","resetTime":"2026-07-31T13:04:40Z"},"limits":[{"window":{"duration":300,"timeUnit":"TIME_UNIT_MINUTE"},"detail":{"limit":"100","used":"76","remaining":"24","resetTime":"2026-07-26T07:04:40Z"}}]})",
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

    KimiProtocol protocol;
    protocol.on_response(HttpResponse{
        200,
        "{}",
        cpr::Header{
            {"x-ratelimit-unified-5h-utilization", "0.76"},
            {"x-ratelimit-limit-tokens", "100"},
            {"x-ratelimit-remaining-tokens", "24"},
        }});
    protocol.enrich_rate_limit(
        std::format("http://127.0.0.1:{}/coding/v1", port),
        cpr::Header{{"Authorization", "Bearer test-token"}},
        HttpResponse{200, "{}", {}});

    const auto info = protocol.last_rate_limit();
    REQUIRE(requests.load() == 1);
    REQUIRE(authorization == "Bearer test-token");
    REQUIRE(info.usage_windows.size() == 2);
    CHECK(info.usage_windows[0].label == "5h");
    CHECK(info.usage_windows[0].utilization
          == Catch::Approx(0.76f).epsilon(0.001f));
    CHECK(info.usage_windows[1].label == "7d");
    CHECK(info.usage_windows[1].utilization
          == Catch::Approx(0.31f).epsilon(0.001f));
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
                       R"({"code":200,"msg":"Operation successful","data":{"limits":[{"type":"TIME_LIMIT","unit":5,"number":1,"usage":100,"currentValue":0,"remaining":100,"percentage":0,"nextResetTime":1785332618972,"usageDetails":[{"modelCode":"search-prime","usage":0},{"modelCode":"web-reader","usage":0},{"modelCode":"zread","usage":0}]},{"type":"CREDIT_LIMIT","unit":3,"number":5,"percentage":11,"nextResetTime":1782759322207},{"type":"TOKENS_LIMIT","unit":6,"number":1,"percentage":2,"nextResetTime":1783345418971}],"level":"lite"},"success":true})",
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

TEST_CASE("Fable 5.1 completes an HTTP tool round trip after a context change",
          "[integration][http][claude][fable51]") {
    httplib::Server server;
    std::vector<httplib::Request> received;
    std::mutex mutex;
    server.Post("/v1/messages", [&](const httplib::Request& request, httplib::Response& response) {
        std::lock_guard lock(mutex);
        received.push_back(request);
        const std::string body = received.size() == 1
            ? "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n\n"
              "event: content_block_start\ndata: {\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n"
              "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"signed-first-turn\"}}\n\n"
              "event: content_block_stop\ndata: {\"index\":0}\n\n"
              "event: content_block_start\ndata: {\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"read-1\",\"name\":\"read_file\",\"input\":{}}}\n\n"
              "event: content_block_delta\ndata: {\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}\n\n"
              "event: content_block_stop\ndata: {\"index\":1}\n\n"
              "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":20}}\n\n"
              "event: message_stop\ndata: {}\n\n"
            : "event: message_start\ndata: {\"type\":\"message_start\",\"input_transformations\":[{\"type\":\"thinking_dropped\",\"reason\":\"prefix_binding_mismatch\",\"path\":\"messages.1.content.0\"}],\"message\":{\"usage\":{\"input_tokens\":30}}}\n\n"
              "event: content_block_delta\ndata: {\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Verified.\"}}\n\n"
              "event: message_delta\ndata: {\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n"
              "event: message_stop\ndata: {}\n\n";
        response.set_chunked_content_provider("text/event-stream",
            [body](std::size_t offset, httplib::DataSink& sink) {
                const auto size = std::min(std::size_t{13}, body.size() - offset);
                if (!sink.write(body.data() + offset, size)) return false;
                if (offset + size == body.size()) sink.done();
                return true;
            });
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);
    auto provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}", port),
        core::auth::ApiKeyCredentialSource::as_custom_header("test-key", "x-api-key"),
        "fable", std::make_unique<AnthropicProtocol>());

    auto request = make_claude_request("fable");
    request.messages.insert(request.messages.begin(), Message{.role = "system", .content = "Read the file."});
    core::tools::ToolDefinition definition;
    definition.name = "read_file";
    definition.description = "Read a file";
    request.tools.push_back(Tool{.type = "function", .function = std::move(definition)});
    Message assistant{.role = "assistant"};
    std::vector<StreamChunk> first;
    provider->stream_response(request, [&](const StreamChunk& chunk) { first.push_back(chunk); });
    REQUIRE_FALSE(first.empty());
    for (const auto& chunk : first) {
        REQUIRE_FALSE(chunk.is_error);
        assistant.content += chunk.content;
        assistant.tool_calls.insert(assistant.tool_calls.end(), chunk.tools.begin(), chunk.tools.end());
        assistant.continuation_items.insert(assistant.continuation_items.end(),
            chunk.continuation_items.begin(), chunk.continuation_items.end());
    }
    REQUIRE(first.back().is_final);
    REQUIRE(assistant.tool_calls.size() == 1);
    REQUIRE(assistant.continuation_items.size() == 1);
    request.messages.push_back(std::move(assistant));
    request.messages.push_back(Message{.role = "tool", .content = "File contents", .tool_call_id = "read-1"});
    request.messages[0].content = "Read the file. The user updated the goal.";
    std::vector<StreamChunk> second;
    provider->stream_response(request, [&](const StreamChunk& chunk) { second.push_back(chunk); });
    REQUIRE_FALSE(second.empty());
    REQUIRE(second.back().is_final);
    std::string text;
    for (const auto& chunk : second) {
        REQUIRE_FALSE(chunk.is_error);
        text += chunk.content;
    }
    CHECK(text == "Verified.");
    std::lock_guard lock(mutex);
    REQUIRE(received.size() == 2);
    for (const auto& actual : received) {
        simdjson::dom::parser parser;
        const auto doc = parser.parse(actual.body);
        CHECK(doc["model"].get_string().value() == "claude-fable-5-1");
        CHECK(doc["thinking"]["block_binding"]["prefix_mismatch_behavior"].get_string().value() == "drop_block");
        CHECK_THAT(actual.get_header_value("anthropic-beta"), Catch::Matchers::ContainsSubstring("thinking-binding-controls-2026-08-01"));
        CHECK_THAT(actual.get_header_value("x-anthropic-billing-header"), Catch::Matchers::StartsWith("cc_version=2.1.255"));
    }
    simdjson::dom::parser parser;
    const auto doc = parser.parse(received.back().body);
    CHECK(doc["messages"].at(1)["content"].at(0)["signature"].get_string().value() == "signed-first-turn");
    CHECK(doc["messages"].at(2)["content"].at(0)["tool_use_id"].get_string().value() == "read-1");
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

TEST_CASE("ClaudeOAuthFlow classifies an expired refresh token as reauthentication",
          "[integration][ClaudeOAuthFlow][reauthentication]") {
    httplib::Server server;
    server.Post("/oauth/token", [&](const httplib::Request&, httplib::Response& res) {
        res.status = 400;
        res.set_content(
            R"({"error": "invalid_grant", "error_description": "Refresh token expired"})",
            "application/json");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Loopback port binding unavailable in this test environment");
    }
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    wait_until_running(server);

    const std::string base = "http://127.0.0.1:" + std::to_string(port);
    core::auth::ClaudeOAuthFlow flow(
        "test-client-id",
        base + "/oauth/authorize",
        base + "/oauth/token",
        {"user:inference"},
        42100,
        42101,
        nullptr);

    REQUIRE_THROWS_AS(
        flow.refresh("expired-refresh-token"),
        core::auth::OAuthRefreshRejected);
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
