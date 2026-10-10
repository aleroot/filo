#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include <simdjson.h>

#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/providers/QwenProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/tools/WebBackendAdapters.hpp"
#include "TestSessionContext.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace {
class QwenServer {
public:
    httplib::Server server;
    int port = 0;
    QwenServer() {
        server.Get("/compatible-mode/v1/models", [](const auto&, auto& response) {
            response.set_content(R"({"data":[]})", "application/json");
        });
        port = server.bind_to_any_port("127.0.0.1");
        if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
        thread_ = std::jthread([&] { server.listen_after_bind(); });
        for (int i = 0; i < 100 && !server.is_running(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        REQUIRE(server.is_running());
    }
    ~QwenServer() { server.stop(); }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port) + "/compatible-mode/v1"; }
private:
    std::jthread thread_;
};

std::shared_ptr<core::llm::providers::QwenProvider> make_provider(const QwenServer& server) {
    return std::make_shared<core::llm::providers::QwenProvider>(core::llm::providers::QwenProvider::Options{
        .base_url = server.url(),
        .credentials = core::auth::ApiKeyCredentialSource::as_bearer("test-qwen-key"),
        .model = "qwen3.7-plus",
        .provider_name = "qwen-audit-fixture",
    });
}

core::tools::ToolInvocationContext invocation(const std::shared_ptr<core::llm::LLMProvider>& provider) {
    return {.session_context = test_support::make_workspace_session_context(core::context::SessionTransport::cli),
            .model_name = "qwen3-coder-plus", .provider = provider};
}
} // namespace

TEST_CASE("Qwen HTTP preserves truncation and usage but rejects a premature EOF",
          "[integration][qwen][regression][http]") {
    QwenServer fixture;
    std::atomic<int> count{0};
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto&, auto& response) {
        if (++count == 1) {
            response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"Partial\"},\"finish_reason\":\"length\"}]}\n\n"
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":15000,\"completion_tokens\":456}}\n\n"
                "data: [DONE]\n\n", "text/event-stream");
        } else {
            response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"Interrupted\"}}]}\n\n", "text/event-stream");
        }
    });
    auto provider = make_provider(fixture);
    core::llm::ChatRequest request;
    request.model = "qwen3.7-plus";
    request.messages = {{.role = "user", .content = "Continue"}};
    bool truncated = false;
    provider->stream_response(request, [&](const auto& chunk) {
        if (chunk.is_final) {
            truncated = chunk.stop_reason == "max_tokens";
            CHECK(provider->get_last_usage().prompt_tokens == 15000);
        }
    });
    CHECK(truncated);
    CHECK(provider->estimate_context_tokens(request.messages) >= 15000);
    provider->reset_conversation_state();
    CHECK(provider->estimate_context_tokens(request.messages) < 15000);
    bool failed = false;
    provider->stream_response(request, [&](const auto& chunk) { failed = failed || chunk.is_error; });
    CHECK(failed);
    CHECK(count == 2);
}

TEST_CASE("Coding Plan quota SSE is an error and is never shown as answer text",
          "[integration][qwen][regression][http][quota]") {
    QwenServer fixture;
    std::atomic<int> count{0};
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto&, auto& response) {
        ++count;
        response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"Allocated quota exceeded\"},\"finish_reason\":\"error_finish\"}]}\n\n", "text/event-stream");
    });
    auto provider = make_provider(fixture);
    core::llm::ChatRequest request;
    request.model = "qwen3.7-plus";
    bool failed = false;
    std::string text;
    provider->stream_response(request, [&](const auto& chunk) {
        failed = failed || chunk.is_error;
        if (!chunk.is_error) text += chunk.content;
    });
    CHECK(failed);
    CHECK(text.empty());
    CHECK(count == 1);
    CHECK(provider->get_last_rate_limit_info().unified_overage_status == "rejected");
}

TEST_CASE("Qwen releases tools only after a complete valid response",
          "[integration][qwen][regression][http][tools]") {
    const bool responses = GENERATE(false, true);
    // Interrupted, malformed, and successful tool responses use the same transport.
    const int outcome = GENERATE(0, 1, 2);
    CAPTURE(responses, outcome);
    QwenServer fixture;
    fixture.server.Post(responses ? "/compatible-mode/v1/responses" : "/compatible-mode/v1/chat/completions",
        [&](const auto&, auto& response) {
            std::string body;
            if (responses) {
                body = "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"read\",\"arguments\":\""
                    + std::string(outcome == 1 ? "{" : "{}") + "\"}}\n\n";
                if (outcome != 0) body += "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\"}}\n\n";
            } else {
                body = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read\",\"arguments\":\""
                    + std::string(outcome == 1 ? "{" : "{}") + "\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n";
                if (outcome != 0) body += "data: [DONE]\n\n";
            }
            response.set_content(body, "text/event-stream");
        });
    auto provider = std::make_shared<core::llm::providers::QwenProvider>(core::llm::providers::QwenProvider::Options{
        .base_url = fixture.url(),
        .credentials = core::auth::ApiKeyCredentialSource::as_bearer("test-qwen-key"),
        .model = "qwen3-coder-plus",
        .provider_name = "qwen-tools-fixture",
        .responses = responses,
    });
    core::llm::ChatRequest request;
    request.model = "qwen3-coder-plus";
    std::vector<core::llm::ToolCall> calls;
    bool failed = false, incomplete = false;
    provider->stream_response(request, [&](const auto& chunk) {
        failed = failed || chunk.is_error;
        incomplete = incomplete || chunk.incomplete_tool_call;
        calls.insert(calls.end(), chunk.tools.begin(), chunk.tools.end());
    });
    CHECK(failed == (outcome == 0));
    CHECK(incomplete == (outcome == 1));
    if (outcome == 2) {
        REQUIRE(calls.size() == 1);
        CHECK(calls.front().function.name == "read");
        CHECK(calls.front().function.arguments == "{}");
    } else {
        CHECK(calls.empty());
    }
}

TEST_CASE("Qwen learns a required-thinking rejection and retries exactly once",
          "[integration][qwen][regression][http][thinking]") {
    QwenServer fixture;
    std::atomic<int> count{0};
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto& request, auto& response) {
        ++count;
        if (request.body.find("\"enable_thinking\":false") != std::string::npos) {
            response.status = 400;
            response.set_content(R"({"error":{"message":"enable_thinking must\u0020be true"}})", "application/json");
        } else {
            CHECK(request.body.find(count == 4 ? "\"reasoning_effort\":\"xhigh\""
                                               : "\"reasoning_effort\":\"low\"") != std::string::npos);
            response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", "text/event-stream");
        }
    });
    auto provider = make_provider(fixture);
    core::llm::ChatRequest request;
    request.model = "qwen3.8-flash";
    request.effort = "none";
    for (int i = 0; i < 2; ++i) {
        std::string text;
        provider->stream_response(request, [&](const auto& chunk) {
            CHECK_FALSE(chunk.is_error);
            text += chunk.content;
        });
        CHECK(text == "OK");
    }
    CHECK(count == 3);
    request.effort = "high";
    provider->stream_response(request, [](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    CHECK(count == 4);
}

TEST_CASE("Qwen search retries an unsearched answer and consumes fragmented SSE",
          "[integration][qwen][regression][web]") {
    QwenServer fixture;
    std::atomic<int> count{0};
    fixture.server.Post("/compatible-mode/v1/responses", [&](const auto& request, auto& response) {
        CHECK(request.body.find("\"model\":\"qwen3.8-flash\"") != std::string::npos);
        CHECK(request.body.find("\"stream\":true") != std::string::npos);
        if (++count == 1) {
            response.set_content(R"({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Unsearched narration"}]}]})", "application/json");
            return;
        }
        const std::string body =
            "event: response.output_item.done\r\ndata: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"sources\":[{\"url\":\"https://example.com/reference\",\"title\":\"Reference\"}]}}}\r\n\r\n"
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Verified facts\"}\n\n"
            "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[]}}\n\n";
        response.set_chunked_content_provider("text/event-stream", [body](std::size_t offset, httplib::DataSink& sink) {
            const auto size = std::min(std::size_t{7}, body.size() - offset);
            if (!sink.write(body.data() + offset, size)) return false;
            if (offset + size == body.size()) sink.done();
            return true;
        });
    });
    const auto backend = core::tools::web::make_dashscope_web_search_backend({.model = "qwen3.8-flash"});
    const auto result = backend->search({.query = "Check facts"}, invocation(make_provider(fixture)));
    REQUIRE(result.has_value());
    CHECK(result->answer == "Verified facts");
    REQUIRE(result->results.size() == 1);
    CHECK(result->results.front().url == "https://example.com/reference");
    CHECK(count == 2);
}

TEST_CASE("Qwen factory honors an explicit Responses route with hosted tools",
          "[integration][qwen][regression][http][responses]") {
    QwenServer fixture;
    std::atomic<int> calls{0};
    fixture.server.Post("/compatible-mode/v1/responses", [&](const auto& request, auto& response) {
        ++calls;
        CHECK(request.body.find("\"input\":[") != std::string::npos);
        CHECK(request.body.find("\"type\":\"web_search\"") != std::string::npos);
        response.set_content("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_test\"}}\n\n"
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Verified\"}\n\n"
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_test\",\"status\":\"completed\",\"output\":[]}}\n\n", "text/event-stream");
    });
    core::config::ProviderConfig config;
    config.base_url = fixture.url();
    config.api_key = "test-qwen-key";
    config.model = "qwen3.8-max";
    config.wire_api = "responses";
    const auto provider = core::llm::ProviderFactory::create_provider("qwen-token-plan", config);
    REQUIRE(provider != nullptr);
    CHECK_FALSE(provider->should_estimate_cost());
    core::llm::ChatRequest request;
    request.model = config.model;
    request.messages = {{.role = "user", .content = "Verify"}};
    std::string text;
    provider->stream_response(request, [&](const auto& chunk) {
        CHECK_FALSE(chunk.is_error);
        text += chunk.content;
    });
    CHECK(text == "Verified");
    CHECK(calls == 1);
}

TEST_CASE("Qwen search refuses unsearched narration and honors cancellation",
          "[integration][qwen][regression][web]") {
    QwenServer fixture;
    std::atomic<int> count{0};
    fixture.server.Post("/compatible-mode/v1/responses", [&](const auto&, auto& response) {
        ++count;
        response.set_content(R"({"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Unsearched answer"}]}]})", "application/json");
    });
    const auto backend = core::tools::web::make_dashscope_web_search_backend({.timeout_ms = 1000});
    auto context = invocation(make_provider(fixture));
    const auto result = backend->search({.query = "Check facts"}, context);
    CHECK_FALSE(result.has_value());
    CHECK(count == 2);
    context.cancellation_requested = [] { return true; };
    CHECK_FALSE(backend->search({.query = "Check facts"}, context).has_value());
    CHECK(count == 2);
}

TEST_CASE("Qwen search labels a verified partial stream and bounds its timeout",
          "[integration][qwen][regression][web]") {
    QwenServer fixture;
    std::atomic<int> calls{0};
    fixture.server.Post("/compatible-mode/v1/responses", [&](const auto&, auto& response) {
        if (++calls == 1) {
            response.set_content("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"sources\":[{\"url\":\"https://example.com/source\"}]}}}\n\n"
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Verified fragment\"}\n\n", "text/event-stream");
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            response.set_content("{}", "application/json");
        }
    });
    const auto context = invocation(make_provider(fixture));
    const auto backend = core::tools::web::make_dashscope_web_search_backend({.model = "qwen3.8-flash", .timeout_ms = 50});
    const auto partial = backend->search({.query = "Check facts"}, context);
    REQUIRE(partial.has_value());
    CHECK_THAT(partial->answer, Catch::Matchers::ContainsSubstring("Partial search result"));
    CHECK_THAT(partial->answer, Catch::Matchers::ContainsSubstring("Verified fragment"));
    REQUIRE(partial->results.size() == 1);
    const auto started = std::chrono::steady_clock::now();
    CHECK_FALSE(backend->search({.query = "Slow search"}, context).has_value());
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(2));
    CHECK(calls == 2);
}

TEST_CASE("Qwen request repair cannot loop when the repaired request also fails",
          "[integration][qwen][regression][http]") {
    QwenServer fixture;
    std::atomic<int> calls{0};
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto&, auto& response) {
        ++calls;
        response.status = 400;
        response.set_content(R"({"error":{"message":"enable_thinking must be true"}})", "application/json");
    });
    const auto provider = make_provider(fixture);
    core::llm::ChatRequest request;
    request.model = "qwen3.7-plus";
    request.effort = "none";
    bool failed = false;
    provider->stream_response(request, [&](const auto& chunk) { failed = failed || chunk.is_error; });
    CHECK(failed);
    CHECK(calls == 2);
}

TEST_CASE("Qwen policy does not change the generic HTTP provider",
          "[integration][qwen][regression][http][isolation]") {
    QwenServer fixture;
    std::atomic<int> calls{0};
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto& request, auto& response) {
        ++calls;
        CHECK(request.body.find("\"max_tokens\":131072") != std::string::npos);
        response.status = 400;
        response.set_content(R"({"error":{"message":"enable_thinking must be true"}})", "application/json");
    });
    const auto generic = std::make_shared<core::llm::HttpLLMProvider>(fixture.url(),
        core::auth::ApiKeyCredentialSource::as_bearer("test-generic-key"), "qwen3.8-flash",
        std::make_unique<core::llm::protocols::OpenAIProtocol>(), core::config::ApiType::OpenAI,
        "generic-policy-isolation");
    CHECK(dynamic_cast<const core::llm::ContextTokenEstimator*>(generic.get()) == nullptr);
    core::llm::ChatRequest request;
    request.model = "qwen3.8-flash";
    request.effort = "none";
    bool failed = false;
    generic->stream_response(request, [&](const auto& chunk) { failed = failed || chunk.is_error; });
    CHECK(failed);
    CHECK(calls == 1);
}

TEST_CASE("Qwen repair remains eligible after a transport retry notice",
          "[integration][qwen][regression][http][isolation]") {
    QwenServer fixture;
    std::atomic<int> calls{0};
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto& request, auto& response) {
        const auto call = ++calls;
        if (call == 1) {
            response.status = 500;
            response.set_content(R"({"error":{"message":"Temporary failure"}})", "application/json");
        } else if (request.body.find("\"enable_thinking\":false") != std::string::npos) {
            response.status = 400;
            response.set_content(R"({"error":{"message":"enable_thinking must be true"}})", "application/json");
        } else {
            response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", "text/event-stream");
        }
    });
    const auto provider = make_provider(fixture);
    core::llm::ChatRequest request;
    request.model = "qwen3.7-plus";
    request.effort = "none";
    bool failed = false, retried = false;
    std::string text;
    provider->stream_response(request, [&](const auto& chunk) {
        failed = failed || chunk.is_error;
        retried = retried || chunk.reset_attempt;
        if (!chunk.reset_attempt) text += chunk.content;
    });
    CHECK_FALSE(failed);
    CHECK(retried);
    CHECK(text == "OK");
    CHECK(calls == 3);
}

TEST_CASE("Parallel Qwen forks retain policy and isolate request state",
          "[integration][qwen][regression][http][isolation]") {
    QwenServer fixture;
    fixture.server.Post("/compatible-mode/v1/chat/completions", [&](const auto& request, auto& response) {
        if (request.body.find("\"enable_thinking\":false") != std::string::npos) {
            response.status = 400;
            response.set_content(R"({"error":{"message":"enable_thinking must be true"}})", "application/json");
        } else {
            const int tokens = request.body.find("Parent") != std::string::npos ? 5000 : 15000;
            response.set_content("data: {\"choices\":[{\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}]}\n\n"
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":" + std::to_string(tokens)
                + ",\"completion_tokens\":5}}\n\ndata: [DONE]\n\n", "text/event-stream");
        }
    });
    const auto parent = make_provider(fixture);
    const auto fork = std::dynamic_pointer_cast<core::llm::providers::QwenProvider>(parent->fork_for_parallel_request());
    REQUIRE(fork != nullptr);
    core::llm::ChatRequest request;
    request.model = "qwen3.7-plus";
    request.effort = "none";
    request.messages = {{.role = "user", .content = "Check"}};
    auto parent_request = request;
    parent_request.effort = "high";
    parent_request.messages.front().content = "Parent";
    std::atomic<bool> failed{false};
    const auto callback = [&](const auto& chunk) { if (chunk.is_error) failed = true; };
    std::jthread parent_thread([&] { parent->stream_response(parent_request, callback); });
    std::jthread fork_thread([&] { fork->stream_response(request, callback); });
    parent_thread.join();
    fork_thread.join();
    CHECK_FALSE(failed);
    CHECK(fork->reasoning_capabilities(request.model).supports(core::llm::ReasoningCapability::Required));
    CHECK_FALSE(parent->reasoning_capabilities(request.model).supports(core::llm::ReasoningCapability::Required));
    CHECK(fork->estimate_context_tokens(request.messages) >= 15000);
    CHECK(parent->estimate_context_tokens(request.messages) < 15000);
    const auto inherited = fork->fork_for_parallel_request();
    CHECK(inherited->reasoning_capabilities(request.model).supports(core::llm::ReasoningCapability::Required));
}
