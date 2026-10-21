#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ModelCatalogDiscovery.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include <atomic>
#include <chrono>
#include <thread>

using namespace core::llm;
using namespace core::llm::protocols;
using Catch::Matchers::ContainsSubstring;

namespace {
class MiniMaxServer {
public:
    httplib::Server server;
    int port;
    std::jthread thread_;
    explicit MiniMaxServer(std::string models_response = R"({"data":[{"id":"MiniMax-M3.1-Flash"}]})") {
        server.Get("/anthropic/v1/models", [models_response = std::move(models_response)](const auto&, auto& response) {
            response.set_content(models_response, "application/json");
        });
        port = server.bind_to_any_port("127.0.0.1");
        if (port <= 0) SKIP("Local sockets unavailable.");
        thread_ = std::jthread([&] { server.listen_after_bind(); });
        for (int i = 0; i < 100 && !server.is_running(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        REQUIRE(server.is_running());
    }
    ~MiniMaxServer() { server.stop(); if (thread_.joinable()) thread_.join(); }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port) + "/anthropic"; }
    std::shared_ptr<LLMProvider> provider(std::string name = "minimax-m-plan") const {
        core::config::ProviderConfig config;
        config.base_url = url(); config.api_key = "fixture-key"; config.model = "MiniMax-M3.1-Flash";
        return ProviderFactory::create_provider(name, config);
    }
};
std::string event(std::string_view json) { return "data: " + std::string(json) + "\n\n"; }
std::string end_turn() {
    return event(R"({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})")
        + event(R"({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Done"}})")
        + event(R"({"type":"content_block_stop","index":0})")
        + event(R"({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}})")
        + event(R"({"type":"message_stop"})");
}
}

TEST_CASE("MiniMax HTTP tool turns preserve reasoning and account for cached tokens", "[integration][minimax][http]") {
    MiniMaxServer fixture;
    std::vector<std::string> bodies;
    std::vector<std::string> keys;
    std::atomic<int> calls{0};
    fixture.server.Post("/anthropic/v1/messages", [&](const auto& req, auto& response) {
        bodies.push_back(req.body);
        keys.push_back(req.get_header_value("x-api-key"));
        const auto start = event(R"({"type":"message_start","message":{"usage":{"input_tokens":20,"cache_read_input_tokens":200}}})");
        if (++calls == 1) {
            response.set_content(start
                + event(R"({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"Plan","signature":"signed"}})")
                + event(R"({"type":"content_block_stop","index":0})")
                + event(R"({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool-1","name":"read","input":{}}})")
                + event(R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}})")
                + event(R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"file\"}"}})")
                + event(R"({"type":"content_block_stop","index":1})")
                + event(R"({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}})")
                + event(R"({"type":"message_stop"})"), "text/event-stream");
        } else response.set_content(start + end_turn(), "text/event-stream");
    });
    auto provider = fixture.provider(); REQUIRE(provider);
    ChatRequest req; req.model = "MiniMax-M3.1-Flash";
    req.messages = {{.role = "system", .content = "Workspace"}, {.role = "user", .content = "Read file"}};
    Message assistant{.role = "assistant"};
    bool failed = false;
    provider->stream_response(req, [&](const StreamChunk& chunk) {
        failed |= chunk.is_error;
        assistant.content += chunk.content;
        assistant.tool_calls.insert(assistant.tool_calls.end(), chunk.tools.begin(), chunk.tools.end());
        assistant.continuation_items.insert(assistant.continuation_items.end(), chunk.continuation_items.begin(), chunk.continuation_items.end());
    });
    REQUIRE_FALSE(failed); REQUIRE(assistant.tool_calls.size() == 1);
    REQUIRE_FALSE(assistant.continuation_items.empty());
    for (const auto& item : assistant.continuation_items) {
        CHECK(item.origin_model == req.model);
        CHECK_THAT(item.origin_service, ContainsSubstring("minimax-m-plan|http://127.0.0.1:"));
    }
    CHECK(provider->get_last_usage().prompt_tokens == 220);
    CHECK(provider->get_last_usage().cached_prompt_tokens == 200);
    req.messages.push_back(assistant);
    req.messages.push_back({.role = "tool", .content = "Contents", .tool_call_id = "tool-1"});
    std::string answer;
    provider->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); answer += chunk.content; });
    CHECK(answer == "Done"); REQUIRE(bodies.size() == 2);
    CHECK_THAT(bodies[1], ContainsSubstring(R"("thinking":"Plan","signature":"signed")"));
    CHECK_THAT(bodies[1], ContainsSubstring(R"("type":"tool_result","tool_use_id":"tool-1")"));
    for (const auto& body : bodies) {
        CHECK_THAT(body, !ContainsSubstring("billing-header"));
        CHECK_THAT(body, !ContainsSubstring("budget_tokens"));
    }
    CHECK(keys == std::vector<std::string>{"fixture-key", "fixture-key"});

    // The same model on a different commercial service must not inherit signatures.
    auto other = fixture.provider("minimax"); REQUIRE(other);
    other->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    REQUIRE(bodies.size() == 3);
    CHECK_THAT(bodies[2], !ContainsSubstring("signed"));
    CHECK_THAT(bodies[2], ContainsSubstring("tool-1"));
}

TEST_CASE("MiniMax HTTP refuses incomplete tools and permanent business errors", "[integration][minimax][http]") {
    MiniMaxServer fixture;
    std::atomic<int> calls{0};
    fixture.server.Post("/anthropic/v1/messages", [&](const auto&, auto& response) {
        ++calls;
        response.set_content(event(R"({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"bad","name":"read","input":{}}})")
            + event(R"({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}})"), "text/event-stream");
    });
    auto provider = fixture.provider();
    ChatRequest req; req.messages = {{.role = "user", .content = "Read"}};
    bool error = false; int tools = 0;
    provider->stream_response(req, [&](const auto& chunk) { error |= chunk.is_error; tools += static_cast<int>(chunk.tools.size()); });
    CHECK(error); CHECK(tools == 0);
    MiniMaxServer exhausted;
    std::atomic<int> exhausted_calls{0};
    exhausted.server.Post("/anthropic/v1/messages", [&](const auto&, auto& response) {
        ++exhausted_calls; response.status = 429;
        response.set_content(R"({"error":{"code":2056,"message":"Usage limit reached"}})", "application/json");
    });
    error = false;
    exhausted.provider()->stream_response(req, [&](const auto& chunk) { error |= chunk.is_error; });
    CHECK(error); CHECK(exhausted_calls == 1);
}

TEST_CASE("Custom Messages factory does not inherit Claude policy", "[integration][anthropic-neutral][minimax]") {
    MiniMaxServer fixture;
    std::string body; bool claude_header = false;
    fixture.server.Post("/anthropic/v1/messages", [&](const auto& req, auto& response) {
        body = req.body; claude_header = req.has_header("x-anthropic-billing-header") || req.has_header("anthropic-beta");
        response.set_content(end_turn(), "text/event-stream");
    });
    core::config::ProviderConfig config;
    config.api_type = core::config::ApiType::Anthropic; config.base_url = fixture.url();
    config.api_key = "fixture"; config.model = "sonnet"; config.thinking_budget = 10000;
    auto provider = ProviderFactory::create_provider("custom-messages", config); REQUIRE(provider);
    ChatRequest req; req.messages = {{.role = "user", .content = "Hello"}};
    provider->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    CHECK_FALSE(claude_header);
    CHECK_THAT(body, ContainsSubstring(R"("model":"sonnet")"));
    CHECK_THAT(body, !ContainsSubstring("billing"));
    CHECK_THAT(body, !ContainsSubstring("thinking"));
}

TEST_CASE("Claude retains legacy reasoning while MiniMax refuses unscoped state", "[integration][claude][minimax][http]") {
    MiniMaxServer fixture;
    std::vector<std::string> bodies;
    std::vector<bool> billing;
    fixture.server.Post("/anthropic/v1/messages", [&](const auto& req, auto& response) {
        bodies.push_back(req.body);
        billing.push_back(req.has_header("x-anthropic-billing-header"));
        response.set_content(end_turn(), "text/event-stream");
    });
    core::config::ProviderConfig config;
    config.api_key = "fixture"; config.base_url = fixture.url(); config.model = "sonnet";
    const auto claude = ProviderFactory::create_provider("claude-legacy-review", config);
    REQUIRE(claude);
    ChatRequest req;
    req.messages = {{.role = "user", .content = "Hello"},
        {.role = "assistant", .content = "Prior answer", .continuation_items = {{
            .provider = "anthropic", .kind = "thinking",
            .payload = R"({"type":"thinking","thinking":"Prior reasoning","signature":"legacy-claude-signature"})"}}},
        {.role = "user", .content = "Continue"}};
    claude->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    fixture.provider()->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    REQUIRE(bodies.size() == 2);
    CHECK(billing[0]); CHECK_FALSE(billing[1]);
    CHECK_THAT(bodies[0], ContainsSubstring("legacy-claude-signature"));
    CHECK_THAT(bodies[0], ContainsSubstring(R"("model":"claude-sonnet-)"));
    CHECK_THAT(bodies[1], !ContainsSubstring("legacy-claude-signature"));
    CHECK_THAT(bodies[1], ContainsSubstring("Prior answer"));
}

TEST_CASE("MiniMax native counting caps generation using the actual input footprint", "[integration][minimax][budget]") {
    MiniMaxServer fixture;
    int input_tokens = 490000;
    std::size_t history_bytes = 360000;
    std::string expected_max = R"("max_tokens":6000)";
    std::optional<int> requested_max;
    SECTION("Exact counts reduce output to fit") {}
    SECTION("Exact counts admit histories overstated by the local estimate") {
        input_tokens = 100000; history_bytes = 1100000;
        expected_max = R"("max_tokens":128000)";
    }
    SECTION("Exact counts restore output room lost to an inflated estimate") {
        input_tokens = 100000; history_bytes = 800000;
        expected_max = R"("max_tokens":128000)";
    }
    SECTION("Explicit smaller output requests remain respected") {
        input_tokens = 100000; history_bytes = 800000; requested_max = 20000;
        expected_max = R"("max_tokens":20000)";
    }
    std::atomic<int> counts{0};
    std::string count_body, generation_body;
    fixture.server.Post("/anthropic/v1/messages/count_tokens", [&](const auto& req, auto& response) {
        ++counts; count_body = req.body;
        response.set_content("{\"input_tokens\":" + std::to_string(input_tokens) + "}", "application/json");
    });
    fixture.server.Post("/anthropic/v1/messages", [&](const auto& req, auto& response) {
        generation_body = req.body;
        response.set_content(end_turn(), "text/event-stream");
    });
    ChatRequest req;
    req.messages = {{.role = "user", .content = std::string(history_bytes, 'a')}};
    req.max_tokens = requested_max;
    fixture.provider()->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    CHECK(counts == 1);
    CHECK_THAT(count_body, !ContainsSubstring("max_tokens"));
    CHECK_THAT(count_body, !ContainsSubstring("output_config"));
    CHECK(generation_body.find(expected_max) != std::string::npos);
}

TEST_CASE("MiniMax failed native counting keeps conservative context admission", "[integration][minimax][budget]") {
    MiniMaxServer fixture;
    std::atomic<int> generations{0}, counts{0};
    std::atomic<bool> full_output_budget{false};
    bool oversized_response = false;
    fixture.server.Post("/anthropic/v1/messages/count_tokens", [&](const auto&, auto& response) {
        ++counts;
        if (oversized_response) {
            response.set_content(R"({"input_tokens":10000})" + std::string(20 * 1024, ' '), "application/json");
        } else {
            response.status = 404;
        }
    });
    fixture.server.Post("/anthropic/v1/messages", [&](const auto& req, auto& response) {
        full_output_budget = req.body.find(R"("max_tokens":128000)") != std::string::npos;
        ++generations; response.set_content(end_turn(), "text/event-stream");
    });
    std::size_t bytes = 800000;
    bool overfull = false;
    SECTION("Local fallback admits a fitting request") {}
    SECTION("Local fallback refuses an overfull request") { bytes = 1100000; overfull = true; }
    SECTION("Oversized count responses use the local fallback") { oversized_response = true; }
    ChatRequest req; req.messages = {{.role = "user", .content = std::string(bytes, 'a')}};
    bool error = false;
    fixture.provider()->stream_response(req, [&](const auto& chunk) { error |= chunk.is_error; });
    CHECK(counts == 1);
    CHECK(error == overfull);
    CHECK(generations == (overfull ? 0 : 1));
    if (oversized_response) CHECK_FALSE(full_output_budget.load());
}

TEST_CASE("MiniMax serving context limits govern both admission and compaction", "[integration][minimax][budget]") {
    MiniMaxServer fixture(R"({"data":[{"id":"MiniMax-M3.1-Flash","max_input_tokens":64000}]})");
    std::string generation_body;
    fixture.server.Post("/anthropic/v1/messages/count_tokens", [](const auto&, auto& response) {
        response.set_content(R"({"input_tokens":10000})", "application/json");
    });
    fixture.server.Post("/anthropic/v1/messages", [&](const auto& req, auto& response) {
        generation_body = req.body; response.set_content(end_turn(), "text/event-stream");
    });
    auto provider = fixture.provider("minimax-m-plan-small-context");
    ChatRequest req; req.messages = {{.role = "user", .content = std::string(60000, 'a')}};
    provider->stream_response(req, [&](const auto& chunk) { CHECK_FALSE(chunk.is_error); });
    CHECK(generation_body.find(R"("max_tokens":38000)") != std::string::npos);
    CHECK(provider->max_context_size() == 64000);
    CHECK(provider->fork_for_parallel_request()->max_context_size() == 64000);
}

TEST_CASE("MiniMax cancellation during context counting prevents generation", "[integration][minimax][cancel]") {
    MiniMaxServer fixture;
    std::atomic<bool> counting{false}, release{false};
    std::atomic<int> generations{0};
    fixture.server.Post("/anthropic/v1/messages/count_tokens", [&](const auto&, auto& response) {
        counting = true;
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        response.set_content(R"({"input_tokens":100000})", "application/json");
    });
    fixture.server.Post("/anthropic/v1/messages", [&](const auto&, auto& response) {
        ++generations; response.set_content(end_turn(), "text/event-stream");
    });
    auto provider = fixture.provider();
    ChatRequest req; req.messages = {{.role = "user", .content = std::string(360000, 'a')}};
    std::jthread worker([&] { provider->stream_response(req, [](const auto&) {}); });
    for (int i = 0; i < 1000 && !counting.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    provider->cancel(); release = true;
    worker.join();
    CHECK(counting.load()); CHECK(generations == 0);
}

TEST_CASE("Parallel MiniMax provider forks isolate streaming state", "[integration][minimax][parallel]") {
    MiniMaxServer fixture;
    std::atomic<int> requests{0};
    fixture.server.Post("/anthropic/v1/messages", [&](const auto&, auto& response) {
        ++requests; response.set_content(end_turn(), "text/event-stream");
    });
    auto parent = fixture.provider();
    auto fork = parent->fork_for_parallel_request(); REQUIRE(fork);
    std::atomic<int> failures{0};
    std::string first, second;
    ChatRequest req; req.messages = {{.role = "user", .content = "Hello"}};
    std::jthread one([&] { parent->stream_response(req, [&](const auto& chunk) { failures += chunk.is_error; first += chunk.content; }); });
    std::jthread two([&] { fork->stream_response(req, [&](const auto& chunk) { failures += chunk.is_error; second += chunk.content; }); });
    one.join(); two.join();
    CHECK(failures == 0); CHECK(requests == 2); CHECK(first == "Done"); CHECK(second == "Done");
}
