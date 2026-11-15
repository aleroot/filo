#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/AnthropicAdvisor.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"
#include "core/llm/protocols/ZaiProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/auth/ICredentialSource.hpp"

#include <httplib.h>
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace core::llm;
using namespace core::llm::protocols;
using namespace core::llm::protocols::anthropic;

namespace {

ChatRequest make_request(std::string model = "claude-opus-5-5") {
    ChatRequest request;
    request.model = std::move(model);
    request.messages.push_back(Message{.role = "user", .content = "Hello"});
    return request;
}

Tool make_tool(std::string name) {
    Tool tool;
    tool.function.name = std::move(name);
    tool.function.description = "A client tool";
    tool.function.input_schema = R"({"type":"object","properties":{}})";
    return tool;
}

void require_valid_json(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string(payload)};
    simdjson::dom::element document;
    REQUIRE(parser.parse(padded).get(document) == simdjson::SUCCESS);
}

/// The value of a JSON member, as text.
[[nodiscard]] std::string json_member(std::string_view payload, std::string_view key) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string(payload)};
    simdjson::dom::element document;
    REQUIRE(parser.parse(padded).get(document) == simdjson::SUCCESS);
    std::string_view value;
    REQUIRE(document[key].get(value) == simdjson::SUCCESS);
    return std::string(value);
}

/// A credential source that hands the transport a fixed key.
class StaticCredentialSource final : public core::auth::ICredentialSource {
public:
    core::auth::AuthInfo get_auth() override {
        core::auth::AuthInfo auth;
        auth.headers["x-api-key"] = "test-key";
        return auth;
    }
};

/// A local endpoint that answers with one canned Anthropic event stream and
/// records what the client sent.
class AdvisorEndpoint {
public:
    AdvisorEndpoint() {
        server_.Post("/v1/messages", [this](const httplib::Request& request,
                                             httplib::Response& response) {
            bodies_.push_back(request.body);
            betas_.push_back(request.get_header_value("anthropic-beta"));
            response.set_content(stream(), "text/event-stream");
        });
    }

    /// The canned stream: an advisor call, its result, and the end of the turn.
    [[nodiscard]] static std::string stream() {
        return
            "event: message_start\n"
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-opus-5-5\","
            "\"usage\":{\"input_tokens\":12}}}\n\n"
            "event: content_block_start\n"
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":"
            "{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"advisor\",\"input\":{}}}\n\n"
            "event: content_block_stop\n"
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
            "event: content_block_start\n"
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":"
            "{\"type\":\"advisor_tool_result\",\"tool_use_id\":\"srvtoolu_1\",\"content\":"
            "{\"type\":\"advisor_result\",\"text\":\"Use a channel-based pattern.\"}}}\n\n"
            "event: content_block_stop\n"
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n"
            "event: message_delta\n"
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},"
            "\"usage\":{\"output_tokens\":7}}\n\n"
            "event: message_stop\n"
            "data: {\"type\":\"message_stop\"}\n\n";
    }

    [[nodiscard]] int start() {
        const int port = server_.bind_to_any_port("127.0.0.1");
        if (port > 0) {
            thread_ = std::thread([this] { server_.listen_after_bind(); });
        }
        return port;
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    [[nodiscard]] const std::vector<std::string>& bodies() const noexcept { return bodies_; }
    [[nodiscard]] const std::vector<std::string>& betas() const noexcept { return betas_; }

private:
    httplib::Server server_;
    std::thread thread_;
    std::vector<std::string> bodies_;
    std::vector<std::string> betas_;
};

/// A local endpoint that refuses the advisor once, then answers normally.
class RefusingAdvisorEndpoint {
public:
    RefusingAdvisorEndpoint() {
        server_.Post("/v1/messages", [this](const httplib::Request& request,
                                             httplib::Response& response) {
            bodies_.push_back(request.body);
            if (bodies_.size() == 1) {
                response.status = 400;
                response.set_content(
                    R"({"type":"error","error":{"type":"invalid_request_error",)"
                    R"("message":"advisor: model claude-fable-5-2 is not available for this account"}})",
                    "application/json");
                return;
            }
            response.set_content(AdvisorEndpoint::stream(), "text/event-stream");
        });
    }

    [[nodiscard]] int start() {
        const int port = server_.bind_to_any_port("127.0.0.1");
        if (port > 0) {
            thread_ = std::thread([this] { server_.listen_after_bind(); });
        }
        return port;
    }

    void stop() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    [[nodiscard]] const std::vector<std::string>& bodies() const noexcept { return bodies_; }

private:
    httplib::Server server_;
    std::thread thread_;
    std::vector<std::string> bodies_;
};

} // namespace

TEST_CASE("Anthropic advisor accepts the pairings Anthropic documents",
          "[advisor][anthropic]") {
    struct Case {
        std::string_view executor;
        std::string_view advisor;
        bool supported;
    };
    static constexpr std::array kCases{
        // Opus 5.5 accepts Fable and Opus 5 or later.
        Case{"claude-opus-5-5", "claude-fable-5-2", true},
        Case{"claude-opus-5-5", "claude-opus-5-5", true},
        Case{"claude-opus-5-5", "claude-opus-5", true},
        Case{"claude-opus-5-5", "claude-sonnet-5-5", false},
        Case{"claude-opus-5-5", "claude-haiku-5-5", false},
        Case{"claude-opus-5", "claude-fable-5-2", true},
        // Opus 4.7 and 4.8 accept Opus 4.7 or later and Sonnet 5.5.
        Case{"claude-opus-4-8", "claude-fable-5-2", true},
        Case{"claude-opus-4-8", "claude-opus-4-8", true},
        Case{"claude-opus-4-8", "claude-sonnet-5-5", true},
        // "Opus 4.7 or later" spans generations, so a newer Opus advises too.
        Case{"claude-opus-4-8", "claude-opus-5-5", true},
        // Opus 4.6 accepts every Opus, plus Sonnet 5 or later and Haiku 5.5.
        Case{"claude-opus-4-6", "claude-sonnet-5", true},
        Case{"claude-opus-4-6", "claude-opus-4-8", true},
        Case{"claude-opus-4-6", "claude-opus-5-5", true},
        Case{"claude-opus-4-6", "claude-sonnet-4-6", false},
        Case{"claude-opus-4-6", "claude-haiku-5-5", true},
        Case{"claude-opus-4-6", "claude-haiku-4-5", false},
        // Sonnet 5.5 accepts Opus 5 or later and Sonnet 5.5.
        Case{"claude-sonnet-5-5", "claude-opus-5-5", true},
        Case{"claude-sonnet-5-5", "claude-sonnet-5-5", true},
        Case{"claude-sonnet-5-5", "claude-opus-4-8", false},
        Case{"claude-sonnet-5-5", "claude-haiku-5-5", false},
        // Sonnet 5 accepts Opus 4.7 or later and Haiku 5.5.
        Case{"claude-sonnet-5", "claude-opus-4-8", true},
        Case{"claude-sonnet-5", "claude-haiku-5-5", true},
        // Haiku 4.5 accepts any Opus or Sonnet.
        Case{"claude-haiku-4-5", "claude-opus-4-6", true},
        Case{"claude-haiku-4-5", "claude-sonnet-4-6", true},
        // Fable advises itself, and only the generation Anthropic documents.
        Case{"claude-fable-5-2", "claude-fable-5-2", true},
        Case{"claude-fable-5-2", "claude-fable-5-1", false},
        Case{"claude-fable-5-2", "claude-opus-5-5", false},
        Case{"claude-fable-5-1", "claude-fable-5-2", true},
        Case{"claude-fable-5", "claude-fable-5-1", true},
        Case{"claude-fable-5", "claude-fable-5-2", true},
        // A model Anthropic does not document has no advisor.
        Case{"claude-3-5-sonnet-20241022", "claude-fable-5-2", false},
        Case{"claude-opus-5-5", "claude-3-5-sonnet-20241022", false},
        // The executor arrives as the id the session runs, which may be the
        // alias the user typed. It names the same generation as the canonical
        // id, so it accepts the same advisors.
        Case{"opus-5-5", "claude-fable-5-2", true},
        Case{"opus-5-5", "claude-opus-5-5", true},
        Case{"opus", "claude-fable-5-2", true},
        Case{"opus-5.5", "claude-fable-5-2", true},
        Case{"fable", "claude-fable-5-2", true},
        Case{"sonnet-5-5", "claude-fable-5-2", true},
        Case{"haiku-4-5", "claude-haiku-5-5", true},
        // A model of another provider is not one Anthropic can advise with.
        Case{"gpt-5", "claude-fable-5-2", false},
        Case{"qwen3-coder", "claude-fable-5-2", false},
    };

    AnthropicAdvisorSupport support;
    for (const auto& test_case : kCases) {
        CAPTURE(test_case.executor, test_case.advisor);
        const AdvisorSpec advisor{.model = std::string(test_case.advisor)};
        CHECK(support.supports(test_case.executor, advisor) == test_case.supported);
        const auto reason = support.unsupported_reason(test_case.executor, advisor);
        if (test_case.supported) {
            CHECK(reason.empty());
        } else {
            CHECK(!reason.empty());
        }
    }
}

TEST_CASE("Anthropic advisor resolves Claude aliases to concrete models",
          "[advisor][anthropic]") {
    CHECK(resolve_advisor_model("fable") == "claude-fable-5-2");
    CHECK(resolve_advisor_model("best") == "claude-fable-5-2");
    CHECK(resolve_advisor_model("claude-fable") == "claude-fable-5-2");
    CHECK(resolve_advisor_model("opus") == "claude-opus-5");
    CHECK(resolve_advisor_model("opus-5.5") == "claude-opus-5-5");
    CHECK(resolve_advisor_model("sonnet") == "claude-sonnet-5");
    CHECK(resolve_advisor_model("haiku") == "claude-haiku-5-5");
    CHECK(resolve_advisor_model("claude-fable-5-2") == "claude-fable-5-2");
    CHECK(resolve_advisor_model("").empty());

    // An alias names the same generation the alias resolves to.
    AnthropicAdvisorSupport support;
    const AdvisorSpec fable{.model = "fable"};
    CHECK(support.supports("claude-opus-5-5", fable));
    const AdvisorSpec haiku{.model = "haiku"};
    CHECK_FALSE(support.supports("claude-opus-5-5", haiku));
}

TEST_CASE("An advisor change needs a wire that can absorb it",
          "[advisor][anthropic][switch]") {
    // A conversation that produced no bound reasoning can change its tools.
    const std::vector<Message> fresh_conversation{
        Message{.role = "user", .content = "Hello"}};
    // A conversation that did carries reasoning a prefix check would reject.
    Message with_reasoning{.role = "assistant", .content = "Hello"};
    with_reasoning.continuation_items.push_back(ContinuationItem{
        .provider = "anthropic",
        .kind = "thinking",
        .payload = R"({"type":"thinking","thinking":"","signature":"sig"})",
    });
    const std::vector<Message> bound_conversation{with_reasoning};

    // The Claude Code wire asks the provider to drop the reasoning a changed
    // request prefix invalidated, so the models whose requests carry that
    // setting absorb a change whatever the conversation holds.
    const ClaudeProtocol claude_wire;
    const auto claude_support = claude_wire.advisor_support();
    REQUIRE(claude_support);
    // The plain Messages API wire sends no thinking configuration at all, so
    // it can only change the tools of a conversation that has produced no
    // bound reasoning yet.
    const AnthropicProtocol plain_wire;
    const auto plain_support = plain_wire.advisor_support();
    REQUIRE(plain_support);

    // A model that runs no prefix check is not affected by a tools change.
    CHECK(claude_support->absorbs_advisor_changes("claude-opus-5", fresh_conversation));
    CHECK(plain_support->absorbs_advisor_changes("claude-opus-5", bound_conversation));

    // A model that does run it.
    CHECK(claude_support->absorbs_advisor_changes("claude-opus-5-5", bound_conversation));
    CHECK_FALSE(plain_support->absorbs_advisor_changes("claude-opus-5-5", bound_conversation));
    CHECK(plain_support->absorbs_advisor_changes("claude-opus-5-5", fresh_conversation));
    CHECK_FALSE(plain_support->absorbs_advisor_changes("claude-sonnet-5-5", bound_conversation));
    CHECK_FALSE(plain_support->absorbs_advisor_changes("claude-fable-5-2", bound_conversation));

    // A model whose requests the wire leaves bound cannot absorb the change
    // either: the provider rejects the request whose prefix moved.
    CHECK_FALSE(claude_support->absorbs_advisor_changes("claude-mythos-5-1", bound_conversation));
    CHECK(claude_support->absorbs_advisor_changes("claude-mythos-5-1", fresh_conversation));
}

TEST_CASE("The advisor's answer agrees with the request the wire sends",
          "[advisor][anthropic][switch][wire]") {
    // A conversation that produced bound reasoning: the hard case.
    Message with_reasoning{.role = "assistant", .content = "Hello"};
    with_reasoning.continuation_items.push_back(ContinuationItem{
        .provider = "anthropic",
        .kind = "thinking",
        .payload = R"({"type":"thinking","thinking":"","signature":"sig"})",
    });
    const std::vector<Message> bound_conversation{with_reasoning};

    ClaudeProtocol wire;
    const auto support = wire.advisor_support();
    REQUIRE(support);

    for (const auto* model : {"claude-opus-5-5", "claude-fable-5-2", "claude-fable-5-1",
                              "claude-mythos-5-2", "claude-mythos-5-1", "claude-sonnet-5-5",
                              "claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5"}) {
        CAPTURE(model);
        auto request = make_request(model);
        wire.prepare_request(request);
        const auto body = wire.serialize(request);
        // The request asks the provider to drop the reasoning a changed
        // prefix invalidated only for the models that bind their reasoning.
        const bool drops = body.find(R"("prefix_mismatch_behavior":"drop_block")")
                           != std::string::npos;
        // The capability must agree with the request: a conversation whose
        // reasoning is bound can only change its tools when the request
        // carries the setting that drops the invalidated reasoning.
        CHECK(support->absorbs_advisor_changes(model, bound_conversation)
              == (!claude::model_runs_prefix_check(model) || drops));
    }
}

TEST_CASE("Anthropic advisor lists the advisors a model accepts",
          "[advisor][anthropic][picker]") {
    AnthropicAdvisorSupport support;

    // The list holds the families the executor accepts, named the way the
    // user names them.
    const auto opus_choices = support.advisor_choices("claude-opus-5-5");
    REQUIRE(opus_choices.size() == 2);
    CHECK(opus_choices[0] == "fable");
    CHECK(opus_choices[1] == "opus");

    const auto sonnet_choices = support.advisor_choices("claude-sonnet-5-5");
    REQUIRE(sonnet_choices.size() == 2);
    CHECK(sonnet_choices[0] == "fable");
    CHECK(sonnet_choices[1] == "opus");

    // A family whose alias resolves outside the accepted range is not
    // offered: the provider would refuse that pairing. The sonnet alias
    // resolves to Sonnet 5, which a Sonnet 5.5 executor does not accept.
    CHECK(std::find(sonnet_choices.begin(), sonnet_choices.end(), "sonnet")
          == sonnet_choices.end());

    // Haiku 4.5 accepts Haiku 5.5, which is what the alias resolves to.
    const auto haiku_choices = support.advisor_choices("claude-haiku-4-5");
    CHECK(std::find(haiku_choices.begin(), haiku_choices.end(), "haiku")
          != haiku_choices.end());
    CHECK(std::find(haiku_choices.begin(), haiku_choices.end(), "opus")
          != haiku_choices.end());

    // A model Anthropic does not document has no advisors at all.
    CHECK(support.advisor_choices("claude-3-5-sonnet-20241022").empty());
    CHECK(support.advisor_choices("").empty());

    // Everything the list offers is a pairing the provider accepts.
    for (const auto& executor : {"claude-opus-5-5", "claude-sonnet-5-5",
                                  "claude-haiku-4-5", "claude-fable-5-2"}) {
        for (const auto& choice : support.advisor_choices(executor)) {
            CAPTURE(executor, choice);
            const AdvisorSpec advisor{.model = choice};
            CHECK(support.supports(executor, advisor));
        }
    }
}

TEST_CASE("A refusal that names the advisor is read as one",
          "[advisor][anthropic][refusal]") {
    CHECK(advisor_refused_by_response(
        400, R"({"type":"error","error":{"type":"invalid_request_error",)"
             R"("message":"advisor: model claude-fable-5-2 is not available for this account"}})"));
    // The word may be capitalized.
    CHECK(advisor_refused_by_response(
        400, R"({"error":{"message":"Advisor is not supported on this plan"}})"));
    // A refusal that does not name the advisor is about the request.
    CHECK_FALSE(advisor_refused_by_response(
        400, R"({"error":{"message":"max_tokens is too large"}})"));
    // A server error is not a refusal of the advisor.
    CHECK_FALSE(advisor_refused_by_response(500, R"({"error":{"message":"advisor"}})"));
    CHECK_FALSE(advisor_refused_by_response(200, R"({"error":{"message":"advisor"}})"));
}

TEST_CASE("Anthropic advisor declares itself as a server tool", "[advisor][anthropic]") {
    const AdvisorSpec advisor{.model = "claude-fable-5-2"};
    const auto definition = advisor_tool_definition(advisor);

    require_valid_json(definition);
    CHECK(json_member(definition, "type") == "advisor_20260301");
    CHECK(json_member(definition, "name") == "advisor");
    CHECK(json_member(definition, "model") == "claude-fable-5-2");
    CHECK(definition.find("max_uses") == std::string::npos);
    CHECK(definition.find("max_tokens") == std::string::npos);

    const AdvisorSpec capped{.model = "fable", .max_uses = 3, .max_output_tokens = 512};
    const auto capped_definition = advisor_tool_definition(capped);
    require_valid_json(capped_definition);
    CHECK(json_member(capped_definition, "model") == "claude-fable-5-2");
    CHECK(capped_definition.find("\"max_uses\":3") != std::string::npos);
    // The API refuses an advisor output cap below 1024 tokens.
    CHECK(capped_definition.find("\"max_tokens\":1024") != std::string::npos);

    CHECK(advisor_tool_definition(AdvisorSpec{}).empty());
    REQUIRE(advisor_feature_flags().size() == 1);
    CHECK(advisor_feature_flags().front() == "advisor-tool-2026-03-01");
}

TEST_CASE("A request with an advisor declares it after the client tools",
          "[advisor][anthropic][serializer]") {
    ClaudeProtocol protocol;
    auto request = make_request();
    request.tools.push_back(make_tool("read"));
    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};
    protocol.prepare_request(request);

    const auto payload = protocol.serialize(request);
    require_valid_json(payload);

    const auto tools_at = payload.find("\"tools\":[");
    REQUIRE(tools_at != std::string::npos);
    const auto client_at = payload.find("\"name\":\"read\"", tools_at);
    const auto advisor_at = payload.find("\"type\":\"advisor_20260301\"", tools_at);
    REQUIRE(client_at != std::string::npos);
    REQUIRE(advisor_at != std::string::npos);
    // The advisor is declared last, so the cache breakpoint stays on the last
    // client tool.
    CHECK(client_at < advisor_at);
    const auto cache_at = payload.find("\"cache_control\"", client_at);
    REQUIRE(cache_at != std::string::npos);
    CHECK(cache_at < advisor_at);
}

TEST_CASE("A request without an advisor declares no server tool",
          "[advisor][anthropic][serializer][regression]") {
    ClaudeProtocol protocol;
    auto request = make_request();
    request.tools.push_back(make_tool("read"));
    protocol.prepare_request(request);

    const auto payload = protocol.serialize(request);

    CHECK(payload.find("advisor_20260301") == std::string::npos);
    CHECK(payload.find("advisor-tool-2026-03-01") == std::string::npos);
}

TEST_CASE("A request with an advisor asks for the advisor beta",
          "[advisor][anthropic][headers]") {
    ClaudeProtocol protocol;
    auto request = make_request();
    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};
    protocol.prepare_request(request);

    const core::auth::AuthInfo auth;
    const auto headers = protocol.build_headers(auth);
    const auto betas = headers.find("anthropic-beta");
    REQUIRE(betas != headers.end());
    CHECK(betas->second.find("advisor-tool-2026-03-01") != std::string::npos);
    // The Claude Code betas stay in place.
    CHECK(betas->second.find("claude-code-20250219") != std::string::npos);
}

TEST_CASE("A request without an advisor asks for no advisor beta",
          "[advisor][anthropic][headers][regression]") {
    ClaudeProtocol protocol;
    auto request = make_request();
    protocol.prepare_request(request);

    const core::auth::AuthInfo auth;
    const auto headers = protocol.build_headers(auth);
    const auto betas = headers.find("anthropic-beta");
    if (betas != headers.end()) {
        CHECK(betas->second.find("advisor-tool-2026-03-01") == std::string::npos);
    }
}

TEST_CASE("The provider reports an advisor consultation", "[advisor][anthropic][sse]") {
    ClaudeProtocol protocol;
    auto request = make_request();
    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};
    protocol.prepare_request(request);

    const auto use = protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":"
        "{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"advisor\",\"input\":{}}}\n");
    REQUIRE(use.chunks.size() == 1);
    REQUIRE(use.chunks.front().advisor_event.has_value());
    CHECK(use.chunks.front().advisor_event->state == AdvisorCallState::Pending);
    CHECK(use.chunks.front().advisor_event->advisor_model == "claude-fable-5-2");

    const auto result = protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":"
        "{\"type\":\"advisor_tool_result\",\"tool_use_id\":\"srvtoolu_1\",\"content\":"
        "{\"type\":\"advisor_redacted_result\",\"encrypted_content\":\"EqQBCk\"}}}\n");
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks.front().advisor_event.has_value());
    CHECK(result.chunks.front().advisor_event->state == AdvisorCallState::Reviewed);
    // The newest advisors answer encrypted: the client cannot read the advice.
    CHECK_FALSE(result.chunks.front().advisor_event->guidance_readable);
}

TEST_CASE("The provider reports a failed advisor consultation",
          "[advisor][anthropic][sse]") {
    ClaudeProtocol protocol;

    const auto result = protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":"
        "{\"type\":\"advisor_tool_result\",\"tool_use_id\":\"srvtoolu_1\",\"content\":"
        "{\"type\":\"advisor_tool_result_error\",\"error_code\":\"overloaded\"}}}\n");
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks.front().advisor_event.has_value());
    CHECK(result.chunks.front().advisor_event->state == AdvisorCallState::Unavailable);
    CHECK(result.chunks.front().advisor_event->error_code == "overloaded");
}

TEST_CASE("Only a provider with Anthropic's server tools offers an advisor",
          "[advisor][providers]") {
    // Anthropic's own wire offers the advisor.
    ClaudeProtocol claude;
    REQUIRE(claude.advisor_support() != nullptr);
    CHECK(claude.advisor_support()->can_pause_turn());

    // A provider that only mimics Anthropic's shape does not.
    MiniMaxProtocol minimax;
    CHECK(minimax.advisor_support() == nullptr);
    ZaiCodingProtocol zai;
    CHECK(zai.advisor_support() == nullptr);
}

TEST_CASE("The advisor reaches the endpoint as a server tool",
          "[advisor][anthropic][http]") {
    AdvisorEndpoint endpoint;
    const int port = endpoint.start();
    REQUIRE(port > 0);

    auto provider = std::make_shared<HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(port),
        std::make_shared<StaticCredentialSource>(),
        "claude-opus-5-5",
        std::make_unique<ClaudeProtocol>());

    auto request = make_request();
    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};

    std::mutex mutex;
    std::condition_variable finished_signal;
    std::vector<StreamChunk> chunks;
    bool finished = false;
    provider->stream_response(request, [&](const StreamChunk& chunk) {
        std::lock_guard lock(mutex);
        chunks.push_back(chunk);
        if (chunk.is_final) {
            finished = true;
            finished_signal.notify_all();
        }
    });
    {
        std::unique_lock lock(mutex);
        REQUIRE(finished_signal.wait_for(lock, std::chrono::seconds(15),
                                         [&] { return finished; }));
    }
    endpoint.stop();

    // The request the endpoint received declares the advisor as a server tool.
    REQUIRE(endpoint.bodies().size() == 1);
    const auto& body = endpoint.bodies().front();
    require_valid_json(body);
    CHECK(body.find("\"type\":\"advisor_20260301\"") != std::string::npos);
    CHECK(body.find("\"name\":\"advisor\"") != std::string::npos);
    CHECK(body.find("\"model\":\"claude-fable-5-2\"") != std::string::npos);

    // The beta travels with the request.
    REQUIRE(endpoint.betas().size() == 1);
    CHECK(endpoint.betas().front().find("advisor-tool-2026-03-01") != std::string::npos);

    // The provider reports the consultation it was told about.
    bool saw_pending = false;
    bool saw_reviewed = false;
    for (const auto& chunk : chunks) {
        if (!chunk.advisor_event) continue;
        saw_pending |= chunk.advisor_event->state == AdvisorCallState::Pending;
        saw_reviewed |= chunk.advisor_event->state == AdvisorCallState::Reviewed;
    }
    CHECK(saw_pending);
    CHECK(saw_reviewed);
}

TEST_CASE("A refused advisor does not fail the turn", "[advisor][anthropic][http][refusal]") {
    RefusingAdvisorEndpoint endpoint;
    const int port = endpoint.start();
    REQUIRE(port > 0);

    auto provider = std::make_shared<HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(port),
        std::make_shared<StaticCredentialSource>(),
        "claude-opus-5-5",
        std::make_unique<ClaudeProtocol>());

    auto request = make_request();
    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};

    std::mutex mutex;
    std::condition_variable finished_signal;
    std::vector<StreamChunk> chunks;
    bool finished = false;
    provider->stream_response(request, [&](const StreamChunk& chunk) {
        std::lock_guard lock(mutex);
        chunks.push_back(chunk);
        if (chunk.is_final) {
            finished = true;
            finished_signal.notify_all();
        }
    });
    {
        std::unique_lock lock(mutex);
        REQUIRE(finished_signal.wait_for(lock, std::chrono::seconds(15),
                                         [&] { return finished; }));
    }
    endpoint.stop();

    // The request went again, this time without the advisor.
    REQUIRE(endpoint.bodies().size() == 2);
    CHECK(endpoint.bodies().front().find("advisor_20260301") != std::string::npos);
    CHECK(endpoint.bodies().back().find("advisor_20260301") == std::string::npos);

    // The client is told the provider refused the advisor, with the reason
    // the provider gave.
    bool saw_refused = false;
    std::string detail;
    for (const auto& chunk : chunks) {
        if (!chunk.advisor_event) continue;
        if (chunk.advisor_event->state == AdvisorCallState::Refused) {
            saw_refused = true;
            detail = chunk.advisor_event->detail;
        }
    }
    CHECK(saw_refused);
    CHECK(detail.find("not available for this account") != std::string::npos);
}

TEST_CASE("The advisor's blocks are replayed verbatim", "[advisor][anthropic][sse]") {
    ClaudeProtocol protocol;

    (void)protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":"
        "{\"type\":\"text\",\"text\":\"Hello\"}}\n");
    (void)protocol.parse_event(
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n");
    (void)protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":"
        "{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"advisor\",\"input\":{}}}\n");
    (void)protocol.parse_event(
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n");
    (void)protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":"
        "{\"type\":\"advisor_tool_result\",\"tool_use_id\":\"srvtoolu_1\",\"content\":"
        "{\"type\":\"advisor_result\",\"text\":\"Use a channel-based pattern.\"}}}\n");
    (void)protocol.parse_event(
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":2}\n");
    const auto stop = protocol.parse_event(
        "event: message_stop\n"
        "data: {\"type\":\"message_stop\"}\n");
    REQUIRE(stop.done);

    // The assistant turn keeps the advisor's blocks, so the next request can
    // echo them back as the API requires.
    std::vector<ContinuationItem> items;
    for (const auto& chunk : stop.chunks) {
        for (const auto& item : chunk.continuation_items) items.push_back(item);
    }
    REQUIRE(!items.empty());

    Message assistant{.role = "assistant", .content = "Hello", .continuation_items = items};
    auto request = make_request();
    request.messages.push_back(std::move(assistant));
    const auto payload = ClaudeSerializer::serialize(request);
    require_valid_json(payload);

    CHECK(payload.find("\"type\":\"server_tool_use\"") != std::string::npos);
    CHECK(payload.find("\"id\":\"srvtoolu_1\"") != std::string::npos);
    CHECK(payload.find("\"type\":\"advisor_tool_result\"") != std::string::npos);
    CHECK(payload.find("Use a channel-based pattern.") != std::string::npos);
}
