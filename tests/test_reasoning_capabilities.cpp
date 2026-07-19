#include <catch2/catch_test_macros.hpp>

#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/MistralProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

[[nodiscard]] bool supports(const ApiProtocolBase& protocol,
                            std::string_view model,
                            ReasoningCapability capability) {
    return protocol.reasoning_capabilities(model).supports(capability);
}

} // namespace

TEST_CASE("OpenAI protocols own their reasoning capability policy", "[llm][effort]") {
    OpenAIProtocol chat;
    OpenAIResponsesProtocol responses;
    for (const auto* protocol : {static_cast<ApiProtocolBase*>(&chat),
                                 static_cast<ApiProtocolBase*>(&responses)}) {
        CHECK(supports(*protocol, "gpt-5", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "o1-preview", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "o4-mini", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "GPT-5", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "gpt-5.6-sol", ReasoningCapability::MaxEffort));
        CHECK_FALSE(supports(*protocol, "gpt-5.5", ReasoningCapability::MaxEffort));
        CHECK_FALSE(supports(*protocol, "gpt-4o", ReasoningCapability::Effort));
        CHECK_FALSE(supports(*protocol, "glm-5.2", ReasoningCapability::Effort));
    }
}

TEST_CASE("Mistral protocol owns its reasoning model allow-list",
          "[llm][effort][mistral]") {
    MistralProtocol protocol;
    for (const auto model : {"mistral-vibe-cli-latest", "mistral-medium-3.5",
                             "mistral-medium-latest", "mistral-small-2603",
                             "MISTRAL-SMALL-LATEST"}) {
        CAPTURE(model);
        CHECK(supports(protocol, model, ReasoningCapability::Effort));
        CHECK(supports(protocol, model, ReasoningCapability::MaxEffort));
    }
    CHECK_FALSE(supports(protocol, "mistral-large-latest", ReasoningCapability::Effort));
    CHECK_FALSE(supports(protocol, "codestral-latest", ReasoningCapability::Effort));
}

TEST_CASE("Kimi protocol reports fixed and required reasoning modes",
          "[llm][effort][kimi]") {
    KimiProtocol protocol;
    for (const auto model : {"k3", "kimi-k3", "K3", "Kimi-K3"}) {
        CAPTURE(model);
        CHECK(supports(protocol, model, ReasoningCapability::Effort));
        CHECK(supports(protocol, model, ReasoningCapability::MaxEffort));
        CHECK(supports(protocol, model, ReasoningCapability::Required));
        CHECK(supports(protocol, model, ReasoningCapability::FixedMax));
    }
    CHECK_FALSE(supports(protocol, "kimi-k3-turbo", ReasoningCapability::Effort));

    for (const auto model : {"kimi-k2.7-code", "kimi-k2.7-code-highspeed",
                             "kimi-for-coding", "kimi-for-coding-highspeed"}) {
        CAPTURE(model);
        CHECK(supports(protocol, model, ReasoningCapability::Effort));
        CHECK(supports(protocol, model, ReasoningCapability::Required));
        CHECK_FALSE(supports(protocol, model, ReasoningCapability::Disable));
    }
    CHECK(supports(protocol, "kimi-k2.6", ReasoningCapability::Disable));
    CHECK(supports(protocol, "some-thinking-model", ReasoningCapability::Disable));
    CHECK_FALSE(supports(protocol, "moonshot-v1-8k", ReasoningCapability::Effort));
}

TEST_CASE("DashScope protocols own Qwen reasoning policy", "[llm][effort][qwen]") {
    DashScopeProtocol chat;
    DashScopeResponsesProtocol responses;
    for (const auto model : {"qwen3.8-max-preview", "qwen3.7-max",
                             "qwen3.7-plus", "qwen3.6-flash", "qwen3-coder-plus"}) {
        CAPTURE(model);
        for (const auto* protocol : {static_cast<ApiProtocolBase*>(&chat),
                                     static_cast<ApiProtocolBase*>(&responses)}) {
            CHECK(supports(*protocol, model, ReasoningCapability::Effort));
            CHECK(supports(*protocol, model, ReasoningCapability::MaxEffort));
            CHECK(supports(*protocol, model, ReasoningCapability::Disable));
        }
    }
    CHECK_FALSE(supports(chat, "qwen2.5-72b-instruct", ReasoningCapability::Effort));
    CHECK_FALSE(supports(chat, "glm-5.2", ReasoningCapability::Effort));
}

TEST_CASE("Anthropic protocol reports per-family effort levels",
          "[llm][effort][anthropic]") {
    AnthropicProtocol protocol;
    for (const auto model : {"claude-fable-5", "claude-mythos-1", "claude-sonnet-5",
                             "claude-opus-4-8", "claude-opus-4-7",
                             "claude-sonnet-4-6"}) {
        CAPTURE(model);
        CHECK(supports(protocol, model, ReasoningCapability::Effort));
        CHECK(supports(protocol, model, ReasoningCapability::MaxEffort));
    }
    CHECK(supports(protocol, "claude-fable-5", ReasoningCapability::XHighEffort));
    CHECK(supports(protocol, "claude-opus-4-7", ReasoningCapability::XHighEffort));
    CHECK_FALSE(supports(protocol, "claude-sonnet-4-6", ReasoningCapability::XHighEffort));
    CHECK_FALSE(supports(protocol, "claude-haiku-4-5", ReasoningCapability::Effort));
    CHECK_FALSE(supports(protocol, "claude-3-5-sonnet", ReasoningCapability::Effort));
}
