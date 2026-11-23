#include <catch2/catch_test_macros.hpp>

#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/MistralProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/ReasoningEffortResolution.hpp"
#include "TestEnvironment.hpp"

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
        CHECK(supports(*protocol, "gpt-5.6-sol", ReasoningCapability::XHighEffort));
        CHECK(supports(*protocol, "gpt-5.6-sol", ReasoningCapability::MaxEffort));
        CHECK(supports(*protocol, "gpt-5.6-sol", ReasoningCapability::Disable));
        CHECK(supports(*protocol, "gpt-5", ReasoningCapability::MinimalEffort));
        CHECK_FALSE(supports(*protocol, "gpt-5", ReasoningCapability::XHighEffort));
        CHECK_FALSE(supports(*protocol, "gpt-5", ReasoningCapability::Disable));
        CHECK_FALSE(supports(*protocol, "gpt-5.6-sol", ReasoningCapability::MinimalEffort));
        CHECK_FALSE(supports(*protocol, "gpt-5.6-sol", ReasoningCapability::UltraEffort));
        CHECK_FALSE(supports(*protocol, "gpt-5.5", ReasoningCapability::MaxEffort));
        CHECK_FALSE(supports(*protocol, "gpt-4o", ReasoningCapability::Effort));
        CHECK_FALSE(supports(*protocol, "glm-5.2", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "gpt-6", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "gpt-6.1", ReasoningCapability::XHighEffort));
        CHECK(supports(*protocol, "gpt-6.1-sol", ReasoningCapability::MaxEffort));
        CHECK_FALSE(supports(*protocol, "gpt-6.1", ReasoningCapability::UltraEffort));
        // Verified against the live backend: GPT-6.1 rejects `none` and `minimal`.
        CHECK_FALSE(supports(*protocol, "gpt-6.1-terra", ReasoningCapability::Disable));
        CHECK_FALSE(supports(*protocol, "gpt-6.1-terra", ReasoningCapability::MinimalEffort));
        // GPT-6.2 rejects `none` the same way. `minimal` is not a 6.2 value.
        CHECK(supports(*protocol, "gpt-6.2-sol", ReasoningCapability::MaxEffort));
        CHECK(supports(*protocol, "gpt-6.2-astra", ReasoningCapability::XHighEffort));
        CHECK_FALSE(supports(*protocol, "gpt-6.2-sol", ReasoningCapability::Disable));
        CHECK_FALSE(supports(*protocol, "gpt-6.2-astra", ReasoningCapability::MinimalEffort));
        CHECK_FALSE(supports(*protocol, "gpt-6.2-sol", ReasoningCapability::UltraEffort));
    }

    // `ultra` is a Codex client mode, not a wire value: the API rejects it.
    CodexResponsesProtocol codex;
    CHECK_FALSE(supports(codex, "gpt-5.6-sol", ReasoningCapability::UltraEffort));
    CHECK_FALSE(supports(codex, "gpt-6.1-terra", ReasoningCapability::UltraEffort));
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

TEST_CASE("Kimi protocol reports selectable and required reasoning modes",
          "[llm][effort][kimi]") {
    KimiProtocol protocol;
    for (const auto model : {"k3", "k3-256k", "kimi-k3", "K3", "Kimi-K3",
                             "kimi-k3.1-pro", "kimi-k3.1-pro-highspeed"}) {
        CAPTURE(model);
        CHECK(supports(protocol, model, ReasoningCapability::Effort));
        CHECK(supports(protocol, model, ReasoningCapability::MaxEffort));
        CHECK(supports(protocol, model, ReasoningCapability::Required));
        CHECK_FALSE(supports(protocol, model, ReasoningCapability::FixedMax));
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

TEST_CASE("DashScope protocols own endpoint-specific model reasoning policy", "[llm][effort][qwen]") {
    DashScopeProtocol chat;
    DashScopeResponsesProtocol responses;
    for (const auto* protocol : {static_cast<ApiProtocolBase*>(&chat), static_cast<ApiProtocolBase*>(&responses)}) {
        CHECK(supports(*protocol, "qwen3.8-max", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "qwen3.8-max", ReasoningCapability::Required));
        CHECK_FALSE(supports(*protocol, "qwen3.8-max", ReasoningCapability::Disable));
        CHECK_FALSE(supports(*protocol, "qwen3.8-max", ReasoningCapability::MaxEffort));
        CHECK(supports(*protocol, "qwen3.8-flash", ReasoningCapability::Disable));
        for (const auto model : {"qwen3.7-max", "qwen3.7-plus", "qwen3.6-flash"}) {
            CHECK_FALSE(supports(*protocol, model, ReasoningCapability::Effort));
            CHECK(supports(*protocol, model, ReasoningCapability::ToggleOnly));
        }
        CHECK_FALSE(supports(*protocol, "qwen3-coder-plus", ReasoningCapability::Effort));
        CHECK_FALSE(supports(*protocol, "glm-5.2", ReasoningCapability::Effort));
        CHECK(supports(*protocol, "glm-5.2", ReasoningCapability::ToggleOnly));
        CHECK(supports(*protocol, "ZHIPU/GLM-5.3", ReasoningCapability::Required));
    }
    CHECK_FALSE(supports(chat, "qwen2.5-72b-instruct", ReasoningCapability::Effort));
}

TEST_CASE("Anthropic protocol reports per-family effort levels",
          "[llm][effort][anthropic]") {
    ClaudeProtocol protocol;
    for (const auto model : {"claude-fable-5", "claude-mythos-1", "claude-sonnet-5",
                             "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
                             "claude-sonnet-4-6"}) {
        CAPTURE(model);
        CHECK(supports(protocol, model, ReasoningCapability::Effort));
        CHECK(supports(protocol, model, ReasoningCapability::MaxEffort));
    }
    CHECK(supports(protocol, "claude-fable-5", ReasoningCapability::XHighEffort));
    CHECK(supports(protocol, "claude-opus-4-7", ReasoningCapability::XHighEffort));
    CHECK(supports(protocol, "claude-opus-5-5", ReasoningCapability::XHighEffort));
    CHECK(supports(protocol, "claude-opus-5-5", ReasoningCapability::MaxEffort));
    CHECK(supports(protocol, "claude-opus-5-5", ReasoningCapability::Required));
    CHECK_FALSE(supports(protocol, "claude-opus-5", ReasoningCapability::Required));
    CHECK_FALSE(supports(protocol, "claude-sonnet-4-6", ReasoningCapability::XHighEffort));
    CHECK_FALSE(supports(protocol, "claude-haiku-4-5", ReasoningCapability::Effort));
    CHECK_FALSE(supports(protocol, "claude-3-5-sonnet", ReasoningCapability::Effort));
}

TEST_CASE("effective reasoning effort is resolved from model capabilities",
          "[llm][effort][resolution]") {
    using Kind = ReasoningEffortResolutionKind;

    CHECK(resolve_reasoning_effort("", {}).kind == Kind::ProviderDefault);

    const auto off_supported = resolve_reasoning_effort(
        "off", ReasoningCapabilities{ReasoningCapability::Disable});
    CHECK(off_supported.effective == "off");
    CHECK(off_supported.kind == Kind::Exact);

    const auto off_required = resolve_reasoning_effort(
        "none", ReasoningCapability::Required | ReasoningCapability::Effort);
    CHECK(off_required.effective == "low");
    CHECK(off_required.kind == Kind::RequiredMinimum);

    const auto off_unknown = resolve_reasoning_effort("off", {});
    CHECK(off_unknown.effective == "provider default");
    CHECK(off_unknown.kind == Kind::UnsupportedProviderDefault);

    const auto low_mapped = resolve_reasoning_effort(
        "low", ReasoningCapabilities{ReasoningCapability::MapsLowToHigh});
    CHECK(low_mapped.effective == "high");
    CHECK(low_mapped.kind == Kind::Mapped);

    const auto minimal_prefers_low = resolve_reasoning_effort(
        "minimal",
        ReasoningCapability::MapsMinimalToLow | ReasoningCapability::MapsMinimalToOff);
    CHECK(minimal_prefers_low.effective == "low");
    CHECK(minimal_prefers_low.kind == Kind::Mapped);

    const auto minimal_maps_off = resolve_reasoning_effort(
        "minimal", ReasoningCapabilities{ReasoningCapability::MapsMinimalToOff});
    CHECK(minimal_maps_off.effective == "off");
    CHECK(minimal_maps_off.kind == Kind::Mapped);

    const auto xhigh_maps_max = resolve_reasoning_effort(
        "xhigh",
        ReasoningCapability::XHighEffort | ReasoningCapability::MapsXHighToMax);
    CHECK(xhigh_maps_max.effective == "max");
    CHECK(xhigh_maps_max.kind == Kind::Mapped);

    const auto max_fallback = resolve_reasoning_effort("max", {});
    CHECK(max_fallback.effective == "high");
    CHECK(max_fallback.kind == Kind::UnsupportedFallback);

    const auto ultra_to_max = resolve_reasoning_effort(
        "ultra", ReasoningCapabilities{ReasoningCapability::MaxEffort});
    CHECK(ultra_to_max.effective == "max");
    CHECK(ultra_to_max.kind == Kind::UnsupportedFallback);

    const auto ultra_to_high = resolve_reasoning_effort("ultra", {});
    CHECK(ultra_to_high.effective == "high");
    CHECK(ultra_to_high.kind == Kind::UnsupportedFallback);
}

TEST_CASE("Effort selections parse the same for /effort and --effort", "[llm][effort][cli]") {
    CHECK(parse_effort_selection("ultra") == "ultra");
    CHECK(parse_effort_selection(" X-High ") == "xhigh");
    CHECK(parse_effort_selection("x_high") == "xhigh");
    CHECK(parse_effort_selection("MAX") == "max");
    CHECK(parse_effort_selection("auto") == "");
    CHECK(parse_effort_selection("default") == "");
    CHECK(parse_effort_selection("off") == "none");
    CHECK(parse_effort_selection("disabled") == "none");
    CHECK_FALSE(parse_effort_selection("extreme"));
    CHECK_FALSE(parse_effort_selection("   "));
}

TEST_CASE("ultrathink is recognised only as a plain word the user wrote", "[llm][effort][ultrathink]") {
    for (const std::string_view prompt : {
             "ultrathink",
             "ULTRATHINK fix the flaky test",
             "please ultrathink: why does the cache miss?",
             "Fix the race. Ultrathink.",
             "(see below) ultrathink",
             "look at it,\nultrathink!",
             "fix the \"foo\" bug, ultrathink",
             "`ultrathink` is the keyword, so ultrathink",
             "an unclosed `tick, ultrathink",
         }) {
        INFO(prompt);
        CHECK(mentions_ultrathink(prompt));
    }
    for (const std::string_view prompt : {
             "",
             "think hard about it",
             "stop ultrathinking",
             "the ultrathinker",
             "`ultrathink`",
             "a ``code ` ultrathink`` span",
             "```\nultrathink\n```",
             "what does \"ultrathink\" do?",
             "say \"please ultrathink now\" to it",
             "what does 'ultrathink' do?",
             "explain the \xe2\x80\x9cultrathink\xe2\x80\x9d keyword",
             "explain the \xe2\x80\x98ultrathink\xe2\x80\x99 keyword",
             "open docs/ultrathink.md",
             "open ultrathink.md",
             "see @ultrathink",
             "run with --ultrathink",
             "ultrathink-mode",
             "ultrathink/notes",
             "my_ultrathink",
             "echo $ultrathink",
             "(ultrathink) then refactor",
             "fix it [ultrathink]",
             "use {ultrathink} here",
             "the <ultrathink> tag",
             "/review ultrathink",
             "  /goal fix it, ultrathink",
             "!echo ultrathink",
         }) {
        INFO(prompt);
        CHECK_FALSE(mentions_ultrathink(prompt));
    }
}

TEST_CASE("raised_effort only ever raises", "[llm][effort][ultrathink]") {
    for (const std::string_view session : {"", "none", "off", "minimal", "low", "medium", "high", " High "}) {
        INFO(session);
        CHECK(raised_effort(session, kUltrathinkEffort) == "xhigh");
    }
    for (const std::string_view session : {"xhigh", "max", "ultra", "extreme"}) {
        INFO(session);
        CHECK_FALSE(raised_effort(session, kUltrathinkEffort));
    }
    CHECK(raised_effort("high", " XHigh ") == "xhigh");
    CHECK_FALSE(raised_effort("low", ""));
    CHECK_FALSE(raised_effort("low", "bogus"));
}

TEST_CASE("FILO_DISABLE_ULTRATHINK turns the keyword off", "[llm][effort][ultrathink]") {
    {
        const test_support::ScopedEnvVar disabled(std::string(kDisableUltrathinkEnv), "1");
        CHECK(mentions_ultrathink("ultrathink"));
        CHECK_FALSE(ultrathink_requested("ultrathink"));
    }
    for (const std::string_view off : {"0", "false", ""}) {
        const test_support::ScopedEnvVar enabled{std::string(kDisableUltrathinkEnv), std::string(off)};
        INFO(off);
        CHECK(ultrathink_requested("ultrathink"));
    }
}

TEST_CASE("ultrathink_spans locates each occurrence mentions_ultrathink counts",
          "[llm][ultrathink]") {
    const std::string prompt = "Ultrathink this, not `ultrathink` or ultrathinking, then ultrathink now";
    const auto spans = core::llm::ultrathink_spans(prompt);
    REQUIRE(spans.size() == 2);
    CHECK(prompt.substr(spans[0].first, spans[0].second - spans[0].first) == "Ultrathink");
    CHECK(prompt.substr(spans[1].first, spans[1].second - spans[1].first) == "ultrathink");
    CHECK(core::llm::mentions_ultrathink(prompt));
    CHECK(core::llm::ultrathink_spans("/effort high ultrathink").empty());
}
