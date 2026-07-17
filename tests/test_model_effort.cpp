#include <catch2/catch_test_macros.hpp>

#include "core/llm/ModelEffort.hpp"

using namespace core::llm;

// ─────────────────────────────────────────────────────────────────────────────
// ModelEffort — single source of truth for effort/thinking capability
// predicates shared by the wire protocols and the TUI effort picker.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ModelEffort - OpenAI reasoning effort allow-list", "[llm][effort]") {
    CHECK(openai_model_supports_reasoning_effort("gpt-5"));
    CHECK(openai_model_supports_reasoning_effort("gpt-5-mini"));
    CHECK(openai_model_supports_reasoning_effort("o1-preview"));
    CHECK(openai_model_supports_reasoning_effort("o3"));
    CHECK(openai_model_supports_reasoning_effort("o4-mini"));
    CHECK(openai_model_supports_reasoning_effort("GPT-5")); // case-insensitive
    CHECK(openai_model_supports_max_effort("gpt-5.6-sol"));
    CHECK(openai_model_supports_max_effort("gpt-5.6-terra"));
    CHECK(openai_model_supports_max_effort("gpt-5.6-luna"));

    CHECK_FALSE(openai_model_supports_reasoning_effort("gpt-4o"));
    CHECK_FALSE(openai_model_supports_max_effort("gpt-5.5"));
    CHECK_FALSE(openai_model_supports_reasoning_effort("glm-5.2"));
    CHECK_FALSE(openai_model_supports_reasoning_effort(""));
}

TEST_CASE("ModelEffort - Kimi K3 identification", "[llm][effort][kimi]") {
    CHECK(kimi_model_is_k3("k3"));
    CHECK(kimi_model_is_k3("kimi-k3"));
    CHECK(kimi_model_is_k3("K3"));
    CHECK(kimi_model_is_k3("Kimi-K3"));

    CHECK_FALSE(kimi_model_is_k3("kimi-k3-turbo")); // exact ids only
    CHECK_FALSE(kimi_model_is_k3("kimi-k2.7-code"));
    CHECK_FALSE(kimi_model_is_k3(""));
}

TEST_CASE("ModelEffort - Mistral reasoning effort allow-list", "[llm][effort][mistral]") {
    CHECK(mistral_model_supports_reasoning_effort("mistral-vibe-cli-latest"));
    CHECK(mistral_model_supports_reasoning_effort("mistral-medium-3.5"));
    CHECK(mistral_model_supports_reasoning_effort("mistral-medium-latest"));
    CHECK(mistral_model_supports_reasoning_effort("mistral-small-2603"));
    CHECK(mistral_model_supports_reasoning_effort("MISTRAL-SMALL-LATEST"));
    CHECK(model_supports_max_effort("mistral-vibe-cli-latest"));

    CHECK_FALSE(mistral_model_supports_reasoning_effort("mistral-large-latest"));
    CHECK_FALSE(mistral_model_supports_reasoning_effort("codestral-latest"));
    CHECK_FALSE(mistral_model_supports_reasoning_effort(""));
}

TEST_CASE("ModelEffort - Kimi thinking support", "[llm][effort][kimi]") {
    CHECK(kimi_model_supports_thinking("k3"));
    CHECK(kimi_model_supports_thinking("kimi-k3"));
    CHECK(kimi_model_supports_thinking("kimi-for-coding"));
    CHECK(kimi_model_supports_thinking("kimi-for-coding-highspeed"));
    CHECK(kimi_model_supports_thinking("kimi-k2.7-code"));
    CHECK(kimi_model_supports_thinking("kimi-k2.6"));
    CHECK(kimi_model_supports_thinking("kimi-k2-5-thinking"));
    CHECK(kimi_model_supports_thinking("some-thinking-model"));

    CHECK_FALSE(kimi_model_supports_thinking("moonshot-v1-8k"));
    CHECK_FALSE(kimi_model_supports_thinking(""));
}

TEST_CASE("ModelEffort - Kimi always-on thinking models", "[llm][effort][kimi]") {
    CHECK(kimi_model_requires_thinking("kimi-k2.7-code"));
    CHECK(kimi_model_requires_thinking("kimi-k2.7-code-highspeed"));
    CHECK(kimi_model_requires_thinking("kimi-for-coding"));
    CHECK(kimi_model_requires_thinking("kimi-for-coding-highspeed"));

    // K3 is always-reasoning too but takes the dedicated K3 path.
    CHECK_FALSE(kimi_model_requires_thinking("k3"));
    CHECK_FALSE(kimi_model_requires_thinking("kimi-k2.6"));
}

TEST_CASE("ModelEffort - Anthropic effort families", "[llm][effort][anthropic]") {
    for (const auto model : {"claude-fable-5", "claude-mythos-1", "claude-sonnet-5",
                             "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"}) {
        CAPTURE(model);
        CHECK(anthropic_model_supports_effort(model));
        CHECK(anthropic_model_supports_max_effort(model));
    }

    // xhigh: everything above except sonnet-4-6.
    CHECK(anthropic_model_supports_xhigh_effort("claude-fable-5"));
    CHECK(anthropic_model_supports_xhigh_effort("claude-opus-4-7"));
    CHECK_FALSE(anthropic_model_supports_xhigh_effort("claude-sonnet-4-6"));

    CHECK_FALSE(anthropic_model_supports_effort("claude-haiku-4-5"));
    CHECK_FALSE(anthropic_model_supports_effort("claude-3-5-sonnet"));
    CHECK_FALSE(anthropic_model_supports_effort(""));
}

TEST_CASE("ModelEffort - cross-vendor max effort", "[llm][effort]") {
    // Anthropic max-capable families.
    CHECK(model_supports_max_effort("claude-fable-5"));
    CHECK(model_supports_max_effort("claude-sonnet-4-6"));
    // Kimi K3 is max-only.
    CHECK(model_supports_max_effort("k3"));
    CHECK(model_supports_max_effort("kimi-k3"));
    CHECK(model_supports_max_effort("gpt-5.6-sol"));

    CHECK_FALSE(model_supports_max_effort("kimi-k2.7-code"));
    CHECK_FALSE(model_supports_max_effort("gpt-5"));
    CHECK_FALSE(model_supports_max_effort("claude-haiku-4-5"));
}
