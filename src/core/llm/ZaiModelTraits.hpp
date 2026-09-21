#pragma once

#include "ReasoningCapabilities.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace core::llm {

[[nodiscard]] bool is_glm_model(std::string_view model) noexcept;
[[nodiscard]] bool is_glm_model_family(
    std::string_view model,
    std::string_view family) noexcept;
[[nodiscard]] bool is_glm_53_model(std::string_view model) noexcept;
[[nodiscard]] bool is_glm_52_model(std::string_view model) noexcept;

/// GLM models that accept video input (GLM-5.3-Flash per ZCode's model
/// capability matrix). Video parts degrade to placeholder text on every
/// other model.
[[nodiscard]] bool is_glm_multimodal_model(std::string_view model) noexcept;

[[nodiscard]] ReasoningCapabilities glm_reasoning_capabilities(
    std::string_view model) noexcept;

// OpenAI Chat Completions extras: thinking + optional reasoning_effort.
void append_glm_openai_thinking_fields(
    std::string& payload,
    std::string_view model,
    std::string_view effort);

// Anthropic Messages extras: thinking + optional output_config.effort.
void append_glm_anthropic_thinking_fields(
    std::string& payload,
    std::string_view model,
    std::string_view effort);

[[nodiscard]] bool zai_is_transient_business_code(int code) noexcept;
[[nodiscard]] bool zai_is_resettable_limit(int code) noexcept;

/// Extracts a BigModel/Z.AI business code from a stream-error payload.
/// Mirrors ZCode's readBigModelBracketedBusinessCode: SSE error chunks
/// sometimes carry only a "[1302][…][request_id]" message prefix, and some
/// gateways send the bare 4-digit code as the error type. Returns 0 when no
/// business code is present.
[[nodiscard]] int zai_stream_business_code(
    std::string_view error_type,
    std::string_view error_message) noexcept;

/// Z.ai business code for "conversation no longer fits the context window"
/// (ZCode maps it to a terminal ContextExceeded failure).
[[nodiscard]] bool zai_is_context_overflow_code(int code) noexcept;

/// Wording-based context-overflow detection for payloads that arrive
/// without a structured business code (Anthropic-style 400s, gateway
/// rewraps). Narrow on purpose: only phrases confirmed in ZCode's
/// classifier and the backend's own messages.
[[nodiscard]] bool zai_error_mentions_context_overflow(
    std::string_view message) noexcept;

/// True for the narrow HTTP 400 wording the GLM backend emits when replayed
/// thinking-block signatures are rejected. Mirrors ZCode's
/// isThinkingSignatureRejection; deliberately does not match generic
/// invalid_request errors.
[[nodiscard]] bool zai_error_is_signature_rejection(
    int http_status,
    std::string_view message) noexcept;

} // namespace core::llm
