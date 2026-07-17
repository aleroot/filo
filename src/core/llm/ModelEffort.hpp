#pragma once

/**
 * @file ModelEffort.hpp
 * @brief Single source of truth for "does this model accept effort/thinking
 *        controls on the wire?" predicates.
 *
 * Each wire protocol decides whether to emit vendor-specific reasoning fields
 * (`reasoning_effort`, `thinking`, Anthropic `output_config.effort`, …) for a
 * given model, and UI layers need the same answers to build effort pickers and
 * status lines. Keeping one implementation here prevents the drift that occurs
 * when serializers and the TUI each maintain their own copy of the model name
 * heuristics.
 *
 * Contract for every predicate:
 *  - Matching is ASCII case-insensitive on the raw model id.
 *  - Never allocates and never throws (`noexcept`).
 *  - Unknown models answer `false`; callers treat that as "send no vendor
 *    effort fields", which is always wire-safe.
 */

#include <string_view>

namespace core::llm {

/// OpenAI-style `reasoning_effort` (conservative allow-list so vendor fields
/// are never sent to unrelated OpenAI-compatible providers/models).
[[nodiscard]] bool openai_model_supports_reasoning_effort(std::string_view model) noexcept;

/// GPT-5.6 models accept the `max` reasoning effort without downgrading it.
[[nodiscard]] bool openai_model_supports_max_effort(std::string_view model) noexcept;

/// Current Mistral reasoning families accepting Mistral's `reasoning_effort`.
[[nodiscard]] bool mistral_model_supports_reasoning_effort(std::string_view model) noexcept;

/// Kimi K3 family (`kimi-k3` on the public API, `k3` on Kimi Code).
/// K3 is always-reasoning and controlled via top-level `reasoning_effort`.
[[nodiscard]] bool kimi_model_is_k3(std::string_view model) noexcept;

/// Any Kimi model that understands thinking controls (K3, K2.x, Kimi Code
/// coding models, or explicit "thinking" variants).
[[nodiscard]] bool kimi_model_supports_thinking(std::string_view model) noexcept;

/// Kimi models whose thinking cannot be switched off by the client.
[[nodiscard]] bool kimi_model_requires_thinking(std::string_view model) noexcept;

/// Anthropic models accepting the effort control at all.
[[nodiscard]] bool anthropic_model_supports_effort(std::string_view model) noexcept;

/// Anthropic models accepting `effort: "max"`.
[[nodiscard]] bool anthropic_model_supports_max_effort(std::string_view model) noexcept;

/// Anthropic models accepting `effort: "xhigh"`.
[[nodiscard]] bool anthropic_model_supports_xhigh_effort(std::string_view model) noexcept;

/// Cross-vendor answer for UI: may this model be configured with "max" effort?
/// Providers can preserve it (GPT-5.6, Anthropic, Kimi K3) or normalize it to
/// their highest supported wire value (Mistral maps it to `high`).
[[nodiscard]] bool model_supports_max_effort(std::string_view model) noexcept;

} // namespace core::llm
