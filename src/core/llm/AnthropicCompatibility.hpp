#pragma once

#include "../utils/StringUtils.hpp"
#include "ModelRegistry.hpp"
#include <array>

#include <string>
#include <string_view>

namespace core::llm::anthropic {

inline constexpr std::string_view kBillingHeader = "cc_version=2.1.280.13b; cc_entrypoint=cli; cch=0;";
inline constexpr std::string_view kThinkingBindingBeta =
    "thinking-binding-controls-2026-08-01";

// Canonical id behind each short Claude alias. The Messages API accepts only
// these ids, so aliases have to be resolved before the request is written.
inline constexpr std::string_view kDefaultSonnet = "claude-sonnet-5";
inline constexpr std::string_view kDefaultFable  = "claude-fable-5-1";
inline constexpr std::string_view kDefaultOpus   = "claude-opus-5";
inline constexpr std::string_view kDefaultOpus55 = "claude-opus-5-5";
inline constexpr std::string_view kDefaultHaiku  = "claude-haiku-5-5";

// Id/alias shaping only. Which request fields a model accepts is metadata, not
// a name: see AnthropicWirePolicy and ModelWireConstraints.
[[nodiscard]] inline bool is_fable_alias(std::string_view model) {
    return model == "fable" || model == "best" || model == "claude-fable"
        || model == "fable-5-1";
}

[[nodiscard]] inline bool is_opus_55_alias(std::string_view model) {
    return model == "opus-5-5" || model == "opus-5.5"
        || model == "claude-opus-5.5";
}

/// Lowercased id with a trailing `[1m]` context suffix removed.
[[nodiscard]] inline std::string normalized_claude_id(std::string_view model) {
    auto normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_copy(model));
    if (normalized.ends_with("[1m]")) normalized.resize(normalized.size() - 4);
    return normalized;
}

// Shared policies for registry cards and generation/deployment fallbacks. Ordered most
// specific first, and matched on generation boundaries so `fable-5` never
// claims `claude-fable-5-1`.
struct ModelPolicy {
    std::string_view token;
    ReasoningCapabilities effort;
    bool adaptive_thinking;
    bool manual_thinking;
    ModelWireConstraints wire;
    [[nodiscard]] constexpr ModelReasoningProfile reasoning() const noexcept {
        return {effort, adaptive_thinking, manual_thinking, true};
    }
};

inline constexpr ModelWireConstraints kWireNone = {};
inline constexpr ModelWireConstraints kWireAdaptive = {
    .reasoning_text_hidden = true,
    .fixed_sampling = true,
};
inline constexpr ModelWireConstraints kWireNoForcedTools = {
    .reasoning_text_hidden = true,
    .fixed_sampling = true,
    .forced_tool_choice_rejected = true,
};
inline constexpr ModelWireConstraints kWireBoundThinking = {
    .thinking_always_on = true,
    .reasoning_text_hidden = true,
    .reasoning_bound_to_prefix = true,
    .fixed_sampling = true,
    .forced_tool_choice_rejected = true,
};

inline constexpr ModelWireConstraints kWireHaiku55 = {
    .reasoning_text_hidden = true,
    .reasoning_bound_to_prefix = true,
    .fixed_sampling = true,
    .disabled_thinking = true,
};
inline constexpr ModelWireConstraints kWireSonnet55 = {
    .reasoning_text_hidden = true,
    .reasoning_bound_to_prefix = true,
    .fixed_sampling = true,
    .forced_tool_choice_rejected = true,
    .between_tools_thinking = true,
};

inline constexpr ReasoningCapabilities kEffortMax =
    ReasoningCapability::Effort | ReasoningCapability::MaxEffort;
inline constexpr ReasoningCapabilities kEffortMaxXHigh =
    kEffortMax | ReasoningCapability::XHighEffort;

inline constexpr std::array<ModelPolicy, 14>
    kModelPolicies{{
        {"fable-5-1", kEffortMaxXHigh, false, false, kWireBoundThinking},
        {"mythos-5-1", kEffortMaxXHigh, false, false, kWireNoForcedTools},
        {"fable-5", kEffortMaxXHigh, false, false, kWireAdaptive},
        {"mythos-5", kEffortMaxXHigh, false, false, kWireAdaptive},
        {"opus-5-5", kEffortMaxXHigh, true, false, kWireBoundThinking},
        {"haiku-5-5", kEffortMaxXHigh, true, false, kWireHaiku55},
        {"sonnet-5-5", kEffortMaxXHigh, true, false, kWireSonnet55},
        {"sonnet-5", kEffortMaxXHigh, false, false, kWireAdaptive},
        {"opus-4-8", kEffortMaxXHigh, true, false, kWireAdaptive},
        {"opus-4-7", kEffortMaxXHigh, true, false, kWireAdaptive},
        {"opus-5", kEffortMaxXHigh, true, false, kWireAdaptive},
        {"sonnet-4-6", kEffortMax, false, true, kWireNone},
        // Unversioned family ids keep the family's adaptive wire.
        {"fable", kEffortMaxXHigh, false, false, kWireAdaptive},
        {"mythos", kEffortMaxXHigh, false, false, kWireAdaptive},
    }};

/// True when `token` names the whole id or one generation segment of it.
[[nodiscard]] inline bool matches_generation(std::string_view normalized,
                                      std::string_view token) {
    const std::size_t start = normalized.find(token);
    if (start == std::string_view::npos) return false;
    const std::size_t end = start + token.size();
    return end == normalized.size()
        || normalized[end] == '-'
        || normalized[end] == '[';
}

/// Token of the generation the bare Fable aliases resolve to.
inline constexpr std::string_view kCurrentFableToken = "fable-5-1";

[[nodiscard]] inline const ModelPolicy* model_policy(
    std::string_view normalized) {
    if (normalized.empty()) return nullptr;
    if (normalized == "haiku" || normalized == "claude-haiku") normalized = kDefaultHaiku;
    if (normalized == "sonnet" || normalized == "claude-sonnet") normalized = kDefaultSonnet;
    if (normalized == "opus" || normalized == "claude-opus") normalized = kDefaultOpus;
    // Bare Fable aliases name the current Fable rather than a generation
    // substring, so they resolve by token instead of by table position.
    const bool bare_fable_alias = is_fable_alias(normalized);
    for (const auto& candidate : kModelPolicies) {
        if (bare_fable_alias) {
            if (candidate.token == kCurrentFableToken) return &candidate;
            continue;
        }
        if (matches_generation(normalized, candidate.token)) {
            return &candidate;
        }
    }
    return nullptr;
}

} // namespace core::llm::anthropic
