#pragma once

#include "../utils/StringUtils.hpp"

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
inline constexpr std::string_view kDefaultHaiku  = "claude-haiku-4-5";

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

} // namespace core::llm::anthropic
