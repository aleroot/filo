#pragma once

#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"
#include "ModelRegistry.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <optional>

#include <string>
#include <string_view>

namespace core::llm::claude {

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

// Shared policies for registry cards and generation/deployment fallbacks. An id
// takes the newest entry of its family whose version does not exceed its own
// within the same major, so an unlisted point release (`claude-fable-5-2`
// before it was added) inherits its predecessor's constraints rather than an
// older generation's (see model_policy).
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

inline constexpr std::array<ModelPolicy, 16>
    kModelPolicies{{
        // Fable/Mythos 5.2 keep 5.1's wire: adaptive thinking always on, no
        // sampling parameters, forced tool_choice rejected with a 400
        // (platform.claude.com/docs/en/models/fable-5-2/migration-guide).
        {"fable-5-2", kEffortMaxXHigh, false, false, kWireBoundThinking},
        {"mythos-5-2", kEffortMaxXHigh, false, false, kWireBoundThinking},
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

namespace detail {

struct GenerationVersion {
    std::array<unsigned, 4> parts{};
    std::size_t size = 0;

    [[nodiscard]] bool operator<(const GenerationVersion& other) const noexcept {
        return std::lexicographical_compare(parts.begin(), parts.begin() + size,
                                            other.parts.begin(),
                                            other.parts.begin() + other.size);
    }
};

/// Version segments after the `family` segment of `id`: 5,2 for
/// `claude-fable-5-2`. Stops at the first segment that is not a one- or
/// two-digit number, so a date snapshot (`-20260801`) is not a version.
/// nullopt when no segment of `id` is `family`.
[[nodiscard]] inline std::optional<GenerationVersion> version_after(
    std::string_view id, std::string_view family) {
    namespace ascii = core::utils::ascii;
    GenerationVersion version;
    bool found = false;
    std::size_t pos = 0;
    while (pos <= id.size()) {
        std::size_t end = pos;
        while (end < id.size() && ascii::is_alnum(static_cast<unsigned char>(id[end]))) ++end;
        const auto segment = id.substr(pos, end - pos);
        if (!found) {
            found = segment == family;
        } else {
            unsigned value = 0;
            if (segment.empty() || segment.size() > 2
                || version.size == version.parts.size()
                || std::from_chars(segment.data(), segment.data() + segment.size(), value).ptr
                       != segment.data() + segment.size()) {
                break;
            }
            version.parts[version.size++] = value;
        }
        pos = end + 1;
    }
    if (!found) return std::nullopt;
    return version;
}

} // namespace detail

/// Policy for a normalized Claude id: the newest entry of the id's family
/// whose version does not exceed the id's, within the same major. Point
/// releases have so far only added wire constraints (Fable 5 -> 5.1 and Opus
/// 5 -> 5.5 both started rejecting forced tool_choice), so an unlisted one is
/// safer with its predecessor's than with an older generation's. A new major
/// matches only an unversioned family entry, or nothing.
[[nodiscard]] inline const ModelPolicy* model_policy(
    std::string_view normalized) {
    if (normalized.empty()) return nullptr;
    if (normalized == "haiku" || normalized == "claude-haiku") normalized = kDefaultHaiku;
    if (normalized == "sonnet" || normalized == "claude-sonnet") normalized = kDefaultSonnet;
    if (normalized == "opus" || normalized == "claude-opus") normalized = kDefaultOpus;
    // Bare Fable aliases name the current Fable rather than a generation
    // substring, so they resolve by token instead of by version.
    if (is_fable_alias(normalized)) {
        for (const auto& candidate : kModelPolicies) {
            if (candidate.token == kCurrentFableToken) return &candidate;
        }
        return nullptr;
    }
    const ModelPolicy* best = nullptr;
    detail::GenerationVersion best_version;
    for (const auto& candidate : kModelPolicies) {
        const auto family = candidate.token.substr(0, candidate.token.find('-'));
        const auto requested = detail::version_after(normalized, family);
        if (!requested) continue;
        const auto offered = detail::version_after(candidate.token, family).value();
        if (offered.size > 0
            && (requested->size == 0 || offered.parts[0] != requested->parts[0])) {
            continue;
        }
        if (*requested < offered) continue;
        if (!best || best_version < offered) {
            best = &candidate;
            best_version = offered;
        }
    }
    return best;
}

} // namespace core::llm::claude
