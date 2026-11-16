#pragma once

#include "ReasoningCapabilities.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace core::llm {

enum class ReasoningEffortResolutionKind {
    ProviderDefault,
    Exact,
    RequiredMinimum,
    Mapped,
    UnsupportedFallback,
    UnsupportedProviderDefault,
};

struct ReasoningEffortResolution {
    std::string effective;
    ReasoningEffortResolutionKind kind = ReasoningEffortResolutionKind::Exact;
};

// Resolve a configured effort to the value supported by a model. Capability
// interpretation belongs to the LLM layer so callers only need to present the
// resulting value and resolution kind.
[[nodiscard]] ReasoningEffortResolution resolve_reasoning_effort(
    std::string_view configured,
    const ReasoningCapabilities& capabilities);

/// True for the effort spellings that ask for no reasoning ("off", "none",
/// "disabled"), ignoring case and surrounding whitespace.
[[nodiscard]] bool is_reasoning_off_effort(std::string_view effort);

/// The effort levels a user can select, strongest last (`auto` and `off` aside).
inline constexpr std::string_view kSelectableEffortLevels =
    "auto, off, low, medium, high, xhigh, max, ultra";

/// Parse a user's effort selection (`/effort`, `--effort`) into the value an
/// Agent carries: "" for auto (provider/catalog default), "none" for off,
/// else the level. Case, spaces, '-' and '_' are ignored. nullopt = unknown.
[[nodiscard]] std::optional<std::string> parse_effort_selection(std::string_view requested);

/// The effort a prompt containing the `ultrathink` keyword runs at.
inline constexpr std::string_view kUltrathinkEffort = "xhigh";

/// Set to a non-empty value other than "0"/"false" to ignore the keyword.
inline constexpr std::string_view kDisableUltrathinkEnv = "FILO_DISABLE_ULTRATHINK";

/// True when the user wrote the word `ultrathink` (any case) as an
/// instruction: not quoted or bracketed, not inside a backtick code span or
/// fence, not part of a longer word, identifier, file name or path, and not in
/// a prompt that starts with `/` (a command) or `!` (a shell command).
[[nodiscard]] bool mentions_ultrathink(std::string_view prompt);

/// mentions_ultrathink() unless FILO_DISABLE_ULTRATHINK is set.
[[nodiscard]] bool ultrathink_requested(std::string_view prompt);

/// `floor` (lowercased) when it is a stronger effort than `current`, where
/// auto ("") and off rank below every level. nullopt when `current` already
/// meets it or either value is unrecognised, so a raise never lowers effort.
[[nodiscard]] std::optional<std::string> raised_effort(
    std::string_view current, std::string_view floor);

} // namespace core::llm
