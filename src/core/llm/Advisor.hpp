#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace core::llm {

/**
 * @brief A provider-side advisor: a second model the provider runs inside the
 * same turn, on the provider's own credential and billing.
 *
 * An advisor is a provider capability, not an agent feature: only a provider's
 * own wire can run one. A request therefore carries a provider-neutral
 * specification and the provider's adapter renders it, so the core never
 * learns how any vendor declares an advisor.
 */
struct AdvisorSpec {
    /// The advisor model, in the provider's own namespace.
    std::string model = {};
    /// Advisor calls per turn. Zero asks for the provider's default.
    int max_uses = 0;
    /// The advisor's output cap per call. Zero asks for the provider's default.
    int max_output_tokens = 0;
};

/// What one advisor consultation produced.
enum class AdvisorCallState {
    Pending,     ///< The provider started the consultation.
    Reviewed,    ///< The advisor returned guidance.
    Unavailable, ///< The consultation failed; `AdvisorEvent::error_code` says why.
    Refused,     ///< The provider declined to run the advisor at all.
};

[[nodiscard]] constexpr std::string_view to_string(AdvisorCallState state) noexcept {
    switch (state) {
        case AdvisorCallState::Pending: return "pending";
        case AdvisorCallState::Reviewed: return "reviewed";
        case AdvisorCallState::Unavailable: return "unavailable";
        case AdvisorCallState::Refused: return "refused";
    }
    return "unknown";
}

/// One consultation, as the provider reported it.
struct AdvisorEvent {
    AdvisorCallState state = AdvisorCallState::Pending;
    /// The advisor that ran, when the provider named it.
    std::string advisor_model = {};
    /// The provider's error code, set when the consultation failed.
    std::string error_code = {};
    /// The provider's own words about what happened, when it gave any. It is
    /// the reason a refusal or a failure carries: an account that cannot run
    /// an advisor says so here.
    std::string detail = {};
    /// False when the provider keeps the advice for itself: the executor
    /// still receives the guidance, but the client cannot read it.
    bool guidance_readable = false;
};

/// What a turn will do about an advisor.
struct AdvisorPlan {
    /// The advisor to declare, when one runs.
    std::optional<AdvisorSpec> spec;
    /// Why no advisor runs; empty when one does.
    std::string refusal;

    [[nodiscard]] bool has_advisor() const noexcept { return spec.has_value(); }
};

/// Normalizes an advisor selection the user typed: "off" and "none" ask for
/// no advisor, anything else names one. Empty when no advisor is asked for.
[[nodiscard]] inline std::optional<std::string> parse_advisor_selection(
    std::string_view value) {
    std::string trimmed;
    trimmed.reserve(value.size());
    for (const char ch : value) {
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
        trimmed.push_back(static_cast<char>(
            (ch >= 'A' && ch <= 'Z') ? ch - 'A' + 'a' : ch));
    }
    if (trimmed.empty() || trimmed == "off" || trimmed == "none") return std::nullopt;
    return trimmed;
}

} // namespace core::llm
