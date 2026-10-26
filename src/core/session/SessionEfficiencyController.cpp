#include "SessionEfficiencyController.hpp"

#include <algorithm>
#include <format>

namespace core::session {

void SessionEfficiencyController::reset() noexcept {
    baseline_prompt_tokens_ = {};
    recent_prompt_tokens_ = {};
    latest_ = {};
    turn_count_ = 0;
}

void SessionEfficiencyController::record_turn(const SessionEfficiencyObservation& observation) noexcept {
    if (turn_count_ < kBaselineWindow)
        baseline_prompt_tokens_[turn_count_] = observation.prompt_tokens;
    recent_prompt_tokens_[turn_count_ % kCurrentWindow] = observation.prompt_tokens;
    latest_ = observation;
    ++turn_count_;
}

double SessionEfficiencyController::average_prompt_tokens(
    std::span<const int32_t> samples) noexcept {
    if (samples.empty()) return 0.0;
    double total = 0.0;
    for (const auto tokens : samples)
        total += static_cast<double>(std::max(tokens, 0));
    return total / static_cast<double>(samples.size());
}

SessionEfficiencyDecision SessionEfficiencyController::current_decision() const {
    SessionEfficiencyDecision decision;
    decision.turn_count = static_cast<int>(turn_count_);
    if (turn_count_ == 0) {
        return decision;
    }

    const auto& latest = latest_;
    decision.estimated_history_tokens = latest.estimated_history_tokens;
    if (latest.max_context_tokens > 0) {
        decision.context_utilization =
            static_cast<double>(latest.estimated_history_tokens)
            / static_cast<double>(latest.max_context_tokens);
    }

    const double baseline = average_prompt_tokens(
        std::span(baseline_prompt_tokens_).first(std::min(kBaselineWindow, turn_count_)));
    const double current = average_prompt_tokens(
        std::span(recent_prompt_tokens_).first(std::min(kCurrentWindow, turn_count_)));
    if (baseline > 0.0 && current > 0.0) {
        decision.waste_factor = current / baseline;
    }

    // Provider-native prompt caching can absorb some growth, so be slightly
    // less aggressive when the active backend already optimizes stable prefixes.
    const double waste_rotate_threshold =
        latest.provider_supports_prompt_caching ? 6.5 : 4.5;
    const double local_waste_rotate_threshold = 4.0;
    const double context_rotate_threshold =
        latest.provider_is_local ? 0.72 : 0.78;
    const int min_turns_before_rotation =
        latest.provider_is_local ? 8 : 7;

    const bool history_is_heavy = latest.provider_is_local
        ? latest.estimated_history_tokens >= 24'000
        : latest.estimated_history_tokens >= 30'000;
    const bool waste_is_high = decision.turn_count >= min_turns_before_rotation
        && decision.waste_factor >= (latest.provider_is_local
            ? local_waste_rotate_threshold
            : waste_rotate_threshold);
    const bool context_is_high = latest.max_context_tokens > 0
        && decision.turn_count >= 6
        && decision.context_utilization >= context_rotate_threshold;

    if (!history_is_heavy || (!waste_is_high && !context_is_high)) {
        return decision;
    }

    decision.action = SessionEfficiencyDecision::Action::Rotate;
    if (context_is_high && waste_is_high) {
        decision.reason = std::format(
            "Session history reached {:.0f}% of the model context and prompts grew to {:.1f}x the baseline.",
            decision.context_utilization * 100.0,
            decision.waste_factor);
    } else if (context_is_high) {
        decision.reason = std::format(
            "Session history reached {:.0f}% of the model context window.",
            decision.context_utilization * 100.0);
    } else {
        decision.reason = std::format(
            "Prompt size grew to {:.1f}x the early-session baseline.",
            decision.waste_factor);
    }
    return decision;
}

} // namespace core::session
