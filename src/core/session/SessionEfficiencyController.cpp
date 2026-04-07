#include "SessionEfficiencyController.hpp"

#include <algorithm>
#include <format>

namespace core::session {

namespace {

constexpr std::size_t kBaselineWindow = 4;
constexpr std::size_t kCurrentWindow = 4;

} // namespace

void SessionEfficiencyController::reset() noexcept {
    turns_.clear();
}

void SessionEfficiencyController::record_turn(const SessionEfficiencyObservation& observation) {
    turns_.push_back(TurnSample{
        .prompt_tokens = observation.prompt_tokens,
        .completion_tokens = observation.completion_tokens,
        .estimated_history_tokens = observation.estimated_history_tokens,
        .max_context_tokens = observation.max_context_tokens,
        .provider_is_local = observation.provider_is_local,
        .provider_supports_prompt_caching = observation.provider_supports_prompt_caching,
    });
}

double SessionEfficiencyController::average_prompt_tokens(
    const std::vector<TurnSample>& turns,
    std::size_t begin,
    std::size_t end) noexcept {
    if (begin >= end || begin >= turns.size()) {
        return 0.0;
    }
    end = std::min(end, turns.size());
    double total = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        total += static_cast<double>(std::max(turns[i].prompt_tokens, 0));
    }
    return total / static_cast<double>(end - begin);
}

SessionEfficiencyDecision SessionEfficiencyController::current_decision() const {
    SessionEfficiencyDecision decision;
    decision.turn_count = static_cast<int>(turns_.size());
    if (turns_.empty()) {
        return decision;
    }

    const TurnSample& latest = turns_.back();
    decision.estimated_history_tokens = latest.estimated_history_tokens;
    if (latest.max_context_tokens > 0) {
        decision.context_utilization =
            static_cast<double>(latest.estimated_history_tokens)
            / static_cast<double>(latest.max_context_tokens);
    }

    const std::size_t baseline_end = std::min(kBaselineWindow, turns_.size());
    const std::size_t current_begin = turns_.size() > kCurrentWindow
        ? turns_.size() - kCurrentWindow
        : 0;
    const double baseline = average_prompt_tokens(turns_, 0, baseline_end);
    const double current = average_prompt_tokens(turns_, current_begin, turns_.size());
    if (baseline > 0.0 && current > 0.0) {
        decision.waste_factor = current / baseline;
    }

    // Provider-native prompt caching can absorb some growth, so be slightly
    // less aggressive when the active backend already optimizes stable prefixes.
    const double waste_rotate_threshold =
        latest.provider_supports_prompt_caching ? 7.0 : 5.0;
    const double local_waste_rotate_threshold = 4.0;
    const double context_rotate_threshold =
        latest.provider_is_local ? 0.72 : 0.82;
    const int min_turns_before_rotation =
        latest.provider_is_local ? 8 : 10;

    const bool history_is_heavy = latest.provider_is_local
        ? latest.estimated_history_tokens >= 24'000
        : latest.estimated_history_tokens >= 40'000;
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
