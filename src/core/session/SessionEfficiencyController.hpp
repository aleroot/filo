#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace core::session {

struct SessionEfficiencyObservation {
    int32_t     prompt_tokens = 0;
    int32_t     completion_tokens = 0;
    std::size_t estimated_history_tokens = 0;
    int32_t     max_context_tokens = 0;
    bool        provider_is_local = false;
    bool        provider_supports_prompt_caching = false;
};

struct SessionEfficiencyDecision {
    enum class Action {
        None,
        Rotate,
    };

    Action      action = Action::None;
    int         turn_count = 0;
    double      waste_factor = 1.0;
    double      context_utilization = 0.0;
    std::size_t estimated_history_tokens = 0;
    std::string reason;
};

class SessionEfficiencyController {
public:
    void reset() noexcept;
    void record_turn(const SessionEfficiencyObservation& observation) noexcept;

    [[nodiscard]] SessionEfficiencyDecision current_decision() const;

private:
    static constexpr std::size_t kBaselineWindow = 4;
    static constexpr std::size_t kCurrentWindow = 4;

    [[nodiscard]] static double average_prompt_tokens(
        std::span<const int32_t> samples) noexcept;

    // The policy uses only the first and latest four prompt counts, plus the
    // latest observation. Retaining every prior observation adds no evidence.
    std::array<int32_t, kBaselineWindow> baseline_prompt_tokens_{};
    std::array<int32_t, kCurrentWindow> recent_prompt_tokens_{};
    SessionEfficiencyObservation latest_{};
    std::size_t turn_count_ = 0;
};

} // namespace core::session
