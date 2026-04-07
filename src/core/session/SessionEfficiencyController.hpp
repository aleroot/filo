#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

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
    void record_turn(const SessionEfficiencyObservation& observation);

    [[nodiscard]] SessionEfficiencyDecision current_decision() const;

private:
    struct TurnSample {
        int32_t     prompt_tokens = 0;
        int32_t     completion_tokens = 0;
        std::size_t estimated_history_tokens = 0;
        int32_t     max_context_tokens = 0;
        bool        provider_is_local = false;
        bool        provider_supports_prompt_caching = false;
    };

    [[nodiscard]] static double average_prompt_tokens(
        const std::vector<TurnSample>& turns,
        std::size_t begin,
        std::size_t end) noexcept;

    std::vector<TurnSample> turns_;
};

} // namespace core::session
