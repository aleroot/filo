#pragma once

#include "SessionData.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace core::session {

class GoalManager {
public:
    using ClockFn = std::string (*)();
    static constexpr std::size_t kMaxObjectiveChars = 4096;
    static constexpr std::size_t kMaxNoteChars = 2048;

    explicit GoalManager(ClockFn clock);

    void restore(std::optional<SessionGoal> goal);
    [[nodiscard]] std::optional<SessionGoal> current() const;
    [[nodiscard]] std::optional<SessionGoal> active_goal() const;

    [[nodiscard]] bool has_goal() const noexcept;
    [[nodiscard]] bool has_active_goal() const noexcept;

    [[nodiscard]] std::optional<SessionGoal> set(std::string_view objective);
    [[nodiscard]] std::optional<SessionGoal> set_status(GoalStatus status,
                                                        std::string_view note);
    void clear() noexcept;

    [[nodiscard]] static std::string trim_copy(std::string_view value);
    [[nodiscard]] static std::string prompt_context(const std::optional<SessionGoal>& goal);

private:
    [[nodiscard]] static std::string clamp_copy(std::string_view value,
                                                std::size_t max_chars);
    [[nodiscard]] static std::optional<SessionGoal> normalize(
        std::optional<SessionGoal> goal);

    ClockFn clock_;
    std::optional<SessionGoal> goal_;
};

} // namespace core::session
