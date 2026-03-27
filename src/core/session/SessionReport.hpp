#pragma once

#include "SessionStats.hpp"
#include "core/budget/BudgetTracker.hpp"
#include <chrono>
#include <string>
#include <string_view>

namespace core::session {

// ---------------------------------------------------------------------------
// SessionReport — renders the end-of-session summary to stdout.
// Must be called AFTER the FTXUI loop exits (terminal in normal mode).
// Suppressed when zero turns were recorded.
// ---------------------------------------------------------------------------
class SessionReport {
public:
    static void print(const core::budget::BudgetTracker& budget,
                      const SessionStats::Snapshot& snap,
                      std::string_view session_id,
                      std::string_view session_file_path);

private:
    static std::string fmt_tokens(int32_t n);
    static std::string fmt_cost(double usd);
    static std::string fmt_duration(std::chrono::seconds s);
    static bool        supports_color() noexcept;
};

} // namespace core::session
