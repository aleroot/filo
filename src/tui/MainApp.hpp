#pragma once

#include <optional>
#include <string>
#include <vector>

namespace tui {

struct StartupTrust {
    bool trust_all_tools = false;
    std::vector<std::string> session_allow_rules;
};

struct RunOptions {
    /// Non-empty: resume by ID or 1-based index.  Empty string: resume most recent.
    std::optional<std::string> resume_session_id;
    /// Optional startup trust settings applied before the first turn.
    StartupTrust startup_trust;
};

struct RunResult {
    std::string session_id;
    std::string session_file_path;
};

/**
 * @brief Run the Filo TUI application.
 * Returns when the user quits.  The result carries session metadata for the post-session report printed by main().
 */
RunResult run(RunOptions opts = {});

} // namespace tui
