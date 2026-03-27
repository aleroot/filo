#pragma once

#include <optional>
#include <string>

namespace tui {

struct RunOptions {
    /// Non-empty: resume by ID or 1-based index.  Empty string: resume most recent.
    std::optional<std::string> resume_session_id;
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
