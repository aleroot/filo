#pragma once

#include <optional>
#include <string>
#include <vector>

#include "core/context/SteeringLoader.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"

namespace tui {

struct StartupTrust {
    bool trust_all_tools = false;
    std::vector<std::string> session_allow_rules;
};

struct RunOptions {
    /// Non-empty: resume by ID or 1-based index.  Empty string: resume most recent.
    std::optional<std::string> resume_session_id;
    /// When true and no resume_session_id is set: open the TUI with the most
    /// recent session scoped to the current project (the "filo -c" experience).
    bool continue_last = false;
    /// Optional startup trust settings applied before the first turn.
    StartupTrust startup_trust;
    /// Optional process-scoped model selector. Accepted forms are MODEL,
    /// PROVIDER, or PROVIDER/MODEL. This never changes saved defaults.
    std::optional<std::string> startup_model;
    /// Explicit process sandbox mode; supplied by the application composition root.
    core::landrun::LandrunMode landrun_mode{core::landrun::LandrunMode::off};
    core::landrun::LandrunPolicyEnvironment landrun_environment;
    /// Startup steering policy (e.g. from --steering flag)
    core::context::SteeringPolicy steering_policy;
    /// True only when the inbound Streamable-HTTP MCP server was requested.
    /// Drives the isolated remote-activity affordance; outbound MCP clients do
    /// not make it visible.
    bool remote_mcp_server_enabled = false;
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
