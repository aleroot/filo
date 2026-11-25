#pragma once

#include "../tools/ToolNames.hpp"

#include <array>
#include <span>
#include <string_view>

namespace core::config::builtin_subagents {

struct Definition {
    std::string_view name;
    std::string_view description;
    std::string_view prompt;
    /// Empty means every tool.
    std::span<const std::string_view> allowed_tools;
    int max_steps = 0;
    bool read_only = false;
};

inline constexpr Definition kGeneral{
    .name = "general",
    .description = "General-purpose subagent for complex multi-step tasks and broad research.",
    .prompt =
        "You are Filo's @general worker subagent.\n"
        "Work independently on delegated tasks and use tools when needed.\n"
        "Final summary: lead with the outcome (what you changed or found), then the "
        "evidence: files changed, commands run with observed results, key findings as "
        "file:line references. Report only what you observed; after the result, say what "
        "remains unfinished or unverified and why.",
    .allowed_tools = {},
    .max_steps = 40,
    .read_only = false,
};

inline constexpr Definition kExplore{
    .name = "explore",
    .description = "Fast read-only codebase explorer for search-heavy investigations.",
    .prompt =
        "You are Filo's @explore worker subagent.\n"
        "Focus on finding information quickly using read/search/list tools.\n"
        "Do not edit files. Your shell runs only commands that change nothing "
        "(git diff/log/show/status, ls, cat, grep...).\n"
        "Final summary: answer first, then file:line citations; quote exact code for "
        "load-bearing claims.",
    .allowed_tools = core::tools::names::kExploreAllowedTools,
    .max_steps = 24,
    .read_only = true,
};

inline constexpr std::array kCodeReviewAllowedTools{
    core::tools::names::kReview,
    core::tools::names::kRead,
    core::tools::names::kConversation,
    core::tools::names::kFileSearch,
    core::tools::names::kGrepSearch,
    core::tools::names::kListDirectory,
    core::tools::names::kGetCurrentTime,
    core::tools::names::kRunTerminalCommand,
};

// The review tool runs the campaign. This profile only reports its result,
// so the step budget stays small and the worker cannot edit the tree.
inline constexpr Definition kCodeReview{
    .name = "code-review",
    .description = "Read-only reviewer. Runs the review tool and splits findings into blocking and follow-ups.",
    .prompt =
        "You are Filo's @code-review worker.\n"
        "Call the review tool once. Do not edit files.\n"
        "Repeat the tool's blocking findings and follow-ups. Do not invent findings it did not return.\n"
        "If the tool says the review is incomplete or failed, say so and do not call the tree clean.\n"
        "Final summary: blocking count, follow-up count, then each finding as severity, path:line, and title.",
    .allowed_tools = kCodeReviewAllowedTools,
    .max_steps = 8,
    .read_only = true,
};

inline constexpr std::array kAll{kGeneral, kExplore, kCodeReview};

} // namespace core::config::builtin_subagents
