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
        "Final summary: one status line (done / blocked / partial), then the evidence: "
        "files changed, commands run with observed results, key findings as file:line "
        "references. Report only what you observed; say plainly what you could not verify.",
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

inline constexpr std::array kAll{kGeneral, kExplore};

} // namespace core::config::builtin_subagents
