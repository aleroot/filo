#pragma once

#include "core/commands/CommandExecutor.hpp"
#include "core/workspace/MentionIndex.hpp"
#include <string>
#include <string_view>
#include <vector>

namespace tui {

// The mention index lives in core so the TUI prompt and the MCP
// `completion/complete` handler rank `@` paths identically.
using core::workspace::MentionSuggestion;
using core::workspace::build_mention_index;
using core::workspace::search_mention_index;

struct CommandSuggestion {
    std::string display_name;
    std::string insertion_text;
    std::string description;
    std::string aliases_label;
    bool accepts_arguments = false;
};

std::vector<CommandSuggestion> search_command_index(
    const std::vector<core::commands::CommandDescriptor>& index,
    std::string_view token,
    std::size_t max_results);

} // namespace tui
