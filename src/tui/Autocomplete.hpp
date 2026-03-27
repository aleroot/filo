#pragma once

#include "core/commands/CommandExecutor.hpp"
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

struct MentionSuggestion {
    std::string display_path;
    std::string insertion_text;
    bool is_directory = false;
};

struct CommandSuggestion {
    std::string display_name;
    std::string insertion_text;
    std::string description;
    std::string aliases_label;
    bool accepts_arguments = false;
};

std::vector<MentionSuggestion> build_mention_index(const std::filesystem::path& root);

std::vector<MentionSuggestion> search_mention_index(const std::vector<MentionSuggestion>& index,
                                                    std::string_view query,
                                                    std::size_t max_results);

std::vector<CommandSuggestion> search_command_index(
    const std::vector<core::commands::CommandDescriptor>& index,
    std::string_view token,
    std::size_t max_results);

} // namespace tui
