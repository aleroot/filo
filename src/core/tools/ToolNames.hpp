#pragma once

#include <array>
#include <string_view>

namespace core::tools::names {

inline constexpr std::string_view kRunTerminalCommand = "run_terminal_command";
inline constexpr std::string_view kReadFile = "read_file";
inline constexpr std::string_view kWriteFile = "write_file";
inline constexpr std::string_view kListDirectory = "list_directory";
inline constexpr std::string_view kReplace = "replace";
inline constexpr std::string_view kReplaceInFile = "replace_in_file";
inline constexpr std::string_view kFileSearch = "file_search";
inline constexpr std::string_view kGrepSearch = "grep_search";
inline constexpr std::string_view kApplyPatch = "apply_patch";
inline constexpr std::string_view kSearchReplace = "search_replace";
inline constexpr std::string_view kDeleteFile = "delete_file";
inline constexpr std::string_view kMoveFile = "move_file";
inline constexpr std::string_view kCreateDirectory = "create_directory";
inline constexpr std::string_view kGetCurrentTime = "get_current_time";
inline constexpr std::string_view kGetWorkspaceConfig = "get_workspace_config";
inline constexpr std::string_view kAskUserQuestion = "AskUserQuestion";
inline constexpr std::string_view kPython = "python";
inline constexpr std::string_view kActivateSkill = "activate_skill";
inline constexpr std::string_view kWebSearch = "web_search";
inline constexpr std::string_view kFetchUrl = "fetch_url";

inline constexpr std::array<std::string_view, 7> kExploreAllowedTools{
    kReadFile,
    kFileSearch,
    kGrepSearch,
    kListDirectory,
    kGetCurrentTime,
    kWebSearch,
    kFetchUrl,
};

[[nodiscard]] constexpr bool is_web_access_tool(std::string_view tool_name) noexcept {
    return tool_name == kWebSearch || tool_name == kFetchUrl;
}

[[nodiscard]] constexpr bool is_replace_tool(std::string_view tool_name) noexcept {
    return tool_name == kReplace || tool_name == kReplaceInFile;
}

[[nodiscard]] constexpr bool is_terminal_tool(std::string_view tool_name) noexcept {
    return tool_name == kRunTerminalCommand;
}

[[nodiscard]] constexpr bool is_file_modification_tool(std::string_view tool_name) noexcept {
    return tool_name == kWriteFile
        || tool_name == kApplyPatch
        || tool_name == kSearchReplace
        || is_replace_tool(tool_name)
        || tool_name == kDeleteFile
        || tool_name == kMoveFile
        || tool_name == kCreateDirectory;
}

[[nodiscard]] constexpr bool is_write_destructive_tool(std::string_view tool_name) noexcept {
    return tool_name == kApplyPatch
        || tool_name == kWriteFile
        || tool_name == kSearchReplace
        || is_replace_tool(tool_name)
        || tool_name == kDeleteFile
        || tool_name == kMoveFile;
}

[[nodiscard]] constexpr bool is_read_search_list_tool(std::string_view tool_name) noexcept {
    return tool_name == kReadFile
        || tool_name == kFileSearch
        || tool_name == kGrepSearch
        || tool_name == kListDirectory;
}

[[nodiscard]] constexpr bool is_path_visibility_constrained_tool(
    std::string_view tool_name) noexcept {
    return is_read_search_list_tool(tool_name)
        || is_file_modification_tool(tool_name);
}

} // namespace core::tools::names
