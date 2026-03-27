#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

enum class DiffLineKind { Add, Delete, Context, Hunk, Header, Other };

struct DiffLinePreview {
    DiffLineKind         kind = DiffLineKind::Other;
    std::optional<int>   old_line = {};
    std::optional<int>   new_line = {};
    std::string          content = {};
};

struct ToolDiffPreview {
    std::string                  title;
    std::vector<DiffLinePreview> lines;
    std::size_t                  hidden_line_count = 0;

    [[nodiscard]] bool empty() const noexcept { return lines.empty(); }
};

ToolDiffPreview build_tool_diff_preview(std::string_view tool_name,
                                        std::string_view tool_args_json,
                                        std::size_t max_lines);

std::size_t diff_line_number_width(const ToolDiffPreview& preview);

} // namespace tui

