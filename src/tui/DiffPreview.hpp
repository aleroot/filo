#pragma once

#include <cstddef>
#include <memory>
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

/// A tool's diff as the UI models it: complete, immutable, and independent of
/// how much of it any given card decides to show.
///
/// Truncation is a *rendering* decision here, never a model decision. Building
/// a clamped diff would make "expand" a lie — the dropped lines would be gone
/// before the disclosure state is even known, and the +A/-D header stats would
/// be computed from the surviving fragment. Renderers clamp what they draw;
/// `added_count`/`deleted_count`/`total_line_count` always describe the whole
/// change.
///
/// Lines are shared and never mutated after construction: the transcript
/// snapshot copies every `UiMessage` on each UI revision, and a by-value vector
/// would copy entire patches on every streaming tick.
struct ToolDiffPreview {
    std::string title;

    // Full-diff accounting. Independent of any clamp applied for display, and
    // unaffected by `truncated_at_source`.
    std::size_t added_count = 0;
    std::size_t deleted_count = 0;
    std::size_t total_line_count = 0;

    /// Lines dropped by a display clamp (see `clamp_diff_preview`). Zero for a
    /// freshly built preview.
    std::size_t hidden_line_count = 0;

    /// True when the diff was so large that the model itself refused to keep
    /// all of it (see `kToolDiffModelMaxLines`). Expanding cannot reveal the
    /// remainder, so the UI must say so rather than imply another click helps.
    bool truncated_at_source = false;

    [[nodiscard]] const std::vector<DiffLinePreview>& lines() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return lines().empty(); }

    void set_lines(std::vector<DiffLinePreview> value);

private:
    std::shared_ptr<const std::vector<DiffLinePreview>> lines_;
};

/// Builds the complete diff a tool call implies, capped only by the model-level
/// ceiling that keeps a pathological generated file from living in the
/// transcript forever.
ToolDiffPreview build_tool_diff_preview(std::string_view tool_name,
                                        std::string_view tool_args_json);

/// Display clamp: keeps at most `max_lines` and reports the rest through
/// `hidden_line_count`. `max_lines == 0` means "no clamp". Stats are preserved.
[[nodiscard]] ToolDiffPreview clamp_diff_preview(const ToolDiffPreview& preview,
                                                 std::size_t max_lines);

std::size_t diff_line_number_width(const ToolDiffPreview& preview);

} // namespace tui

