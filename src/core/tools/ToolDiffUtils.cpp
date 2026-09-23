#include "ToolDiffUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace core::tools::detail {
namespace {

constexpr std::size_t kMaxMyersWorkUnits = 4'000'000;
constexpr std::size_t kUnifiedContextLines = 3;

struct DiffLine {
    std::string_view text;
    bool             terminated = false;
};

enum class EditKind : std::uint8_t { Keep, Delete, Insert };

struct Edit {
    EditKind kind = EditKind::Keep;
    std::uint32_t line = 0;
};

std::size_t line_count_for_unified_hunk(std::string_view text) noexcept {
    if (text.empty()) {
        return 0;
    }
    return static_cast<std::size_t>(std::ranges::count(text, '\n'))
        + (text.back() == '\n' ? 0 : 1);
}

std::vector<DiffLine> split_diff_lines(std::string_view text) {
    std::vector<DiffLine> lines;
    lines.reserve(line_count_for_unified_hunk(text));

    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t newline = text.find('\n', start);
        if (newline == std::string_view::npos) {
            lines.push_back({.text = text.substr(start), .terminated = false});
            break;
        }
        lines.push_back({
            .text = text.substr(start, newline - start),
            .terminated = true,
        });
        start = newline + 1;
    }
    return lines;
}

bool lines_equal(const DiffLine& lhs, const DiffLine& rhs) noexcept {
    return lhs.text == rhs.text && lhs.terminated == rhs.terminated;
}

struct Frontier {
    std::int32_t distance = 0;
    std::vector<std::int32_t> furthest_x;
    std::vector<EditKind>     direction;

    [[nodiscard]] std::int32_t x_at(std::int32_t diagonal) const noexcept {
        const std::int32_t offset = diagonal + distance;
        if (offset < 0 || offset % 2 != 0) {
            return -1;
        }
        const auto index = static_cast<std::size_t>(offset / 2);
        return index < furthest_x.size() ? furthest_x[index] : -1;
    }

    [[nodiscard]] EditKind direction_at(std::int32_t diagonal) const noexcept {
        const std::int32_t offset = diagonal + distance;
        if (offset < 0 || offset % 2 != 0) {
            return EditKind::Keep;
        }
        const auto index = static_cast<std::size_t>(offset / 2);
        return index < direction.size() ? direction[index] : EditKind::Keep;
    }
};

bool consume_work(std::size_t& work, std::size_t units = 1) noexcept {
    if (units > kMaxMyersWorkUnits - work) {
        return false;
    }
    work += units;
    return true;
}

std::optional<std::vector<Edit>> shortest_edit_script(
    std::span<const DiffLine> old_lines,
    std::span<const DiffLine> new_lines)
{
    const auto old_size = static_cast<std::int32_t>(old_lines.size());
    const auto new_size = static_cast<std::int32_t>(new_lines.size());

    std::vector<Edit> edits;
    if (old_size == 0) {
        edits.reserve(new_lines.size());
        for (std::int32_t i = 0; i < new_size; ++i) {
            edits.push_back({EditKind::Insert, static_cast<std::uint32_t>(i)});
        }
        return edits;
    }
    if (new_size == 0) {
        edits.reserve(old_lines.size());
        for (std::int32_t i = 0; i < old_size; ++i) {
            edits.push_back({EditKind::Delete, static_cast<std::uint32_t>(i)});
        }
        return edits;
    }

    std::vector<Frontier> trace;
    trace.reserve(128);
    std::size_t work = 0;
    std::int32_t found_distance = -1;

    for (std::int32_t distance = 0;
         distance <= old_size + new_size;
         ++distance) {
        Frontier current;
        current.distance = distance;
        current.furthest_x.reserve(static_cast<std::size_t>(distance) + 1);
        current.direction.reserve(static_cast<std::size_t>(distance) + 1);

        const Frontier* previous = trace.empty() ? nullptr : &trace.back();
        for (std::int32_t diagonal = -distance;
             diagonal <= distance;
             diagonal += 2) {
            if (!consume_work(work)) {
                return std::nullopt;
            }

            std::int32_t x = 0;
            EditKind direction = EditKind::Keep;
            if (distance != 0) {
                const std::int32_t insert_x = previous->x_at(diagonal + 1);
                const std::int32_t delete_x = previous->x_at(diagonal - 1);
                const bool can_insert = insert_x >= 0
                    && insert_x - (diagonal + 1) < new_size;
                const bool can_delete = delete_x >= 0 && delete_x < old_size;

                if (!can_insert && !can_delete) {
                    current.furthest_x.push_back(-1);
                    current.direction.push_back(EditKind::Keep);
                    continue;
                }

                if (can_insert && (!can_delete || delete_x < insert_x)) {
                    x = insert_x;
                    direction = EditKind::Insert;
                } else {
                    x = delete_x + 1;
                    direction = EditKind::Delete;
                }
            }

            std::int32_t y = x - diagonal;
            while (x < old_size && y < new_size) {
                const DiffLine& old_line = old_lines[static_cast<std::size_t>(x)];
                const DiffLine& new_line = new_lines[static_cast<std::size_t>(y)];
                const std::size_t comparison_cost =
                    std::min(old_line.text.size(), new_line.text.size()) + 1;
                if (!consume_work(work, comparison_cost)) {
                    return std::nullopt;
                }
                if (!lines_equal(old_line, new_line)) {
                    break;
                }
                ++x;
                ++y;
            }

            current.furthest_x.push_back(x);
            current.direction.push_back(direction);
            if (x == old_size && y == new_size) {
                found_distance = distance;
                break;
            }
        }

        trace.push_back(std::move(current));
        if (found_distance >= 0) {
            break;
        }
    }

    if (found_distance < 0) {
        return std::nullopt;
    }

    edits.reserve(static_cast<std::size_t>(found_distance)
                  + old_lines.size() + new_lines.size());
    std::int32_t x = old_size;
    std::int32_t y = new_size;
    for (std::int32_t distance = found_distance; distance > 0; --distance) {
        const Frontier& current = trace[static_cast<std::size_t>(distance)];
        const Frontier& previous = trace[static_cast<std::size_t>(distance - 1)];
        const std::int32_t diagonal = x - y;
        const EditKind direction = current.direction_at(diagonal);

        if (direction == EditKind::Insert) {
            const std::int32_t previous_x = previous.x_at(diagonal + 1);
            if (previous_x < 0) {
                return std::nullopt;
            }
            const std::int32_t previous_y = previous_x - (diagonal + 1);
            while (x > previous_x && y > previous_y + 1) {
                --x;
                --y;
                edits.push_back({EditKind::Keep, static_cast<std::uint32_t>(x)});
            }
            if (x != previous_x || y != previous_y + 1 || previous_y < 0) {
                return std::nullopt;
            }
            --y;
            edits.push_back({EditKind::Insert, static_cast<std::uint32_t>(y)});
        } else if (direction == EditKind::Delete) {
            const std::int32_t previous_x = previous.x_at(diagonal - 1);
            if (previous_x < 0) {
                return std::nullopt;
            }
            const std::int32_t previous_y = previous_x - (diagonal - 1);
            while (x > previous_x + 1 && y > previous_y) {
                --x;
                --y;
                edits.push_back({EditKind::Keep, static_cast<std::uint32_t>(x)});
            }
            if (x != previous_x + 1 || y != previous_y || previous_x < 0) {
                return std::nullopt;
            }
            --x;
            edits.push_back({EditKind::Delete, static_cast<std::uint32_t>(x)});
        } else {
            return std::nullopt;
        }
    }

    while (x > 0 && y > 0) {
        --x;
        --y;
        if (!lines_equal(old_lines[static_cast<std::size_t>(x)],
                         new_lines[static_cast<std::size_t>(y)])) {
            return std::nullopt;
        }
        edits.push_back({EditKind::Keep, static_cast<std::uint32_t>(x)});
    }
    while (x > 0) {
        --x;
        edits.push_back({EditKind::Delete, static_cast<std::uint32_t>(x)});
    }
    while (y > 0) {
        --y;
        edits.push_back({EditKind::Insert, static_cast<std::uint32_t>(y)});
    }
    std::ranges::reverse(edits);
    return edits;
}

struct HunkRange {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::size_t old_begin = 0;
    std::size_t new_begin = 0;
    std::size_t old_count = 0;
    std::size_t new_count = 0;
};

bool append_bounded(std::string& output, std::string_view text) {
    if (text.size() > kMaxToolDiffOutputBytes - output.size()) {
        return false;
    }
    output.append(text);
    return true;
}

bool append_bounded(std::string& output, char value) {
    if (output.size() == kMaxToolDiffOutputBytes) {
        return false;
    }
    output.push_back(value);
    return true;
}

bool append_patch_line(std::string& output, char prefix, const DiffLine& line) {
    if (!append_bounded(output, prefix)
        || !append_bounded(output, line.text)
        || !append_bounded(output, '\n')) {
        return false;
    }
    if (!line.terminated) {
        return append_bounded(output, "\\ No newline at end of file\n");
    }
    return true;
}

const DiffLine& line_for_edit(const Edit& edit,
                              std::span<const DiffLine> old_lines,
                              std::span<const DiffLine> new_lines) {
    return edit.kind == EditKind::Insert
        ? new_lines[edit.line]
        : old_lines[edit.line];
}

std::optional<std::string> build_unified_diff_impl(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content)
{
    if (old_content.size() > kMaxToolDiffInputBytes
        || new_content.size() > kMaxToolDiffInputBytes - old_content.size()
        || file_path.size() > (kMaxToolDiffOutputBytes - 64) / 2) {
        return std::nullopt;
    }

    if (old_content == new_content
        || file_path.find_first_of("\r\n") != std::string_view::npos
        || !is_text_like_for_diff(old_content)
        || !is_text_like_for_diff(new_content)) {
        return std::nullopt;
    }

    const auto old_lines = split_diff_lines(old_content);
    const auto new_lines = split_diff_lines(new_content);
    std::size_t prefix = 0;
    while (prefix < old_lines.size()
           && prefix < new_lines.size()
           && lines_equal(old_lines[prefix], new_lines[prefix])) {
        ++prefix;
    }

    std::size_t suffix = 0;
    while (suffix < old_lines.size() - prefix
           && suffix < new_lines.size() - prefix
           && lines_equal(old_lines[old_lines.size() - suffix - 1],
                          new_lines[new_lines.size() - suffix - 1])) {
        ++suffix;
    }

    const auto old_middle = std::span{old_lines}.subspan(
        prefix, old_lines.size() - prefix - suffix);
    const auto new_middle = std::span{new_lines}.subspan(
        prefix, new_lines.size() - prefix - suffix);
    auto middle_script = shortest_edit_script(old_middle, new_middle);
    if (!middle_script) {
        return std::nullopt;
    }

    std::vector<Edit> edits;
    edits.reserve(prefix + middle_script->size() + suffix);
    for (std::size_t i = 0; i < prefix; ++i) {
        edits.push_back({EditKind::Keep, static_cast<std::uint32_t>(i)});
    }
    for (const Edit& edit : *middle_script) {
        edits.push_back({
            edit.kind,
            static_cast<std::uint32_t>(prefix + edit.line),
        });
    }
    for (std::size_t i = 0; i < suffix; ++i) {
        edits.push_back({
            EditKind::Keep,
            static_cast<std::uint32_t>(old_lines.size() - suffix + i),
        });
    }

    std::vector<HunkRange> hunks;
    for (std::size_t i = 0; i < edits.size(); ++i) {
        if (edits[i].kind == EditKind::Keep) {
            continue;
        }
        const std::size_t begin = i > kUnifiedContextLines
            ? i - kUnifiedContextLines
            : 0;
        const std::size_t end = std::min(
            edits.size(), i + kUnifiedContextLines + 1);
        if (!hunks.empty() && begin <= hunks.back().end) {
            hunks.back().end = std::max(hunks.back().end, end);
        } else {
            hunks.push_back({.begin = begin, .end = end});
        }
    }
    if (hunks.empty()) {
        return std::nullopt;
    }

    std::size_t operation_index = 0;
    std::size_t old_position = 0;
    std::size_t new_position = 0;
    const auto advance_position = [&](const Edit& edit) {
        old_position += edit.kind != EditKind::Insert ? 1 : 0;
        new_position += edit.kind != EditKind::Delete ? 1 : 0;
    };
    for (HunkRange& hunk : hunks) {
        while (operation_index < hunk.begin) {
            advance_position(edits[operation_index++]);
        }
        hunk.old_begin = old_position;
        hunk.new_begin = new_position;
        while (operation_index < hunk.end) {
            advance_position(edits[operation_index++]);
        }
        hunk.old_count = old_position - hunk.old_begin;
        hunk.new_count = new_position - hunk.new_begin;
    }

    std::string diff;
    diff.reserve(std::min<std::size_t>(kMaxToolDiffOutputBytes, 1024));
    if (!append_bounded(diff, "--- a/")
        || !append_bounded(diff, file_path)
        || !append_bounded(diff, "\n+++ b/")
        || !append_bounded(diff, file_path)
        || !append_bounded(diff, '\n')) {
        return std::nullopt;
    }

    for (const HunkRange& hunk : hunks) {
        const std::size_t old_start = hunk.old_count == 0
            ? hunk.old_begin
            : hunk.old_begin + 1;
        const std::size_t new_start = hunk.new_count == 0
            ? hunk.new_begin
            : hunk.new_begin + 1;

        const std::string header = std::format(
            "@@ -{},{} +{},{} @@\n",
            old_start,
            hunk.old_count,
            new_start,
            hunk.new_count);
        if (!append_bounded(diff, header)) {
            return std::nullopt;
        }

        for (std::size_t i = hunk.begin; i < hunk.end; ++i) {
            const Edit& edit = edits[i];
            if (edit.kind == EditKind::Keep) {
                if (!append_patch_line(
                        diff, ' ', line_for_edit(edit, old_lines, new_lines))) {
                    return std::nullopt;
                }
            } else if (edit.kind == EditKind::Delete) {
                if (!append_patch_line(
                        diff, '-', line_for_edit(edit, old_lines, new_lines))) {
                    return std::nullopt;
                }
            } else {
                if (!append_patch_line(
                        diff, '+', line_for_edit(edit, old_lines, new_lines))) {
                    return std::nullopt;
                }
            }
        }
    }

    return diff;
}

} // namespace

std::optional<std::string> build_unified_diff(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content)
{
    try {
        return build_unified_diff_impl(file_path, old_content, new_content);
    } catch (const std::bad_alloc&) {
        // Reporting is optional; an allocation failure must not make an edit
        // that has already been applied appear to have failed.
        return std::nullopt;
    }
}

} // namespace core::tools::detail
