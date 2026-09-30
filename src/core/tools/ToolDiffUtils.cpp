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

constexpr std::size_t kUnifiedContextLines = 3;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

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

} // namespace

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

std::string join_diff_lines(std::span<const DiffLine> lines) {
    std::string text;
    for (const DiffLine& line : lines) {
        text += line.text;
        if (line.terminated) {
            text += '\n';
        }
    }
    return text;
}

bool lines_equal(const DiffLine& lhs, const DiffLine& rhs) noexcept {
    return lhs.text == rhs.text && lhs.terminated == rhs.terminated;
}

void LineHasher::feed(std::string_view chunk) {
    for (const unsigned char byte : chunk) {
        if (byte == '\n') {
            hashes.push_back(pending_);
            pending_ = kFnvOffsetBasis;
            in_line_ = false;
        } else {
            pending_ ^= byte;
            pending_ *= kFnvPrime;
            in_line_ = true;
        }
    }
}

void LineHasher::finish() {
    if (in_line_) {
        // The read ended inside a line, so it carries no terminator. A line
        // never contains '\n', so folding one in cannot collide with any
        // terminated line: the hash then separates "a" from "a\n" exactly as
        // lines_equal does, and a count of the two agrees with a diff of them.
        pending_ ^= static_cast<std::uint64_t>('\n');
        pending_ *= kFnvPrime;
        hashes.push_back(pending_);
        pending_ = kFnvOffsetBasis;
        in_line_ = false;
    }
}

std::vector<std::uint64_t> hash_text_lines(std::string_view text) {
    LineHasher hasher;
    hasher.feed(text);
    hasher.finish();
    return std::move(hasher.hashes);
}

std::optional<std::pair<std::size_t, std::size_t>> count_line_edits(
    std::span<const std::uint64_t> before,
    std::span<const std::uint64_t> after,
    WorkBudget& budget)
{
    // Equal ends carry no edit. Trimming them leaves the search the span that
    // actually moved, which is where a turn's edits sit in a file it otherwise
    // kept.
    std::size_t prefix = 0;
    while (prefix < before.size() && prefix < after.size()
           && before[prefix] == after[prefix]) {
        ++prefix;
    }
    std::size_t suffix = 0;
    while (suffix < before.size() - prefix && suffix < after.size() - prefix
           && before[before.size() - 1 - suffix] == after[after.size() - 1 - suffix]) {
        ++suffix;
    }
    const auto old_lines = before.subspan(prefix, before.size() - prefix - suffix);
    const auto new_lines = after.subspan(prefix, after.size() - prefix - suffix);
    const auto old_size = static_cast<std::int32_t>(old_lines.size());
    const auto new_size = static_cast<std::int32_t>(new_lines.size());

    // Myers' frontier, one row per edit distance: row `d` holds diagonal
    // `-d + 2j` in slot `j`. A row only reads the row before it, so two rows
    // suffice however far the search runs.
    std::vector<std::int32_t> previous;
    std::vector<std::int32_t> current;
    for (std::int32_t distance = 0; distance <= old_size + new_size; ++distance) {
        current.assign(static_cast<std::size_t>(distance) + 1, -1);
        for (std::int32_t j = 0; j <= distance; ++j) {
            if (!budget.spend()) {
                return std::nullopt;
            }
            const std::int32_t diagonal = -distance + 2 * j;
            std::int32_t x = 0;
            if (distance != 0) {
                // Diagonal + 1 of the previous row is this slot; diagonal - 1
                // is the one before it.
                const std::int32_t insert_x =
                    static_cast<std::size_t>(j) < previous.size() ? previous[static_cast<std::size_t>(j)] : -1;
                const std::int32_t delete_x = j > 0 ? previous[static_cast<std::size_t>(j) - 1] : -1;
                const bool can_insert =
                    insert_x >= 0 && insert_x - (diagonal + 1) < new_size;
                const bool can_delete = delete_x >= 0 && delete_x < old_size;
                if (!can_insert && !can_delete) {
                    continue;  // Unreachable diagonal; the slot stays -1.
                }
                x = can_insert && (!can_delete || delete_x < insert_x)
                    ? insert_x
                    : delete_x + 1;
            }
            std::int32_t y = x - diagonal;
            while (x < old_size && y < new_size) {
                if (!budget.spend()) {
                    return std::nullopt;
                }
                if (old_lines[static_cast<std::size_t>(x)]
                    != new_lines[static_cast<std::size_t>(y)]) {
                    break;
                }
                ++x;
                ++y;
            }
            current[static_cast<std::size_t>(j)] = x;
            if (x == old_size && y == new_size) {
                // distance is additions plus deletions; the size difference
                // is additions minus deletions.
                const auto edits = static_cast<std::ptrdiff_t>(distance);
                const auto delta = static_cast<std::ptrdiff_t>(new_size) - old_size;
                return std::pair<std::size_t, std::size_t>{
                    static_cast<std::size_t>((edits + delta) / 2),
                    static_cast<std::size_t>((edits - delta) / 2),
                };
            }
        }
        previous.swap(current);
    }
    return std::nullopt;  // distance == old + new always arrives; this is a backstop.
}

namespace {

std::optional<std::vector<Edit>> shortest_edit_script(
    std::span<const DiffLine> old_lines,
    std::span<const DiffLine> new_lines,
    WorkBudget& budget)
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
            if (!budget.spend()) {
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
                if (!budget.spend(comparison_cost)) {
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

/// Additions and deletions one edit script carries: what a diff of it would
/// count, known as soon as the search finishes and before any text is written.
std::pair<std::size_t, std::size_t> count_edits(std::span<const Edit> edits) noexcept {
    std::pair<std::size_t, std::size_t> counts{};
    for (const Edit& edit : edits) {
        counts.first += edit.kind == EditKind::Insert ? 1 : 0;
        counts.second += edit.kind == EditKind::Delete ? 1 : 0;
    }
    return counts;
}

DiffOutcome build_diff_outcome(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content)
{
    if (old_content.size() > kMaxToolDiffInputBytes
        || new_content.size() > kMaxToolDiffInputBytes - old_content.size()
        || file_path.size() > (kMaxToolDiffOutputBytes - 64) / 2) {
        return {};
    }

    if (old_content == new_content
        || file_path.find_first_of("\r\n") != std::string_view::npos
        || !is_text_like_for_diff(old_content)
        || !is_text_like_for_diff(new_content)) {
        return {};
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
    WorkBudget budget(kMaxMyersWorkUnits);
    auto middle_script = shortest_edit_script(old_middle, new_middle, budget);
    if (!middle_script) {
        // Either the search ran out of work or the frontier did not add up on
        // the way back. Neither establishes anything about the change, so
        // nothing is claimed: no text, and no numbers either.
        return {};
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

    // What the search proved, kept whether or not its text turns out to be
    // affordable: a summary can then report the size of a change it cannot
    // show instead of falling back to a second search for the same numbers.
    const auto counts = count_edits(edits);

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
        return {};
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
    // The change is measured; only its text is unaffordable. Handing back the
    // numbers without it is what lets a summary say how big a change it cannot
    // show.
    const auto counts_only = [&] { return DiffOutcome{.counts = counts}; };
    if (!append_bounded(diff, "--- a/")
        || !append_bounded(diff, file_path)
        || !append_bounded(diff, "\n+++ b/")
        || !append_bounded(diff, file_path)
        || !append_bounded(diff, '\n')) {
        return counts_only();
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
            return counts_only();
        }

        for (std::size_t i = hunk.begin; i < hunk.end; ++i) {
            const Edit& edit = edits[i];
            if (edit.kind == EditKind::Keep) {
                if (!append_patch_line(
                        diff, ' ', line_for_edit(edit, old_lines, new_lines))) {
                    return counts_only();
                }
            } else if (edit.kind == EditKind::Delete) {
                if (!append_patch_line(
                        diff, '-', line_for_edit(edit, old_lines, new_lines))) {
                    return counts_only();
                }
            } else {
                if (!append_patch_line(
                        diff, '+', line_for_edit(edit, old_lines, new_lines))) {
                    return counts_only();
                }
            }
        }
    }

    return DiffOutcome{.diff = std::move(diff), .counts = counts};
}

} // namespace

DiffOutcome build_unified_diff_detailed(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content)
{
    try {
        return build_diff_outcome(file_path, old_content, new_content);
    } catch (const std::bad_alloc&) {
        // Reporting is optional; an allocation failure must not make an edit
        // that has already been applied appear to have failed.
        return {};
    }
}

} // namespace core::tools::detail
