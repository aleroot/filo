#include "ToolDiffUtils.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
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

/// A line kept by both sides: its index in the old text and in the new one.
struct Match {
    std::uint32_t old_line = 0;
    std::uint32_t new_line = 0;
};

/// Lines reduced to symbols: equal lines share one, different lines never do.
/// Comparing two lines is then one integer comparison however long the lines
/// are, which is what lets a search be bounded by the shape of a change rather
/// than by the width of the text it moves.
using Symbols = std::span<const std::uint32_t>;

/// How two symbol sequences line up. `matches` increase strictly on both
/// sides; every line not in them is a deletion or an insertion.
struct Alignment {
    std::vector<Match> matches;
    /// The alignment keeps as many lines as any could, i.e. its edit script is
    /// a shortest one. False when the exact search ran past its bound and the
    /// anchored alignment took over.
    bool minimal = true;
};

/// Myers' shortest edit script between `a` and `b`, appended to `out` as the
/// matches it keeps, offset by `a_base` and `b_base`. False, with `out`
/// untouched, when `budget` runs out first: a frontier slot and a comparison
/// each cost one unit.
///
/// Row `d` of the frontier holds diagonal `-d + 2j` in slot `j`, and all rows
/// share one buffer: the walk back reads every row once, and the bound on the
/// work is also the bound on the memory.
bool shortest_alignment(Symbols a, Symbols b,
                        std::uint32_t a_base, std::uint32_t b_base,
                        WorkBudget& budget, std::vector<Match>& out) {
    const auto n = static_cast<std::int32_t>(a.size());
    const auto m = static_cast<std::int32_t>(b.size());
    if (n == 0 || m == 0) {
        return true;
    }

    std::vector<std::int32_t> frontier;
    std::vector<std::size_t> row_start;
    // Where the previous row reached on the diagonals either side of slot `j`
    // of row `d`, and which of the two steps the slot takes. The walk back
    // decides it again from the same row, so the step is never stored.
    struct Step {
        std::int32_t x = -1;
        bool insert = false;
    };
    const auto step_into = [&](std::int32_t d, std::int32_t j) -> Step {
        const std::size_t previous = row_start[static_cast<std::size_t>(d) - 1];
        const std::int32_t diagonal = -d + 2 * j;
        const std::int32_t insert_x = j < d ? frontier[previous + static_cast<std::size_t>(j)] : -1;
        const std::int32_t delete_x = j > 0 ? frontier[previous + static_cast<std::size_t>(j) - 1] : -1;
        const bool can_insert = insert_x >= 0 && insert_x - (diagonal + 1) < m;
        const bool can_delete = delete_x >= 0 && delete_x < n;
        if (!can_insert && !can_delete) {
            return {};
        }
        if (can_insert && (!can_delete || delete_x < insert_x)) {
            return {.x = insert_x, .insert = true};
        }
        return {.x = delete_x + 1, .insert = false};
    };

    std::int32_t found = -1;
    for (std::int32_t d = 0; d <= n + m && found < 0; ++d) {
        const std::size_t start = frontier.size();
        row_start.push_back(start);
        frontier.resize(start + static_cast<std::size_t>(d) + 1, -1);
        for (std::int32_t j = 0; j <= d; ++j) {
            if (!budget.spend()) {
                return false;
            }
            std::int32_t x = 0;
            if (d != 0) {
                const Step step = step_into(d, j);
                if (step.x < 0) {
                    continue;  // Unreachable diagonal; the slot stays -1.
                }
                x = step.x;
            }
            const std::int32_t diagonal = -d + 2 * j;
            std::int32_t y = x - diagonal;
            while (x < n && y < m) {
                if (!budget.spend()) {
                    return false;
                }
                if (a[static_cast<std::size_t>(x)] != b[static_cast<std::size_t>(y)]) {
                    break;
                }
                ++x;
                ++y;
            }
            frontier[start + static_cast<std::size_t>(j)] = x;
            if (x == n && y == m) {
                found = d;
                break;
            }
        }
    }
    if (found < 0) {
        return false;  // d == n + m always arrives; this is a backstop.
    }

    const std::size_t first = out.size();
    std::int32_t x = n;
    std::int32_t y = m;
    for (std::int32_t d = found; d > 0; --d) {
        const std::int32_t diagonal = x - y;
        const Step step = step_into(d, (diagonal + d) / 2);
        // The diagonal run that followed the step is what this row kept.
        while (x > step.x) {
            --x;
            --y;
            out.push_back({a_base + static_cast<std::uint32_t>(x),
                           b_base + static_cast<std::uint32_t>(y)});
        }
        if (step.insert) {
            --y;
        } else {
            --x;
        }
    }
    while (x > 0 && y > 0) {
        --x;
        --y;
        out.push_back({a_base + static_cast<std::uint32_t>(x),
                       b_base + static_cast<std::uint32_t>(y)});
    }
    std::reverse(out.begin() + static_cast<std::ptrdiff_t>(first), out.end());
    return true;
}

/// Lines that pin a stretch of a change in place: those that occur exactly
/// once on each side, kept in the longest order both sides agree on. This is
/// patience diff's anchoring; it needs no search, so it scales with the length
/// of the stretch rather than with the size of the change.
class Anchoring {
public:
    Anchoring(Symbols a, Symbols b, std::size_t symbols)
        : a_(a), b_(b), seen_(symbols) {}

    [[nodiscard]] std::vector<Match> unique_in_order(std::uint32_t a0, std::uint32_t a1,
                                                     std::uint32_t b0, std::uint32_t b1) {
        for (std::uint32_t i = a0; i < a1; ++i) {
            Seen& entry = touch(a_[i]);
            ++entry.in_a;
            entry.a_line = i;
        }
        for (std::uint32_t j = b0; j < b1; ++j) {
            Seen& entry = touch(b_[j]);
            ++entry.in_b;
            entry.b_line = j;
        }
        std::vector<Match> candidates;
        for (std::uint32_t i = a0; i < a1; ++i) {
            const Seen& entry = seen_[a_[i]];
            if (entry.in_a == 1 && entry.in_b == 1) {
                candidates.push_back({entry.a_line, entry.b_line});
            }
        }
        for (const std::uint32_t symbol : touched_) {
            seen_[symbol] = {};
        }
        touched_.clear();
        return longest_increasing(candidates);
    }

private:
    struct Seen {
        std::uint32_t in_a = 0;
        std::uint32_t in_b = 0;
        std::uint32_t a_line = 0;
        std::uint32_t b_line = 0;
    };

    Seen& touch(std::uint32_t symbol) {
        Seen& entry = seen_[symbol];
        if (entry.in_a == 0 && entry.in_b == 0) {
            touched_.push_back(symbol);
        }
        return entry;
    }

    /// The longest run of `candidates`, already ordered by old line, whose new
    /// lines also increase: patience sorting, O(k log k).
    [[nodiscard]] static std::vector<Match> longest_increasing(const std::vector<Match>& candidates) {
        std::vector<std::size_t> tails;  // Index of the smallest tail of each length.
        std::vector<std::size_t> previous(candidates.size(), SIZE_MAX);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            const auto position = std::ranges::lower_bound(
                tails, candidates[i].new_line, {},
                [&](std::size_t index) { return candidates[index].new_line; });
            if (position != tails.begin()) {
                previous[i] = *(position - 1);
            }
            if (position == tails.end()) {
                tails.push_back(i);
            } else {
                *position = i;
            }
        }
        std::vector<Match> run;
        run.reserve(tails.size());
        for (std::size_t i = tails.empty() ? SIZE_MAX : tails.back(); i != SIZE_MAX; i = previous[i]) {
            run.push_back(candidates[i]);
        }
        std::ranges::reverse(run);
        return run;
    }

    Symbols a_;
    Symbols b_;
    std::vector<Seen> seen_;
    std::vector<std::uint32_t> touched_;
};

/// How deep the anchored alignment splits a stretch before it stops looking
/// for anchors inside it. Each level costs one pass over what is left, so the
/// depth bounds the work; real files are pinned within a few levels.
constexpr int kMaxAnchorDepth = 32;
/// Search one unpinned stretch may spend per line it holds. Linear in its
/// size, so one stretch cannot spend what the others need, and generous enough
/// that an ordinary edit between two anchors is still aligned exactly.
constexpr std::size_t kStretchWorkPerLine = 64;

/// The alignment used when the exact search runs past its bound. Anchors pin
/// the change in place; the stretches between them are searched exactly when
/// that is affordable, and otherwise shown rewritten whole. The result is
/// always a correct edit script, only not necessarily a shortest one.
///
/// Anchoring is not charged to `budget`: it is a pass over each stretch per
/// level, and the levels are bounded, so its cost is bounded by the input the
/// caller already bounds. Only the searches spend `budget`, and running out of
/// it only leaves the remaining stretches rewritten — this path cannot fail.
class AnchoredAlignment {
public:
    AnchoredAlignment(Symbols a, Symbols b, std::size_t symbols, WorkBudget& budget)
        : a_(a), b_(b), anchoring_(a, b, symbols), budget_(budget) {}

    [[nodiscard]] std::vector<Match> run() && {
        align(0, static_cast<std::uint32_t>(a_.size()),
              0, static_cast<std::uint32_t>(b_.size()), 0);
        return std::move(matches_);
    }

private:
    void align(std::uint32_t a0, std::uint32_t a1,
               std::uint32_t b0, std::uint32_t b1, int depth) {
        while (a0 < a1 && b0 < b1 && a_[a0] == b_[b0]) {
            matches_.push_back({a0++, b0++});
        }
        std::uint32_t kept_tail = 0;
        while (a0 < a1 && b0 < b1 && a_[a1 - 1] == b_[b1 - 1]) {
            --a1;
            --b1;
            ++kept_tail;
        }
        if (a0 < a1 && b0 < b1) {
            const auto anchors = depth < kMaxAnchorDepth
                ? anchoring_.unique_in_order(a0, a1, b0, b1)
                : std::vector<Match>{};
            if (!anchors.empty()) {
                std::uint32_t next_a = a0;
                std::uint32_t next_b = b0;
                for (const Match& anchor : anchors) {
                    align(next_a, anchor.old_line, next_b, anchor.new_line, depth + 1);
                    matches_.push_back(anchor);
                    next_a = anchor.old_line + 1;
                    next_b = anchor.new_line + 1;
                }
                align(next_a, a1, next_b, b1, depth + 1);
            } else {
                // Nothing pins this stretch. Search it exactly if it is cheap
                // enough; otherwise it stays a rewrite, which is still a
                // correct description of what changed between its anchors.
                const std::size_t lines = std::size_t{a1 - a0} + (b1 - b0);
                const std::size_t allowance =
                    std::min(budget_.remaining(), lines * kStretchWorkPerLine);
                WorkBudget stretch(allowance);
                (void)shortest_alignment(a_.subspan(a0, a1 - a0), b_.subspan(b0, b1 - b0),
                                         a0, b0, stretch, matches_);
                budget_.charge(allowance - stretch.remaining());
            }
        }
        for (std::uint32_t k = 0; k < kept_tail; ++k) {
            matches_.push_back({a1 + k, b1 + k});
        }
    }

    Symbols a_;
    Symbols b_;
    Anchoring anchoring_;
    WorkBudget& budget_;
    std::vector<Match> matches_;
};

/// How many lines `a` and `b` share at their start and, of what is left, at
/// their end.
std::pair<std::size_t, std::size_t> common_ends(Symbols a, Symbols b) noexcept {
    std::size_t prefix = 0;
    while (prefix < a.size() && prefix < b.size() && a[prefix] == b[prefix]) {
        ++prefix;
    }
    std::size_t suffix = 0;
    while (suffix < a.size() - prefix && suffix < b.size() - prefix
           && a[a.size() - 1 - suffix] == b[b.size() - 1 - suffix]) {
        ++suffix;
    }
    return {prefix, suffix};
}

/// Lines `a` and `b` keep in common, within `budget`. Symbols are below
/// `symbols`.
///
/// Exact steps shrink the problem before any search, and cost no budget: they
/// are linear in the input, which every caller already bounds. Equal ends are
/// kept outright. A line that does not occur on the other side can never be
/// kept, so it is set aside; that leaves the longest common subsequence
/// unchanged, and often exposes more equal ends, which are kept too. Then the
/// shortest edit script of what remains gets half the allowance, and past that
/// the anchored alignment takes the rest, marking `budget` exhausted: the
/// result is then correct but possibly longer than it must be.
Alignment align_symbols(Symbols a, Symbols b, std::size_t symbols,
                                       WorkBudget& budget) {
    Alignment result;
    const auto [prefix, suffix] = common_ends(a, b);
    const auto old_middle = a.subspan(prefix, a.size() - prefix - suffix);
    const auto new_middle = b.subspan(prefix, b.size() - prefix - suffix);

    // What may still be kept of each middle, and where each line of it sits in
    // its text.
    std::vector<std::uint32_t> old_kept;
    std::vector<std::uint32_t> old_at;
    std::vector<std::uint32_t> new_kept;
    std::vector<std::uint32_t> new_at;
    if (!old_middle.empty() && !new_middle.empty()) {
        std::vector<std::uint8_t> sides(symbols, 0);
        constexpr std::uint8_t kInOld = 1;
        constexpr std::uint8_t kInNew = 2;
        for (const auto symbol : old_middle) {
            sides[symbol] |= kInOld;
        }
        for (const auto symbol : new_middle) {
            sides[symbol] |= kInNew;
        }
        for (std::size_t i = 0; i < old_middle.size(); ++i) {
            if (sides[old_middle[i]] & kInNew) {
                old_kept.push_back(old_middle[i]);
                old_at.push_back(static_cast<std::uint32_t>(prefix + i));
            }
        }
        for (std::size_t j = 0; j < new_middle.size(); ++j) {
            if (sides[new_middle[j]] & kInOld) {
                new_kept.push_back(new_middle[j]);
                new_at.push_back(static_cast<std::uint32_t>(prefix + j));
            }
        }
    }

    std::vector<Match> kept;
    const auto [kept_prefix, kept_suffix] = common_ends(old_kept, new_kept);
    for (std::size_t i = 0; i < kept_prefix; ++i) {
        kept.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(i)});
    }
    const auto old_search = Symbols(old_kept).subspan(
        kept_prefix, old_kept.size() - kept_prefix - kept_suffix);
    const auto new_search = Symbols(new_kept).subspan(
        kept_prefix, new_kept.size() - kept_prefix - kept_suffix);
    const std::size_t exact_allowance = budget.remaining() / 2;
    WorkBudget exact(exact_allowance);
    const bool shortest = shortest_alignment(
        old_search, new_search, static_cast<std::uint32_t>(kept_prefix),
        static_cast<std::uint32_t>(kept_prefix), exact, kept);
    budget.charge(exact_allowance - exact.remaining());
    if (shortest) {
        for (std::size_t k = kept_suffix; k > 0; --k) {
            kept.push_back({static_cast<std::uint32_t>(old_kept.size() - k),
                            static_cast<std::uint32_t>(new_kept.size() - k)});
        }
    } else {
        result.minimal = false;
        budget.mark_exhausted();
        kept = AnchoredAlignment(old_kept, new_kept, symbols, budget).run();
    }

    result.matches.reserve(prefix + kept.size() + suffix);
    for (std::size_t i = 0; i < prefix; ++i) {
        result.matches.push_back({static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(i)});
    }
    for (const Match& match : kept) {
        result.matches.push_back({old_at[match.old_line], new_at[match.new_line]});
    }
    for (std::size_t k = suffix; k > 0; --k) {
        result.matches.push_back({static_cast<std::uint32_t>(a.size() - k),
                                  static_cast<std::uint32_t>(b.size() - k)});
    }
    return result;
}

/// Additions and deletions an alignment implies: every line it does not keep.
std::pair<std::size_t, std::size_t> alignment_counts(const Alignment& alignment,
                                                     std::size_t old_size,
                                                     std::size_t new_size) noexcept {
    const std::size_t kept = alignment.matches.size();
    return {new_size - kept, old_size - kept};
}

/// Dense symbols for a sequence of keys, numbered in the order first met
/// across both sides, so two runs over equal inputs align identically.
template <typename Key, typename Hash = std::hash<Key>>
class SymbolTable {
public:
    explicit SymbolTable(std::size_t expected) { ids_.reserve(expected); }

    std::uint32_t operator()(const Key& key) {
        return ids_.try_emplace(key, static_cast<std::uint32_t>(ids_.size())).first->second;
    }

    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }

private:
    std::unordered_map<Key, std::uint32_t, Hash> ids_;
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
    // The same alignment a diff of the same lines runs, on the same symbols:
    // equal hashes are equal lines, so the count is the diff's count. With no
    // diff to show for it, only a shortest script's count is reported.
    SymbolTable<std::uint64_t> table(before.size() + after.size());
    std::vector<std::uint32_t> old_symbols;
    std::vector<std::uint32_t> new_symbols;
    old_symbols.reserve(before.size());
    new_symbols.reserve(after.size());
    for (const auto hash : before) {
        old_symbols.push_back(table(hash));
    }
    for (const auto hash : after) {
        new_symbols.push_back(table(hash));
    }
    const auto alignment = align_symbols(old_symbols, new_symbols, table.size(), budget);
    if (!alignment.minimal) {
        return std::nullopt;
    }
    return alignment_counts(alignment, before.size(), after.size());
}

namespace {

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

/// The edit script an alignment describes. Between two kept lines, the old
/// lines go first and the new ones follow, as a unified diff prints them.
std::vector<Edit> edits_from(const Alignment& alignment,
                             std::size_t old_size, std::size_t new_size) {
    std::vector<Edit> edits;
    edits.reserve(old_size + new_size - alignment.matches.size());
    std::uint32_t old_line = 0;
    std::uint32_t new_line = 0;
    const auto flush = [&](std::size_t old_end, std::size_t new_end) {
        for (; old_line < old_end; ++old_line) {
            edits.push_back({EditKind::Delete, old_line});
        }
        for (; new_line < new_end; ++new_line) {
            edits.push_back({EditKind::Insert, new_line});
        }
    };
    for (const Match& match : alignment.matches) {
        flush(match.old_line, match.new_line);
        edits.push_back({EditKind::Keep, old_line});
        ++old_line;
        ++new_line;
    }
    flush(old_size, new_size);
    return edits;
}

DiffOutcome build_diff_outcome(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content)
{
    // Each side is bounded on its own, as every caller bounds what it keeps of
    // one file: a file that may be retained for a diff may also be diffed.
    if (old_content.size() > kMaxToolDiffInputBytes
        || new_content.size() > kMaxToolDiffInputBytes
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
    // A line's identity is its bytes and its terminator, which is exactly what
    // lines_equal compares: the view is widened over the '\n' that follows a
    // terminated line, so "a" and "a\n" are different symbols.
    const auto identity = [](const DiffLine& line) {
        return std::string_view(line.text.data(), line.text.size() + (line.terminated ? 1 : 0));
    };
    SymbolTable<std::string_view> table(old_lines.size() + new_lines.size());
    std::vector<std::uint32_t> old_symbols;
    std::vector<std::uint32_t> new_symbols;
    old_symbols.reserve(old_lines.size());
    new_symbols.reserve(new_lines.size());
    for (const DiffLine& line : old_lines) {
        old_symbols.push_back(table(identity(line)));
    }
    for (const DiffLine& line : new_lines) {
        new_symbols.push_back(table(identity(line)));
    }

    WorkBudget budget(kMaxMyersWorkUnits);
    const auto alignment = align_symbols(old_symbols, new_symbols, table.size(), budget);
    const std::vector<Edit> edits = edits_from(alignment, old_lines.size(), new_lines.size());
    const auto counts = alignment_counts(alignment, old_lines.size(), new_lines.size());

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
    // Only the text is unaffordable. Numbers handed back without it are what
    // let a summary say how big a change it cannot show, so they must stand
    // on their own: a shortest script's, or none.
    const auto counts_only = [&] {
        if (!alignment.minimal) {
            return DiffOutcome{.minimal = false};
        }
        return DiffOutcome{.counts = counts};
    };
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

    return DiffOutcome{.diff = std::move(diff), .counts = counts, .minimal = alignment.minimal};
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
