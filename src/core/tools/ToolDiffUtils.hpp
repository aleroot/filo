#pragma once

#include "../utils/JsonUtils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::tools::detail {

inline constexpr std::size_t kMaxToolDiffInputBytes = 512 * 1024;
inline constexpr std::size_t kMaxToolDiffOutputBytes = 512 * 1024;
/// Seed of every FNV-1a hash Filo keeps: content fingerprints, file stamps and
/// the per-line hashes a change too large to diff is still counted from.
inline constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ull;
/// Work one diff or count may spend aligning two texts. Lines are compared as
/// symbols, so a frontier slot and a line comparison each cost one unit
/// whatever the lines hold: the bound follows the shape of a change, never the
/// width of its lines. Half of it goes to the shortest edit script; past that
/// an anchored alignment takes the rest and still yields a correct, if longer,
/// script. Detail is optional; giving it up always beats hanging a turn for
/// it, and measured the cap costs milliseconds.
inline constexpr std::size_t kMaxMyersWorkUnits = 4'000'000;

[[nodiscard]] inline bool is_text_like_for_diff(std::string_view value) noexcept {
    const auto is_continuation = [](unsigned char byte) noexcept {
        return byte >= 0x80 && byte <= 0xBF;
    };

    std::size_t i = 0;
    while (i < value.size()) {
        const auto byte = static_cast<unsigned char>(value[i]);
        if (byte == '\0') {
            return false;
        }
        if (byte <= 0x7F) {
            ++i;
            continue;
        }

        const auto remaining = value.size() - i;
        const auto at = [&](std::size_t offset) noexcept {
            return static_cast<unsigned char>(value[i + offset]);
        };

        if (byte >= 0xC2 && byte <= 0xDF) {
            if (remaining < 2 || !is_continuation(at(1))) {
                return false;
            }
            i += 2;
        } else if (byte == 0xE0) {
            if (remaining < 3 || at(1) < 0xA0 || at(1) > 0xBF || !is_continuation(at(2))) {
                return false;
            }
            i += 3;
        } else if (byte >= 0xE1 && byte <= 0xEC) {
            if (remaining < 3 || !is_continuation(at(1)) || !is_continuation(at(2))) {
                return false;
            }
            i += 3;
        } else if (byte == 0xED) {
            if (remaining < 3 || at(1) < 0x80 || at(1) > 0x9F || !is_continuation(at(2))) {
                return false;
            }
            i += 3;
        } else if (byte >= 0xEE && byte <= 0xEF) {
            if (remaining < 3 || !is_continuation(at(1)) || !is_continuation(at(2))) {
                return false;
            }
            i += 3;
        } else if (byte == 0xF0) {
            if (remaining < 4 || at(1) < 0x90 || at(1) > 0xBF
                || !is_continuation(at(2)) || !is_continuation(at(3))) {
                return false;
            }
            i += 4;
        } else if (byte >= 0xF1 && byte <= 0xF3) {
            if (remaining < 4 || !is_continuation(at(1)) || !is_continuation(at(2))
                || !is_continuation(at(3))) {
                return false;
            }
            i += 4;
        } else if (byte == 0xF4) {
            if (remaining < 4 || at(1) < 0x80 || at(1) > 0x8F
                || !is_continuation(at(2)) || !is_continuation(at(3))) {
                return false;
            }
            i += 4;
        } else {
            return false;
        }
    }

    return true;
}

/// One line of a file or of a patch side. `terminated` is false only for a
/// final line that carries no newline, which a unified diff marks explicitly
/// and a rewrite must reproduce byte for byte.
struct DiffLine {
    std::string_view text;
    bool             terminated = false;
};

/// Splits content the way a unified diff counts lines: a final line without a
/// terminator is still a line. The views borrow `text`.
[[nodiscard]] std::vector<DiffLine> split_diff_lines(std::string_view text);

/// Inverse of `split_diff_lines`: concatenates lines, terminating only those
/// that were terminated.
[[nodiscard]] std::string join_diff_lines(std::span<const DiffLine> lines);

[[nodiscard]] bool lines_equal(const DiffLine& lhs, const DiffLine& rhs) noexcept;

/// The FNV-1a of every line of `text`, terminators excluded but recorded: a
/// final line without a terminator is still a line, and empty text is no line
/// at all. Two texts hash alike line for line exactly when a unified diff
/// would find nothing between them, terminators included.
[[nodiscard]] std::vector<std::uint64_t> hash_text_lines(std::string_view text);

/// The incremental form of `hash_text_lines`, for a bounded read that sees a
/// file chunk by chunk: `feed` the chunks in order, then `finish` once the
/// read ends.
struct LineHasher {
    void feed(std::string_view chunk);
    void finish();

    std::vector<std::uint64_t> hashes;

private:
    std::uint64_t pending_ = kFnvOffsetBasis;
    bool in_line_ = false;
};

/// A bound on one bounded search, spent as it runs. Shared by the diff and by
/// a count of the same lines so both answer "is this search finishing?" the
/// same way, and so a caller can hand several searches one allowance: a file
/// that cannot be counted then costs what it used rather than the whole cap.
class WorkBudget {
public:
    explicit constexpr WorkBudget(std::size_t units) noexcept : remaining_(units) {}

    /// Spends `units`. False when they do not fit, which ends the search; the
    /// budget remembers that it was the reason, so a caller can tell "gave up"
    /// from "found nothing".
    [[nodiscard]] bool spend(std::size_t units = 1) noexcept {
        if (units > remaining_) {
            exhausted_ = true;
            return false;
        }
        remaining_ -= units;
        return true;
    }

    /// Records that a search this budget bounded was cut short, for a caller
    /// that ran it on an allowance carved out of this one.
    void mark_exhausted() noexcept { exhausted_ = true; }

    [[nodiscard]] constexpr bool exhausted() const noexcept { return exhausted_; }
    [[nodiscard]] constexpr std::size_t remaining() const noexcept { return remaining_; }

    /// Charges what a search carved out of this allowance actually used. It
    /// fits by construction: it was reserved out of what was left.
    void charge(std::size_t units) noexcept {
        remaining_ -= std::min(units, remaining_);
    }

private:
    std::size_t remaining_;
    bool exhausted_ = false;
};

/// The additions and deletions between two line-hash sequences: exactly what
/// a unified diff of the same lines would count, without building one — the
/// two run the same alignment. No value when `budget` cannot pay for a
/// shortest script: with no diff to check them against, the numbers of a
/// longer one would be a guess, and a count is never a guess.
[[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> count_line_edits(
    std::span<const std::uint64_t> before,
    std::span<const std::uint64_t> after,
    WorkBudget& budget);

/// What one bounded diff attempt established. `counts` is what the alignment
/// proved rather than what the text says, so a caller that cannot afford the
/// diff still keeps the numbers — the reason a summary can report the size of
/// a change it cannot show. Absent counts mean the inputs were not worth
/// diffing (unchanged, not text, a side over `kMaxToolDiffInputBytes`), or the
/// text did not fit and the script was not a shortest one: numbers without a
/// diff to check them against are reported only when they are exact.
struct DiffOutcome {
    /// The unified diff, absent when the change could not be shown: nothing
    /// was aligned, or the text would exceed `kMaxToolDiffOutputBytes`.
    std::optional<std::string> diff;
    /// Additions and deletions. Always present with `diff`, and always what
    /// `diff` itself counts; without it, present only when `minimal`.
    std::optional<std::pair<std::size_t, std::size_t>> counts;
    /// The script is a shortest one. False when the change was too reshaped
    /// for the exact search and the anchored alignment described it: the diff
    /// still applies exactly, it may only show more lines than it must.
    bool minimal = true;

    [[nodiscard]] bool has_diff() const noexcept { return diff.has_value(); }
};

[[nodiscard]] DiffOutcome build_unified_diff_detailed(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content);

/// Builds a bounded, line-based unified diff. Returns no value when either
/// input is unsuitable for a text diff or the diff text would exceed its size
/// limit; the work it does is bounded either way. A change too reshaped for a
/// shortest script is anchored on the lines it left unique rather than
/// dropped (`DiffOutcome::minimal` is then false). A stretch is shown as
/// rewritten whole only when its sides share no line, which is then the
/// shortest script, or when no line in it is unique to both sides and it is
/// too reshaped to search.
[[nodiscard]] inline std::optional<std::string> build_unified_diff(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content) {
    return std::move(build_unified_diff_detailed(file_path, old_content, new_content).diff);
}

[[nodiscard]] inline std::string json_diff_field(std::string_view diff) {
    return std::string{R"(,"diff":")"}
        + core::utils::escape_json_string(diff)
        + '"';
}

} // namespace core::tools::detail
