#pragma once

#include "../tools/ToolDiffUtils.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// Reading back the unified diffs a TurnChangeTracker stored. Shared by the
/// revert, which writes what unwinding them yields, and by the conversation
/// comparison, which only looks at it.
namespace core::changes::detail {

/// A diff is unwound against a file read whole. Past this it is cheaper to
/// give up than to pull a file into memory that no stored diff could have
/// described (the tracker only diffs up to 512 KiB a side).
inline constexpr std::uintmax_t kMaxWholeReadBytes = 8 * 1024 * 1024;

/// One hunk of a stored unified diff: the two sides it relates and where the
/// "after" side sits in the file. The line views borrow the diff text.
struct Hunk {
    /// Zero-based line the after side starts at.
    std::size_t position = 0;
    std::size_t before_count = 0;
    std::size_t after_count = 0;
    std::vector<core::tools::detail::DiffLine> before;  ///< Context and removed lines
    std::vector<core::tools::detail::DiffLine> after;   ///< Context and added lines
};

/// Reads the hunks of a diff the tracker stored. Anything that is not one of
/// our own diffs is refused whole: a reader that half understands a patch is
/// how a file gets rewritten wrongly. The hunks borrow `diff`.
[[nodiscard]] std::optional<std::vector<Hunk>> parse_hunks(std::string_view diff);

/// Puts every hunk's before side back where its after side is now. No value
/// when `current` no longer holds what the diff's after side says, line for
/// line and terminators included: something edited the file since.
[[nodiscard]] std::optional<std::string> reverse_apply(std::string_view current,
                                                       const std::vector<Hunk>& hunks);

/// Where a display path lives: as is when absolute, under `root` otherwise.
[[nodiscard]] std::filesystem::path absolute_in(const std::filesystem::path& root,
                                                const std::string& display);

enum class ReadFailure { Missing, TooLarge, Unreadable };

/// The whole content of a regular file, bounded by kMaxWholeReadBytes.
[[nodiscard]] std::expected<std::string, ReadFailure> read_whole(
    const std::filesystem::path& path);

} // namespace core::changes::detail
