#include "StoredDiff.hpp"

#include <charconv>
#include <fstream>
#include <system_error>

namespace core::changes::detail {
namespace {

namespace fs = std::filesystem;
using core::tools::detail::DiffLine;
using core::tools::detail::join_diff_lines;
using core::tools::detail::lines_equal;
using core::tools::detail::split_diff_lines;

constexpr std::string_view kNoNewlineMarker = "\\ No newline at end of file";

/// Which side of a hunk the last line read belonged to, so a following "no
/// newline" marker can flag the line it describes.
enum class HunkSide { None, Before, After, Both };

bool read_number(std::string_view& text, std::size_t& out) {
    int value = 0;
    const char* const first = text.data();
    const char* const last = text.data() + text.size();
    const auto result = std::from_chars(first, last, value);
    if (result.ec != std::errc{} || value < 0) {
        return false;
    }
    out = static_cast<std::size_t>(value);
    text.remove_prefix(static_cast<std::size_t>(result.ptr - first));
    return true;
}

/// Reads `start[,count]`, where an omitted count means one line.
bool read_range(std::string_view& text, std::size_t& start, std::size_t& count) {
    if (!read_number(text, start)) {
        return false;
    }
    count = 1;
    if (text.empty() || text.front() != ',') {
        return true;
    }
    text.remove_prefix(1);
    return read_number(text, count);
}

/// Parses `@@ -old +new @@`. Only the new side matters: a revert walks the
/// file as the turn left it and puts each hunk's before side back.
std::optional<Hunk> parse_hunk_header(std::string_view header) {
    if (!header.starts_with("@@ ")) {
        return std::nullopt;
    }
    header.remove_prefix(3);
    if (header.empty() || header.front() != '-') {
        return std::nullopt;
    }
    header.remove_prefix(1);
    Hunk hunk;
    std::size_t old_start = 0;
    if (!read_range(header, old_start, hunk.before_count)) {
        return std::nullopt;
    }
    if (header.empty() || header.front() != ' ') {
        return std::nullopt;
    }
    header.remove_prefix(1);
    if (header.empty() || header.front() != '+') {
        return std::nullopt;
    }
    header.remove_prefix(1);
    std::size_t new_start = 0;
    if (!read_range(header, new_start, hunk.after_count)) {
        return std::nullopt;
    }
    // A side with no lines names the line it sits after, so it is already
    // zero-based; a side with lines is one-based. This is the exact inverse of
    // the header the diff builder wrote.
    hunk.position = hunk.after_count == 0 ? new_start : new_start - 1;
    return hunk;
}

} // namespace

/// Reads the hunks of a diff the tracker stored. Anything that is not one of
/// our own diffs is refused whole: a revert that half understands a patch is
/// how a file gets rewritten wrongly.
std::optional<std::vector<Hunk>> parse_hunks(std::string_view diff) {
    std::vector<Hunk> hunks;
    std::optional<Hunk> current;
    auto side = HunkSide::None;

    // A hunk is complete when its two sides hold as many lines as its header
    // promised, and when it starts after the previous one ends: the builder
    // emits ordered, non-overlapping hunks, and a revert applies them back to
    // front on that assumption.
    const auto finish = [&hunks](Hunk& hunk) {
        if (hunk.before.size() != hunk.before_count || hunk.after.size() != hunk.after_count) {
            return false;
        }
        if (!hunks.empty()) {
            const Hunk& previous = hunks.back();
            if (hunk.position < previous.position + previous.after.size()) {
                return false;
            }
        }
        hunks.push_back(std::move(hunk));
        return true;
    };

    while (!diff.empty()) {
        const auto newline = diff.find('\n');
        const std::string_view line = diff.substr(0, newline);
        diff.remove_prefix(newline == std::string_view::npos ? diff.size() : newline + 1);

        if (line.starts_with("@@")) {
            auto header = parse_hunk_header(line);
            if (!header) {
                return std::nullopt;
            }
            if (current && !finish(*current)) {
                return std::nullopt;
            }
            current = std::move(header);
            side = HunkSide::None;
            continue;
        }
        if (!current) {
            continue;  // The ---/+++ pair, which names paths this revert already has.
        }
        if (line == kNoNewlineMarker) {
            const auto flag = [](std::vector<DiffLine>& lines) {
                if (!lines.empty()) {
                    lines.back().terminated = false;
                }
            };
            switch (side) {
                case HunkSide::Before: flag(current->before); break;
                case HunkSide::After:  flag(current->after); break;
                case HunkSide::Both:   flag(current->before); flag(current->after); break;
                case HunkSide::None:   return std::nullopt;
            }
            continue;
        }
        if (line.empty()) {
            return std::nullopt;  // Every hunk line carries its prefix, so this is not ours.
        }
        const DiffLine parsed{.text = line.substr(1), .terminated = true};
        switch (line.front()) {
            case ' ':
                current->before.push_back(parsed);
                current->after.push_back(parsed);
                side = HunkSide::Both;
                break;
            case '-':
                current->before.push_back(parsed);
                side = HunkSide::Before;
                break;
            case '+':
                current->after.push_back(parsed);
                side = HunkSide::After;
                break;
            default:
                return std::nullopt;
        }
    }
    if (current && !finish(*current)) {
        return std::nullopt;
    }
    return hunks.empty() ? std::nullopt : std::optional(std::move(hunks));
}

/// Puts every hunk's before side back where its after side is now. No value
/// when the file no longer holds what the turn left, which is the answer that
/// matters: it means something edited the file since, and a revert that
/// proceeded would erase that edit.
std::optional<std::string> reverse_apply(std::string_view current,
                                         const std::vector<Hunk>& hunks) {
    auto lines = split_diff_lines(current);
    // Back to front, so the positions the header recorded stay valid for the
    // hunks that have not been applied yet.
    for (auto it = hunks.rbegin(); it != hunks.rend(); ++it) {
        const Hunk& hunk = *it;
        if (hunk.position + hunk.after.size() > lines.size()) {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < hunk.after.size(); ++i) {
            if (!lines_equal(lines[hunk.position + i], hunk.after[i])) {
                return std::nullopt;
            }
        }
        const auto at = lines.begin() + static_cast<std::ptrdiff_t>(hunk.position);
        lines.erase(at, at + static_cast<std::ptrdiff_t>(hunk.after.size()));
        lines.insert(at, hunk.before.begin(), hunk.before.end());
    }
    return join_diff_lines(lines);
}

fs::path absolute_in(const fs::path& root, const std::string& display) {
    const fs::path path(display);
    return path.is_absolute() ? path : root / path;
}

std::expected<std::string, ReadFailure> read_whole(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) {
        return std::unexpected(fs::exists(path, ec) ? ReadFailure::Unreadable
                                                    : ReadFailure::Missing);
    }
    if (size > kMaxWholeReadBytes) {
        return std::unexpected(ReadFailure::TooLarge);
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(ReadFailure::Unreadable);
    }
    std::string content(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !in.read(content.data(), static_cast<std::streamsize>(size))) {
        return std::unexpected(ReadFailure::Unreadable);
    }
    return content;
}

} // namespace core::changes::detail
