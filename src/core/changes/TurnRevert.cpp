#include "TurnRevert.hpp"

#include "../tools/ToolDiffUtils.hpp"
#include "../utils/InterprocessFile.hpp"

#include <charconv>
#include <cstdint>
#include <format>
#include <fstream>
#include <system_error>

namespace core::changes {
namespace {

namespace fs = std::filesystem;
using core::tools::detail::DiffLine;
using core::tools::detail::join_diff_lines;
using core::tools::detail::lines_equal;
using core::tools::detail::split_diff_lines;

/// A revert reads a file whole to prove it still matches the turn. Past this it
/// is cheaper to refuse than to pull a file into memory that no diff of this
/// turn could have described (the tracker only diffs up to 512 KiB a side).
constexpr std::uintmax_t kMaxRevertReadBytes = 8 * 1024 * 1024;

constexpr std::string_view kNoNewlineMarker = "\\ No newline at end of file";

/// One hunk of a stored unified diff: the two sides it relates and where the
/// "after" side sits in the file. The line views borrow the diff text.
struct Hunk {
    /// Zero-based line the after side starts at.
    std::size_t position = 0;
    std::size_t before_count = 0;
    std::size_t after_count = 0;
    std::vector<DiffLine> before;  ///< Context and removed lines
    std::vector<DiffLine> after;   ///< Context and added lines
};

enum class ContentAction { Keep, Write, Remove };

/// What one file needs in order to be back where the turn found it.
struct Restoration {
    std::string display_path;
    /// Where the file belongs once the revert is done.
    fs::path origin;
    /// Where it is now, when that is somewhere else: a rename to undo.
    std::optional<fs::path> misplaced;
    ContentAction action = ContentAction::Keep;
    std::string content;  ///< Write only
};

/// Which side of a hunk the last line read belonged to, so a following "no
/// newline" marker can flag the line it describes.
enum class HunkSide { None, Before, After, Both };

[[nodiscard]] fs::path absolute_in(const fs::path& root, const std::string& display) {
    const fs::path path(display);
    return path.is_absolute() ? path : root / path;
}

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

std::optional<std::string> read_whole(const fs::path& path, std::string& reason) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) {
        reason = fs::exists(path, ec) ? "could not be read" : "was deleted since that turn";
        return std::nullopt;
    }
    if (size > kMaxRevertReadBytes) {
        reason = "grew past what a revert reads";
        return std::nullopt;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        reason = "could not be read";
        return std::nullopt;
    }
    std::string content(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !in.read(content.data(), static_cast<std::streamsize>(size))) {
        reason = "could not be read";
        return std::nullopt;
    }
    return content;
}

[[nodiscard]] std::string_view undiffed_phrase(FileChangeContent content) noexcept {
    switch (content) {
        case FileChangeContent::Binary:
            return "is a binary file, whose previous content was never kept";
        case FileChangeContent::TooLarge:
            return "is too large for the diff a revert reads";
        case FileChangeContent::BudgetSpent:
            return "kept no diff, because that turn's diff budget was spent";
        case FileChangeContent::Text:
            break;
    }
    return "kept no diff";
}

/// Plans one file's part of the revert, or says why it cannot be honoured.
std::optional<Restoration> plan_restoration(const FileChange& change,
                                            const fs::path& root,
                                            std::string& reason) {
    const fs::path target = absolute_in(root, change.path);
    Restoration step{.display_path = change.path, .origin = target};
    std::error_code ec;
    const bool target_exists = fs::exists(target, ec);

    // An empty diff with text content means the two sides were byte-identical:
    // a rename that edited nothing, or a file the turn created or deleted
    // while it was empty. Each of those has an exact pre-image without a hunk.
    // Any other change without a diff (binary, too large, budget spent) has
    // none, and is refused rather than approximated.
    if (change.diff.empty()) {
        if (change.content != FileChangeContent::Text) {
            reason = std::string(undiffed_phrase(change.content));
            return std::nullopt;
        }
        switch (change.kind) {
            case FileChangeKind::Renamed: {
                if (change.previous_path.empty()) {
                    reason = "kept no record of where it was moved from";
                    return std::nullopt;
                }
                step.origin = absolute_in(root, change.previous_path);
                if (target_exists) {
                    step.misplaced = target;
                } else if (!fs::exists(step.origin, ec)) {
                    reason = "was deleted since that turn";
                    return std::nullopt;
                }
                return step;
            }
            case FileChangeKind::Added: {
                if (!target_exists) {
                    return step;  // Already gone; nothing to undo.
                }
                const auto current = read_whole(target, reason);
                if (!current) {
                    return std::nullopt;  // read_whole named the reason.
                }
                if (!current->empty()) {
                    reason = "changed since that turn";
                    return std::nullopt;
                }
                step.action = ContentAction::Remove;
                return step;
            }
            case FileChangeKind::Deleted: {
                if (target_exists) {
                    std::string unreadable;
                    const auto current = read_whole(target, unreadable);
                    if (!current || !current->empty()) {
                        reason = "was recreated with different content since that turn";
                        return std::nullopt;
                    }
                    return step;  // Already back; nothing to write.
                }
                step.action = ContentAction::Write;  // An empty file, restored.
                return step;
            }
            case FileChangeKind::Modified:
                break;
        }
        reason = std::string(undiffed_phrase(change.content));
        return std::nullopt;
    }

    const auto hunks = parse_hunks(change.diff);
    if (!hunks) {
        reason = "kept a diff this build cannot read";
        return std::nullopt;
    }

    switch (change.kind) {
        case FileChangeKind::Added: {
            const auto current = read_whole(target, reason);
            if (!current) {
                return std::nullopt;
            }
            const auto restored = reverse_apply(*current, *hunks);
            // The turn created this file, so undoing it leaves nothing. Text
            // still standing after the hunks went back means the file is not
            // only what the turn wrote.
            if (!restored || !restored->empty()) {
                reason = "changed since that turn";
                return std::nullopt;
            }
            step.action = ContentAction::Remove;
            return step;
        }
        case FileChangeKind::Deleted: {
            // Deleting a file diffs its whole content, so one hunk holds the
            // pre-image. More than one means gaps this revert would not see.
            if (hunks->size() != 1) {
                reason = "kept no diff of its whole previous content";
                return std::nullopt;
            }
            std::string content = join_diff_lines(hunks->front().before);
            std::error_code ec;
            if (fs::exists(target, ec)) {
                std::string unused;
                const auto current = read_whole(target, unused);
                if (!current || *current != content) {
                    reason = "was recreated with different content since that turn";
                    return std::nullopt;
                }
                return step;  // Already back; nothing to write.
            }
            step.action = ContentAction::Write;
            step.content = std::move(content);
            return step;
        }
        case FileChangeKind::Renamed: {
            if (change.previous_path.empty()) {
                reason = "kept no record of where it was moved from";
                return std::nullopt;
            }
            step.origin = absolute_in(root, change.previous_path);
            std::error_code ec;
            if (!fs::exists(target, ec)) {
                if (fs::exists(step.origin, ec)) {
                    return step;  // Already back; nothing to write.
                }
                reason = "was deleted since that turn";
                return std::nullopt;
            }
            if (step.origin != target && fs::exists(step.origin, ec)) {
                reason = std::format("cannot be moved back: '{}' exists again",
                                     change.previous_path);
                return std::nullopt;
            }
            const auto current = read_whole(target, reason);
            if (!current) {
                return std::nullopt;
            }
            const auto restored = reverse_apply(*current, *hunks);
            if (!restored) {
                reason = "changed since that turn";
                return std::nullopt;
            }
            step.misplaced = target;
            step.action = ContentAction::Write;
            step.content = std::move(*restored);
            return step;
        }
        case FileChangeKind::Modified: {
            const auto current = read_whole(target, reason);
            if (!current) {
                return std::nullopt;
            }
            const auto restored = reverse_apply(*current, *hunks);
            if (!restored) {
                reason = "changed since that turn";
                return std::nullopt;
            }
            step.action = ContentAction::Write;
            step.content = std::move(*restored);
            return step;
        }
    }
    reason = "is not a change a revert understands";
    return std::nullopt;
}

/// Writes one planned step. The rename goes first, so a file that has to move
/// back is written at the path it is restored to.
bool perform(const Restoration& step, std::string& error) {
    if (step.misplaced && *step.misplaced != step.origin) {
        std::error_code ec;
        if (const auto parent = step.origin.parent_path(); !parent.empty()) {
            fs::create_directories(parent, ec);
        }
        fs::rename(*step.misplaced, step.origin, ec);
        if (ec) {
            // A rename across filesystems fails; copy and remove, as the move
            // tool does, so a revert is not refused by a mount point.
            fs::copy_file(*step.misplaced, step.origin,
                          fs::copy_options::overwrite_existing, ec);
            if (ec) {
                error = std::format("could not be moved back: {}", ec.message());
                return false;
            }
            fs::remove(*step.misplaced, ec);
        }
    }
    switch (step.action) {
        case ContentAction::Keep:
            return true;
        case ContentAction::Remove: {
            std::error_code ec;
            fs::remove(step.origin, ec);
            if (ec) {
                error = std::format("could not be removed: {}", ec.message());
                return false;
            }
            return true;
        }
        case ContentAction::Write:
            // The same crash-safe write the rest of Filo uses: it keeps the
            // file's mode and never leaves a half-written source file behind.
            if (core::utils::atomic_write_file(step.origin, step.content, &error)) {
                return true;
            }
            error = std::format("could not be written: {}", error);
            return false;
    }
    return false;
}

} // namespace

RevertResult revert_turn(const TurnChanges& changes, const fs::path& workspace_root) {
    RevertResult result;
    std::vector<Restoration> plan;
    plan.reserve(changes.files.size());
    for (const auto& change : changes.files) {
        std::string reason;
        if (auto step = plan_restoration(change, workspace_root, reason)) {
            plan.push_back(std::move(*step));
            continue;
        }
        result.refused.push_back(RevertRefusal{.path = change.path, .reason = std::move(reason)});
    }
    // Nothing is written until every change is known to be honourable.
    if (!result.refused.empty()) {
        return result;
    }

    for (const auto& step : plan) {
        std::string error;
        if (!perform(step, error)) {
            result.error = std::format("{} {}.", step.display_path, error);
            return result;
        }
        result.restored.push_back(step.display_path);
    }
    return result;
}

} // namespace core::changes
