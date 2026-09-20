#include "Plan.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <map>
#include <ranges>
#include <utility>

namespace core::review {
namespace {

using core::utils::str::trim_ascii_view;

[[nodiscard]] std::string_view trim_cr(std::string_view line) noexcept {
    if (!line.empty() && line.back() == '\r') {
        line.remove_suffix(1);
    }
    return line;
}

[[nodiscard]] bool is_diff_git_line(std::string_view line) noexcept {
    return line.starts_with("diff --git ");
}

[[nodiscard]] std::string unquote_git_path(std::string_view raw) {
    raw = trim_ascii_view(raw);
    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
        raw.remove_prefix(1);
        raw.remove_suffix(1);
    }
    if (raw.starts_with("a/") || raw.starts_with("b/")) {
        raw.remove_prefix(2);
    }
    return std::string(raw);
}

[[nodiscard]] std::pair<std::string, std::string>
parse_diff_git_paths(std::string_view line) {
    constexpr std::string_view kPrefix = "diff --git ";
    if (!line.starts_with(kPrefix)) {
        return {};
    }
    line.remove_prefix(kPrefix.size());

    std::string first;
    std::string second;
    if (!line.empty() && line.front() == '"') {
        const auto end = line.find('"', 1);
        if (end == std::string_view::npos) {
            return {unquote_git_path(line), {}};
        }
        first = unquote_git_path(line.substr(0, end + 1));
        line.remove_prefix(end + 1);
        line = trim_ascii_view(line);
        second = unquote_git_path(line);
        return {std::move(first), std::move(second)};
    }

    const auto space = line.find(' ');
    if (space == std::string_view::npos) {
        first = unquote_git_path(line);
        return {first, first};
    }
    first = unquote_git_path(line.substr(0, space));
    second = unquote_git_path(line.substr(space + 1));
    if (second.empty()) second = first;
    return {std::move(first), std::move(second)};
}

[[nodiscard]] bool is_null_path(std::string_view path) noexcept {
    return path == "/dev/null" || path == "NUL" || path.empty();
}

[[nodiscard]] bool looks_like_plus_plus_plus(std::string_view line) noexcept {
    return line.starts_with("+++ ");
}

[[nodiscard]] bool looks_like_minus_minus_minus(std::string_view line) noexcept {
    return line.starts_with("--- ");
}

[[nodiscard]] std::string path_from_unified_header(std::string_view line) {
    if (line.size() < 4) return {};
    auto rest = trim_ascii_view(line.substr(4));
    const auto tab = rest.find('\t');
    if (tab != std::string_view::npos) {
        rest = rest.substr(0, tab);
    }
    return unquote_git_path(rest);
}

void finalize_kind(FileChange& file) {
    if (file.untracked) {
        file.kind = FileChangeKind::Added;
        return;
    }
    if (!file.old_path.empty() && file.old_path != file.path
        && !is_null_path(file.old_path) && !is_null_path(file.path)) {
        file.kind = FileChangeKind::Renamed;
        return;
    }
    if (is_null_path(file.old_path) || file.old_path.empty()) {
        if (!is_null_path(file.path)) {
            file.kind = FileChangeKind::Added;
            return;
        }
    }
    if (is_null_path(file.path)) {
        file.kind = FileChangeKind::Deleted;
        if (!file.old_path.empty()) {
            file.path = file.old_path;
        }
        return;
    }
    file.kind = FileChangeKind::Modified;
}

[[nodiscard]] bool is_tiny(const FileChange& file, const GrouperConfig& config) noexcept {
    return file.changed_lines() <= config.tiny_file_line_limit
        && file.patch.size() <= config.tiny_file_char_limit;
}

[[nodiscard]] bool group_can_take(const ReviewGroup& group,
                                  const FileChange& file,
                                  const GrouperConfig& config) noexcept {
    if (group.files.size() >= config.max_packed_files) return false;
    if (static_cast<std::size_t>(group.changed_lines()) + static_cast<std::size_t>(file.changed_lines())
        > static_cast<std::size_t>(config.max_packed_lines)) {
        return false;
    }
    if (group.patch_chars() + file.patch.size() > config.max_packed_chars) {
        return false;
    }
    return true;
}

void apply_plan_flag(ReviewGroup& group, const GrouperConfig& config) {
    if (group.files.empty()) {
        group.needs_plan = false;
        return;
    }
    int largest = 0;
    for (const auto& file : group.files) {
        largest = std::max(largest, file.changed_lines());
    }
    group.needs_plan =
        largest >= config.plan_file_line_threshold
        || (group.files.size() >= 2
            && group.changed_lines() >= config.plan_group_line_threshold);
}

} // namespace

std::vector<FileChange> parse_unified_diff(std::string_view patch) {
    std::vector<FileChange> files;
    FileChange current;
    bool in_file = false;
    bool untracked_mode = false;
    std::size_t file_start = 0;
    bool saw_hunk = false;

    auto flush = [&](std::size_t end) {
        if (!in_file) return;
        if (end > file_start) {
            current.patch = std::string(patch.substr(file_start, end - file_start));
        }
        if (current.path.empty() && !current.old_path.empty()) {
            current.path = current.old_path;
        }
        if (!current.path.empty() || !current.patch.empty()) {
            current.untracked = current.untracked || untracked_mode;
            finalize_kind(current);
            if (current.hunks.empty()) {
                current.hunks = parse_hunk_ranges(current.patch);
            }
            files.push_back(std::move(current));
        }
        current = {};
        in_file = false;
        saw_hunk = false;
    };

    std::size_t cursor = 0;
    while (cursor <= patch.size()) {
        const auto next = patch.find('\n', cursor);
        const auto end = next == std::string_view::npos ? patch.size() : next;
        const auto line = trim_cr(patch.substr(cursor, end - cursor));
        const std::size_t line_begin = cursor;

        if (line == "# Untracked file patches") {
            flush(line_begin);
            untracked_mode = true;
        } else if (is_diff_git_line(line)) {
            flush(line_begin);
            in_file = true;
            file_start = line_begin;
            auto [old_path, new_path] = parse_diff_git_paths(line);
            current.old_path = std::move(old_path);
            current.path = std::move(new_path);
            current.untracked = untracked_mode;
        } else if (!in_file && (looks_like_minus_minus_minus(line) || looks_like_plus_plus_plus(line))) {
            in_file = true;
            file_start = line_begin;
            current.untracked = untracked_mode;
            if (looks_like_minus_minus_minus(line)) {
                current.old_path = path_from_unified_header(line);
            } else {
                current.path = path_from_unified_header(line);
            }
        } else if (in_file && looks_like_minus_minus_minus(line) && !saw_hunk) {
            current.old_path = path_from_unified_header(line);
        } else if (in_file && looks_like_plus_plus_plus(line) && !saw_hunk) {
            const auto path = path_from_unified_header(line);
            if (!is_null_path(path)) {
                current.path = path;
            }
        } else if (in_file && line.starts_with("@@")) {
            saw_hunk = true;
            const auto hunks = parse_hunk_ranges(line);
            current.hunks.insert(current.hunks.end(), hunks.begin(), hunks.end());
        } else if (in_file && saw_hunk) {
            if (line.starts_with('+') && !looks_like_plus_plus_plus(line)) {
                ++current.added_lines;
            } else if (line.starts_with('-') && !looks_like_minus_minus_minus(line)) {
                ++current.deleted_lines;
            }
        } else if (in_file && line.starts_with("rename from ")) {
            current.old_path = unquote_git_path(line.substr(12));
        } else if (in_file && line.starts_with("rename to ")) {
            current.path = unquote_git_path(line.substr(10));
        } else if (in_file && line.starts_with("new file mode ")) {
            current.kind = FileChangeKind::Added;
        } else if (in_file && line.starts_with("deleted file mode ")) {
            current.kind = FileChangeKind::Deleted;
        }

        if (next == std::string_view::npos) {
            flush(patch.size());
            break;
        }
        cursor = next + 1;
    }

    return files;
}

ReviewPlan plan_review(std::span<const FileChange> files, const GrouperConfig& config) {
    ReviewPlan plan;
    const auto skip_chars = static_cast<std::size_t>(
        static_cast<double>(config.prompt_budget_chars) * config.skip_budget_ratio);

    std::vector<FileChange> accepted;
    accepted.reserve(files.size());
    for (const auto& file : files) {
        if (file.patch.size() > skip_chars) {
            const std::string path = file.path.empty() ? file.old_path : file.path;
            plan.skipped_paths.push_back(path);
            plan.warnings.push_back(std::format(
                "Skipped '{}': diff ({} chars) exceeds {:.0f}% of the per-group review budget.",
                path,
                file.patch.size(),
                config.skip_budget_ratio * 100.0));
            continue;
        }
        accepted.push_back(file);
    }

    std::ranges::sort(accepted, [](const FileChange& a, const FileChange& b) {
        if (a.changed_lines() != b.changed_lines()) {
            return a.changed_lines() > b.changed_lines();
        }
        return a.path < b.path;
    });

    // Related-file families first (foo.cpp + foo.hpp + test_foo.cpp). OCR's
    // quality comes from reviewing a unit of meaning, not a bag of tiny diffs.
    // Families use the group prompt budget rather than the tiny-file packer:
    // a 80-line .cpp plus a 40-line .hpp is exactly the unit we want together.
    std::map<std::string, std::vector<FileChange>> families;
    for (auto& file : accepted) {
        families[file_family_key(file_display_path(file))].push_back(std::move(file));
    }

    std::vector<FileChange> leftovers;
    for (auto& [key, members] : families) {
        std::ranges::sort(members, [](const FileChange& a, const FileChange& b) {
            return file_display_path(a) < file_display_path(b);
        });
        if (key.empty() || members.size() < 2) {
            for (auto& file : members) leftovers.push_back(std::move(file));
            continue;
        }
        ReviewGroup family;
        for (auto& file : members) {
            if (!family.files.empty()
                && family.patch_chars() + file.patch.size() > config.prompt_budget_chars) {
                leftovers.push_back(std::move(file));
                continue;
            }
            family.files.push_back(std::move(file));
        }
        if (family.files.size() >= 2) {
            apply_plan_flag(family, config);
            plan.groups.push_back(std::move(family));
        } else {
            for (auto& file : family.files) leftovers.push_back(std::move(file));
        }
    }

    std::vector<FileChange> tiny;
    tiny.reserve(leftovers.size());
    for (auto& file : leftovers) {
        if (is_tiny(file, config)) {
            tiny.push_back(std::move(file));
            continue;
        }
        ReviewGroup group;
        group.files.push_back(std::move(file));
        apply_plan_flag(group, config);
        plan.groups.push_back(std::move(group));
    }

    std::vector<ReviewGroup> tiny_groups;
    for (auto& file : tiny) {
        bool packed = false;
        for (auto& group : tiny_groups) {
            if (group_can_take(group, file, config)) {
                group.files.push_back(std::move(file));
                packed = true;
                break;
            }
        }
        if (!packed) {
            ReviewGroup group;
            group.files.push_back(std::move(file));
            tiny_groups.push_back(std::move(group));
        }
    }
    for (auto& group : tiny_groups) {
        apply_plan_flag(group, config);
        plan.groups.push_back(std::move(group));
    }

    std::ranges::sort(plan.groups, [](const ReviewGroup& a, const ReviewGroup& b) {
        if (a.changed_lines() != b.changed_lines()) {
            return a.changed_lines() > b.changed_lines();
        }
        return join_group_paths(a) < join_group_paths(b);
    });
    return plan;
}

std::string_view file_display_path(const FileChange& file) noexcept {
    return file.path.empty() ? file.old_path : file.path;
}

std::string join_group_paths(const ReviewGroup& group) {
    std::string out;
    for (std::size_t i = 0; i < group.files.size(); ++i) {
        if (i > 0) out += ", ";
        out += file_display_path(group.files[i]);
    }
    return out;
}

std::string absolute_review_path(std::string_view worktree_root,
                                 std::string_view relative_path) {
    if (relative_path.empty()) {
        return std::string(worktree_root);
    }
    if (relative_path.starts_with('/')
#if defined(_WIN32)
        || (relative_path.size() > 1 && std::isalpha(static_cast<unsigned char>(relative_path[0]))
            && relative_path[1] == ':')
#endif
    ) {
        return std::string(relative_path);
    }
    if (worktree_root.empty()) {
        return std::string(relative_path);
    }
    if (worktree_root.ends_with('/') || worktree_root.ends_with('\\')) {
        return std::format("{}{}", worktree_root, relative_path);
    }
    return std::format("{}/{}", worktree_root, relative_path);
}

std::string file_family_key(std::string_view path) {
    auto slash = path.find_last_of("/");
#if defined(_WIN32)
    const auto bslash = path.find_last_of('\\');
    if (bslash != std::string_view::npos
        && (slash == std::string_view::npos || bslash > slash)) {
        slash = bslash;
    }
#endif
    auto name = slash == std::string_view::npos ? path : path.substr(slash + 1);
    const auto dot = name.rfind('.');
    if (dot != std::string_view::npos && dot > 0) {
        name = name.substr(0, dot);
    }
    const std::string stem(name);
    std::string key(stem);
    for (char& ch : key) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    // Swift/JUnit/Kotlin type names: FooTests.swift, FooTest.java. Matched on
    // the original stem so "States.swift" / "Latest.swift" stay intact.
    if (stem.size() > 5 && stem.ends_with("Tests")) {
        key.resize(key.size() - 5);
    } else if (stem.size() > 4 && stem.ends_with("Test")) {
        key.resize(key.size() - 4);
    }
    constexpr std::string_view kPrefixes[] = {"test_", "test", "spec_"};
    for (const auto prefix : kPrefixes) {
        if (key.starts_with(prefix) && key.size() > prefix.size()) {
            key.erase(0, prefix.size());
            break;
        }
    }
    constexpr std::string_view kSuffixes[] = {"_test", "_spec", "_tests"};
    for (const auto suffix : kSuffixes) {
        if (key.size() > suffix.size() && key.ends_with(suffix)) {
            key.resize(key.size() - suffix.size());
            break;
        }
    }
    return key;
}

std::vector<LineRange> parse_hunk_ranges(std::string_view patch) {
    std::vector<LineRange> ranges;
    std::size_t cursor = 0;
    while (cursor < patch.size()) {
        const auto at = patch.find("@@", cursor);
        if (at == std::string_view::npos) break;
        const auto plus = patch.find('+', at);
        const auto end = patch.find("@@", at + 2);
        if (plus == std::string_view::npos || plus > (end == std::string_view::npos ? patch.size() : end)) {
            cursor = at + 2;
            continue;
        }
        std::size_t i = plus + 1;
        int start = 0;
        while (i < patch.size() && std::isdigit(static_cast<unsigned char>(patch[i]))) {
            start = start * 10 + (patch[i] - '0');
            ++i;
        }
        int count = 1;
        if (i < patch.size() && patch[i] == ',') {
            ++i;
            count = 0;
            while (i < patch.size() && std::isdigit(static_cast<unsigned char>(patch[i]))) {
                count = count * 10 + (patch[i] - '0');
                ++i;
            }
        }
        if (start > 0 && count > 0) {
            ranges.push_back(LineRange{.start = start, .end = start + count - 1});
        } else if (start > 0) {
            ranges.push_back(LineRange{.start = start, .end = start});
        }
        cursor = i;
    }
    return ranges;
}

namespace {

[[nodiscard]] std::string normalize_path_key(std::string_view path) {
    std::string out(path);
    for (char& ch : out) {
        if (ch == '\\') ch = '/';
        else ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

[[nodiscard]] bool path_matches_file(std::string_view finding_path,
                                     const FileChange& file,
                                     std::string_view worktree_root) {
    const auto file_path = normalize_path_key(file_display_path(file));
    const auto abs = normalize_path_key(absolute_review_path(worktree_root, file_display_path(file)));
    const auto needle = normalize_path_key(finding_path);
    if (needle.empty()) return false;
    return needle == file_path
        || needle == abs
        || needle.ends_with(file_path)
        || file_path.ends_with(needle);
}

[[nodiscard]] bool ranges_overlap(int a0, int a1, int b0, int b1) noexcept {
    if (a1 < a0) std::swap(a0, a1);
    if (b1 < b0) std::swap(b0, b1);
    return a0 <= b1 && b0 <= a1;
}

[[nodiscard]] int range_distance(int a0, int a1, int b0, int b1) noexcept {
    if (a1 < a0) std::swap(a0, a1);
    if (b1 < b0) std::swap(b0, b1);
    if (a0 <= b1 && b0 <= a1) return 0;
    if (a1 < b0) return b0 - a1;
    return a0 - b1;
}

} // namespace

std::size_t localize_findings(std::vector<Finding>& findings,
                              std::span<const FileChange> files,
                              std::string_view worktree_root,
                              int snap_slack_lines) {
    if (files.empty()) return 0;
    std::vector<Finding> kept;
    kept.reserve(findings.size());
    std::size_t dropped = 0;

    for (auto& finding : findings) {
        const FileChange* match = nullptr;
        for (const auto& file : files) {
            if (path_matches_file(finding.absolute_file_path, file, worktree_root)) {
                match = &file;
                break;
            }
        }
        if (match == nullptr) {
            // Path is required to overlap the diff. Unanchored comments are the
            // position-drift OCR exists to kill.
            ++dropped;
            continue;
        }
        if (match->hunks.empty()) {
            kept.push_back(std::move(finding));
            continue;
        }
        if (finding.line_start <= 0 && finding.line_end <= 0) {
            finding.line_start = match->hunks.front().start;
            finding.line_end = match->hunks.front().end;
            kept.push_back(std::move(finding));
            continue;
        }

        const int start = finding.line_start;
        const int end = finding.line_end > 0 ? finding.line_end : finding.line_start;
        const LineRange* best = nullptr;
        int best_distance = snap_slack_lines + 1;
        for (const auto& hunk : match->hunks) {
            if (ranges_overlap(start, end, hunk.start, hunk.end)) {
                best = &hunk;
                best_distance = 0;
                break;
            }
            const int distance = range_distance(start, end, hunk.start, hunk.end);
            if (distance < best_distance) {
                best_distance = distance;
                best = &hunk;
            }
        }
        if (best == nullptr || best_distance > snap_slack_lines) {
            ++dropped;
            continue;
        }
        if (best_distance > 0) {
            finding.line_start = best->start;
            finding.line_end = std::min(best->end, best->start + 4);
        }
        kept.push_back(std::move(finding));
    }

    findings = std::move(kept);
    return dropped;
}

} // namespace core::review
