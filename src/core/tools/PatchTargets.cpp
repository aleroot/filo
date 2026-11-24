#include "PatchTargets.hpp"

#include <cctype>
#include <format>
#include <optional>
#include <string>
#include <vector>

namespace core::tools {
namespace {

std::string trim_ascii(std::string_view input) {
    std::size_t start = 0;
    while (start < input.size()
           && std::isspace(static_cast<unsigned char>(input[start])) != 0) {
        ++start;
    }

    std::size_t end = input.size();
    while (end > start
           && std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return std::string(input.substr(start, end - start));
}

std::optional<std::string> extract_patch_path(std::string_view line) {
    if (!(line.starts_with("--- ") || line.starts_with("+++ "))) {
        return std::nullopt;
    }

    std::string raw = trim_ascii(line.substr(4));
    if (raw.empty()) {
        return std::nullopt;
    }

    const auto tab_pos = raw.find('\t');
    if (tab_pos != std::string::npos) {
        raw.erase(tab_pos);
        raw = trim_ascii(raw);
    }

    if (raw == "/dev/null") {
        return std::nullopt;
    }

    if ((raw.starts_with("a/") || raw.starts_with("b/")) && raw.size() > 2) {
        raw.erase(0, 2);
    }

    if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
        raw = raw.substr(1, raw.size() - 2);
    }

    if (raw.empty()) {
        return std::nullopt;
    }
    return raw;
}

template <typename Visitor>
void for_each_line(std::string_view text, Visitor&& visit) {
    while (!text.empty()) {
        const auto newline = text.find('\n');
        auto line = text.substr(0, newline == std::string_view::npos ? text.size() : newline);
        if (line.ends_with('\r')) line.remove_suffix(1);
        visit(line);
        if (newline == std::string_view::npos) break;
        text.remove_prefix(newline + 1);
    }
}

[[nodiscard]] std::string_view trim_trailing(std::string_view line) noexcept {
    while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.remove_suffix(1);
    return line;
}

[[nodiscard]] bool is_codex_envelope(std::string_view text) noexcept {
    bool any = false;
    bool codex = false;
    for_each_line(text, [&](std::string_view line) {
        if (any) return;
        if (trim_trailing(line).empty()) return;
        any = true;
        codex = trim_trailing(line) == "*** Begin Patch";
    });
    return codex;
}

[[nodiscard]] std::optional<std::string> codex_path(std::string_view line, std::string_view prefix) {
    const auto trimmed = trim_trailing(line);
    if (!trimmed.starts_with(prefix)) return std::nullopt;
    auto path = trim_ascii(trimmed.substr(prefix.size()));
    if (path.empty()) return std::nullopt;
    return path;
}

std::vector<std::string> codex_target_paths(std::string_view patch_text) {
    std::vector<std::string> paths;
    for_each_line(patch_text, [&](std::string_view line) {
        for (const auto prefix : {
                 std::string_view{"*** Update File: "},
                 std::string_view{"*** Add File: "},
                 std::string_view{"*** Delete File: "},
                 std::string_view{"*** Move to: "}}) {
            if (auto path = codex_path(line, prefix)) {
                paths.push_back(std::move(*path));
                return;
            }
        }
    });
    return paths;
}

enum class HunkSide { None, Old, New, Both, Meta };

[[nodiscard]] HunkSide hunk_side(std::string_view line) noexcept {
    if (line.starts_with("\\")) return HunkSide::Meta;
    if (line.empty() || line.front() == ' ') return HunkSide::Both;
    if (line.front() == '-' && !line.starts_with("--- ")) return HunkSide::Old;
    if (line.front() == '+' && !line.starts_with("+++ ")) return HunkSide::New;
    return HunkSide::None;
}

struct HunkCounts {
    int old_count = 0;
    int new_count = 0;
};

[[nodiscard]] std::optional<HunkCounts> parse_hunk_counts(std::string_view line) {
    if (!line.starts_with("@@ ")) return std::nullopt;
    auto rest = line.substr(3);
    auto range = [&](char sign, int& count) {
        if (rest.empty() || rest.front() != sign) return false;
        rest.remove_prefix(1);
        if (rest.empty() || std::isdigit(static_cast<unsigned char>(rest.front())) == 0) return false;
        std::size_t index = 0;
        while (index < rest.size() && std::isdigit(static_cast<unsigned char>(rest[index])) != 0) ++index;
        rest.remove_prefix(index);
        count = 1;
        if (!rest.empty() && rest.front() == ',') {
            rest.remove_prefix(1);
            if (rest.empty() || std::isdigit(static_cast<unsigned char>(rest.front())) == 0) return false;
            int value = 0;
            index = 0;
            while (index < rest.size() && std::isdigit(static_cast<unsigned char>(rest[index])) != 0) {
                value = value * 10 + (rest[index] - '0');
                ++index;
            }
            count = value;
            rest.remove_prefix(index);
        }
        return true;
    };
    HunkCounts counts;
    if (!range('-', counts.old_count) || rest.empty() || rest.front() != ' ') return std::nullopt;
    rest.remove_prefix(1);
    if (!range('+', counts.new_count) || !rest.starts_with(" @@")) return std::nullopt;
    return counts;
}

} // namespace

std::vector<std::string> patch_target_paths(std::string_view patch_text) {
    if (is_codex_patch(patch_text)) return codex_target_paths(patch_text);

    std::vector<std::string> paths;
    for_each_line(patch_text, [&](std::string_view line) {
        if (auto path = extract_patch_path(line)) paths.push_back(std::move(*path));
    });
    return paths;
}

bool is_codex_patch(std::string_view patch_text) noexcept {
    return is_codex_envelope(patch_text);
}

std::optional<std::string> unified_diff_diagnostic(std::string_view patch_text) {
    if (is_codex_patch(patch_text)) return std::nullopt;

    std::vector<std::string> lines;
    for_each_line(patch_text, [&](std::string_view line) {
        lines.emplace_back(line);
    });

    std::string file;
    int hunk_index = 0;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (lines[i].starts_with("+++ ")) {
            if (auto path = extract_patch_path(lines[i])) file = std::move(*path);
            hunk_index = 0;
            continue;
        }
        if (!lines[i].starts_with("@@ ")) continue;
        const auto counts = parse_hunk_counts(lines[i]);
        if (!counts) return std::string{"Malformed unified diff: a hunk header is not '@@ -old +new @@'."};
        ++hunk_index;

        int old_seen = 0;
        int new_seen = 0;
        std::size_t j = i + 1;
        auto mismatch = [&](int old_lines, int new_lines) {
            const auto where = file.empty()
                ? std::format("hunk {}", hunk_index)
                : std::format("hunk {} of {}", hunk_index, file);
            return std::format(
                "Malformed unified diff in {}: it declares {} old lines and {} new lines, "
                "but the body has {} old lines and {} new lines.",
                where, counts->old_count, counts->new_count, old_lines, new_lines);
        };
        for (; j < lines.size(); ++j) {
            const auto side = hunk_side(lines[j]);
            if (side == HunkSide::Meta) continue;
            if (side == HunkSide::None) break;
            if (side == HunkSide::Old || side == HunkSide::Both) ++old_seen;
            if (side == HunkSide::New || side == HunkSide::Both) ++new_seen;
            if (old_seen > counts->old_count || new_seen > counts->new_count) {
                return mismatch(old_seen, new_seen);
            }
            if (old_seen == counts->old_count && new_seen == counts->new_count) {
                std::size_t look = j + 1;
                while (look < lines.size() && hunk_side(lines[look]) == HunkSide::Meta) ++look;
                if (look < lines.size()) {
                    const auto extra = hunk_side(lines[look]);
                    if (extra == HunkSide::Old || extra == HunkSide::New || extra == HunkSide::Both) {
                        const auto extra_old = old_seen + (extra == HunkSide::New ? 0 : 1);
                        const auto extra_new = new_seen + (extra == HunkSide::Old ? 0 : 1);
                        return mismatch(extra_old, extra_new);
                    }
                }
                j = look;
                break;
            }
        }
        if (old_seen != counts->old_count || new_seen != counts->new_count) {
            return mismatch(old_seen, new_seen);
        }
        if (j > i) i = j - 1;
    }
    return std::nullopt;
}

} // namespace core::tools
