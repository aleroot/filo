#pragma once

#include "../utils/JsonUtils.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace core::tools::detail {

inline constexpr std::size_t kMaxToolDiffInputBytes = 512 * 1024;

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

[[nodiscard]] inline std::size_t line_count_for_unified_hunk(std::string_view value) noexcept {
    if (value.empty()) {
        return 0;
    }

    return static_cast<std::size_t>(std::ranges::count(value, '\n'))
        + (value.back() == '\n' ? 0 : 1);
}

inline void append_prefixed_unified_lines(
    std::string& output,
    char prefix,
    std::string_view value)
{
    std::size_t start = 0;
    while (start < value.size()) {
        const auto next = value.find('\n', start);
        const auto end = next == std::string_view::npos ? value.size() : next;
        output.push_back(prefix);
        output.append(value.substr(start, end - start));
        output.push_back('\n');

        if (next == std::string_view::npos) {
            output += "\\ No newline at end of file\n";
            break;
        }
        start = next + 1;
    }
}

[[nodiscard]] inline std::optional<std::string> build_full_content_unified_diff(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content)
{
    if (old_content == new_content) {
        return std::nullopt;
    }

    if (old_content.size() + new_content.size() > kMaxToolDiffInputBytes) {
        return std::nullopt;
    }

    if (!is_text_like_for_diff(old_content) || !is_text_like_for_diff(new_content)) {
        return std::nullopt;
    }

    const auto old_lines = line_count_for_unified_hunk(old_content);
    const auto new_lines = line_count_for_unified_hunk(new_content);

    std::string diff;
    diff.reserve(old_content.size() + new_content.size() + file_path.size() * 2 + 96);
    diff += "--- a/";
    diff += file_path;
    diff += '\n';
    diff += "+++ b/";
    diff += file_path;
    diff += '\n';
    diff += "@@ -1,";
    diff += std::to_string(old_lines);
    diff += " +1,";
    diff += std::to_string(new_lines);
    diff += " @@\n";

    append_prefixed_unified_lines(diff, '-', old_content);
    append_prefixed_unified_lines(diff, '+', new_content);

    if (!diff.empty() && diff.back() == '\n') {
        diff.pop_back();
    }

    return diff;
}

[[nodiscard]] inline std::string json_diff_field(std::string_view diff) {
    return std::string{R"(,"diff":")"}
        + core::utils::escape_json_string(diff)
        + '"';
}

} // namespace core::tools::detail
