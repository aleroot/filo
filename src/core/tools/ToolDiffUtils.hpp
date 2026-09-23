#pragma once

#include "../utils/JsonUtils.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace core::tools::detail {

inline constexpr std::size_t kMaxToolDiffInputBytes = 512 * 1024;
inline constexpr std::size_t kMaxToolDiffOutputBytes = 512 * 1024;

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

/// Builds a bounded, line-based unified diff. Returns no value when either
/// input is unsuitable for a text diff or the bounded Myers search exceeds its
/// work limit. In particular, a failed diff is never replaced with a
/// whole-file remove/add hunk.
[[nodiscard]] std::optional<std::string> build_unified_diff(
    std::string_view file_path,
    std::string_view old_content,
    std::string_view new_content);

[[nodiscard]] inline std::string json_diff_field(std::string_view diff) {
    return std::string{R"(,"diff":")"}
        + core::utils::escape_json_string(diff)
        + '"';
}

} // namespace core::tools::detail
