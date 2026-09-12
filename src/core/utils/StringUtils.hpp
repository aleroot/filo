#pragma once

#include "AsciiUtils.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace core::utils::str {

class CaseInsensitiveAsciiSearcher {
public:
    explicit CaseInsensitiveAsciiSearcher(std::string_view needle)
        : needle_(needle) {
        skip_.fill(needle_.size());
        for (std::size_t i = 0; i + 1 < needle_.size(); ++i) {
            skip_[folded_index(needle_[i])] = needle_.size() - i - 1;
        }
    }

    [[nodiscard]] std::size_t find(std::string_view haystack,
                                   std::size_t from = 0) const noexcept {
        if (needle_.empty()) return std::min(from, haystack.size());
        if (from >= haystack.size() || needle_.size() > haystack.size() - from) {
            return std::string_view::npos;
        }

        if (needle_.size() == 1) {
            const auto wanted = folded_index(needle_.front());
            for (std::size_t i = from; i < haystack.size(); ++i) {
                if (folded_index(haystack[i]) == wanted) return i;
            }
            return std::string_view::npos;
        }

        const std::size_t last_start = haystack.size() - needle_.size();
        for (std::size_t start = from; start <= last_start;) {
            std::size_t index = needle_.size();
            while (index > 0
                   && core::utils::ascii::to_lower(haystack[start + index - 1])
                       == core::utils::ascii::to_lower(needle_[index - 1])) {
                --index;
            }
            if (index == 0) return start;
            start += skip_[folded_index(haystack[start + needle_.size() - 1])];
        }
        return std::string_view::npos;
    }

private:
    [[nodiscard]] static constexpr unsigned char folded_index(char ch) noexcept {
        return static_cast<unsigned char>(core::utils::ascii::to_lower(ch));
    }

    std::string_view needle_;
    std::array<std::size_t, 256> skip_{};
};

[[nodiscard]] inline std::string to_lower_ascii_copy(std::string_view value) {
    std::string out(value);
    std::ranges::transform(
        out,
        out.begin(),
        [](char ch) { return core::utils::ascii::to_lower(ch); });
    return out;
}

[[nodiscard]] inline std::string_view trim_ascii_view(std::string_view value) {
    while (!value.empty()
           && core::utils::ascii::is_space(
               static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty()
           && core::utils::ascii::is_space(
               static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return value;
}

[[nodiscard]] inline std::string trim_ascii_copy(std::string_view value) {
    return std::string(trim_ascii_view(value));
}

[[nodiscard]] inline std::string collapse_ascii_whitespace_copy(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    bool previous_space = true;
    for (const unsigned char ch : value) {
        if (core::utils::ascii::is_space(ch)) {
            if (!previous_space) out.push_back(' ');
            previous_space = true;
            continue;
        }
        out.push_back(static_cast<char>(ch));
        previous_space = false;
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

[[nodiscard]] inline bool contains_case_insensitive(std::string_view haystack,
                                                    std::string_view needle) noexcept {
    return CaseInsensitiveAsciiSearcher(needle).find(haystack) != std::string_view::npos;
}

[[nodiscard]] inline std::optional<std::size_t>
find_case_insensitive(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) return std::nullopt;
    const auto pos = CaseInsensitiveAsciiSearcher(needle).find(haystack);
    if (pos == std::string::npos) return std::nullopt;
    return pos;
}

[[nodiscard]] inline std::string trim_trailing(std::string_view value, char ch) {
    std::size_t end = value.size();
    while (end > 0 && value[end - 1] == ch) --end;
    return std::string(value.substr(0, end));
}

[[nodiscard]] inline std::string trim_trailing_slashes(std::string_view value) {
    return trim_trailing(value, '/');
}

} // namespace core::utils::str
