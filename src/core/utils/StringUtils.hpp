#pragma once

#include "AsciiUtils.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace core::utils::str {

[[nodiscard]] inline std::string to_lower_ascii_copy(std::string_view value) {
    std::string out(value);
    std::ranges::transform(
        out,
        out.begin(),
        [](char ch) { return core::utils::ascii::to_lower(ch); });
    return out;
}

[[nodiscard]] inline std::string_view trim_ascii_view(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
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
        if (std::isspace(ch)) {
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
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;
    for (std::size_t pos = 0; pos + needle.size() <= haystack.size(); ++pos) {
        if (core::utils::ascii::iequals(haystack.substr(pos, needle.size()), needle)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline std::optional<std::size_t>
find_case_insensitive(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) return std::nullopt;
    const std::string lowered_haystack = to_lower_ascii_copy(haystack);
    const std::string lowered_needle = to_lower_ascii_copy(needle);
    const auto pos = lowered_haystack.find(lowered_needle);
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
