#pragma once

#include <algorithm>
#include <string_view>

namespace core::utils::ascii {

[[nodiscard]] constexpr char to_lower(char ch) noexcept {
    if (ch >= 'A' && ch <= 'Z') {
        return static_cast<char>(ch + ('a' - 'A'));
    }
    return ch;
}

[[nodiscard]] constexpr bool is_alnum(unsigned char ch) noexcept {
    return (ch >= '0' && ch <= '9')
        || (ch >= 'A' && ch <= 'Z')
        || (ch >= 'a' && ch <= 'z');
}

[[nodiscard]] inline bool iequals(std::string_view lhs, std::string_view rhs) noexcept {
    return lhs.size() == rhs.size()
        && std::ranges::equal(
            lhs,
            rhs,
            [](char a, char b) { return to_lower(a) == to_lower(b); });
}

[[nodiscard]] inline bool istarts_with(std::string_view value, std::string_view prefix) noexcept {
    return value.size() >= prefix.size()
        && std::ranges::equal(
            value.substr(0, prefix.size()),
            prefix,
            [](char a, char b) { return to_lower(a) == to_lower(b); });
}

} // namespace core::utils::ascii
