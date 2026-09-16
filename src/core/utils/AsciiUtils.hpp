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

[[nodiscard]] inline bool is_ascii(std::string_view value) noexcept {
    return std::ranges::all_of(value, [](const unsigned char ch) {
        return ch < 0x80;
    });
}

[[nodiscard]] constexpr bool is_alnum(unsigned char ch) noexcept {
    return (ch >= '0' && ch <= '9')
        || (ch >= 'A' && ch <= 'Z')
        || (ch >= 'a' && ch <= 'z');
}

[[nodiscard]] constexpr bool is_space(unsigned char ch) noexcept {
    return ch == ' ' || ch == '\t' || ch == '\n'
        || ch == '\r' || ch == '\f' || ch == '\v';
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

/// Case-insensitive substring test. ASCII only, like the rest of this header:
/// it exists for filename and flag matching, not for prose.
[[nodiscard]] inline bool icontains(std::string_view value, std::string_view needle) noexcept {
    if (needle.empty()) {
        return true;
    }
    if (value.size() < needle.size()) {
        return false;
    }
    const auto last = value.size() - needle.size();
    for (std::size_t offset = 0; offset <= last; ++offset) {
        if (std::ranges::equal(
                value.substr(offset, needle.size()),
                needle,
                [](char a, char b) { return to_lower(a) == to_lower(b); })) {
            return true;
        }
    }
    return false;
}

} // namespace core::utils::ascii
