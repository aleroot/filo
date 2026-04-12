#pragma once

#include "AsciiUtils.hpp"

#include <algorithm>
#include <cctype>
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

[[nodiscard]] inline std::string trim_trailing(std::string_view value, char ch) {
    std::size_t end = value.size();
    while (end > 0 && value[end - 1] == ch) --end;
    return std::string(value.substr(0, end));
}

[[nodiscard]] inline std::string trim_trailing_slashes(std::string_view value) {
    return trim_trailing(value, '/');
}

} // namespace core::utils::str
