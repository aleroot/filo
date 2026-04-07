#pragma once

#include "AsciiUtils.hpp"

#include <algorithm>
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

[[nodiscard]] inline std::string trim_trailing(std::string_view value, char ch) {
    std::size_t end = value.size();
    while (end > 0 && value[end - 1] == ch) --end;
    return std::string(value.substr(0, end));
}

[[nodiscard]] inline std::string trim_trailing_slashes(std::string_view value) {
    return trim_trailing(value, '/');
}

} // namespace core::utils::str
