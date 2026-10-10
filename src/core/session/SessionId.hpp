#pragma once

#include "../utils/AsciiUtils.hpp"

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace core::session {

inline constexpr std::size_t kSessionIdLength = 8;

[[nodiscard]] constexpr bool is_session_id(std::string_view text) noexcept {
    return text.size() == kSessionIdLength && std::ranges::all_of(text, [](char ch) {
        return core::utils::ascii::is_digit(static_cast<unsigned char>(ch))
            || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
    });
}

} // namespace core::session
