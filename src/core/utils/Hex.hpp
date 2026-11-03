#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace core::utils {

[[nodiscard]] constexpr std::string hex_encode(std::span<const std::uint8_t> bytes) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        result[2 * i] = digits[bytes[i] >> 4];
        result[2 * i + 1] = digits[bytes[i] & 15];
    }
    return result;
}

/// The value of one hexadecimal digit (either case), or -1 for any other byte.
[[nodiscard]] constexpr int hex_digit_value(unsigned char ch) noexcept {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    return -1;
}

} // namespace core::utils
