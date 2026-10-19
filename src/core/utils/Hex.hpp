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

} // namespace core::utils
