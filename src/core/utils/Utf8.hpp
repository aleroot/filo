#pragma once

#include <cstddef>
#include <string>

namespace core::utils::utf8 {

/// Writes the UTF-8 encoding of @p code_point (at most U+10FFFF) to @p out and
/// returns its length, 1 to 4 bytes. Validating the code point is the caller's
/// job: surrogates are encoded as given.
[[nodiscard]] constexpr std::size_t encode(char32_t code_point, char out[4]) noexcept {
    if (code_point < 0x80) {
        out[0] = static_cast<char>(code_point);
        return 1;
    }
    if (code_point < 0x800) {
        out[0] = static_cast<char>(0xC0 | (code_point >> 6));
        out[1] = static_cast<char>(0x80 | (code_point & 0x3F));
        return 2;
    }
    if (code_point < 0x10000) {
        out[0] = static_cast<char>(0xE0 | (code_point >> 12));
        out[1] = static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
        out[2] = static_cast<char>(0x80 | (code_point & 0x3F));
        return 3;
    }
    out[0] = static_cast<char>(0xF0 | (code_point >> 18));
    out[1] = static_cast<char>(0x80 | ((code_point >> 12) & 0x3F));
    out[2] = static_cast<char>(0x80 | ((code_point >> 6) & 0x3F));
    out[3] = static_cast<char>(0x80 | (code_point & 0x3F));
    return 4;
}

/// Appends the UTF-8 encoding of @p code_point to @p out; see @ref encode.
inline void append(std::string& out, char32_t code_point) {
    char bytes[4];
    out.append(bytes, encode(code_point, bytes));
}

} // namespace core::utils::utf8
