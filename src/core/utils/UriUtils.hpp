#pragma once

#include "AsciiUtils.hpp"

#include <charconv>
#include <string>
#include <string_view>
#include <system_error>

namespace core::utils::uri {

[[nodiscard]] inline bool percent_decode(std::string_view encoded, std::string& decoded_out) {
    decoded_out.clear();
    decoded_out.reserve(encoded.size());

    for (std::size_t i = 0; i < encoded.size(); ++i) {
        const char ch = encoded[i];
        if (ch != '%') {
            decoded_out.push_back(ch);
            continue;
        }
        if (i + 2 >= encoded.size()) {
            return false;
        }

        unsigned value = 0;
        const char* first = encoded.data() + i + 1;
        const char* last = first + 2;
        const auto [ptr, ec] = std::from_chars(first, last, value, 16);
        if (ec != std::errc{} || ptr != last || value > 0xFF) {
            return false;
        }

        decoded_out.push_back(static_cast<char>(value));
        i += 2;
    }
    return true;
}

[[nodiscard]] inline std::string percent_encode_uri_path(std::string_view raw) {
    std::string out;
    out.reserve(raw.size() + 16);

    constexpr char kHex[] = "0123456789ABCDEF";
    for (const unsigned char ch : raw) {
        const bool safe = ascii::is_alnum(ch)
            || ch == '-'
            || ch == '.'
            || ch == '_'
            || ch == '~'
            || ch == '/'
            || ch == ':';

        if (safe) {
            out.push_back(static_cast<char>(ch));
            continue;
        }

        out.push_back('%');
        out.push_back(kHex[(ch >> 4) & 0x0F]);
        out.push_back(kHex[ch & 0x0F]);
    }
    return out;
}

} // namespace core::utils::uri
