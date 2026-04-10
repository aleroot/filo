#pragma once

#include "AsciiUtils.hpp"

#include <charconv>
#include <optional>
#include <string>
#include <string_view>

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

[[nodiscard]] inline std::optional<std::string_view> extract_http_host(std::string_view url) noexcept {
    std::size_t scheme_len = 0;
    if (ascii::istarts_with(url, "http://")) {
        scheme_len = 7;
    } else if (ascii::istarts_with(url, "https://")) {
        scheme_len = 8;
    } else {
        return std::nullopt;
    }

    const std::string_view rest = url.substr(scheme_len);
    if (rest.empty()) return std::nullopt;

    const std::size_t authority_end = rest.find_first_of("/?#");
    const std::string_view authority =
        authority_end == std::string_view::npos ? rest : rest.substr(0, authority_end);
    if (authority.empty()) return std::nullopt;

    if (authority.find('@') != std::string_view::npos) {
        // Reject userinfo to keep host matching strict and predictable.
        return std::nullopt;
    }

    if (authority.front() == '[') {
        const std::size_t closing_bracket = authority.find(']');
        if (closing_bracket == std::string_view::npos || closing_bracket == 1) {
            return std::nullopt;
        }

        if (closing_bracket + 1 < authority.size()) {
            if (authority[closing_bracket + 1] != ':'
                || closing_bracket + 2 >= authority.size()) {
                return std::nullopt;
            }
        }

        return authority.substr(1, closing_bracket - 1);
    }

    const std::size_t colon = authority.find(':');
    if (colon == 0) return std::nullopt;
    if (colon != std::string_view::npos && colon + 1 >= authority.size()) {
        return std::nullopt;
    }
    return colon == std::string_view::npos ? authority : authority.substr(0, colon);
}

[[nodiscard]] inline bool is_loopback_host(std::string_view host) noexcept {
    return ascii::iequals(host, "localhost")
        || host == "127.0.0.1"
        || host == "::1"
        || host == "[::1]";
}

[[nodiscard]] inline bool is_loopback_http_url(std::string_view url) noexcept {
    const auto host = extract_http_host(url);
    return host.has_value() && is_loopback_host(*host);
}

} // namespace core::utils::uri
