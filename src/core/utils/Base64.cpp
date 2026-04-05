#include "Base64.hpp"

#include <array>
#include <cstddef>

namespace core::utils {

namespace {

constexpr std::string_view kStandardAlphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
constexpr std::string_view kUrlAlphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

constexpr std::array<std::int8_t, 256> make_decode_table(std::string_view alphabet) {
    std::array<std::int8_t, 256> table{};
    table.fill(-1);
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        table[static_cast<unsigned char>(alphabet[i])] = static_cast<std::int8_t>(i);
    }
    return table;
}

constexpr auto kStandardDecodeTable = make_decode_table(kStandardAlphabet);

[[nodiscard]] std::string encode_impl(std::span<const std::uint8_t> input,
                                      std::string_view alphabet,
                                      bool include_padding) {
    std::string out;
    out.reserve(((input.size() + 2) / 3) * 4);

    std::size_t i = 0;
    while (i + 3 <= input.size()) {
        const std::uint32_t chunk =
            (static_cast<std::uint32_t>(input[i]) << 16)
            | (static_cast<std::uint32_t>(input[i + 1]) << 8)
            | static_cast<std::uint32_t>(input[i + 2]);
        out.push_back(alphabet[(chunk >> 18) & 0x3F]);
        out.push_back(alphabet[(chunk >> 12) & 0x3F]);
        out.push_back(alphabet[(chunk >> 6) & 0x3F]);
        out.push_back(alphabet[chunk & 0x3F]);
        i += 3;
    }

    const std::size_t remaining = input.size() - i;
    if (remaining == 1) {
        const std::uint32_t chunk = static_cast<std::uint32_t>(input[i]) << 16;
        out.push_back(alphabet[(chunk >> 18) & 0x3F]);
        out.push_back(alphabet[(chunk >> 12) & 0x3F]);
        if (include_padding) {
            out.push_back('=');
            out.push_back('=');
        }
    } else if (remaining == 2) {
        const std::uint32_t chunk =
            (static_cast<std::uint32_t>(input[i]) << 16)
            | (static_cast<std::uint32_t>(input[i + 1]) << 8);
        out.push_back(alphabet[(chunk >> 18) & 0x3F]);
        out.push_back(alphabet[(chunk >> 12) & 0x3F]);
        out.push_back(alphabet[(chunk >> 6) & 0x3F]);
        if (include_padding) {
            out.push_back('=');
        }
    }

    return out;
}

[[nodiscard]] std::optional<std::string> decode_impl(
    std::string_view input,
    const std::array<std::int8_t, 256>& table) {
    if (input.empty()) {
        return std::string{};
    }
    if (input.size() % 4 != 0) {
        return std::nullopt;
    }

    std::size_t padding = 0;
    if (!input.empty() && input.back() == '=') {
        padding = 1;
        if (input.size() >= 2 && input[input.size() - 2] == '=') {
            padding = 2;
        }
    }

    const std::size_t non_padding_length = input.size() - padding;
    for (std::size_t i = 0; i < non_padding_length; ++i) {
        const unsigned char ch = static_cast<unsigned char>(input[i]);
        if (ch == '=' || table[ch] < 0) {
            return std::nullopt;
        }
    }
    for (std::size_t i = non_padding_length; i < input.size(); ++i) {
        if (input[i] != '=') {
            return std::nullopt;
        }
    }

    std::string out;
    out.reserve((input.size() / 4) * 3 - padding);

    for (std::size_t i = 0; i < input.size(); i += 4) {
        const unsigned char c0 = static_cast<unsigned char>(input[i]);
        const unsigned char c1 = static_cast<unsigned char>(input[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(input[i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(input[i + 3]);

        if (c0 == '=' || c1 == '=') {
            return std::nullopt;
        }

        const int v0 = table[c0];
        const int v1 = table[c1];
        if (v0 < 0 || v1 < 0) {
            return std::nullopt;
        }

        const bool pad2 = c2 == '=';
        const bool pad3 = c3 == '=';
        if (pad2 && !pad3) {
            return std::nullopt;
        }

        const int v2 = pad2 ? 0 : table[c2];
        const int v3 = pad3 ? 0 : table[c3];
        if ((!pad2 && v2 < 0) || (!pad3 && v3 < 0)) {
            return std::nullopt;
        }

        out.push_back(static_cast<char>((v0 << 2) | (v1 >> 4)));
        if (!pad2) {
            out.push_back(static_cast<char>(((v1 & 0x0F) << 4) | (v2 >> 2)));
        }
        if (!pad3) {
            out.push_back(static_cast<char>(((v2 & 0x03) << 6) | v3));
        }
    }

    return out;
}

} // namespace

std::string Base64::encode(std::string_view input) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(input.data());
    return encode(std::span<const std::uint8_t>(bytes, input.size()));
}

std::string Base64::encode(std::span<const std::uint8_t> input) {
    return encode_impl(input, kStandardAlphabet, true);
}

std::string Base64::encode_url(std::span<const std::uint8_t> input,
                               bool include_padding) {
    return encode_impl(input, kUrlAlphabet, include_padding);
}

std::optional<std::string> Base64::decode(std::string_view input) {
    return decode_impl(input, kStandardDecodeTable);
}

std::optional<std::string> Base64::decode_url(std::string_view input) {
    std::string normalized;
    normalized.reserve(input.size() + 2);

    for (const unsigned char ch : input) {
        if (ch == '-') {
            normalized.push_back('+');
            continue;
        }
        if (ch == '_') {
            normalized.push_back('/');
            continue;
        }
        if (ch == '=' || kStandardDecodeTable[ch] >= 0) {
            normalized.push_back(static_cast<char>(ch));
            continue;
        }
        return std::nullopt;
    }

    if (normalized.size() % 4 == 1) {
        return std::nullopt;
    }
    while (normalized.size() % 4 != 0) {
        normalized.push_back('=');
    }

    return decode_impl(normalized, kStandardDecodeTable);
}

} // namespace core::utils
