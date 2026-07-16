#include "OAuthPkce.hpp"

#include "core/utils/Base64.hpp"
#include "core/utils/AsciiUtils.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace core::auth::oauth_pkce {
namespace {

constexpr std::array<std::uint32_t, 64> kSha256RoundConstants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

[[nodiscard]] std::uint32_t rotate_right(std::uint32_t value,
                                          std::uint32_t count) noexcept {
    return (value >> count) | (value << (32u - count));
}

[[nodiscard]] std::array<std::uint8_t, 32> sha256(std::string_view message) {
    std::array<std::uint32_t, 8> hash{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    std::vector<std::uint8_t> data(message.begin(), message.end());
    const std::uint64_t bit_length = static_cast<std::uint64_t>(message.size()) * 8u;
    data.push_back(0x80u);
    while (data.size() % 64u != 56u) data.push_back(0x00u);
    for (int i = 7; i >= 0; --i) {
        data.push_back(static_cast<std::uint8_t>((bit_length >> (i * 8)) & 0xffu));
    }

    for (std::size_t offset = 0; offset < data.size(); offset += 64u) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            words[i] = (std::uint32_t(data[offset + i * 4u]) << 24u)
                     | (std::uint32_t(data[offset + i * 4u + 1u]) << 16u)
                     | (std::uint32_t(data[offset + i * 4u + 2u]) << 8u)
                     | std::uint32_t(data[offset + i * 4u + 3u]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const auto s0 = rotate_right(words[i - 15], 7u)
                          ^ rotate_right(words[i - 15], 18u)
                          ^ (words[i - 15] >> 3u);
            const auto s1 = rotate_right(words[i - 2], 17u)
                          ^ rotate_right(words[i - 2], 19u)
                          ^ (words[i - 2] >> 10u);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        auto a = hash[0]; auto b = hash[1]; auto c = hash[2]; auto d = hash[3];
        auto e = hash[4]; auto f = hash[5]; auto g = hash[6]; auto h = hash[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const auto upper_sigma1 = rotate_right(e, 6u)
                                    ^ rotate_right(e, 11u)
                                    ^ rotate_right(e, 25u);
            const auto choose = (e & f) ^ (~e & g);
            const auto temp1 = h + upper_sigma1 + choose
                             + kSha256RoundConstants[i] + words[i];
            const auto upper_sigma0 = rotate_right(a, 2u)
                                    ^ rotate_right(a, 13u)
                                    ^ rotate_right(a, 22u);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = upper_sigma0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
        hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
    }

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t i = 0; i < hash.size(); ++i) {
        digest[i * 4u] = static_cast<std::uint8_t>((hash[i] >> 24u) & 0xffu);
        digest[i * 4u + 1u] = static_cast<std::uint8_t>((hash[i] >> 16u) & 0xffu);
        digest[i * 4u + 2u] = static_cast<std::uint8_t>((hash[i] >> 8u) & 0xffu);
        digest[i * 4u + 3u] = static_cast<std::uint8_t>(hash[i] & 0xffu);
    }
    return digest;
}

[[nodiscard]] std::string random_from_alphabet(std::size_t length,
                                                std::string_view alphabet) {
    if (alphabet.empty()) throw std::invalid_argument("OAuth random alphabet is empty");
    std::random_device random_device;
    std::uniform_int_distribution<std::size_t> distribution(0, alphabet.size() - 1);
    std::string result;
    result.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        result.push_back(alphabet[distribution(random_device)]);
    }
    return result;
}

} // namespace

std::string generate_code_verifier(std::size_t length) {
    if (length < 43 || length > 128) {
        throw std::invalid_argument("PKCE verifier length must be between 43 and 128");
    }
    return random_from_alphabet(
        length,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~");
}

std::string compute_code_challenge(std::string_view verifier) {
    return core::utils::Base64::encode_url(sha256(verifier));
}

std::string generate_correlation_token(std::size_t length) {
    return random_from_alphabet(
        length,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_");
}

std::string url_encode(std::string_view value) {
    static constexpr char hex[] = "0123456789ABCDEF";
    std::string encoded;
    encoded.reserve(value.size() * 3u);
    for (const unsigned char ch : value) {
        if (core::utils::ascii::is_alnum(ch)
            || ch == '-' || ch == '_' || ch == '.' || ch == '~') {
            encoded.push_back(static_cast<char>(ch));
        } else {
            encoded.push_back('%');
            encoded.push_back(hex[ch >> 4u]);
            encoded.push_back(hex[ch & 0x0fu]);
        }
    }
    return encoded;
}

} // namespace core::auth::oauth_pkce
