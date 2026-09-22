#include "OAuthPkce.hpp"

#include "core/utils/Base64.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "core/utils/Crypto.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace core::auth::oauth_pkce {
namespace {

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
    return core::utils::Base64::encode_url(core::utils::crypto::sha256(verifier));
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
