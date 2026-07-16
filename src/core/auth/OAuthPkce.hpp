#pragma once

#include <string>
#include <string_view>

namespace core::auth::oauth_pkce {

/** Generate an RFC 7636 verifier using only unreserved URI characters. */
[[nodiscard]] std::string generate_code_verifier(std::size_t length = 128);

/** Return BASE64URL(SHA256(verifier)) without padding. */
[[nodiscard]] std::string compute_code_challenge(std::string_view verifier);

/** Generate a cryptographically unpredictable URL-safe correlation value. */
[[nodiscard]] std::string generate_correlation_token(std::size_t length = 32);

/** Percent-encode one OAuth query/form value using RFC 3986 rules. */
[[nodiscard]] std::string url_encode(std::string_view value);

} // namespace core::auth::oauth_pkce
