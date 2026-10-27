#pragma once

/**
 * @file IdTokenClaims.hpp
 * @brief Reader for the claims of an OAuth `id_token` (a JWT).
 *
 * Shared by the flows that learn something about the signed-in account from the
 * id_token: `XaiOAuthFlow` (subject, email, principal and team claims) and the
 * Antigravity import (the `aud` claim, which is the OAuth client the Antigravity
 * app signed in with — the same client Filo authenticates as).
 */

#include <string>
#include <string_view>

namespace core::auth {

struct IdTokenClaims {
    std::string subject;
    std::string email;
    std::string nonce;
    /// `aud` — the client id the token was issued to.
    std::string audience;
    std::string principal_type;
    std::string principal_id;
    std::string team_id;
};

/**
 * @brief Decode the payload segment of `jwt` and read the known claims.
 *
 * Never throws: a token that is not a JWT, is not base64url, or carries no JSON
 * payload yields empty claims, because a missing claim is not fatal to a login —
 * callers decide what they cannot proceed without.
 */
[[nodiscard]] IdTokenClaims parse_id_token_claims(std::string_view jwt);

} // namespace core::auth
