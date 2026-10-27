#pragma once

#include "OAuthToken.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

struct GoogleManualAuthInput {
    std::string code;
    std::optional<std::string> state;
};

/**
 * @brief Side-effect-free Google OAuth primitives shared by the Google flows.
 *
 * The pieces every Google OAuth client needs and none of which depend on a
 * particular client identity: building the authorization URL, parsing a token
 * response, parsing a pasted callback, and normalizing a loopback redirect URI.
 * `GoogleAntigravityOAuthFlow` composes them into an actual login; they are also
 * pure functions, so they are unit-tested without network or browser side
 * effects.
 *
 * The gemini-cli flow that used to live here is gone: Google deprecated it for
 * individual accounts, nothing constructed it any more, and it was the last
 * place a Google client secret was embedded in this source tree. GitHub secret
 * scanning matches both halves of a Google OAuth client and push protection
 * rejects any commit carrying them, so client credentials are configured per
 * machine instead — see GoogleAntigravityIdentity.hpp and
 * AntigravityClientCredentials.hpp.
 */
class GoogleOAuthFlow final {
public:
    GoogleOAuthFlow() = delete;

    static std::string build_auth_url(std::string_view client_id,
                                      std::string_view redirect_uri,
                                      const std::vector<std::string>& scopes,
                                      std::string_view state,
                                      std::string_view code_challenge = {});

    static OAuthToken parse_token_response(std::string_view json,
                                           int64_t request_time_unix);

    static GoogleManualAuthInput parse_manual_auth_input(std::string_view raw_input);

    static std::string build_loopback_redirect_uri(std::string_view listener_host,
                                                   int port,
                                                   std::string_view path = "/oauth2callback",
                                                   std::string_view redirect_host_override = {});
};

} // namespace core::auth
