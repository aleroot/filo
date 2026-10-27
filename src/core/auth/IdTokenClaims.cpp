#include "IdTokenClaims.hpp"

#include "../utils/Base64.hpp"

#include <simdjson.h>

namespace core::auth {

IdTokenClaims parse_id_token_claims(std::string_view jwt) {
    IdTokenClaims claims;
    const auto first_dot = jwt.find('.');
    if (first_dot == std::string_view::npos) return claims;
    const auto second_dot = jwt.find('.', first_dot + 1);
    if (second_dot == std::string_view::npos || second_dot <= first_dot + 1) return claims;
    const auto decoded = core::utils::Base64::decode_url(
        jwt.substr(first_dot + 1, second_dot - first_dot - 1));
    if (!decoded.has_value()) return claims;

    simdjson::dom::parser parser;
    simdjson::padded_string padded(decoded->data(), decoded->size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return claims;

    const auto read = [&doc](std::string_view key) -> std::string {
        std::string_view value;
        if (doc[key].get_string().get(value) == simdjson::SUCCESS) {
            return std::string(value);
        }
        return {};
    };
    claims.subject = read("sub");
    claims.email = read("email");
    claims.nonce = read("nonce");
    claims.audience = read("aud");
    claims.principal_type = read("principal_type");
    if (claims.principal_type.empty()) claims.principal_type = read("principalType");
    claims.principal_id = read("principal_id");
    if (claims.principal_id.empty()) claims.principal_id = read("principalId");
    claims.team_id = read("team_id");
    return claims;
}

} // namespace core::auth
