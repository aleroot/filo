#pragma once

#include "core/utils/Uuid.hpp"

#include <string>

namespace core::auth::xai_grok {

inline constexpr const char* kClientIdentifier = "filo";
inline constexpr const char* kClientVersion = "0.1.0";
inline constexpr const char* kUserAgent = "filo/0.1.0";

[[nodiscard]] inline const std::string& process_agent_id() {
    static const std::string value = core::utils::random_uuid_v4();
    return value;
}

template <typename HeaderMap>
void apply_proxy_identity_headers(HeaderMap& headers) {
    headers["X-XAI-Token-Auth"] = "xai-grok-cli";
    headers["x-authenticateresponse"] = "authenticate-response";
    headers["x-grok-client-identifier"] = kClientIdentifier;
    headers["x-grok-client-version"] = kClientVersion;
    headers["x-grok-agent-id"] = process_agent_id();
    headers["User-Agent"] = kUserAgent;
}

} // namespace core::auth::xai_grok
