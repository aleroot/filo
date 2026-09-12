#pragma once

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <string_view>

namespace core::llm::protocols::grok_build {

[[nodiscard]] inline bool is_https_host(
    std::string_view base_url,
    std::string_view host) {
    const auto extracted = core::utils::uri::extract_http_host(base_url);
    return core::utils::ascii::istarts_with(base_url, "https://")
        && extracted.has_value()
        && core::utils::ascii::iequals(*extracted, host);
}

/**
 * The Grok Build OAuth token and client-identity headers are credentials for
 * this exact first-party HTTPS endpoint. Keep this policy provider-local so a
 * custom OpenAI-compatible URL can never receive them.
 */
[[nodiscard]] inline bool is_official_proxy(std::string_view base_url) {
    return is_https_host(base_url, "cli-chat-proxy.grok.com");
}

/** Public xAI inference API (API keys). Cache-sticky `x-grok-conv-id` is safe here. */
[[nodiscard]] inline bool is_public_api(std::string_view base_url) {
    return is_https_host(base_url, "api.x.ai");
}

} // namespace core::llm::protocols::grok_build
