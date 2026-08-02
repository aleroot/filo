#pragma once

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <string_view>

namespace core::llm::protocols::grok_build {

/**
 * The Grok Build OAuth token and client-identity headers are credentials for
 * this exact first-party HTTPS endpoint. Keep this policy provider-local so a
 * custom OpenAI-compatible URL can never receive them.
 */
[[nodiscard]] inline bool is_official_proxy(std::string_view base_url) {
    const auto host = core::utils::uri::extract_http_host(base_url);
    return core::utils::ascii::istarts_with(base_url, "https://")
        && host.has_value()
        && core::utils::ascii::iequals(*host, "cli-chat-proxy.grok.com");
}

} // namespace core::llm::protocols::grok_build
