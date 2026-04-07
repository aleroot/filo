#pragma once

#include "../utils/StringUtils.hpp"

#include <string>
#include <string_view>

namespace core::llm::openai_endpoint {

[[nodiscard]] inline std::string extract_host_lower(std::string_view url) {
    std::size_t start = url.find("://");
    start = (start == std::string_view::npos) ? 0 : start + 3;

    std::size_t end = url.find_first_of("/?#", start);
    if (end == std::string_view::npos) end = url.size();

    std::string_view authority = url.substr(start, end - start);
    const std::size_t at = authority.rfind('@');
    if (at != std::string_view::npos) authority.remove_prefix(at + 1);

    if (authority.empty()) return {};

    if (authority.front() == '[') {
        const std::size_t close = authority.find(']');
        if (close == std::string_view::npos) return {};
        return core::utils::str::to_lower_ascii_copy(authority.substr(1, close - 1));
    }

    const std::size_t colon = authority.find(':');
    return core::utils::str::to_lower_ascii_copy(authority.substr(0, colon));
}

[[nodiscard]] inline bool is_azure_openai_base_url(std::string_view base_url) {
    const std::string host = extract_host_lower(base_url);
    if (host.empty()) return false;
    if (!host.ends_with(".azure.com")) return false;
    return host.find("cognitiveservices") != std::string::npos
        || host.find("openai") != std::string::npos
        || host.find("services.ai") != std::string::npos;
}

[[nodiscard]] inline bool is_native_openai_responses_base_url(std::string_view base_url) {
    const std::string normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_trailing_slashes(base_url));
    return normalized == "https://api.openai.com/v1"
        || normalized == "https://chatgpt.com/backend-api/codex";
}

[[nodiscard]] inline std::string build_azure_chat_completions_url(std::string_view base_url,
                                                                   std::string_view model,
                                                                   std::string_view api_version) {
    std::string base = core::utils::str::trim_trailing_slashes(base_url);
    const std::string base_lower = core::utils::str::to_lower_ascii_copy(base);
    if (base_lower.find("/deployments/") != std::string::npos) {
        return base + "/chat/completions?api-version=" + std::string(api_version);
    }

    if (base_lower.ends_with("/openai/v1")) {
        base.resize(base.size() - std::string("/openai/v1").size());
    } else if (base_lower.ends_with("/v1")) {
        base.resize(base.size() - std::string("/v1").size());
    }
    base = core::utils::str::trim_trailing_slashes(base);

    const std::string deployment = model.empty() ? "gpt-4o" : std::string(model);
    return base + "/openai/deployments/" + deployment
        + "/chat/completions?api-version=" + std::string(api_version);
}

} // namespace core::llm::openai_endpoint
