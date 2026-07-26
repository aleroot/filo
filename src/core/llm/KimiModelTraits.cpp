#include "KimiModelTraits.hpp"

#include "core/utils/AsciiUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <algorithm>
#include <array>

namespace core::llm {
namespace {

constexpr std::array<std::string_view, 4> kCodeModelIds{{
    "k3",
    "k3-256k",
    "kimi-for-coding",
    "kimi-for-coding-highspeed",
}};

constexpr std::string_view kPublicApiServiceId = "kimi:public-api";
constexpr std::string_view kCodeServiceId = "kimi:code";

[[nodiscard]] std::string_view trimmed(std::string_view value) noexcept {
    while (!value.empty()
           && (value.front() == ' ' || value.front() == '\t'
               || value.front() == '\r' || value.front() == '\n')) {
        value.remove_prefix(1);
    }
    while (!value.empty()
           && (value.back() == ' ' || value.back() == '\t'
               || value.back() == '\r' || value.back() == '\n')) {
        value.remove_suffix(1);
    }
    return value;
}

[[nodiscard]] bool has_path_prefix(
    std::string_view url,
    std::string_view prefix) noexcept {
    const std::size_t scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos) return false;
    const std::size_t path_start = url.find('/', scheme_end + 3);
    if (path_start == std::string_view::npos) return false;
    const std::size_t path_end = url.find_first_of("?#", path_start);
    const std::string_view path = url.substr(
        path_start,
        path_end == std::string_view::npos
            ? std::string_view::npos
            : path_end - path_start);
    return core::utils::ascii::iequals(path, prefix)
        || (path.size() > prefix.size()
            && core::utils::ascii::iequals(
                path.substr(0, prefix.size()),
                prefix)
            && path[prefix.size()] == '/');
}

} // namespace

std::span<const std::string_view> kimi_code_model_ids() noexcept {
    return kCodeModelIds;
}

bool is_kimi_code_provider_name(std::string_view provider_name) noexcept {
    const std::string_view value = trimmed(provider_name);
    return core::utils::ascii::iequals(value, "kimi-code")
        || core::utils::ascii::istarts_with(value, "kimi-code-")
        || core::utils::ascii::iequals(value, "kimi-for-coding")
        || core::utils::ascii::istarts_with(
            value,
            "kimi-for-coding-");
}

bool is_kimi_code_model(std::string_view model_id) noexcept {
    const std::string_view value = trimmed(model_id);
    return std::ranges::any_of(
        kCodeModelIds,
        [&](std::string_view candidate) {
            return core::utils::ascii::iequals(value, candidate);
        });
}

bool is_kimi_k3_model(std::string_view model_id) noexcept {
    const std::string_view value = trimmed(model_id);
    return core::utils::ascii::iequals(value, "k3")
        || core::utils::ascii::iequals(value, "k3-256k")
        || core::utils::ascii::iequals(value, "kimi-k3");
}

bool is_kimi_k3_256k_model(std::string_view model_id) noexcept {
    return core::utils::ascii::iequals(trimmed(model_id), "k3-256k");
}

bool is_kimi_public_k3_model(std::string_view model_id) noexcept {
    return core::utils::ascii::iequals(trimmed(model_id), "kimi-k3");
}

int32_t kimi_model_context_window(std::string_view model_id) noexcept {
    const std::string_view value = trimmed(model_id);
    if (core::utils::ascii::iequals(value, "k3-256k")) return 262'144;
    if (core::utils::ascii::iequals(value, "k3")
        || core::utils::ascii::iequals(value, "kimi-k3")) {
        return 1'048'576;
    }
    return 0;
}

int32_t kimi_model_max_output_tokens(std::string_view model_id) noexcept {
    const int32_t context_window = kimi_model_context_window(model_id);
    return context_window > 0 ? context_window : 8'192;
}

bool is_kimi_code_endpoint_path(std::string_view base_url) noexcept {
    return has_path_prefix(trimmed(base_url), "/coding");
}

KimiService kimi_service_for_endpoint(std::string_view base_url) noexcept {
    base_url = trimmed(base_url);
    const auto host = core::utils::uri::extract_http_host(base_url);
    if (!host.has_value()) return KimiService::Unknown;

    if (core::utils::ascii::iequals(*host, "api.kimi.com")
        && is_kimi_code_endpoint_path(base_url)) {
        return KimiService::Code;
    }
    if (core::utils::ascii::iequals(*host, "api.moonshot.ai")
        || core::utils::ascii::iequals(*host, "api.moonshot.cn")) {
        return KimiService::PublicApi;
    }
    return KimiService::Unknown;
}

std::string_view kimi_service_id(KimiService service) noexcept {
    switch (service) {
        case KimiService::PublicApi:
            return kPublicApiServiceId;
        case KimiService::Code:
            return kCodeServiceId;
        case KimiService::Unknown:
            return {};
    }
    return {};
}

} // namespace core::llm
