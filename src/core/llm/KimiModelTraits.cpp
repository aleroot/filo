#include "KimiModelTraits.hpp"

#include "core/auth/KimiClientIdentity.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "core/utils/UriUtils.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>

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

constexpr KimiRegionProfile kUnknownProfile{
    {}, {}, {},
};
constexpr KimiRegionProfile kMainlandProfile{
    "mainland-cn",
    core::auth::kimi_code::kMainlandOAuthHost,
    core::auth::kimi_code::kMainlandCodingBaseUrl,
};
constexpr KimiRegionProfile kGlobalProfile{
    "global",
    core::auth::kimi_code::kGlobalOAuthHost,
    core::auth::kimi_code::kGlobalCodingBaseUrl,
};

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

[[nodiscard]] std::string strip_trailing_slashes(std::string_view value) {
    value = trimmed(value);
    while (!value.empty() && value.back() == '/') {
        value.remove_suffix(1);
    }
    return std::string(value);
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

[[nodiscard]] bool is_kimi_code_host(std::string_view host) noexcept {
    return core::utils::ascii::iequals(host, "api.kimi.com")
        || core::utils::ascii::iequals(host, "api.kimi.ai");
}

[[nodiscard]] std::optional<std::string> env_nonempty(const char* name) {
    if (name == nullptr) return std::nullopt;
    const char* value = std::getenv(name);
    if (value == nullptr) return std::nullopt;
    const std::string_view trimmed_value = trimmed(value);
    if (trimmed_value.empty()) return std::nullopt;
    return std::string(trimmed_value);
}

[[nodiscard]] KimiRegionResolution resolution_for(KimiRegion region) {
    const auto& profile = kimi_region_profile(region);
    return {
        .region = region,
        .oauth_host = std::string(profile.oauth_host),
        .coding_base_url = std::string(profile.coding_base_url),
    };
}

[[nodiscard]] KimiRegion fallback_region() {
    std::string locale;
    if (const auto lc_all = env_nonempty("LC_ALL"); lc_all.has_value()) {
        locale = *lc_all;
    } else if (const auto lang = env_nonempty("LANG"); lang.has_value()) {
        locale = *lang;
    }
    const auto from_locale = kimi_region_from_locale(locale);
    return from_locale == KimiRegion::Unknown
        ? KimiRegion::Global
        : from_locale;
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

    if (is_kimi_code_host(*host) && is_kimi_code_endpoint_path(base_url)) {
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

const KimiRegionProfile& kimi_region_profile(KimiRegion region) noexcept {
    switch (region) {
        case KimiRegion::MainlandCn:
            return kMainlandProfile;
        case KimiRegion::Global:
            return kGlobalProfile;
        case KimiRegion::Unknown:
            return kUnknownProfile;
    }
    return kUnknownProfile;
}

KimiRegion kimi_region_from_id(std::string_view id) noexcept {
    id = trimmed(id);
    if (core::utils::ascii::iequals(id, "mainland-cn")
        || core::utils::ascii::iequals(id, "cn")
        || core::utils::ascii::iequals(id, "china")) {
        return KimiRegion::MainlandCn;
    }
    if (core::utils::ascii::iequals(id, "global")
        || core::utils::ascii::iequals(id, "international")
        || core::utils::ascii::iequals(id, "intl")) {
        return KimiRegion::Global;
    }
    return KimiRegion::Unknown;
}

KimiRegion kimi_region_for_oauth_host(std::string_view oauth_host) noexcept {
    const std::string normalized = strip_trailing_slashes(oauth_host);
    if (normalized.empty()) return KimiRegion::Unknown;
    if (core::utils::ascii::iequals(normalized, kMainlandProfile.oauth_host)) {
        return KimiRegion::MainlandCn;
    }
    if (core::utils::ascii::iequals(normalized, kGlobalProfile.oauth_host)) {
        return KimiRegion::Global;
    }
    const auto host = core::utils::uri::extract_http_host(normalized);
    if (!host.has_value()) return KimiRegion::Unknown;
    if (core::utils::ascii::iequals(*host, "auth.kimi.com")) {
        return KimiRegion::MainlandCn;
    }
    if (core::utils::ascii::iequals(*host, "auth.kimi.ai")) {
        return KimiRegion::Global;
    }
    return KimiRegion::Unknown;
}

KimiRegion kimi_region_for_endpoint(std::string_view base_url) noexcept {
    const auto host = core::utils::uri::extract_http_host(trimmed(base_url));
    if (!host.has_value()) return KimiRegion::Unknown;
    if (core::utils::ascii::iequals(*host, "api.kimi.com")) {
        return KimiRegion::MainlandCn;
    }
    if (core::utils::ascii::iequals(*host, "api.kimi.ai")) {
        return KimiRegion::Global;
    }
    return KimiRegion::Unknown;
}

KimiRegion kimi_region_from_locale(std::string_view locale) noexcept {
    locale = trimmed(locale);
    if (locale.empty()) return KimiRegion::Unknown;
    // Taiwan/HK typically use the international product. Only mainland China
    // locales pin the .com deployment.
    if (core::utils::ascii::istarts_with(locale, "zh_CN")
        || core::utils::ascii::istarts_with(locale, "zh-CN")
        || core::utils::ascii::iequals(locale, "zh")) {
        return KimiRegion::MainlandCn;
    }
    return KimiRegion::Global;
}

KimiRegionResolution resolve_kimi_region(
    std::string_view persisted_oauth_host,
    KimiRegionResolveMode mode) {
    if (const auto base = env_nonempty("KIMI_CODE_BASE_URL");
        base.has_value()) {
        auto resolved = resolution_for(
            kimi_region_for_endpoint(*base) == KimiRegion::Unknown
                ? fallback_region()
                : kimi_region_for_endpoint(*base));
        resolved.coding_base_url = strip_trailing_slashes(*base);
        if (const auto host = env_nonempty("KIMI_CODE_OAUTH_HOST");
            host.has_value()) {
            resolved.oauth_host = strip_trailing_slashes(*host);
        } else if (const auto alias = env_nonempty("KIMI_OAUTH_HOST");
                   alias.has_value()) {
            resolved.oauth_host = strip_trailing_slashes(*alias);
        }
        if (const auto region = kimi_region_for_endpoint(*base);
            region != KimiRegion::Unknown) {
            resolved.region = region;
        }
        return resolved;
    }

    if (const auto host = env_nonempty("KIMI_CODE_OAUTH_HOST");
        host.has_value()) {
        const auto region = kimi_region_for_oauth_host(*host);
        if (region != KimiRegion::Unknown) {
            return resolution_for(region);
        }
        auto resolved = resolution_for(fallback_region());
        resolved.oauth_host = strip_trailing_slashes(*host);
        return resolved;
    }
    if (const auto host = env_nonempty("KIMI_OAUTH_HOST"); host.has_value()) {
        const auto region = kimi_region_for_oauth_host(*host);
        if (region != KimiRegion::Unknown) {
            return resolution_for(region);
        }
        auto resolved = resolution_for(fallback_region());
        resolved.oauth_host = strip_trailing_slashes(*host);
        return resolved;
    }

    if (const auto id = env_nonempty("KIMI_CODE_REGION"); id.has_value()) {
        const auto region = kimi_region_from_id(*id);
        if (region != KimiRegion::Unknown) {
            return resolution_for(region);
        }
    }

    const auto persisted_region = kimi_region_for_oauth_host(persisted_oauth_host);
    if (persisted_region != KimiRegion::Unknown) {
        return resolution_for(persisted_region);
    }

    (void)mode;
    return resolution_for(fallback_region());
}

std::string resolve_kimi_code_base_url(
    std::string_view persisted_oauth_host,
    KimiRegionResolveMode mode) {
    return resolve_kimi_region(persisted_oauth_host, mode).coding_base_url;
}

std::optional<std::string> kimi_managed_endpoint_override(
    std::string_view configured_base_url,
    std::string_view model,
    bool oauth,
    std::string_view persisted_oauth_host) {
    const auto service = kimi_service_for_endpoint(configured_base_url);
    const bool code_model = is_kimi_code_model(model);
    const bool needs_managed = oauth
        || (service == KimiService::PublicApi && code_model);
    if (!needs_managed) return std::nullopt;

    const auto resolved = resolve_kimi_region(
        persisted_oauth_host,
        KimiRegionResolveMode::Session);
    if (resolved.coding_base_url.empty()) return std::nullopt;

    if (service == KimiService::PublicApi || service == KimiService::Unknown) {
        if (oauth || code_model) {
            return resolved.coding_base_url;
        }
        return std::nullopt;
    }

    if (service == KimiService::Code) {
        const auto configured_region = kimi_region_for_endpoint(configured_base_url);
        if (configured_region != KimiRegion::Unknown
            && configured_region != resolved.region
            && (oauth
                || env_nonempty("KIMI_CODE_BASE_URL").has_value()
                || env_nonempty("KIMI_CODE_REGION").has_value()
                || env_nonempty("KIMI_CODE_OAUTH_HOST").has_value()
                || env_nonempty("KIMI_OAUTH_HOST").has_value()
                || !trimmed(persisted_oauth_host).empty())) {
            return resolved.coding_base_url;
        }
    }
    return std::nullopt;
}

} // namespace core::llm
