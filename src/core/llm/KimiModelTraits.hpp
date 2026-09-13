#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::llm {

enum class KimiService {
    Unknown,
    PublicApi,
    Code,
};

/// Managed Kimi Code deployments. Public Moonshot API hosts are not a region.
enum class KimiRegion {
    Unknown,
    MainlandCn,
    Global,
};

struct KimiRegionProfile {
    std::string_view id;
    std::string_view oauth_host;
    std::string_view coding_base_url;
};

/// How to pick a region when env/persisted state is silent.
/// Both modes default to the international deployment; zh_CN locales still
/// resolve to mainland China. `NewLogin` exists so auth can document intent.
enum class KimiRegionResolveMode {
    Session,
    NewLogin,
};

struct KimiRegionResolution {
    KimiRegion region = KimiRegion::Unknown;
    std::string oauth_host;
    std::string coding_base_url;
};

[[nodiscard]] std::span<const std::string_view> kimi_code_model_ids() noexcept;
[[nodiscard]] bool is_kimi_code_provider_name(
    std::string_view provider_name) noexcept;
[[nodiscard]] bool is_kimi_code_model(std::string_view model_id) noexcept;
[[nodiscard]] bool is_kimi_k3_model(std::string_view model_id) noexcept;
[[nodiscard]] bool is_kimi_k3_256k_model(std::string_view model_id) noexcept;
[[nodiscard]] bool is_kimi_public_k3_model(
    std::string_view model_id) noexcept;
[[nodiscard]] int32_t kimi_model_context_window(
    std::string_view model_id) noexcept;
[[nodiscard]] int32_t kimi_model_max_output_tokens(
    std::string_view model_id) noexcept;

[[nodiscard]] bool is_kimi_code_endpoint_path(
    std::string_view base_url) noexcept;
[[nodiscard]] KimiService kimi_service_for_endpoint(
    std::string_view base_url) noexcept;
[[nodiscard]] std::string_view kimi_service_id(KimiService service) noexcept;

[[nodiscard]] const KimiRegionProfile& kimi_region_profile(
    KimiRegion region) noexcept;
[[nodiscard]] KimiRegion kimi_region_from_id(std::string_view id) noexcept;
[[nodiscard]] KimiRegion kimi_region_for_oauth_host(
    std::string_view oauth_host) noexcept;
[[nodiscard]] KimiRegion kimi_region_for_endpoint(
    std::string_view base_url) noexcept;
[[nodiscard]] KimiRegion kimi_region_from_locale(
    std::string_view locale) noexcept;

/// Env + persisted host + mode default. `KIMI_CODE_BASE_URL` / OAuth host /
/// `KIMI_CODE_REGION` win, matching kimi-code's region resolver.
[[nodiscard]] KimiRegionResolution resolve_kimi_region(
    std::string_view persisted_oauth_host = {},
    KimiRegionResolveMode mode = KimiRegionResolveMode::Session);

/// Subscription chat endpoint for the current process (env + persisted host).
[[nodiscard]] std::string resolve_kimi_code_base_url(
    std::string_view persisted_oauth_host = {},
    KimiRegionResolveMode mode = KimiRegionResolveMode::Session);

/// Rewrite public-API or mismatched coding hosts onto the managed endpoint.
[[nodiscard]] std::optional<std::string> kimi_managed_endpoint_override(
    std::string_view configured_base_url,
    std::string_view model,
    bool oauth,
    std::string_view persisted_oauth_host = {});

} // namespace core::llm
