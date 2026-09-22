#include "MimoModelTraits.hpp"

#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"

#include <array>
#include <cstdlib>

namespace core::llm {
namespace {

using core::utils::ascii::iequals;

[[nodiscard]] std::string env_value(const char* name) {
    if (const char* raw = std::getenv(name); raw != nullptr && *raw != '\0') {
        return std::string(raw);
    }
    return {};
}

[[nodiscard]] std::string lowered_host(std::string_view base_url) {
    const auto host = core::utils::uri::extract_http_host(base_url);
    if (!host) return {};
    return core::utils::str::to_lower_ascii_copy(*host);
}

} // namespace

std::string_view mimo_token_plan_endpoint(MimoRegion region) noexcept {
    switch (region) {
        case MimoRegion::Singapore:
            return "https://token-plan-sgp.xiaomimimo.com/v1";
        case MimoRegion::China:
            return "https://token-plan-cn.xiaomimimo.com/v1";
        case MimoRegion::Europe:
            break;
    }
    return "https://token-plan-ams.xiaomimimo.com/v1";
}

std::string_view mimo_region_provider_name(MimoRegion region) noexcept {
    switch (region) {
        case MimoRegion::Singapore:
            return "mimo-token-plan-sgp";
        case MimoRegion::China:
            return "mimo-token-plan-cn";
        case MimoRegion::Europe:
            break;
    }
    return "mimo-token-plan";
}

std::string_view mimo_region_label(MimoRegion region) noexcept {
    switch (region) {
        case MimoRegion::Singapore:
            return "Singapore";
        case MimoRegion::China:
            return "China";
        case MimoRegion::Europe:
            break;
    }
    return "Europe";
}

std::optional<MimoRegion> mimo_region_from_string(
    std::string_view value) noexcept {
    if (iequals(value, "ams") || iequals(value, "eu")
        || iequals(value, "europe") || iequals(value, "amsterdam")) {
        return MimoRegion::Europe;
    }
    if (iequals(value, "sgp") || iequals(value, "sg")
        || iequals(value, "singapore") || iequals(value, "intl")
        || iequals(value, "international")) {
        return MimoRegion::Singapore;
    }
    if (iequals(value, "cn") || iequals(value, "china")
        || iequals(value, "mainland") || iequals(value, "mainland-cn")) {
        return MimoRegion::China;
    }
    return std::nullopt;
}

MimoRegion resolve_mimo_region() {
    if (const std::string configured = env_value("MIMO_REGION");
        !configured.empty()) {
        if (const auto region = mimo_region_from_string(configured)) {
            return *region;
        }
    }

    // A mainland locale is the one case where the CN gateway is the better
    // default; everything else is served faster from Amsterdam.
    for (const char* name : {"LC_ALL", "LC_MESSAGES", "LANG"}) {
        const std::string locale =
            core::utils::str::to_lower_ascii_copy(env_value(name));
        if (locale.starts_with("zh_cn") || locale.starts_with("zh-cn")) {
            return MimoRegion::China;
        }
    }
    return MimoRegion::Europe;
}

std::optional<MimoRegion> mimo_region_for_endpoint(std::string_view base_url) {
    const std::string host = lowered_host(base_url);
    if (host.empty() || !host.ends_with("xiaomimimo.com")) return std::nullopt;

    if (host.starts_with("token-plan-ams.")) return MimoRegion::Europe;
    if (host.starts_with("token-plan-sgp.")) return MimoRegion::Singapore;
    if (host.starts_with("token-plan-cn.")) return MimoRegion::China;
    return std::nullopt;
}

bool is_mimo_token_plan_endpoint(std::string_view base_url) {
    return mimo_region_for_endpoint(base_url).has_value();
}

bool is_mimo_endpoint(std::string_view base_url) {
    const std::string host = lowered_host(base_url);
    return !host.empty() && host.ends_with("xiaomimimo.com");
}

bool is_mimo_model(std::string_view model) noexcept {
    // Model ids are `mimo-v2.6-pro`, `mimo-v2-flash`, and friends. A bare
    // "mimo" prefix check would also claim unrelated ids such as
    // "mimosa-7b", so require the separator the family always uses.
    if (model.size() < 5) return false;
    if (!iequals(model.substr(0, 4), "mimo")) return false;
    return model[4] == '-' || model[4] == '_' || model[4] == '.';
}

std::string mimo_platform_url() {
    if (std::string configured = env_value("MIMO_PLATFORM_URL");
        !configured.empty()) {
        while (!configured.empty() && configured.back() == '/') {
            configured.pop_back();
        }
        if (!configured.empty()) return configured;
    }
    return "https://platform.xiaomimimo.com";
}

std::string_view mimo_gateway_code_label(std::string_view code) noexcept {
    if (code == "421") return "Request blocked by content moderation";
    if (code == "441") return "Request blocked by risk control";
    return {};
}

} // namespace core::llm
