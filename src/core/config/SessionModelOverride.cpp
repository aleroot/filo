#include "SessionModelOverride.hpp"

#include "core/utils/StringUtils.hpp"

#include <format>

namespace core::config {

std::expected<void, std::string> apply_session_model_override(
    AppConfig& config,
    std::string_view raw_selector) {
    const std::string selector = core::utils::str::trim_ascii_copy(raw_selector);
    if (selector.empty()) {
        return std::unexpected("--model requires a non-empty selector.");
    }

    if (selector == "router" || selector == "auto") {
        config.default_model_selection = selector;
        return {};
    }

    std::string provider = config.default_provider;
    std::string model = selector;
    if (config.providers.contains(selector)) {
        provider = selector;
        model = config.providers.at(provider).model;
    } else if (const auto slash = selector.find('/');
               slash != std::string::npos
               && config.providers.contains(selector.substr(0, slash))) {
        provider = selector.substr(0, slash);
        model = selector.substr(slash + 1);
    }

    if (!config.providers.contains(provider)) {
        return std::unexpected(std::format(
            "--model selected unknown provider '{}'.", provider));
    }
    if (model.empty()) {
        return std::unexpected(std::format(
            "--model requires a model after '{}/'.", provider));
    }

    config.default_provider = provider;
    config.default_model_selection = "manual";
    config.providers[provider].model = model;
    return {};
}

} // namespace core::config
