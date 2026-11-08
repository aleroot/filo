#include "ProviderRuntime.hpp"

#include "core/config/ConfigManager.hpp"
#include "core/config/SessionModelOverride.hpp"
#include "core/llm/ModelCatalogDiscovery.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ProviderManager.hpp"

#include <algorithm>
#include <format>

namespace exec {

ProviderRuntime load_provider_runtime(std::optional<std::string> model_override) {
    ProviderRuntime runtime;

    auto& config_manager = core::config::ConfigManager::get_instance();
    auto config = config_manager.get_config();
    if (model_override.has_value()) {
        if (auto applied = core::config::apply_session_model_override(
                config, *model_override); !applied) {
            runtime.error = applied.error();
            return runtime;
        }
    }

    auto& provider_manager = core::llm::ProviderManager::get_instance();

    std::vector<std::string> provider_names;
    provider_names.reserve(config.providers.size());
    for (const auto& [name, pconfig] : config.providers) {
        if (auto provider = core::llm::ProviderFactory::create_provider(name, pconfig)) {
            provider_manager.register_provider(name, provider);
            provider_names.push_back(name);
        }
    }
    if (provider_names.empty()) {
        runtime.error = "No providers could be initialized from configuration.";
        return runtime;
    }
    std::ranges::sort(provider_names);

    std::string provider_name = config.default_provider;
    if (!std::ranges::contains(provider_names, provider_name)) {
        provider_name = provider_names.front();
    }

    try {
        runtime.provider = provider_manager.get_provider(provider_name);
    } catch (const std::exception& ex) {
        runtime.error = std::format(
            "Failed to initialize provider '{}': {}", provider_name, ex.what());
        return runtime;
    }

    core::llm::request_model_catalog_discovery(runtime.provider, {.timeout_ms = 3000});

    if (const auto it = config.providers.find(provider_name);
        it != config.providers.end()) {
        runtime.model_name = it->second.model;
    }
    runtime.provider_name = std::move(provider_name);
    runtime.default_mode = config.default_mode.empty() ? "BUILD" : config.default_mode;
    return runtime;
}

} // namespace exec
