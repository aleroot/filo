#include "ModelMetadata.hpp"

#include "core/config/ConfigManager.hpp"
#include "ProviderDefinition.hpp"

#include <algorithm>
#include <ranges>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] const ModelInfo* find_model(
    std::span<const ModelInfo> models,
    std::string_view model_id) noexcept {
    const auto matches = [model_id](const ModelInfo& model) {
        return model.canonical_id == model_id
            || std::ranges::find(model.aliases, model_id)
                != model.aliases.end();
    };
    const auto it = std::ranges::find_if(models, matches);
    return it == models.end() ? nullptr : std::addressof(*it);
}

} // namespace

ModelInfo merge_model_metadata(ModelInfo baseline, ModelInfo discovered) {
    if (!discovered.display_name.empty()
        && (baseline.display_name.empty()
            || discovered.display_name != discovered.canonical_id)) {
        baseline.display_name = std::move(discovered.display_name);
    }
    if (!discovered.provider.empty() && baseline.provider.empty()) {
        baseline.provider = std::move(discovered.provider);
    }
    if (discovered.context_window > 0) {
        baseline.context_window = discovered.context_window;
    }
    if (discovered.max_output_tokens > 0) {
        baseline.max_output_tokens = discovered.max_output_tokens;
    }
    if (discovered.max_reasoning_tokens > 0) {
        baseline.max_reasoning_tokens = discovered.max_reasoning_tokens;
    }

    if (discovered.capabilities_complete) {
        baseline.capabilities = discovered.capabilities;
        baseline.capabilities_complete = true;
    } else if (discovered.capabilities != 0) {
        baseline.capabilities |= discovered.capabilities;
    }

    if (discovered.reasoning.complete) {
        baseline.reasoning = discovered.reasoning;
    } else {
        if (!discovered.reasoning.effort.empty()) {
            baseline.reasoning.effort =
                ReasoningCapabilities::from_bits(
                    static_cast<ReasoningCapabilities::Storage>(
                        baseline.reasoning.effort.bits()
                        | discovered.reasoning.effort.bits()));
        }
        baseline.reasoning.adaptive_thinking =
            baseline.reasoning.adaptive_thinking
            || discovered.reasoning.adaptive_thinking;
        baseline.reasoning.manual_thinking =
            baseline.reasoning.manual_thinking
            || discovered.reasoning.manual_thinking;
    }

    if (discovered.tier != ModelTier::Balanced
        || baseline.tier == ModelTier::Balanced) {
        baseline.tier = discovered.tier;
    }
    if (discovered.pricing.input_per_mtok > 0.0) {
        baseline.pricing.input_per_mtok =
            discovered.pricing.input_per_mtok;
    }
    if (discovered.pricing.output_per_mtok > 0.0) {
        baseline.pricing.output_per_mtok =
            discovered.pricing.output_per_mtok;
    }
    if (discovered.pricing.cached_input_per_mtok >= 0.0) {
        baseline.pricing.cached_input_per_mtok =
            discovered.pricing.cached_input_per_mtok;
    }
    if (discovered.pricing.prompt_caching_write_per_mtok >= 0.0) {
        baseline.pricing.prompt_caching_write_per_mtok =
            discovered.pricing.prompt_caching_write_per_mtok;
    }
    if (!discovered.knowledge_cutoff.empty()) {
        baseline.knowledge_cutoff =
            std::move(discovered.knowledge_cutoff);
    }
    if (!discovered.deprecation_date.empty()) {
        baseline.deprecation_date =
            std::move(discovered.deprecation_date);
    }
    if (!discovered.expected_completion_date.empty()) {
        baseline.expected_completion_date =
            std::move(discovered.expected_completion_date);
    }

    if (discovered.constraints.temperature) {
        baseline.constraints.temperature =
            discovered.constraints.temperature;
    }
    if (discovered.constraints.top_p) {
        baseline.constraints.top_p = discovered.constraints.top_p;
    }
    if (discovered.constraints.frequency_penalty) {
        baseline.constraints.frequency_penalty =
            discovered.constraints.frequency_penalty;
    }
    if (discovered.constraints.presence_penalty) {
        baseline.constraints.presence_penalty =
            discovered.constraints.presence_penalty;
    }
    if (discovered.constraints.max_tokens_min != 1) {
        baseline.constraints.max_tokens_min =
            discovered.constraints.max_tokens_min;
    }
    if (discovered.constraints.max_tokens_max > 0) {
        baseline.constraints.max_tokens_max =
            discovered.constraints.max_tokens_max;
    }
    if (discovered.max_tool_calls != 32) {
        baseline.max_tool_calls = discovered.max_tool_calls;
    }

    for (auto& alias : discovered.aliases) {
        if (std::ranges::find(baseline.aliases, alias)
            == baseline.aliases.end()) {
            baseline.aliases.push_back(std::move(alias));
        }
    }
    return baseline;
}

ResolvedModelCatalog resolve_model_catalog(
    std::span<const ModelInfo> provider_api_models,
    std::span<const ModelInfo> registry_models) noexcept {
    if (!provider_api_models.empty()) {
        return {
            .models = provider_api_models,
            .origin = ModelMetadataOrigin::ProviderApi,
        };
    }
    if (!registry_models.empty()) {
        return {
            .models = registry_models,
            .origin = ModelMetadataOrigin::InternalRegistry,
        };
    }
    return {};
}

ResolvedModelMetadata resolve_model_metadata(
    std::string_view model_id,
    std::span<const ModelInfo> provider_api_models,
    std::shared_ptr<const ModelInfo> registry_model) {
    if (const ModelInfo* provider_model =
            find_model(provider_api_models, model_id)) {
        ModelInfo resolved = registry_model
            ? merge_model_metadata(*registry_model, *provider_model)
            : *provider_model;
        return {
            .model = std::move(resolved),
            .origin = ModelMetadataOrigin::ProviderApi,
        };
    }
    if (registry_model) {
        return {
            .model = *registry_model,
            .origin = ModelMetadataOrigin::InternalRegistry,
        };
    }
    return {};
}

std::string model_registry_provider_key(
    std::string_view provider_name,
    config::ApiType api_type) {
    if (const auto* definition =
            find_builtin_provider_definition(provider_name)) {
        return std::string(definition->registry_provider);
    }

    switch (api_type) {
        case config::ApiType::Anthropic:
            return "anthropic";
        case config::ApiType::Kimi:
            return "kimi";
        case config::ApiType::Gemini:
            return "gemini";
        case config::ApiType::DashScope:
            return "qwen";
        case config::ApiType::Ollama:
        case config::ApiType::LlamaCppLocal:
            return "local";
        case config::ApiType::OpenAI:
        case config::ApiType::Unknown:
            return std::string(provider_name);
    }
    return std::string(provider_name);
}

} // namespace core::llm
