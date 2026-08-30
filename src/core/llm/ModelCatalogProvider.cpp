#include "ModelCatalogProvider.hpp"

namespace core::llm {

std::unique_ptr<ModelCatalogProvider>
make_model_catalog_provider(
    config::ApiType api_type,
    std::string_view provider_name) {
    return make_model_catalog_provider(api_type, provider_name, false);
}

std::unique_ptr<ModelCatalogProvider>
make_model_catalog_provider(
    config::ApiType api_type,
    std::string_view provider_name,
    bool subscription_session) {
    const std::string name(provider_name);
    switch (api_type) {
        case config::ApiType::Gemini:
            return std::make_unique<GeminiModelCatalogProvider>(name);
        case config::ApiType::Anthropic:
            return std::make_unique<AnthropicModelCatalogProvider>(name);
        case config::ApiType::Kimi:
            return std::make_unique<KimiModelCatalogProvider>(name);
        case config::ApiType::OpenAI:
            if (provider_name == "openai-codex") {
                return std::make_unique<CodexModelCatalogProvider>(name);
            }
            if (provider_name.starts_with("grok")) {
                return std::make_unique<XaiModelCatalogProvider>(
                    name,
                    subscription_session);
            }
            if (provider_name.starts_with("mistral")) {
                return std::make_unique<MistralModelCatalogProvider>(name);
            }
            return std::make_unique<OpenAICompatibleModelCatalogProvider>(
                name);
        case config::ApiType::DashScope:
            return std::make_unique<OpenAICompatibleModelCatalogProvider>(
                name);
        case config::ApiType::Ollama:
            return std::make_unique<OllamaModelCatalogProvider>(name);
        case config::ApiType::Unknown:
        case config::ApiType::LlamaCppLocal:
            return nullptr;
    }
    return nullptr;
}

} // namespace core::llm
