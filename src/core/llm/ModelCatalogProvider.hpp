#pragma once

#include "ModelRegistry.hpp"
#include "core/config/ConfigManager.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm {

struct ModelCatalogResult {
    std::vector<ModelInfo> models;
    std::string error;
    std::string next_page_token;

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

class ModelCatalogProvider {
public:
    virtual ~ModelCatalogProvider() = default;

    [[nodiscard]] virtual std::string_view provider_name() const noexcept = 0;
    [[nodiscard]] virtual std::string model_list_path(std::string_view page_token = {}) const = 0;
    [[nodiscard]] virtual ModelCatalogResult parse_models_response(std::string_view body) const = 0;
};

class GeminiModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit GeminiModelCatalogProvider(std::string provider_name = "gemini");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

class OpenAICompatibleModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit OpenAICompatibleModelCatalogProvider(std::string provider_name = "openai");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

class KimiModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit KimiModelCatalogProvider(std::string provider_name = "kimi");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

class AnthropicModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit AnthropicModelCatalogProvider(std::string provider_name = "anthropic");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

[[nodiscard]] std::unique_ptr<ModelCatalogProvider>
make_model_catalog_provider(config::ApiType api_type, std::string_view provider_name);

} // namespace core::llm
