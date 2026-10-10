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

/**
 * @brief HTTP envelope for fetching one catalog page.
 *
 * Most catalogs are plain GETs; the default `catalog_request()` reflects
 * that. POST-based catalogs (Cloud Code Assist) override only this hook.
 */
struct ModelCatalogRequest {
    std::string_view method = "GET"; ///< HTTP method for the page fetch.
    std::string body;                ///< Request body (empty for GET).
};

/**
 * Provider-specific adapter for the remote model-catalog contract.
 *
 * Transport orchestration depends only on this interface. Each implementation
 * owns one provider schema and maps it into the provider-neutral ModelInfo
 * domain record; registry precedence and HTTP execution live elsewhere.
 */
class ModelCatalogProvider {
public:
    virtual ~ModelCatalogProvider() = default;

    [[nodiscard]] virtual std::string_view provider_name() const noexcept = 0;
    [[nodiscard]] virtual std::string model_list_path(std::string_view page_token = {}) const = 0;
    [[nodiscard]] virtual ModelCatalogResult parse_models_response(std::string_view body) const = 0;

    /// Request envelope for `model_list_path()`; defaults to a bodiless GET.
    [[nodiscard]] virtual ModelCatalogRequest
    catalog_request(std::string_view /*page_token*/ = {}) const {
        return {};
    }
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

/**
 * Cloud Code Assist catalog adapter (`POST /v1internal:fetchAvailableModels`).
 *
 * Used by the unofficial Antigravity session path against the daily
 * `cloudcode-pa` hosts, where `/v1beta/models` does not exist. The response is
 * an id-keyed `models` map rather than a list; internal and retired ids are
 * dropped the way the real hub client filters its picker.
 */
class CodeAssistModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit CodeAssistModelCatalogProvider(
        std::string provider_name = "gemini-antigravity");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogRequest catalog_request(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

class OpenAICompatibleModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit OpenAICompatibleModelCatalogProvider(
        std::string provider_name = "openai");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

/** Private ChatGPT Codex subscription catalog adapter. */
class CodexModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit CodexModelCatalogProvider(
        std::string provider_name = "openai-codex");

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

class XaiModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit XaiModelCatalogProvider(
        std::string provider_name = "grok",
        bool use_session_catalog = false);

    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
    bool use_session_catalog_ = false;
};

class MistralModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit MistralModelCatalogProvider(
        std::string provider_name = "mistral");

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

class OllamaModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit OllamaModelCatalogProvider(
        std::string provider_name = "ollama");

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

[[nodiscard]] std::unique_ptr<ModelCatalogProvider>
make_model_catalog_provider(config::ApiType api_type,
                            std::string_view provider_name,
                            bool subscription_session);

} // namespace core::llm
