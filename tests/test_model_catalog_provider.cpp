#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <httplib.h>

#include "core/llm/LLMProvider.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ModelCatalogDiscovery.hpp"
#include "core/llm/ModelCatalogProvider.hpp"
#include "core/llm/ModelMetadata.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/GeminiAntigravityProtocol.hpp"
#include "core/llm/protocols/GeminiCodeAssistProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"

#include <atomic>
#include <chrono>
#include <thread>

using namespace core::llm;

namespace {

class CatalogCapabilityProbe final : public LLMProvider,
                                     public ModelCatalogDiscoverable {
public:
    void stream_response(const ChatRequest&,
                         std::function<void(const StreamChunk&)>) override {}

    void discover_models(const ModelCatalogDiscoveryOptions& options) const override {
        ++refresh_count;
        last_timeout_ms = options.timeout_ms;
    }

    mutable int refresh_count = 0;
    mutable int last_timeout_ms = 0;
};

class ProviderWithoutCatalog final : public LLMProvider {
public:
    void stream_response(const ChatRequest&,
                         std::function<void(const StreamChunk&)>) override {}
};

class DeferredCatalogProbe final : public LLMProvider,
                                   public ModelCatalogDiscoverable {
public:
    explicit DeferredCatalogProbe(std::string provider_name)
        : provider_name_(std::move(provider_name)) {}

    void stream_response(const ChatRequest&,
                         std::function<void(const StreamChunk&)>) override {}

    void discover_models(
        const ModelCatalogDiscoveryOptions&) const override {
        auto& availability = ModelCatalogAvailability::instance();
        if (!availability.try_mark_refreshing(provider_name_)) {
            return;
        }

        worker_ = std::jthread([provider_name = provider_name_] {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
            ModelInfo model;
            model.canonical_id = "deferred-live-model";
            model.provider = provider_name;

            ModelCatalogDiscoveryResult success;
            success.attempted = true;
            success.fetched = 1;
            ModelCatalogAvailability::instance().record_result(
                provider_name,
                success,
                {std::move(model)});
        });
    }

private:
    std::string provider_name_;
    mutable std::jthread worker_;
};

} // namespace

TEST_CASE("Model discovery dispatches through an optional provider capability",
          "[llm][model-catalog][discovery]") {
    auto discoverable = std::make_shared<CatalogCapabilityProbe>();
    request_model_catalog_discovery(discoverable, {.timeout_ms = 1234});

    CHECK(discoverable->refresh_count == 1);
    CHECK(discoverable->last_timeout_ms == 1234);

    auto unsupported = std::make_shared<ProviderWithoutCatalog>();
    CHECK_NOTHROW(request_model_catalog_discovery(unsupported));
}

TEST_CASE("Interactive catalog snapshot waits for first asynchronous result",
          "[llm][model-catalog][discovery]") {
    constexpr std::string_view provider_name =
        "deferred-picker-catalog-provider";
    auto provider = std::make_shared<DeferredCatalogProbe>(
        std::string(provider_name));

    const auto snapshot = request_model_catalog_snapshot(
        provider,
        provider_name,
        {.timeout_ms = 1000},
        std::chrono::milliseconds{500});

    REQUIRE(snapshot.state == ModelCatalogDiscoveryState::Succeeded);
    REQUIRE(snapshot.models.size() == 1);
    CHECK(snapshot.models.front().canonical_id == "deferred-live-model");
}

TEST_CASE("GeminiModelCatalogProvider parses live model catalog shape", "[llm][model-catalog]") {
    GeminiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "models": [
        {
          "name": "models/gemini-3.2-flash-001",
          "baseModelId": "gemini-3.2-flash",
          "displayName": "Gemini 3.2 Flash",
          "inputTokenLimit": 1048576,
          "outputTokenLimit": 65536,
          "maxTemperature": 1.5,
          "supportedGenerationMethods": [
            "generateContent",
            "countTokens",
            "createCachedContent",
            "batchGenerateContent"
          ],
          "thinking": true
        },
        {
          "name": "models/text-embedding-005",
          "baseModelId": "text-embedding-005",
          "displayName": "Text Embedding 005",
          "inputTokenLimit": 2048,
          "outputTokenLimit": 1,
          "supportedGenerationMethods": ["embedContent"]
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& flash = result.models[0];
    CHECK(flash.canonical_id == "gemini-3.2-flash");
    CHECK(flash.display_name == "Gemini 3.2 Flash");
    REQUIRE(flash.aliases.size() == 1);
    CHECK(flash.aliases.front() == "gemini-3.2-flash-001");
    CHECK(flash.provider == "gemini");
    CHECK(flash.context_window == 1048576);
    CHECK(flash.max_output_tokens == 65536);
    CHECK(flash.tier == ModelTier::Reasoning);
    CHECK_FALSE(flash.supports(ModelCapability::FunctionCalling));
    CHECK_FALSE(flash.supports(ModelCapability::JsonMode));
    CHECK(flash.supports(ModelCapability::TokenCounting));
    CHECK(flash.supports(ModelCapability::PromptCaching));
    CHECK(flash.supports(ModelCapability::Batch));
    CHECK_FALSE(flash.supports(ModelCapability::Vision));
    CHECK_FALSE(flash.capabilities_complete);
    REQUIRE(flash.constraints.temperature.has_value());
    CHECK(flash.constraints.temperature->max == 1.5);

    const auto& embedding = result.models[1];
    CHECK(embedding.canonical_id == "text-embedding-005");
    CHECK(embedding.supports(ModelCapability::Embeddings));
    CHECK_FALSE(embedding.supports(ModelCapability::TextOutput));
}

TEST_CASE("GeminiModelCatalogProvider captures pagination token", "[llm][model-catalog]") {
    GeminiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "models": [],
      "nextPageToken": "page-2"
    })JSON");

    REQUIRE(result.ok());
    CHECK(result.next_page_token == "page-2");
}

TEST_CASE("Model catalog providers preserve configured provider names",
          "[llm][model-catalog]") {
    GeminiModelCatalogProvider gemini("google-work");
    KimiModelCatalogProvider kimi("moonshot-work");
    AnthropicModelCatalogProvider anthropic("claude-work");

    CHECK(gemini.provider_name() == "google-work");
    CHECK(kimi.provider_name() == "moonshot-work");
    CHECK(anthropic.provider_name() == "claude-work");

    const auto gemini_result = gemini.parse_models_response(R"JSON({
      "models": [{"name": "models/gemini-3.2-flash", "baseModelId": "gemini-3.2-flash"}]
    })JSON");
    REQUIRE(gemini_result.ok());
    REQUIRE(gemini_result.models.size() == 1);
    CHECK(gemini_result.models[0].provider == "google-work");

    const auto kimi_result = kimi.parse_models_response(R"JSON({
      "data": [{"id": "kimi-k2.7"}]
    })JSON");
    REQUIRE(kimi_result.ok());
    REQUIRE(kimi_result.models.size() == 1);
    CHECK(kimi_result.models[0].provider == "moonshot-work");

    const auto anthropic_result = anthropic.parse_models_response(R"JSON({
      "data": [{"id": "claude-sonnet-4-6-20260514"}],
      "has_more": false
    })JSON");
    REQUIRE(anthropic_result.ok());
    REQUIRE(anthropic_result.models.size() == 1);
    CHECK(anthropic_result.models[0].provider == "claude-work");
}

TEST_CASE("OpenAICompatibleModelCatalogProvider parses data array and preserves provider", "[llm][model-catalog]") {
    OpenAICompatibleModelCatalogProvider provider("openai");

    const auto result = provider.parse_models_response(R"JSON({
      "object": "list",
      "data": [
        {"id": "gpt-5.9-future", "object": "model"},
        {"id": "text-embedding-4-large", "object": "model"}
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& gpt = result.models[0];
    CHECK(gpt.canonical_id == "gpt-5.9-future");
    CHECK(gpt.provider == "openai");
    CHECK(gpt.context_window == 0);
    CHECK(gpt.max_output_tokens == 0);
    CHECK(gpt.max_reasoning_tokens == 0);
    CHECK(gpt.capabilities == 0);

    const auto& embedding = result.models[1];
    CHECK(embedding.provider == "openai");
    CHECK(embedding.context_window == 0);
    CHECK(embedding.supports(ModelCapability::Embeddings));
}

TEST_CASE("OpenAI OAuth model catalog sends the client version",
          "[llm][model-catalog][openai]") {
    CodexModelCatalogProvider codex_provider;
    CHECK_THAT(codex_provider.model_list_path(),
               Catch::Matchers::StartsWith("/models?client_version="));

    OpenAICompatibleModelCatalogProvider api_key_provider("openai");
    CHECK(api_key_provider.model_list_path() == "/models");
}

TEST_CASE("OpenAI-compatible catalogs consume optional limits and advertised capabilities",
          "[llm][model-catalog][openai]") {
    OpenAICompatibleModelCatalogProvider provider("compatible");

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "provider-agent-1",
          "display_name": "Provider Agent 1",
          "context_length": 262144,
          "max_tokens": 32768,
          "reasoning_max_tokens": 12000,
          "input_modalities": ["text", "image"],
          "output_modalities": ["text"],
          "supports_streaming": true,
          "supports_system_prompts": true,
          "supports_function_calling": true,
          "supports_parallel_tool_calls": true,
          "supports_structured_outputs": true,
          "capabilities": {
            "reasoning": {"supported": true},
            "prompt_caching": true,
            "pdf_input": {"supported": true}
          }
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    const auto& model = result.models.front();
    CHECK(model.context_window == 262144);
    CHECK(model.max_output_tokens == 32768);
    CHECK(model.max_reasoning_tokens == 12000);
    CHECK(model.tier == ModelTier::Reasoning);
    CHECK(model.supports(ModelCapability::TextInput));
    CHECK(model.supports(ModelCapability::TextOutput));
    CHECK(model.supports(ModelCapability::Streaming));
    CHECK(model.supports(ModelCapability::SystemPrompts));
    CHECK(model.supports(ModelCapability::FunctionCalling));
    CHECK(model.supports(ModelCapability::ParallelToolCalls));
    CHECK(model.supports(ModelCapability::JsonMode));
    CHECK(model.supports(ModelCapability::Vision));
    CHECK(model.supports(ModelCapability::Reasoning));
    CHECK(model.supports(ModelCapability::PromptCaching));
    CHECK(model.supports(ModelCapability::PdfInput));
}

TEST_CASE("CodexModelCatalogProvider parses the subscription model catalog", "[llm][model-catalog][openai]") {
    CodexModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "models": [
        {
          "slug": "gpt-5.5-codex",
          "display_name": "GPT-5.5 Codex",
          "description": "Latest coding model",
          "default_reasoning_level": "high",
          "supported_reasoning_levels": [
            {"effort": "medium", "description": "Balanced"},
            {"effort": "high", "description": "Deep"}
          ],
          "visibility": "list",
          "supported_in_api": true,
          "supports_parallel_tool_calls": true,
          "supports_image_detail_original": true,
          "context_window": 272000,
          "max_context_window": 400000,
          "max_output_tokens": 100000,
          "max_reasoning_tokens": 25000,
          "experimental_supported_tools": ["apply_patch"]
        },
        {
          "slug": "gpt-6.1",
          "display_name": "GPT-6.1",
          "visibility": "list",
          "supported_reasoning_levels": [
            {"effort": "high"},
            {"effort": "xhigh"},
            {"effort": "max"}
          ],
          "context_window": 1000000
        },
        {
          "slug": "internal-hidden",
          "display_name": "Hidden",
          "visibility": "hide"
        },
        {
          "slug": "internal-chatgpt-only",
          "display_name": "Internal ChatGPT Only",
          "supported_in_api": false
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 3);

    const auto& codex = result.models[0];
    CHECK(codex.canonical_id == "gpt-5.5-codex");
    CHECK(codex.display_name == "GPT-5.5 Codex");
    CHECK(codex.provider == "openai-codex");
    CHECK(codex.context_window == 272000);
    CHECK(codex.max_output_tokens == 100000);
    CHECK(codex.max_reasoning_tokens == 25000);
    CHECK(codex.tier == ModelTier::Reasoning);
    CHECK(codex.supports(ModelCapability::FunctionCalling));
    CHECK(codex.supports(ModelCapability::ParallelToolCalls));
    CHECK(codex.supports(ModelCapability::JsonMode));
    CHECK(codex.supports(ModelCapability::Reasoning));
    CHECK(codex.supports(ModelCapability::Vision));

    CHECK(result.models[1].canonical_id == "gpt-6.1");
    CHECK(result.models[1].context_window == 1000000);
    CHECK(result.models[1].reasoning.effort.supports(ReasoningCapability::XHighEffort));
    CHECK(result.models[1].reasoning.effort.supports(ReasoningCapability::MaxEffort));
    CHECK(result.models[2].canonical_id == "internal-chatgpt-only");
}

TEST_CASE("xAI catalog loads callable language models, limits, aliases, and pricing",
          "[llm][model-catalog][xai]") {
    XaiModelCatalogProvider provider("grok");

    const auto result = provider.parse_models_response(R"JSON({
      "object": "list",
      "data": [
        {
          "id": "grok-4.3-latest",
          "aliases": ["grok-latest"],
          "context_length": 131072,
          "prompt_text_token_price": 12500,
          "cached_prompt_text_token_price": 2000,
          "prompt_image_token_price": 12500,
          "completion_text_token_price": 25000
        },
        {
          "id": "grok-420-reasoning",
          "aliases": [],
          "context_length": 256000,
          "prompt_text_token_price": 20000,
          "cached_prompt_text_token_price": 2000,
          "prompt_image_token_price": 0,
          "completion_text_token_price": 80000
        },
        {
          "id": "grok-imagine-image",
          "context_length": 1024,
          "image_price": 200000000
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& multimodal = result.models[0];
    CHECK(multimodal.canonical_id == "grok-4.3-latest");
    CHECK(multimodal.context_window == 131072);
    REQUIRE(multimodal.aliases.size() == 1);
    CHECK(multimodal.aliases.front() == "grok-latest");
    CHECK(multimodal.supports(ModelCapability::Vision));
    CHECK(multimodal.supports(ModelCapability::FunctionCalling));
    CHECK(multimodal.pricing.input_per_mtok == 1.25);
    CHECK(multimodal.pricing.output_per_mtok == 2.5);
    CHECK(multimodal.pricing.cached_input_per_mtok == 0.2);

    const auto& reasoning = result.models[1];
    CHECK(reasoning.context_window == 256000);
    CHECK(reasoning.supports(ModelCapability::Reasoning));
    CHECK_FALSE(reasoning.supports(ModelCapability::Vision));
}

TEST_CASE("xAI session catalog retains the Codex-style proxy schema",
          "[llm][model-catalog][xai]") {
    XaiModelCatalogProvider provider("grok", true);

    const auto result = provider.parse_models_response(R"JSON({
      "models": [{
        "slug": "grok-account-model",
        "display_name": "Grok Account Model",
        "supported_in_api": false,
        "context_window": 200000
      }]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    CHECK(result.models.front().canonical_id == "grok-account-model");
    CHECK(result.models.front().context_window == 200000);
}

TEST_CASE("xAI session catalog decodes the grok-build proxy data[] rows",
          "[llm][model-catalog][xai]") {
    // Mirrors xai-org/grok-build remote/client.rs parse_remote_model_value:
    // the proxy serves {"data":[...]} with rich camelCase "models-v2" rows.
    XaiModelCatalogProvider provider("grok", true);

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "grok-4.7",
          "model": "grok-4.7",
          "name": "Grok 4.7",
          "contextWindow": 500000,
          "maxCompletionTokens": 131072,
          "supportsReasoningEffort": true,
          "reasoningEfforts": [
            {"value": "xhigh", "label": "Extra High Effort"},
            {"value": "high", "label": "High Effort", "default": true},
            {"value": "medium", "label": "Medium Effort"},
            {"value": "low", "label": "Low Effort"}
          ],
          "supportsBackendSearch": true,
          "apiBackend": "responses"
        },
        {
          "id": "grok-account-legacy",
          "model": "grok-account-legacy",
          "contextWindow": 256000
        },
        {
          "id": "grok-hidden-plan-model",
          "model": "grok-hidden-plan-model",
          "hidden": true
        },
        {
          "id": "grok-web-only-model",
          "model": "grok-web-only-model",
          "supportedInApi": false
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& grok47 = result.models[0];
    CHECK(grok47.canonical_id == "grok-4.7");
    CHECK(grok47.display_name == "Grok 4.7");
    CHECK(grok47.context_window == 500000);
    CHECK(grok47.max_output_tokens == 131072);
    CHECK(grok47.supports(ModelCapability::Reasoning));
    CHECK(grok47.supports(ModelCapability::FunctionCalling));
    CHECK(grok47.reasoning.effort.supports_effort());
    CHECK(grok47.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));
    REQUIRE(grok47.reasoning.catalog_default_effort.has_value());
    CHECK(*grok47.reasoning.catalog_default_effort == "high");

    const auto& legacy = result.models[1];
    CHECK(legacy.canonical_id == "grok-account-legacy");
    CHECK(legacy.context_window == 256000);
    // No effort menu and no flag: the profile stays empty and the model is
    // still listed (the registry baseline decides reasoning elsewhere).
    CHECK(legacy.reasoning.effort.empty());
    CHECK_FALSE(legacy.reasoning.catalog_default_effort.has_value());
}

TEST_CASE("xAI public catalog reads the capabilities reasoning-effort menu",
          "[llm][model-catalog][xai]") {
    // Mirrors the public /v1/models row from grok-build's client tests: the
    // effort menu arrives under capabilities.reasoning_effort as bare strings.
    XaiModelCatalogProvider provider("grok");

    const auto result = provider.parse_models_response(R"JSON({
      "object": "list",
      "data": [
        {
          "id": "grok-4.7",
          "object": "model",
          "owned_by": "xai",
          "context_length": 500000,
          "capabilities": {
            "reasoning_effort": ["low", "medium", "high", "xhigh"],
            "default_reasoning_effort": "high"
          },
          "prompt_text_token_price": 20000,
          "completion_text_token_price": 60000
        },
        {
          "id": "grok-imagine-image",
          "context_length": 1024,
          "image_price": 200000000
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);

    const auto& grok47 = result.models.front();
    CHECK(grok47.canonical_id == "grok-4.7");
    CHECK(grok47.context_window == 500000);
    CHECK(grok47.supports(ModelCapability::Reasoning));
    CHECK(grok47.reasoning.effort.supports_effort());
    CHECK(grok47.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));
    REQUIRE(grok47.reasoning.catalog_default_effort.has_value());
    CHECK(*grok47.reasoning.catalog_default_effort == "high");
    CHECK(grok47.pricing.input_per_mtok == 2.0);
    CHECK(grok47.pricing.output_per_mtok == 6.0);
}

TEST_CASE("xAI catalog effort default matches grok-build",
          "[llm][model-catalog][xai]") {
    XaiModelCatalogProvider provider("grok", true);

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "grok-unmarked-menu",
          "model": "grok-unmarked-menu",
          "reasoningEfforts": [
            {"value": "nope"},
            {"value": "xhigh"},
            {"value": "high"}
          ]
        },
        {
          "id": "grok-explicit-wins",
          "model": "grok-explicit-wins",
          "reasoning_effort": "low",
          "reasoningEfforts": [
            {"value": "high", "default": true},
            {"value": "low"}
          ]
        },
        {
          "id": "grok-flag-only",
          "model": "grok-flag-only",
          "supportsReasoningEffort": true
        },
        {
          "id": "grok-public-unmarked",
          "model": "grok-public-unmarked",
          "completion_text_token_price": 60000,
          "capabilities": {
            "reasoning_effort": ["low", "medium", "high", "xhigh"],
            "default_reasoning_effort": "turbo"
          }
        },
        {
          "id": "grok-public-server-default",
          "model": "grok-public-server-default",
          "completion_text_token_price": 60000,
          "capabilities": {
            "reasoning_effort": ["low", "high"]
          }
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 5);

    const auto& unmarked = result.models[0];
    REQUIRE(unmarked.reasoning.catalog_default_effort.has_value());
    CHECK(*unmarked.reasoning.catalog_default_effort == "xhigh");
    CHECK(unmarked.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));

    const auto& explicit_wins = result.models[1];
    REQUIRE(explicit_wins.reasoning.catalog_default_effort.has_value());
    CHECK(*explicit_wins.reasoning.catalog_default_effort == "low");

    const auto& flag_only = result.models[2];
    CHECK(flag_only.reasoning.effort.supports_effort());
    REQUIRE(flag_only.reasoning.catalog_default_effort.has_value());
    CHECK(flag_only.reasoning.catalog_default_effort->empty());

    const auto& unknown_mark = result.models[3];
    REQUIRE(unknown_mark.reasoning.catalog_default_effort.has_value());
    CHECK(unknown_mark.reasoning.catalog_default_effort->empty());
    CHECK(unknown_mark.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));

    const auto& server_default = result.models[4];
    REQUIRE(server_default.reasoning.catalog_default_effort.has_value());
    CHECK(server_default.reasoning.catalog_default_effort->empty());
    CHECK_FALSE(server_default.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));
}

TEST_CASE("A catalog effort policy replaces the baseline without erasing it when absent",
          "[llm][model-catalog][metadata-merge]") {
    ModelInfo baseline;
    baseline.canonical_id = "grok-4.7";
    baseline.reasoning.catalog_default_effort = "high";

    ModelInfo silent;
    silent.canonical_id = baseline.canonical_id;
    silent.reasoning.effort = ReasoningCapabilities{ReasoningCapability::Effort};
    const ModelInfo kept =
        merge_model_metadata(baseline, silent);
    REQUIRE(kept.reasoning.catalog_default_effort.has_value());
    CHECK(*kept.reasoning.catalog_default_effort == "high");

    ModelInfo discovered;
    discovered.canonical_id = baseline.canonical_id;
    discovered.reasoning.catalog_default_effort = "";
    const ModelInfo omitted =
        merge_model_metadata(std::move(baseline), std::move(discovered));
    REQUIRE(omitted.reasoning.catalog_default_effort.has_value());
    CHECK(omitted.reasoning.catalog_default_effort->empty());
}

TEST_CASE("Mistral catalog loads its capability matrix and context limit",
          "[llm][model-catalog][mistral]") {
    MistralModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "object": "list",
      "data": [
        {
          "id": "mistral-future-vision",
          "aliases": ["mistral-future-latest"],
          "max_context_length": 262144,
          "archived": false,
          "capabilities": {
            "completion_chat": true,
            "completion_fim": false,
            "function_calling": true,
            "fine_tuning": false,
            "vision": true,
            "classification": false
          }
        },
        {
          "id": "archived-model",
          "archived": true,
          "capabilities": {"completion_chat": true}
        },
        {
          "id": "classifier-only",
          "archived": false,
          "capabilities": {
            "completion_chat": false,
            "classification": true
          }
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    const auto& model = result.models.front();
    CHECK(model.canonical_id == "mistral-future-vision");
    CHECK(model.context_window == 262144);
    CHECK_FALSE(model.capabilities_complete);
    CHECK(model.supports(ModelCapability::TextInput));
    CHECK(model.supports(ModelCapability::Streaming));
    CHECK(model.supports(ModelCapability::JsonMode));
    CHECK(model.supports(ModelCapability::FunctionCalling));
    CHECK(model.supports(ModelCapability::ParallelToolCalls));
    CHECK(model.supports(ModelCapability::Vision));
    REQUIRE(model.aliases.size() == 1);
    CHECK(model.aliases.front() == "mistral-future-latest");
}

TEST_CASE("Mistral catalog accepts the documented bare-array response",
          "[llm][model-catalog][mistral]") {
    MistralModelCatalogProvider provider;
    const auto result = provider.parse_models_response(R"JSON([
      {
        "id": "mistral-bare-array",
        "max_context_length": 32768,
        "capabilities": {"completion_chat": true}
      }
    ])JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    CHECK(result.models.front().context_window == 32768);
}

TEST_CASE("Ollama catalog loads locally installed model identities",
          "[llm][model-catalog][ollama]") {
    OllamaModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "models": [
        {
          "name": "qwen3:30b",
          "modified_at": "2026-07-26T10:00:00Z",
          "size": 18000000000,
          "digest": "sha256",
          "details": {
            "format": "gguf",
            "family": "qwen3",
            "parameter_size": "30B",
            "quantization_level": "Q4_K_M"
          }
        },
        {"model": "nomic-embed-text:latest"}
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);
    CHECK(result.models[0].canonical_id == "qwen3:30b");
    CHECK(result.models[0].provider == "ollama");
    CHECK(result.models[0].supports(ModelCapability::TextInput));
    CHECK(result.models[0].supports(ModelCapability::JsonMode));
    CHECK_FALSE(result.models[0].capabilities_complete);
    CHECK(result.models[1].supports(ModelCapability::Embeddings));
}

TEST_CASE("KimiModelCatalogProvider parses Moonshot enriched models response", "[llm][model-catalog]") {
    KimiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "kimi-k2.7-thinking-turbo",
          "display_name": "Kimi K2.7 Thinking Turbo",
          "context_length": 262144,
          "supports_reasoning": true,
          "supports_image_in": true,
          "supports_video_in": true
        },
        {
          "id": "moonshot-v1-32k",
          "context_length": 32768,
          "supports_reasoning": false,
          "supports_image_in": false,
          "supports_video_in": false
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& thinking = result.models[0];
    CHECK(thinking.canonical_id == "kimi-k2.7-thinking-turbo");
    CHECK(thinking.display_name == "Kimi K2.7 Thinking Turbo");
    CHECK(thinking.provider == "kimi");
    CHECK(thinking.context_window == 262144);
    CHECK(thinking.max_output_tokens == 8192);
    CHECK(thinking.tier == ModelTier::Reasoning);
    CHECK(thinking.supports(ModelCapability::FunctionCalling));
    CHECK(thinking.supports(ModelCapability::JsonMode));
    CHECK(thinking.supports(ModelCapability::Reasoning));
    CHECK(thinking.supports(ModelCapability::Vision));
    CHECK(thinking.supports(ModelCapability::VideoInput));

    const auto& legacy = result.models[1];
    CHECK(legacy.canonical_id == "moonshot-v1-32k");
    CHECK(legacy.context_window == 32768);
    CHECK_FALSE(legacy.supports(ModelCapability::Reasoning));
    CHECK_FALSE(legacy.supports(ModelCapability::Vision));
    CHECK_FALSE(legacy.supports(ModelCapability::VideoInput));
}

TEST_CASE("KimiModelCatalogProvider parses current Kimi Code reasoning metadata",
          "[llm][model-catalog][kimi]") {
    KimiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "k3",
          "display_name": "K3",
          "context_length": 262144,
          "supports_reasoning": true,
          "supports_thinking_type": "only",
          "think_efforts": {
            "support": true,
            "valid_efforts": ["low", "high", "max"],
            "default_effort": "high"
          },
          "supports_image_in": true,
          "supports_video_in": true
        },
        {
          "id": "kimi-for-coding",
          "display_name": "K2.7 Coding",
          "context_length": 262144,
          "supports_reasoning": true,
          "supports_thinking_type": "only",
          "think_efforts": null,
          "supports_image_in": true,
          "supports_video_in": true
        },
        {
          "id": "K3-256K",
          "supports_reasoning": true,
          "supports_thinking_type": "only",
          "think_efforts": {
            "support": true,
            "valid_efforts": ["low", "high", "max"]
          },
          "supports_image_in": true,
          "supports_video_in": false
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 3);

    const auto& k3 = result.models[0];
    CHECK(k3.context_window == 262144);
    CHECK(k3.max_output_tokens == 262144);
    CHECK(k3.reasoning.complete);
    CHECK(k3.reasoning.manual_thinking);
    CHECK(k3.reasoning.effort.supports(ReasoningCapability::Effort));
    CHECK(k3.reasoning.effort.supports(ReasoningCapability::Required));
    CHECK(k3.reasoning.effort.supports(ReasoningCapability::MaxEffort));
    CHECK_FALSE(k3.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));

    const auto& k27 = result.models[1];
    CHECK(k27.reasoning.complete);
    CHECK(k27.reasoning.effort.supports(ReasoningCapability::Effort));
    CHECK(k27.reasoning.effort.supports(ReasoningCapability::Required));
    CHECK_FALSE(k27.reasoning.effort.supports(
        ReasoningCapability::MaxEffort));

    const auto& k3_256k = result.models[2];
    CHECK(k3_256k.canonical_id == "K3-256K");
    CHECK(k3_256k.context_window == 262144);
    CHECK(k3_256k.max_output_tokens == 262144);
    CHECK(k3_256k.supports(ModelCapability::Vision));
    CHECK_FALSE(k3_256k.supports(ModelCapability::VideoInput));
}

TEST_CASE("KimiModelCatalogProvider parses the managed Kimi Code catalog dialect",
          "[llm][model-catalog][kimi]") {
    KimiModelCatalogProvider provider;

    // The managed /models response nests what kimi-code reads: a mandatory
    // limit.context, one `reasoning` object, `modalities.input`,
    // `supports_tool_use`, and the wire the endpoint serves the model on.
    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "k3",
          "display_name": "K3",
          "limit": { "context": 1048576 },
          "reasoning": {
            "type": "only",
            "effort": { "valid": ["low", "high", "max"], "default": "max" }
          },
          "modalities": { "input": ["text", "image", "video"] },
          "supports_tool_use": true
        },
        {
          "id": "kimi-for-coding",
          "limit": { "context": 262144 },
          "reasoning": { "type": "both" },
          "modalities": { "input": ["text"] },
          "protocol": "anthropic"
        },
        {
          "id": "kimi-k2.6",
          "limit": { "context": 262144 },
          "reasoning": { "type": "none" },
          "supports_tool_use": false
        },
        {
          "id": "k3-256k",
          "limit": { "context": 262144 },
          "reasoning": {
            "type": "unrecognized",
            "effort": { "valid": ["low"], "default": "not-listed" }
          },
          "protocol": "response"
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 4);

    const auto& k3 = result.models[0];
    CHECK(k3.display_name == "K3");
    CHECK(k3.context_window == 1048576);
    CHECK(k3.max_output_tokens == 1048576);
    CHECK(k3.reasoning.complete);
    CHECK(k3.reasoning.manual_thinking);
    CHECK(k3.reasoning.effort.supports(ReasoningCapability::Required));
    CHECK(k3.reasoning.effort.supports(ReasoningCapability::MaxEffort));
    CHECK(k3.reasoning.catalog_default_effort == "max");
    CHECK(k3.reasoning.supported_efforts
          == std::vector<std::string>{"low", "high", "max"});
    CHECK(k3.supports(ModelCapability::Vision));
    CHECK(k3.supports(ModelCapability::VideoInput));
    CHECK(k3.supports(ModelCapability::FunctionCalling));
    CHECK(k3.wire_protocol.empty());

    const auto& coding = result.models[1];
    CHECK(coding.context_window == 262144);
    // "both": thinking can be switched off, so effort is offered but not forced.
    CHECK(coding.reasoning.complete);
    CHECK(coding.reasoning.effort.supports(ReasoningCapability::Effort));
    CHECK_FALSE(coding.reasoning.effort.supports(
        ReasoningCapability::Required));
    // A published modality list that omits image/video is authoritative, even
    // for an id whose name would otherwise infer vision.
    CHECK_FALSE(coding.supports(ModelCapability::Vision));
    CHECK_FALSE(coding.supports(ModelCapability::VideoInput));
    CHECK(coding.wire_protocol == "anthropic");

    const auto& k26 = result.models[2];
    // "none" publishes a non-thinking model, and supports_tool_use:false strips
    // tool calling even though the id would otherwise infer it.
    CHECK(k26.reasoning.complete);
    CHECK_FALSE(k26.reasoning.effort.supports(ReasoningCapability::Effort));
    CHECK_FALSE(k26.supports(ModelCapability::FunctionCalling));
    CHECK_FALSE(k26.supports(ModelCapability::ParallelToolCalls));

    const auto& k3_256k = result.models[3];
    // An unrecognized reasoning type stays unknown rather than being guessed
    // at, and a default outside `valid` is dropped; `response` normalizes to
    // the Responses wire.
    CHECK_FALSE(k3_256k.reasoning.complete);
    CHECK_FALSE(k3_256k.reasoning.catalog_default_effort.has_value());
    CHECK(k3_256k.wire_protocol == "openai_responses");
}

TEST_CASE("Kimi catalog reasoning is authoritative and matched exactly",
          "[llm][model-catalog][kimi]") {
    KimiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "brand-new-kimi",
          "limit": { "context": 131072 },
          "reasoning": {
            "type": "only",
            "effort": { "valid": ["low", "high"], "default": "high" }
          }
        },
        {
          "id": "kimi-k2.6",
          "supports_thinking_type": "only"
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    // Nothing about this id implies reasoning. The published block must still
    // win, and "only" must not be read as "no".
    const auto& unseen = result.models[0];
    CHECK(unseen.supports(ModelCapability::Reasoning));
    CHECK(unseen.reasoning.complete);
    CHECK(unseen.reasoning.effort.supports(ReasoningCapability::Required));
    CHECK(unseen.reasoning.catalog_default_effort == "high");
    CHECK_FALSE(unseen.supports(ModelCapability::Vision));

    // Open-platform dialect, and no supports_reasoning boolean to paper over a
    // substring match of "no" inside "only".
    const auto& k26 = result.models[1];
    CHECK(k26.supports(ModelCapability::Reasoning));
    CHECK(k26.reasoning.complete);
    CHECK(k26.reasoning.effort.supports(ReasoningCapability::Required));
}

TEST_CASE("Merging a catalog row does not publish its endpoint wire",
          "[llm][model-catalog][kimi]") {
    ModelInfo baseline;
    baseline.canonical_id = "k3";
    baseline.context_window = 1'048'576;

    ModelInfo discovered = baseline;
    discovered.wire_protocol = "anthropic";
    discovered.context_window = 262'144;

    const ModelInfo merged =
        merge_model_metadata(baseline, std::move(discovered));
    CHECK(merged.context_window == 262'144);
    CHECK(merged.wire_protocol.empty());
}

TEST_CASE("KimiModelCatalogProvider infers context when Moonshot omits it", "[llm][model-catalog]") {
    KimiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {"id": "k3"},
        {"id": "kimi-k3"},
        {"id": "moonshot-v1-8k"},
        {"id": "kimi-for-coding"}
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 4);
    for (std::size_t index : {std::size_t{0}, std::size_t{1}}) {
        CHECK(result.models[index].context_window == 1'048'576);
        CHECK(result.models[index].max_output_tokens == 1'048'576);
        CHECK(result.models[index].supports(ModelCapability::Reasoning));
        CHECK(result.models[index].supports(ModelCapability::Vision));
        CHECK(result.models[index].supports(ModelCapability::VideoInput));
    }
    CHECK(result.models[2].context_window == 8192);
    // Managed Kimi Code row with no published limit: the K2-series window
    // Moonshot documents (262,144) is inferred, and the completion ceiling
    // follows it because the API accepts max_completion_tokens up to the window.
    CHECK(result.models[3].context_window == 262144);
    CHECK(result.models[3].max_output_tokens == 262144);
    CHECK(result.models[3].supports(ModelCapability::Reasoning));
    CHECK(result.models[3].supports(ModelCapability::Vision));
}

TEST_CASE("Kimi catalog treats explicit capability flags as authoritative",
          "[llm][model-catalog][kimi]") {
    KimiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [{
        "id": "kimi-k2-no-media",
        "supports_reasoning": false,
        "supports_image_in": false,
        "supports_video_in": false
      }]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    const auto& model = result.models.front();
    CHECK_FALSE(model.supports(ModelCapability::Reasoning));
    CHECK_FALSE(model.supports(ModelCapability::Vision));
    CHECK_FALSE(model.supports(ModelCapability::VideoInput));
}

TEST_CASE("AnthropicModelCatalogProvider parses limits and capabilities",
          "[llm][model-catalog][anthropic]") {
    AnthropicModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "claude-opus-5",
          "display_name": "Claude Opus 5",
          "max_input_tokens": 1000000,
          "max_tokens": 128000,
          "capabilities": {
            "batch": {"supported": true},
            "citations": {"supported": true},
            "code_execution": {"supported": true},
            "context_management": {"supported": true},
            "effort": {
              "supported": true,
              "low": {"supported": true},
              "medium": {"supported": true},
              "high": {"supported": true},
              "max": {"supported": true},
              "xhigh": {"supported": true}
            },
            "image_input": {"supported": true},
            "pdf_input": {"supported": true},
            "structured_outputs": {"supported": true},
            "thinking": {
              "supported": true,
              "types": {
                "adaptive": {"supported": true},
                "enabled": {"supported": false}
              }
            }
          }
        },
        {
          "id": "claude-legacy-sparse",
          "display_name": "Claude Legacy Sparse"
        }
      ],
      "has_more": false
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& opus = result.models[0];
    CHECK(opus.canonical_id == "claude-opus-5");
    CHECK(opus.display_name == "Claude Opus 5");
    CHECK(opus.provider == "anthropic");
    CHECK(opus.context_window == 1000000);
    CHECK(opus.max_output_tokens == 128000);
    CHECK(opus.capabilities_complete);
    CHECK(opus.tier == ModelTier::Reasoning);
    CHECK(opus.supports(ModelCapability::TextInput));
    CHECK(opus.supports(ModelCapability::TextOutput));
    CHECK(opus.supports(ModelCapability::Streaming));
    CHECK(opus.supports(ModelCapability::SystemPrompts));
    CHECK(opus.supports(ModelCapability::FunctionCalling));
    CHECK(opus.supports(ModelCapability::ParallelToolCalls));
    CHECK(opus.supports(ModelCapability::PromptCaching));
    CHECK(opus.supports(ModelCapability::TokenCounting));
    CHECK(opus.supports(ModelCapability::Vision));
    CHECK(opus.supports(ModelCapability::PdfInput));
    CHECK(opus.supports(ModelCapability::JsonMode));
    CHECK(opus.supports(ModelCapability::Reasoning));
    CHECK(opus.supports(ModelCapability::Citations));
    CHECK(opus.supports(ModelCapability::CodeExecution));
    CHECK(opus.supports(ModelCapability::Batch));
    CHECK(opus.supports(ModelCapability::ContextManagement));
    CHECK(opus.reasoning.complete);
    CHECK(opus.reasoning.effort.supports_effort());
    CHECK(opus.reasoning.effort.supports(ReasoningCapability::MaxEffort));
    CHECK(opus.reasoning.effort.supports(ReasoningCapability::XHighEffort));
    CHECK(opus.reasoning.adaptive_thinking);
    CHECK_FALSE(opus.reasoning.manual_thinking);
    REQUIRE(opus.constraints.temperature.has_value());
    CHECK(opus.constraints.temperature->max == 1.0);

    const auto& sparse = result.models[1];
    CHECK(sparse.context_window == 0);
    CHECK(sparse.max_output_tokens == 0);
    CHECK(sparse.capabilities == 0);
    CHECK_FALSE(sparse.capabilities_complete);
    CHECK_FALSE(sparse.reasoning.complete);
    CHECK_FALSE(sparse.supports(ModelCapability::Vision));
    CHECK_FALSE(sparse.supports(ModelCapability::Reasoning));
}

TEST_CASE("A newly released Claude model is data-driven end to end",
          "[llm][model-catalog][anthropic][future-model]") {
    AnthropicModelCatalogProvider provider;
    auto result = provider.parse_models_response(R"JSON({
      "data": [{
        "id": "claude-future-dynamic-2030",
        "display_name": "Claude Future Dynamic",
        "max_input_tokens": 2000000,
        "max_tokens": 192000,
        "capabilities": {
          "effort": {
            "supported": true,
            "low": {"supported": true},
            "medium": {"supported": true},
            "high": {"supported": true},
            "max": {"supported": false},
            "xhigh": {"supported": true}
          },
          "thinking": {
            "supported": true,
            "types": {
              "adaptive": {"supported": true},
              "enabled": {"supported": false}
            }
          },
          "image_input": {"supported": true},
          "pdf_input": {"supported": true}
        }
      }],
      "has_more": false
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    auto& registry = ModelRegistry::instance();
    CHECK(registry.merge_model(std::move(result.models.front())));

    const auto discovered = registry.lookup("claude-future-dynamic-2030");
    REQUIRE(discovered);
    CHECK(discovered->context_window == 2'000'000);
    CHECK(discovered->max_output_tokens == 192'000);
    CHECK(discovered->reasoning.complete);
    CHECK(discovered->reasoning.adaptive_thinking);
    CHECK_FALSE(discovered->reasoning.manual_thinking);

    ChatRequest request;
    request.model = "claude-future-dynamic-2030";
    request.effort = "xhigh";
    request.messages.push_back({.role = "user", .content = "Hello"});
    const std::string payload =
        protocols::ClaudeSerializer::serialize(request);

    CHECK(payload.find(R"("max_tokens":192000)") != std::string::npos);
    CHECK(payload.find(R"("output_config":{"effort":"xhigh"})")
          != std::string::npos);
    // Adaptive-only reasoning is the marker of the generations that return
    // thinking blocks with empty text, so the request opts into the summarized
    // display even though Filo has no curated card for this model.
    CHECK(payload.find(
              R"("thinking":{"type":"adaptive","display":"summarized"})")
          != std::string::npos);
    CHECK(payload.find(R"("thinking":{"type":"enabled")")
          == std::string::npos);
}

TEST_CASE("AnthropicModelCatalogProvider captures next cursor", "[llm][model-catalog]") {
    AnthropicModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [],
      "has_more": true,
      "last_id": "model-cursor"
    })JSON");

    REQUIRE(result.ok());
    CHECK(result.next_page_token == "model-cursor");
}

TEST_CASE("make_model_catalog_provider selects supported catalog implementations", "[llm][model-catalog]") {
    auto gemini = make_model_catalog_provider(core::config::ApiType::Gemini, "gemini");
    REQUIRE(gemini != nullptr);
    CHECK(gemini->provider_name() == "gemini");
    CHECK(gemini->model_list_path() == "/v1beta/models?pageSize=1000");
    CHECK(gemini->model_list_path("next") == "/v1beta/models?pageSize=1000&pageToken=next");

    auto antigravity = make_model_catalog_provider(
        core::config::ApiType::Gemini, "gemini-antigravity");
    REQUIRE(antigravity != nullptr);
    CHECK(dynamic_cast<CodeAssistModelCatalogProvider*>(antigravity.get()) != nullptr);
    CHECK(antigravity->model_list_path() == "/v1internal:fetchAvailableModels");
    CHECK(antigravity->catalog_request().method == "POST");

    auto grok = make_model_catalog_provider(core::config::ApiType::OpenAI, "grok");
    REQUIRE(grok != nullptr);
    CHECK(grok->provider_name() == "grok");
    CHECK(grok->model_list_path() == "/models");

    auto codex = make_model_catalog_provider(
        core::config::ApiType::OpenAI, "openai-codex");
    REQUIRE(codex != nullptr);
    CHECK(dynamic_cast<CodexModelCatalogProvider*>(codex.get()) != nullptr);

    core::llm::protocols::CodexResponsesProtocol codex_protocol;
    auto chatgpt = codex_protocol.model_catalog("openai");
    REQUIRE(chatgpt != nullptr);
    CHECK(dynamic_cast<CodexModelCatalogProvider*>(chatgpt.get()) != nullptr);
    CHECK(chatgpt->provider_name() == "openai");
    CHECK(chatgpt->model_list_path() == "/models?client_version=0.160.0");

    core::llm::protocols::OpenAIResponsesProtocol public_openai;
    CHECK(public_openai.model_catalog("openai") == nullptr);

    auto mistral = make_model_catalog_provider(
        core::config::ApiType::OpenAI,
        "mistral");
    REQUIRE(mistral != nullptr);
    CHECK(dynamic_cast<MistralModelCatalogProvider*>(mistral.get()) != nullptr);

    auto kimi = make_model_catalog_provider(core::config::ApiType::Kimi, "kimi");
    REQUIRE(kimi != nullptr);
    CHECK(kimi->provider_name() == "kimi");
    CHECK(kimi->model_list_path() == "/models");

    auto qwen = make_model_catalog_provider(core::config::ApiType::DashScope, "qwen");
    REQUIRE(qwen != nullptr);
    CHECK(qwen->provider_name() == "qwen");
    CHECK(qwen->model_list_path() == "/models");

    auto claude = make_model_catalog_provider(core::config::ApiType::Anthropic, "claude");
    REQUIRE(claude != nullptr);
    CHECK(claude->model_list_path() == "/v1/models?limit=1000");
    CHECK(claude->model_list_path("cursor") == "/v1/models?limit=1000&after_id=cursor");

    auto ollama = make_model_catalog_provider(core::config::ApiType::Ollama, "ollama");
    REQUIRE(ollama != nullptr);
    CHECK(ollama->model_list_path() == "/api/tags");

    auto zai = make_model_catalog_provider(
        core::config::ApiType::OpenAI,
        "zai");
    REQUIRE(zai != nullptr);
    CHECK(zai->provider_name() == "zai");
    CHECK(zai->model_list_path() == "/models");

    auto zai_coding = make_model_catalog_provider(
        core::config::ApiType::OpenAI,
        "zai-coding");
    REQUIRE(zai_coding != nullptr);
    CHECK(zai_coding->provider_name() == "zai-coding");
    CHECK(zai_coding->model_list_path() == "/models");

    const auto zai_result = zai_coding->parse_models_response(R"JSON({
      "object": "list",
      "data": [
        {"id": "glm-5.3", "object": "model"},
        {"id": "glm-future-live", "object": "model"}
      ]
    })JSON");
    REQUIRE(zai_result.ok());
    REQUIRE(zai_result.models.size() == 2);
    CHECK(zai_result.models[0].canonical_id == "glm-5.3");
    CHECK(zai_result.models[1].canonical_id == "glm-future-live");
}

TEST_CASE("Remote model discovery skips quietly without credentials",
          "[llm][model-catalog][discovery]") {
    using core::config::ApiType;

    core::llm::protocols::OpenAIProtocol openai;
    CHECK_FALSE(discover_and_register_models(
        "openai",
        ApiType::OpenAI,
        "https://api.openai.com/v1",
        nullptr,
        openai).attempted);

    core::llm::protocols::DashScopeProtocol dashscope;
    CHECK_FALSE(discover_and_register_models(
        "qwen",
        ApiType::DashScope,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        nullptr,
        dashscope).attempted);

    core::llm::protocols::KimiProtocol kimi_protocol;
    CHECK_FALSE(discover_and_register_models(
        "kimi",
        ApiType::Kimi,
        "https://api.moonshot.cn/v1",
        nullptr,
        kimi_protocol).attempted);

    core::llm::protocols::ClaudeProtocol anthropic;
    CHECK_FALSE(discover_and_register_models(
        "claude",
        ApiType::Anthropic,
        "https://api.anthropic.com",
        nullptr,
        anthropic).attempted);

    core::llm::protocols::GeminiProtocol gemini;
    CHECK_FALSE(discover_and_register_models(
        "gemini",
        ApiType::Gemini,
        "https://generativelanguage.googleapis.com",
        nullptr,
        gemini).attempted);
}

TEST_CASE("ModelRegistry loads JSON model overrides and aliases", "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();
    const auto before = registry.size();

    const int loaded = registry.load_from_json(R"JSON({
      "models": [
        {
          "canonical_id": "provider-live-model-2026",
          "aliases": ["provider-live-model", "live-model"],
          "display_name": "Provider Live Model",
          "provider": "custom",
          "context_window": 262144,
          "max_output_tokens": 32768,
          "max_reasoning_tokens": 12000,
          "capabilities": ["text_input", "text_output", "streaming", "function_calling", "json_mode", "reasoning"],
          "tier": "reasoning",
          "pricing": {"input": 1.5, "output": 6.0, "cached_input": 0.5},
          "constraints": {"temperature": [0.0, 1.0], "max_tokens_min": 1}
        }
      ]
    })JSON");

    REQUIRE(loaded == 1);
    CHECK(registry.size() == before + 1);
    REQUIRE(registry.has_model("provider-live-model-2026"));
    REQUIRE(registry.has_model("provider-live-model"));

    const auto info = registry.get_info("live-model");
    REQUIRE(info.has_value());
    CHECK(info->canonical_id == "provider-live-model-2026");
    CHECK(info->provider == "custom");
    CHECK(info->context_window == 262144);
    CHECK(info->max_output_tokens == 32768);
    CHECK(info->max_reasoning_tokens == 12000);
    CHECK(info->tier == ModelTier::Reasoning);
    CHECK(info->supports(ModelCapability::FunctionCalling));
    CHECK(info->supports(ModelCapability::JsonMode));
    CHECK(info->pricing.input_per_mtok == 1.5);
    CHECK(info->pricing.cached_input_per_mtok == 0.5);
    CHECK(info->validate_parameter("temperature", 1.0));
    CHECK_FALSE(info->validate_parameter("temperature", 1.1));
}

TEST_CASE("ModelRegistry model replacement removes stale aliases", "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();

    ModelInfo info;
    info.canonical_id = "alias-refresh-model";
    info.display_name = "Alias Refresh Model";
    info.provider = "test";
    info.aliases = {"alias-refresh-old"};
    registry.register_model(info);
    REQUIRE(registry.has_model("alias-refresh-old"));

    info.aliases = {"alias-refresh-new"};
    registry.register_model(info);

    CHECK_FALSE(registry.has_model("alias-refresh-old"));
    CHECK(registry.has_model("alias-refresh-new"));
}

TEST_CASE("ModelRegistry merge preserves catalog metadata when discovery is sparse", "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();

    ModelInfo catalog;
    catalog.canonical_id = "merge-known-model";
    catalog.display_name = "Merge Known Model";
    catalog.provider = "known";
    catalog.context_window = 128000;
    catalog.max_output_tokens = 8192;
    catalog.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput) |
        static_cast<uint32_t>(ModelCapability::FunctionCalling);
    registry.register_model(catalog);

    ModelInfo discovered;
    discovered.canonical_id = "merge-known-model";
    discovered.display_name = "merge-known-model";
    discovered.provider = "known";
    CHECK_FALSE(registry.merge_model(std::move(discovered)));

    const auto merged = registry.get_info("merge-known-model");
    REQUIRE(merged.has_value());
    CHECK(merged->display_name == "Merge Known Model");
    CHECK(merged->context_window == 128000);
    CHECK(merged->max_output_tokens == 8192);
    CHECK(merged->supports(ModelCapability::FunctionCalling));
}

TEST_CASE("Sparse capability catalogs never create false validation failures",
          "[llm][model-catalog][capabilities]") {
    constexpr std::string_view model_id =
        "partial-capability-validation-model";

    ModelInfo partial;
    partial.canonical_id = std::string(model_id);
    partial.provider = "partial-capability-provider";
    partial.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput);
    partial.capabilities_complete = false;
    ModelRegistry::instance().register_model(partial);

    HttpLLMProvider provider(
        "https://example.invalid/v1",
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        std::string(model_id),
        std::make_unique<core::llm::protocols::OpenAIProtocol>(),
        core::config::ApiType::OpenAI,
        partial.provider);

    ChatRequest request;
    request.model = std::string(model_id);
    request.messages.push_back({.role = "user", .content = "Hello"});
    Tool tool;
    tool.function.name = "test_tool";
    tool.function.input_schema = R"JSON({"type":"object"})JSON";
    request.tools.push_back(std::move(tool));

    CHECK(provider.validate_request(request).empty());

    partial.capabilities_complete = true;
    ModelRegistry::instance().register_model(std::move(partial));
    const auto errors = provider.validate_request(request);
    REQUIRE(errors.size() == 1);
    CHECK(errors.front() == "model does not support function calling");
}

TEST_CASE("HTTP request validation uses provider metadata before registry cards",
          "[llm][model-catalog][metadata][fallback]") {
    constexpr std::string_view provider_name =
        "api-first-validation-provider";
    constexpr std::string_view model_id =
        "api-first-validation-model";

    ModelInfo registry_card;
    registry_card.canonical_id = std::string(model_id);
    registry_card.provider = "test";
    registry_card.max_output_tokens = 8'192;
    registry_card.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput) |
        static_cast<uint32_t>(ModelCapability::FunctionCalling);
    registry_card.capabilities_complete = true;
    ModelRegistry::instance().register_model(registry_card);

    ModelInfo provider_card;
    provider_card.canonical_id = std::string(model_id);
    provider_card.max_output_tokens = 128'000;
    provider_card.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput);
    provider_card.capabilities_complete = true;

    ModelCatalogDiscoveryResult success;
    success.attempted = true;
    success.fetched = 1;
    ModelCatalogAvailability::instance().record_result(
        provider_name, success, {provider_card});

    HttpLLMProvider provider(
        "https://example.invalid/v1",
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        std::string(model_id),
        std::make_unique<core::llm::protocols::OpenAIProtocol>(),
        core::config::ApiType::OpenAI,
        std::string(provider_name));

    ChatRequest request;
    request.model = std::string(model_id);
    request.max_tokens = 64'000;
    request.messages.push_back({.role = "user", .content = "Hello"});
    Tool tool;
    tool.function.name = "test_tool";
    tool.function.input_schema = R"JSON({"type":"object"})JSON";
    request.tools.push_back(std::move(tool));

    const auto errors = provider.validate_request(request);
    REQUIRE(errors.size() == 1);
    CHECK(errors.front() == "model does not support function calling");
    REQUIRE(provider.get_model_info().has_value());
    CHECK(provider.get_model_info()->max_output_tokens == 128'000);
}

TEST_CASE("HTTP request validation falls back after permanent catalog skip",
          "[llm][model-catalog][metadata][fallback]") {
    constexpr std::string_view provider_name =
        "registry-fallback-validation-provider";
    constexpr std::string_view model_id =
        "registry-fallback-validation-model";

    ModelInfo registry_card;
    registry_card.canonical_id = std::string(model_id);
    registry_card.provider = "test";
    registry_card.max_output_tokens = 8'192;
    ModelRegistry::instance().register_model(registry_card);

    ModelCatalogDiscoveryResult unsupported;
    unsupported.permanent_skip = true;
    ModelCatalogAvailability::instance().record_result(
        provider_name, unsupported, {});

    HttpLLMProvider provider(
        "https://example.invalid/v1",
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        std::string(model_id),
        std::make_unique<core::llm::protocols::OpenAIProtocol>(),
        core::config::ApiType::OpenAI,
        std::string(provider_name));

    ChatRequest request;
    request.model = std::string(model_id);
    request.max_tokens = 64'000;
    request.messages.push_back({.role = "user", .content = "Hello"});

    const auto errors = provider.validate_request(request);
    REQUIRE(errors.size() == 1);
    CHECK(errors.front().find("exceeds model limit of 8192")
          != std::string::npos);
}

TEST_CASE("ModelRegistry replaces stale capabilities from a complete provider catalog",
          "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();

    ModelInfo stale;
    stale.canonical_id = "complete-capability-refresh-model";
    stale.provider = "test";
    stale.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput) |
        static_cast<uint32_t>(ModelCapability::Vision);
    registry.register_model(stale);

    ModelInfo discovered;
    discovered.canonical_id = stale.canonical_id;
    discovered.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput);
    discovered.capabilities_complete = true;
    CHECK_FALSE(registry.merge_model(std::move(discovered)));

    const auto merged = registry.get_info(stale.canonical_id);
    REQUIRE(merged.has_value());
    CHECK(merged->capabilities_complete);
    CHECK(merged->supports(ModelCapability::TextInput));
    CHECK(merged->supports(ModelCapability::TextOutput));
    CHECK_FALSE(merged->supports(ModelCapability::Vision));
}

TEST_CASE("Complete reasoning metadata replaces stale wire-mode inference",
          "[llm][model-catalog][metadata-merge]") {
    ModelInfo baseline;
    baseline.canonical_id = "metadata-merge-reasoning";
    baseline.reasoning.effort =
        ReasoningCapability::Effort
        | ReasoningCapability::MaxEffort;
    baseline.reasoning.manual_thinking = true;

    ModelInfo discovered;
    discovered.canonical_id = baseline.canonical_id;
    discovered.reasoning.effort =
        ReasoningCapability::Effort
        | ReasoningCapability::XHighEffort;
    discovered.reasoning.adaptive_thinking = true;
    discovered.reasoning.complete = true;

    const ModelInfo merged =
        merge_model_metadata(std::move(baseline), std::move(discovered));
    CHECK(merged.reasoning.complete);
    CHECK(merged.reasoning.effort.supports_effort());
    CHECK(merged.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));
    CHECK_FALSE(merged.reasoning.effort.supports(
        ReasoningCapability::MaxEffort));
    CHECK(merged.reasoning.adaptive_thinking);
    CHECK_FALSE(merged.reasoning.manual_thinking);
}

TEST_CASE("A live catalog refresh keeps curated wire constraints",
          "[llm][model-catalog][metadata-merge]") {
    ModelInfo baseline;
    baseline.canonical_id = "metadata-merge-wire";
    baseline.reasoning.complete = true;
    baseline.reasoning.adaptive_thinking = true;
    baseline.wire.thinking_always_on = true;
    baseline.wire.reasoning_text_hidden = true;
    baseline.wire.reasoning_bound_to_prefix = true;
    baseline.wire.fixed_sampling = true;
    baseline.wire.forced_tool_choice_rejected = true;

    // The Models API reports effort levels and thinking modes, but nothing
    // about the request fields a model rejects, so its card has no wire data.
    ModelInfo discovered;
    discovered.canonical_id = baseline.canonical_id;
    discovered.reasoning.effort =
        ReasoningCapability::Effort
        | ReasoningCapability::XHighEffort;
    discovered.reasoning.adaptive_thinking = true;
    discovered.reasoning.complete = true;

    const ModelInfo merged =
        merge_model_metadata(std::move(baseline), std::move(discovered));
    CHECK(merged.reasoning.complete);
    CHECK(merged.reasoning.effort.supports(
        ReasoningCapability::XHighEffort));
    CHECK(merged.wire.thinking_always_on);
    CHECK(merged.wire.reasoning_text_hidden);
    CHECK(merged.wire.reasoning_bound_to_prefix);
    CHECK(merged.wire.fixed_sampling);
    CHECK(merged.wire.forced_tool_choice_rejected);
}

TEST_CASE("A live catalog supplies effort support the protocol does not know",
          "[llm][model-catalog][anthropic][effort]") {
    constexpr std::string_view provider_name = "anthropic-live-effort-provider";
    constexpr std::string_view model_id = "claude-opus-6";

    ModelInfo live_card;
    live_card.canonical_id = std::string(model_id);
    live_card.provider = "anthropic";
    live_card.reasoning.effort =
        ReasoningCapability::Effort | ReasoningCapability::XHighEffort;
    live_card.reasoning.adaptive_thinking = true;
    live_card.reasoning.complete = true;

    ModelCatalogDiscoveryResult success;
    success.attempted = true;
    success.fetched = 1;
    ModelCatalogAvailability::instance().record_result(
        provider_name, success, {live_card});

    HttpLLMProvider provider(
        "https://example.invalid/v1",
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        std::string(model_id),
        std::make_unique<protocols::ClaudeProtocol>(),
        core::config::ApiType::Anthropic,
        std::string(provider_name));

    // The protocol has neither a card nor a generation row for this id, so the
    // catalog is what makes /effort work on a model released after this build.
    CHECK(protocols::ClaudeProtocol{}.reasoning_capabilities(model_id).empty());
    const auto capabilities = provider.reasoning_capabilities(model_id);
    CHECK(capabilities.supports_effort());
    CHECK(capabilities.supports(ReasoningCapability::XHighEffort));

    // A model the protocol already describes keeps its answer: Haiku 4.5 has a
    // curated card with no effort control, so the catalog cannot add one.
    CHECK_FALSE(
        provider.reasoning_capabilities("claude-haiku-4-5").supports_effort());
}

TEST_CASE("A live Anthropic catalog keeps Opus 5.5 on its curated wire",
          "[llm][model-catalog][anthropic][opus55]") {
    AnthropicModelCatalogProvider provider;
    const auto result = provider.parse_models_response(R"JSON({
      "data": [{
        "id": "claude-opus-5-5",
        "display_name": "Claude Opus 5.5",
        "max_input_tokens": 1000000,
        "max_tokens": 128000,
        "capabilities": {
          "effort": {
            "supported": true,
            "low": {"supported": true},
            "medium": {"supported": true},
            "high": {"supported": true},
            "xhigh": {"supported": true},
            "max": {"supported": true}
          },
          "thinking": {
            "supported": true,
            "types": {
              "adaptive": {"supported": true},
              "enabled": {"supported": false}
            }
          }
        }
      }],
      "has_more": false
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);

    // Discovery is provider-scoped: it never writes to the global registry, so
    // the request shape still resolves from the curated card.
    const auto policy = protocols::claude_wire_policy("claude-opus-5-5");
    CHECK(policy.known);
    CHECK(policy.adaptive_thinking);
    CHECK_FALSE(policy.manual_thinking);
    CHECK(policy.thinking_always_on);
    CHECK(policy.reasoning_bound_to_prefix);
    CHECK(policy.forced_tool_choice_rejected);
    CHECK(policy.effort.supports(ReasoningCapability::XHighEffort));

    // The live card is authoritative for effort and thinking modes, and the
    // curated constraints survive the merge because the API cannot report them.
    const auto& live = result.models.front();
    CHECK(live.wire.empty());
    const auto resolved = resolve_model_metadata(
        "claude-opus-5-5",
        result.models,
        ModelRegistry::instance().lookup("claude-opus-5-5"));
    REQUIRE(resolved.model.has_value());
    CHECK(resolved.model->reasoning.complete);
    CHECK(resolved.model->reasoning.adaptive_thinking);
    CHECK_FALSE(resolved.model->reasoning.manual_thinking);
    CHECK(resolved.model->wire.thinking_always_on);
    CHECK(resolved.model->wire.reasoning_text_hidden);
    CHECK(resolved.model->wire.reasoning_bound_to_prefix);
    CHECK(resolved.model->wire.fixed_sampling);
    CHECK(resolved.model->wire.forced_tool_choice_rejected);
}

TEST_CASE("Model catalog resolution is provider API first with registry fallback",
          "[llm][model-catalog][metadata][fallback]") {
    std::vector<ModelInfo> provider_models(1);
    provider_models[0].canonical_id = "provider-live-model";
    provider_models[0].provider = "provider-api";

    std::vector<ModelInfo> registry_models(1);
    registry_models[0].canonical_id = "registry-fallback-model";
    registry_models[0].provider = "internal-registry";

    const auto primary =
        resolve_model_catalog(provider_models, registry_models);
    REQUIRE_FALSE(primary.empty());
    CHECK(primary.origin == ModelMetadataOrigin::ProviderApi);
    CHECK_FALSE(primary.uses_registry_fallback());
    REQUIRE(primary.models.size() == 1);
    CHECK(primary.models.front().canonical_id == "provider-live-model");

    const auto fallback = resolve_model_catalog({}, registry_models);
    REQUIRE_FALSE(fallback.empty());
    CHECK(fallback.origin == ModelMetadataOrigin::InternalRegistry);
    CHECK(fallback.uses_registry_fallback());
    REQUIRE(fallback.models.size() == 1);
    CHECK(fallback.models.front().canonical_id
          == "registry-fallback-model");

    const auto unavailable = resolve_model_catalog({}, {});
    CHECK(unavailable.empty());
    CHECK(unavailable.origin == ModelMetadataOrigin::None);
}

TEST_CASE("Single-model resolution merges API truth over registry fallback",
          "[llm][model-catalog][metadata][fallback]") {
    auto registry = std::make_shared<ModelInfo>();
    registry->canonical_id = "api-first-model";
    registry->context_window = 128'000;
    registry->max_output_tokens = 8'192;
    registry->capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::Vision);
    registry->capabilities_complete = true;

    ModelInfo provider;
    provider.canonical_id = registry->canonical_id;
    provider.context_window = 1'000'000;
    provider.max_output_tokens = 128'000;
    provider.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::Reasoning);
    provider.capabilities_complete = true;
    const std::vector provider_models{provider};

    const auto primary = resolve_model_metadata(
        "api-first-model", provider_models, registry);
    REQUIRE(primary);
    CHECK(primary.origin == ModelMetadataOrigin::ProviderApi);
    REQUIRE(primary.model.has_value());
    CHECK(primary.model->context_window == 1'000'000);
    CHECK(primary.model->max_output_tokens == 128'000);
    CHECK(primary.model->supports(ModelCapability::Reasoning));
    CHECK_FALSE(primary.model->supports(ModelCapability::Vision));

    const auto fallback =
        resolve_model_metadata("api-first-model", {}, registry);
    REQUIRE(fallback);
    CHECK(fallback.origin == ModelMetadataOrigin::InternalRegistry);
    REQUIRE(fallback.model.has_value());
    CHECK(fallback.model->max_output_tokens == 8'192);

    const auto missing =
        resolve_model_metadata("unknown-model", provider_models, nullptr);
    CHECK_FALSE(missing);
    CHECK(missing.origin == ModelMetadataOrigin::None);
}

TEST_CASE("Registry fallback family mapping is centralized",
          "[llm][model-catalog][metadata][fallback]") {
    CHECK(model_registry_provider_key(
              "claude-work", core::config::ApiType::Anthropic)
          == "anthropic");
    CHECK(model_registry_provider_key(
              "qwen-token-plan", core::config::ApiType::DashScope)
          == "qwen");
    CHECK(model_registry_provider_key(
              "mistral-team", core::config::ApiType::OpenAI)
          == "mistral");
    CHECK(model_registry_provider_key(
              "ollama-remote", core::config::ApiType::Ollama)
          == "local");
    CHECK(model_registry_provider_key(
              "custom-gateway", core::config::ApiType::OpenAI)
          == "custom-gateway");
}

TEST_CASE("ModelRegistry merge resolves discovered IDs through existing aliases",
          "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();
    const auto before = registry.size();

    ModelInfo catalog;
    catalog.canonical_id = "alias-shadow-canonical-model";
    catalog.aliases = {"alias-shadow-live-id"};
    catalog.display_name = "Alias Shadow Canonical Model";
    catalog.provider = "catalog-provider";
    catalog.context_window = 128000;
    catalog.max_output_tokens = 8192;
    catalog.capabilities =
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput) |
        static_cast<uint32_t>(ModelCapability::FunctionCalling);
    registry.register_model(catalog);
    REQUIRE(registry.size() == before + 1);

    ModelInfo discovered;
    discovered.canonical_id = "alias-shadow-live-id";
    discovered.display_name = "alias-shadow-live-id";
    discovered.provider = "live-provider";
    CHECK_FALSE(registry.merge_model(std::move(discovered)));

    CHECK(registry.size() == before + 1);
    const auto merged = registry.get_info("alias-shadow-live-id");
    REQUIRE(merged.has_value());
    CHECK(merged->canonical_id == "alias-shadow-canonical-model");
    CHECK(merged->display_name == "Alias Shadow Canonical Model");
    CHECK(merged->provider == "catalog-provider");
    CHECK(merged->context_window == 128000);
    CHECK(merged->max_output_tokens == 8192);
    CHECK(merged->supports(ModelCapability::FunctionCalling));
}

TEST_CASE("ModelRegistry merge does not overwrite existing provider ownership",
          "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();

    ModelInfo catalog;
    catalog.canonical_id = "provider-preserve-model";
    catalog.display_name = "Provider Preserve Model";
    catalog.provider = "catalog-provider";
    catalog.context_window = 64000;
    registry.register_model(catalog);

    ModelInfo discovered;
    discovered.canonical_id = "provider-preserve-model";
    discovered.display_name = "Provider Preserve Model Live";
    discovered.provider = "live-provider";
    discovered.context_window = 128000;
    CHECK_FALSE(registry.merge_model(std::move(discovered)));

    const auto merged = registry.get_info("provider-preserve-model");
    REQUIRE(merged.has_value());
    CHECK(merged->provider == "catalog-provider");
    CHECK(merged->display_name == "Provider Preserve Model Live");
    CHECK(merged->context_window == 128000);
}

TEST_CASE("ModelCatalogAvailability stores provider-scoped live models",
          "[llm][model-catalog][discovery]") {
    ModelCatalogDiscoveryResult result;
    result.attempted = true;
    result.fetched = 1;

    ModelInfo model;
    model.canonical_id = "live-provider-model";
    model.provider = "runtime-provider";

    auto& availability = ModelCatalogAvailability::instance();
    REQUIRE(availability.try_mark_refreshing("runtime-provider"));
    CHECK_FALSE(availability.try_mark_refreshing("runtime-provider"));

    availability.record_result("runtime-provider", result, {model});
    const auto snapshot = availability.snapshot("runtime-provider");

    CHECK_FALSE(snapshot.refresh_in_progress);
    CHECK(snapshot.checked);
    CHECK(snapshot.attempted);
    CHECK(snapshot.fetched == 1);
    REQUIRE(snapshot.models.size() == 1);
    CHECK(snapshot.models[0].canonical_id == "live-provider-model");
    CHECK(snapshot.models[0].provider == "runtime-provider");
}

TEST_CASE("An unknown requested model can bypass a fresh catalog TTL",
          "[llm][model-catalog][discovery]") {
    constexpr std::string_view provider = "forced-refresh-provider";
    auto& availability = ModelCatalogAvailability::instance();

    REQUIRE(availability.try_mark_refreshing(provider));
    ModelCatalogDiscoveryResult success;
    success.attempted = true;
    availability.record_result(provider, success, {});

    CHECK_FALSE(availability.try_mark_refreshing(provider));
    CHECK(availability.try_mark_refreshing(provider, true));
    availability.record_result(provider, success, {});
}

TEST_CASE("Forced discovery still respects transient failure backoff",
          "[llm][model-catalog][discovery]") {
    constexpr std::string_view provider = "forced-refresh-backoff-provider";
    auto& availability = ModelCatalogAvailability::instance();

    REQUIRE(availability.try_mark_refreshing(provider));
    ModelCatalogDiscoveryResult failure;
    failure.attempted = true;
    failure.error = "temporary";
    availability.record_result(provider, failure, {});

    CHECK_FALSE(availability.try_mark_refreshing(provider, true));
}

TEST_CASE("ModelCatalogAvailability keeps stale models and schedules transient retries",
          "[llm][model-catalog][discovery]") {
    auto& availability = ModelCatalogAvailability::instance();
    constexpr std::string_view kProvider = "retry-provider";

    ModelCatalogDiscoveryResult success;
    success.attempted = true;
    success.fetched = 1;

    ModelInfo model;
    model.canonical_id = "retry-provider-model";
    model.provider = std::string(kProvider);
    availability.record_result(kProvider, success, {model});

    ModelCatalogDiscoveryResult failure;
    failure.attempted = true;
    failure.error = "timeout";
    failure.retry_after_seconds = 60;
    availability.record_result(kProvider, failure, {});

    const auto snapshot = availability.snapshot(kProvider);
    CHECK(snapshot.state == ModelCatalogDiscoveryState::TransientFailure);
    CHECK(snapshot.checked);
    CHECK(snapshot.consecutive_failures == 1);
    CHECK_FALSE(snapshot.refresh_due());
    CHECK(snapshot.next_retry_at != std::chrono::steady_clock::time_point::max());
    REQUIRE(snapshot.models.size() == 1);
    CHECK(snapshot.models[0].canonical_id == "retry-provider-model");
}

TEST_CASE("ModelCatalogAvailability treats unsupported catalog probes as permanent skips",
          "[llm][model-catalog][discovery]") {
    auto& availability = ModelCatalogAvailability::instance();
    constexpr std::string_view kProvider = "permanent-skip-provider";

    ModelCatalogDiscoveryResult result;
    result.permanent_skip = true;
    availability.record_result(kProvider, result, {});

    const auto snapshot = availability.snapshot(kProvider);
    CHECK(snapshot.state == ModelCatalogDiscoveryState::PermanentSkip);
    CHECK_FALSE(snapshot.refresh_due());
    CHECK(snapshot.next_retry_at == std::chrono::steady_clock::time_point::max());
    CHECK_FALSE(availability.try_mark_refreshing(kProvider));
}

TEST_CASE("Model discovery follows the protocol catalog capability",
          "[llm][model-catalog][discovery]") {
    core::llm::protocols::GeminiCodeAssistProtocol protocol;
    core::llm::protocols::GeminiAntigravityProtocol antigravity;

    CHECK_FALSE(protocol.supports_model_catalog());
    CHECK(antigravity.supports_model_catalog());

    const auto result = discover_and_register_models(
        "gemini",
        core::config::ApiType::Gemini,
        "https://cloudcode-pa.googleapis.com",
        nullptr,
        protocol);

    CHECK_FALSE(result.attempted);
    CHECK(result.permanent_skip);
    CHECK(result.fetched == 0);
    CHECK(result.ok());

    // Antigravity sessions discover through /v1internal:fetchAvailableModels;
    // without credentials the probe is held back, not permanently skipped.
    const auto antigravity_result = discover_and_register_models(
        "gemini-antigravity",
        core::config::ApiType::Gemini,
        "https://cloudcode-pa.googleapis.com",
        nullptr,
        antigravity);

    CHECK_FALSE(antigravity_result.attempted);
    CHECK_FALSE(antigravity_result.permanent_skip);
    CHECK(antigravity_result.error == "missing credentials for remote model discovery");
}

TEST_CASE("CodeAssistModelCatalogProvider serves the hub fetchAvailableModels dialect",
          "[llm][model-catalog][antigravity]") {
    CodeAssistModelCatalogProvider catalog;
    CHECK(catalog.provider_name() == "gemini-antigravity");
    CHECK(catalog.model_list_path() == "/v1internal:fetchAvailableModels");
    CHECK(catalog.model_list_path("ignored") == "/v1internal:fetchAvailableModels");

    const ModelCatalogRequest request = catalog.catalog_request();
    CHECK(request.method == "POST");
    CHECK(request.body == "{}");

    const auto result = catalog.parse_models_response(R"JSON({
      "models": {
        "gemini-3.1-pro-low": {
          "displayName": "Gemini 3.1 Pro",
          "supportsImages": true,
          "supportsThinking": true,
          "maxTokens": 200000,
          "maxOutputTokens": 65535
        },
        "chat_20706": {"displayName": "Retired chat build"},
        "internal-preview": {"displayName": "Internal", "isInternal": true}
      },
      "agentModelSorts": [{"groups": [{"modelIds": ["gemini-3.1-pro-low"]}]}],
      "imageGenerationModelIds": ["gemini-3-pro-image"]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);
    const auto& model = result.models.front();
    CHECK(model.canonical_id == "gemini-3.1-pro-low");
    CHECK(model.display_name == "Gemini 3.1 Pro");
    CHECK(model.provider == "gemini-antigravity");
    CHECK(model.context_window == 200000);
    CHECK(model.max_output_tokens == 65535);
    CHECK(model.supports(ModelCapability::Vision));
    CHECK(model.supports(ModelCapability::Reasoning));
    CHECK(model.supports(ModelCapability::FunctionCalling));

    // A payload without the models map is a parse failure like the other
    // catalogs report.
    const auto missing = catalog.parse_models_response(R"({"agentModelSorts":[]})");
    CHECK_FALSE(missing.ok());
}

TEST_CASE("Discovery reads the ChatGPT catalog only through the Codex protocol",
          "[llm][model-catalog][discovery][openai]") {
    httplib::Server server;
    std::atomic<bool> versioned{false};
    server.Get("/models", [&](const httplib::Request& req, httplib::Response& res) {
        versioned = req.has_param("client_version");
        res.set_content(R"JSON({"models":[
            {"slug":"codex-discovery-test-model","display_name":"Visible","visibility":"list"},
            {"slug":"codex-discovery-test-hidden","visibility":"hide"}
        ]})JSON", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port < 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::thread worker([&] { server.listen_after_bind(); });
    for (int i = 0; i < 200 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
    const std::string base_url = "http://127.0.0.1:" + std::to_string(port);

    core::llm::protocols::CodexResponsesProtocol codex;
    const auto discovered = discover_and_register_models(
        "codex-discovery-test", core::config::ApiType::OpenAI, base_url, nullptr, codex);
    CHECK(discovered.ok());
    CHECK(discovered.fetched == 1);
    CHECK(versioned.load());
    const auto snapshot =
        ModelCatalogAvailability::instance().snapshot("codex-discovery-test");
    REQUIRE(snapshot.models.size() == 1);
    CHECK(snapshot.models.front().canonical_id == "codex-discovery-test-model");

    // The same payload is not a public OpenAI list, so a protocol without its
    // own catalog must keep using the API-type default and reject it.
    core::llm::protocols::OpenAIResponsesProtocol public_openai;
    const auto mismatched = discover_and_register_models(
        "public-discovery-test", core::config::ApiType::OpenAI, base_url, nullptr, public_openai);
    CHECK_FALSE(mismatched.ok());

    server.stop();
    worker.join();
}

TEST_CASE("Model discovery distinguishes authenticated catalogs from unsupported ones",
          "[llm][model-catalog][discovery]") {
    core::llm::protocols::OpenAIProtocol protocol;

    const auto zai = discover_and_register_models(
        "zai",
        core::config::ApiType::OpenAI,
        "https://api.z.ai/api/paas/v4",
        nullptr,
        protocol);
    CHECK_FALSE(zai.permanent_skip);
    CHECK_FALSE(zai.attempted);
    CHECK(zai.error == "missing credentials for remote model discovery");

    // Azure's data-plane model list contains base models, while requests
    // require user-chosen deployment names. Listing it would populate the
    // picker with identifiers that are not necessarily callable.
    const auto azure = discover_and_register_models(
        "azure-work",
        core::config::ApiType::OpenAI,
        "https://example.openai.azure.com",
        nullptr,
        protocol);
    CHECK(azure.permanent_skip);
    CHECK_FALSE(azure.attempted);
}

TEST_CASE("ModelRegistry JSON export escapes strings", "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();

    ModelInfo info;
    info.canonical_id = "json-escape-model";
    info.display_name = "Quote \" and newline\n model";
    info.provider = "custom\\provider";
    info.capabilities =
        static_cast<uint32_t>(ModelCapability::Reasoning);
    info.capabilities_complete = true;
    info.reasoning.effort =
        ReasoningCapability::Effort
        | ReasoningCapability::XHighEffort;
    info.reasoning.adaptive_thinking = true;
    info.reasoning.complete = true;
    registry.register_model(std::move(info));

    const auto exported = registry.export_to_json();
    CHECK(exported.find("Quote \\\" and newline\\n model") != std::string::npos);
    CHECK(exported.find("custom\\\\provider") != std::string::npos);
    CHECK(registry.load_from_json(exported) > 0);
    const auto round_tripped = registry.lookup("json-escape-model");
    REQUIRE(round_tripped);
    CHECK(round_tripped->capabilities_complete);
    CHECK(round_tripped->reasoning.complete);
    CHECK(round_tripped->reasoning.adaptive_thinking);
    CHECK(round_tripped->reasoning.effort.supports(
        ReasoningCapability::XHighEffort));
}

TEST_CASE("ModelRegistry load_from_json rejects malformed catalog JSON", "[llm][model-catalog][registry]") {
    CHECK(ModelRegistry::instance().load_from_json("{not json") == -1);
    CHECK(ModelRegistry::instance().load_from_json(R"JSON({"items":[]})JSON") == -1);
}

TEST_CASE("Codex catalog retains authoritative reasoning defaults and exact levels",
          "[llm][model-catalog][openai][regression]") {
    CodexModelCatalogProvider catalog;
    const auto result = catalog.parse_models_response(R"({"models":[
        {"slug":"gpt-future","default_reasoning_level":"high","supported_reasoning_levels":[{"effort":"high"},{"effort":"ultra"}]},
        {"slug":"no-controls","supported_reasoning_levels":[]}
    ]})");
    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);
    const auto& profile = result.models[0].reasoning;
    CHECK(profile.complete);
    CHECK(profile.catalog_default_effort == "high");
    CHECK(profile.supported_efforts == std::vector<std::string>{"high", "ultra"});
    CHECK(profile.effort.supports(ReasoningCapability::UltraEffort));
    CHECK(result.models[1].reasoning.complete);
    CHECK_FALSE(result.models[1].reasoning.effort.supports_effort());
}
