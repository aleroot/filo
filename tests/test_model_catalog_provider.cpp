#include <catch2/catch_test_macros.hpp>

#include "core/llm/ModelCatalogDiscovery.hpp"
#include "core/llm/ModelCatalogProvider.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/GeminiCodeAssistProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"

using namespace core::llm;

TEST_CASE("GeminiModelCatalogProvider parses live model catalog shape", "[llm][model-catalog]") {
    GeminiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "models": [
        {
          "name": "models/gemini-3.2-flash",
          "baseModelId": "gemini-3.2-flash",
          "displayName": "Gemini 3.2 Flash",
          "inputTokenLimit": 1048576,
          "outputTokenLimit": 65536,
          "supportedGenerationMethods": ["generateContent", "countTokens"],
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
    CHECK(flash.provider == "gemini");
    CHECK(flash.context_window == 1048576);
    CHECK(flash.max_output_tokens == 65536);
    CHECK(flash.tier == ModelTier::Reasoning);
    CHECK(flash.supports(ModelCapability::FunctionCalling));
    CHECK(flash.supports(ModelCapability::JsonMode));
    CHECK(flash.supports(ModelCapability::TokenCounting));
    CHECK(flash.supports(ModelCapability::PromptCaching));

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

TEST_CASE("OpenAICompatibleModelCatalogProvider parses Codex remote model catalog", "[llm][model-catalog][openai]") {
    OpenAICompatibleModelCatalogProvider provider("openai");

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
          "slug": "internal-chatgpt-only",
          "display_name": "Internal ChatGPT Only",
          "supported_in_api": false
        }
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 1);

    const auto& codex = result.models[0];
    CHECK(codex.canonical_id == "gpt-5.5-codex");
    CHECK(codex.display_name == "GPT-5.5 Codex");
    CHECK(codex.provider == "openai");
    CHECK(codex.context_window == 272000);
    CHECK(codex.max_output_tokens == 100000);
    CHECK(codex.max_reasoning_tokens == 25000);
    CHECK(codex.tier == ModelTier::Reasoning);
    CHECK(codex.supports(ModelCapability::FunctionCalling));
    CHECK(codex.supports(ModelCapability::ParallelToolCalls));
    CHECK(codex.supports(ModelCapability::JsonMode));
    CHECK(codex.supports(ModelCapability::Reasoning));
    CHECK(codex.supports(ModelCapability::Vision));
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

    const auto& legacy = result.models[1];
    CHECK(legacy.canonical_id == "moonshot-v1-32k");
    CHECK(legacy.context_window == 32768);
    CHECK_FALSE(legacy.supports(ModelCapability::Reasoning));
    CHECK_FALSE(legacy.supports(ModelCapability::Vision));
}

TEST_CASE("KimiModelCatalogProvider infers context when Moonshot omits it", "[llm][model-catalog]") {
    KimiModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {"id": "moonshot-v1-8k"},
        {"id": "kimi-for-coding"}
      ]
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);
    CHECK(result.models[0].context_window == 8192);
    CHECK(result.models[1].context_window == 256000);
    CHECK(result.models[1].supports(ModelCapability::Reasoning));
    CHECK(result.models[1].supports(ModelCapability::Vision));
}

TEST_CASE("AnthropicModelCatalogProvider parses v1 models response", "[llm][model-catalog]") {
    AnthropicModelCatalogProvider provider;

    const auto result = provider.parse_models_response(R"JSON({
      "data": [
        {
          "id": "claude-sonnet-4-6-20260514",
          "display_name": "Claude Sonnet 4.6"
        },
        {
          "id": "claude-haiku-4-5-20251101",
          "display_name": "Claude Haiku 4.5"
        }
      ],
      "has_more": false
    })JSON");

    REQUIRE(result.ok());
    REQUIRE(result.models.size() == 2);

    const auto& sonnet = result.models[0];
    CHECK(sonnet.canonical_id == "claude-sonnet-4-6-20260514");
    CHECK(sonnet.display_name == "Claude Sonnet 4.6");
    CHECK(sonnet.provider == "anthropic");
    CHECK(sonnet.context_window == 0);
    CHECK(sonnet.max_output_tokens == 0);
    CHECK(sonnet.capabilities == 0);

    const auto& haiku = result.models[1];
    CHECK(haiku.tier == ModelTier::Balanced);
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

    auto grok = make_model_catalog_provider(core::config::ApiType::OpenAI, "grok");
    REQUIRE(grok != nullptr);
    CHECK(grok->provider_name() == "grok");
    CHECK(grok->model_list_path() == "/models");

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
    CHECK(ollama == nullptr);
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

    core::llm::protocols::AnthropicProtocol anthropic;
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
    result.inserted = 1;

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

TEST_CASE("ModelCatalogAvailability keeps stale models and schedules transient retries",
          "[llm][model-catalog][discovery]") {
    auto& availability = ModelCatalogAvailability::instance();
    constexpr std::string_view kProvider = "retry-provider";

    ModelCatalogDiscoveryResult success;
    success.attempted = true;
    success.fetched = 1;
    success.inserted = 1;

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

TEST_CASE("Model discovery skips Gemini Code Assist catalog probes",
          "[llm][model-catalog][discovery]") {
    core::llm::protocols::GeminiCodeAssistProtocol protocol;

    const auto result = discover_and_register_models(
        "gemini",
        core::config::ApiType::Gemini,
        "https://cloudcode-pa.googleapis.com",
        nullptr,
        protocol);

    CHECK_FALSE(result.attempted);
    CHECK(result.fetched == 0);
    CHECK(result.ok());
}

TEST_CASE("ModelRegistry JSON export escapes strings", "[llm][model-catalog][registry]") {
    auto& registry = ModelRegistry::instance();

    ModelInfo info;
    info.canonical_id = "json-escape-model";
    info.display_name = "Quote \" and newline\n model";
    info.provider = "custom\\provider";
    registry.register_model(std::move(info));

    const auto exported = registry.export_to_json();
    CHECK(exported.find("Quote \\\" and newline\\n model") != std::string::npos);
    CHECK(exported.find("custom\\\\provider") != std::string::npos);
    CHECK(registry.load_from_json(exported) > 0);
}

TEST_CASE("ModelRegistry load_from_json rejects malformed catalog JSON", "[llm][model-catalog][registry]") {
    CHECK(ModelRegistry::instance().load_from_json("{not json") == -1);
    CHECK(ModelRegistry::instance().load_from_json(R"JSON({"items":[]})JSON") == -1);
}
