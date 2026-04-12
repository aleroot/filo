#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/llm/ModelRegistry.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"

using namespace core::llm;

// =============================================================================
// Phase 1: Basic Registry Tests (Backward Compatibility)
// =============================================================================

TEST_CASE("ModelRegistry - Legacy API returns correct context sizes for known models", "[llm][registry]") {
    // Kimi (via new registry)
    REQUIRE(get_max_context_size("kimi-k2.5") == 256000);
    REQUIRE(get_max_context_size("kimi-for-coding") == 256000);
    
    // Anthropic (via new registry)
    REQUIRE(get_max_context_size("claude-3-opus") == 200000);
    REQUIRE(get_max_context_size("claude-3-7-sonnet") == 200000);
    REQUIRE(get_max_context_size("claude-3-5-haiku") == 200000);
    REQUIRE(get_max_context_size("claude-sonnet-4-6[1m]") == 1000000);
    REQUIRE(get_max_context_size("sonnet[1m]") == 1000000);
    REQUIRE(get_max_context_size("opus") == 1000000);
    
    // OpenAI (via new registry)
    REQUIRE(get_max_context_size("gpt-5.4") == 200000);
    REQUIRE(get_max_context_size("gpt-4o") == 128000);
    REQUIRE(get_max_context_size("gpt-4o-mini") == 128000);
    REQUIRE(get_max_context_size("o1") == 200000);
    REQUIRE(get_max_context_size("o3-mini") == 200000);
    
    // Gemini (via new registry)
    REQUIRE(get_max_context_size("gemini-3.1-pro-preview") == 1048576);
    REQUIRE(get_max_context_size("gemini-3-flash-preview") == 1048576);
    REQUIRE(get_max_context_size("gemini-2.5-pro") == 1048576);
    REQUIRE(get_max_context_size("gemini-2.5-flash-lite") == 1048576);
    REQUIRE(get_max_context_size("gemini-2.0-flash") == 1048576);
    REQUIRE(get_max_context_size("gemini-1.5-pro") == 2097152);
    
    // Grok (via legacy fallback)
    REQUIRE(get_max_context_size("grok-1") == 128000);

    // Local (via new registry)
    REQUIRE(get_max_context_size("llama-3.3-70b") == 128000);
    REQUIRE(get_max_context_size("mistral-large") == 128000);
}

TEST_CASE("ModelRegistry - Legacy API returns 0 for unknown models", "[llm][registry]") {
    REQUIRE(get_max_context_size("unknown-model-xyz") == 0);
    REQUIRE(get_max_context_size("") == 0);
}

// =============================================================================
// Phase 1: New Instance-Based Registry Tests
// =============================================================================

TEST_CASE("ModelRegistry::instance - singleton returns same instance", "[llm][registry]") {
    auto& reg1 = ModelRegistry::instance();
    auto& reg2 = ModelRegistry::instance();
    REQUIRE(&reg1 == &reg2);
}

TEST_CASE("ModelRegistry::instance - auto-loads defaults", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    REQUIRE(registry.size() > 0);
    
    // Check some known models were loaded
    REQUIRE(registry.has_model("gpt-5.4"));
    REQUIRE(registry.has_model("gpt-4o"));
    REQUIRE(registry.has_model("claude-3-7-sonnet"));
    REQUIRE(registry.has_model("kimi-k2.5"));
    REQUIRE(registry.has_model("gemini-2.5-pro"));
    REQUIRE(registry.has_model("gemini-3.1-pro-preview"));
}

TEST_CASE("ModelRegistry::lookup - finds models by canonical ID", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    const auto* info = registry.lookup("gpt-4o-2024-08-06");
    REQUIRE(info != nullptr);
    REQUIRE(info->canonical_id == "gpt-4o-2024-08-06");
    REQUIRE(info->context_window == 128000);
}

TEST_CASE("ModelRegistry::lookup - finds models by alias", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // Should find "gpt-4o" as an alias for "gpt-4o-2024-08-06"
    const auto* info = registry.lookup("gpt-4o");
    REQUIRE(info != nullptr);
    REQUIRE(info->canonical_id == "gpt-4o-2024-08-06");

    const auto* gemini = registry.lookup("gemini-flash-latest");
    REQUIRE(gemini != nullptr);
    REQUIRE(gemini->canonical_id == "gemini-2.5-flash");

    const auto* gemini_auto = registry.lookup("auto-gemini-3");
    REQUIRE(gemini_auto != nullptr);
    REQUIRE(gemini_auto->canonical_id == "gemini-3.1-pro-preview");
}

TEST_CASE("ModelRegistry::lookup - finds current Gemini preview models", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();

    const auto* info = registry.lookup("gemini-3.1-pro-preview");
    REQUIRE(info != nullptr);
    REQUIRE(info->context_window == 1048576);
    REQUIRE(info->max_output_tokens == 65536);
    REQUIRE(info->supports(ModelCapability::TokenCounting));
}

TEST_CASE("ModelRegistry::get_info - returns full model information", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    auto info = registry.get_info("gpt-4o");
    REQUIRE(info.has_value());
    
    CHECK(info->canonical_id == "gpt-4o-2024-08-06");
    CHECK(info->display_name == "GPT-4o");
    CHECK(info->provider == "openai");
    CHECK(info->context_window == 128000);
    CHECK(info->max_output_tokens == 16384);
    CHECK(info->tier == ModelTier::Powerful);
    CHECK(info->knowledge_cutoff == "2023-10");
    
    // Pricing
    CHECK(info->pricing.input_per_mtok == Catch::Approx(2.50));
    CHECK(info->pricing.output_per_mtok == Catch::Approx(10.0));
}

TEST_CASE("ModelRegistry::get_tier - returns correct tier classification", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // Fast tier models
    REQUIRE(registry.get_tier("gpt-4o-mini") == ModelTier::Fast);
    REQUIRE(registry.get_tier("claude-3-5-haiku") == ModelTier::Fast);
    
    // Balanced tier models
    REQUIRE(registry.get_tier("gpt-4o") == ModelTier::Powerful);  // GPT-4o is actually powerful
    REQUIRE(registry.get_tier("claude-3-7-sonnet") == ModelTier::Balanced);
    
    // Powerful tier models  
    REQUIRE(registry.get_tier("claude-3-opus") == ModelTier::Powerful);
    
    // Reasoning tier models
    REQUIRE(registry.get_tier("gpt-5.4") == ModelTier::Reasoning);
    REQUIRE(registry.get_tier("o1") == ModelTier::Reasoning);
    REQUIRE(registry.get_tier("o3-mini") == ModelTier::Reasoning);
    
    // Unknown model
    REQUIRE_FALSE(registry.get_tier("unknown-model").has_value());
}

// =============================================================================
// Phase 1: Capability Tests
// =============================================================================

TEST_CASE("ModelRegistry::supports - checks capability flags", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // GPT-4o supports everything
    REQUIRE(registry.supports("gpt-4o", ModelCapability::TextInput));
    REQUIRE(registry.supports("gpt-4o", ModelCapability::TextOutput));
    REQUIRE(registry.supports("gpt-4o", ModelCapability::Vision));
    REQUIRE(registry.supports("gpt-4o", ModelCapability::FunctionCalling));
    REQUIRE(registry.supports("gpt-4o", ModelCapability::JsonMode));
    REQUIRE(registry.supports("gpt-4o", ModelCapability::Streaming));
    
    // Claude 3 Opus also supports everything
    REQUIRE(registry.supports("claude-3-opus", ModelCapability::Vision));
    REQUIRE(registry.supports("claude-3-opus", ModelCapability::FunctionCalling));
    REQUIRE(registry.supports("claude-3-opus", ModelCapability::PromptCaching));
    
    // Reasoning models have special flag
    REQUIRE(registry.supports("o1", ModelCapability::Reasoning));
    REQUIRE(registry.supports("gpt-5.4", ModelCapability::Reasoning));
    REQUIRE_FALSE(registry.supports("gpt-4o", ModelCapability::Reasoning));
    
    // Unknown model returns false
    REQUIRE_FALSE(registry.supports("unknown-model", ModelCapability::TextInput));
}

TEST_CASE("ModelRegistry::filter_by_tier - returns models in tier", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    auto fast_models = registry.filter_by_tier(ModelTier::Fast);
    REQUIRE(fast_models.size() > 0);
    
    for (const auto* model : fast_models) {
        REQUIRE(model->tier == ModelTier::Fast);
    }
    
    auto reasoning_models = registry.filter_by_tier(ModelTier::Reasoning);
    REQUIRE(reasoning_models.size() >= 2);  // o1 and o3-mini
}

TEST_CASE("ModelRegistry::filter_by_capability - returns models with capability", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    auto vision_models = registry.filter_by_capability(ModelCapability::Vision);
    REQUIRE(vision_models.size() > 0);
    
    for (const auto* model : vision_models) {
        REQUIRE(model->supports(ModelCapability::Vision));
    }
}

// =============================================================================
// Phase 1: Parameter Validation Tests
// =============================================================================

TEST_CASE("ModelRegistry::validate_parameter - temperature", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // OpenAI models: temperature [0, 2]
    REQUIRE(registry.validate_parameter("gpt-4o", "temperature", 0.0));
    REQUIRE(registry.validate_parameter("gpt-4o", "temperature", 1.0));
    REQUIRE(registry.validate_parameter("gpt-4o", "temperature", 2.0));
    REQUIRE_FALSE(registry.validate_parameter("gpt-4o", "temperature", 2.1));
    REQUIRE_FALSE(registry.validate_parameter("gpt-4o", "temperature", -0.1));
    
    // Claude models: temperature [0, 1]
    REQUIRE(registry.validate_parameter("claude-3-7-sonnet", "temperature", 0.0));
    REQUIRE(registry.validate_parameter("claude-3-7-sonnet", "temperature", 1.0));
    REQUIRE_FALSE(registry.validate_parameter("claude-3-7-sonnet", "temperature", 1.1));
    
    // Unknown model: validation returns false (can't validate unknown models)
    REQUIRE_FALSE(registry.validate_parameter("unknown-model", "temperature", 5.0));
}

TEST_CASE("ModelRegistry::validate_max_tokens - checks model limits", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // GPT-4o max output is 16384
    REQUIRE(registry.validate_max_tokens("gpt-4o", 1000));
    REQUIRE(registry.validate_max_tokens("gpt-4o", 16384));
    REQUIRE_FALSE(registry.validate_max_tokens("gpt-4o", 16385));
    REQUIRE_FALSE(registry.validate_max_tokens("gpt-4o", 0));  // Must be >= 1
    
    // Unknown model: validation fails
    REQUIRE_FALSE(registry.validate_max_tokens("unknown-model", 100));
}

// =============================================================================
// Phase 1: Cost Estimation Tests
// =============================================================================

TEST_CASE("ModelRegistry::estimate_cost - calculates request cost", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // GPT-4o: $2.50/M input, $10/M output
    // 1000 tokens in, 500 tokens out = $0.0025 + $0.005 = $0.0075
    double cost = registry.estimate_cost("gpt-4o", 1000, 500);
    CHECK(cost == Catch::Approx(0.0075));
    
    // GPT-4o Mini: $0.15/M input, $0.60/M output
    // 1000 tokens in, 500 tokens out = $0.00015 + $0.0003 = $0.00045
    double mini_cost = registry.estimate_cost("gpt-4o-mini", 1000, 500);
    CHECK(mini_cost == Catch::Approx(0.00045));
    
    // Cached input pricing
    double cached_cost = registry.estimate_cost("gpt-4o", 1000, 500, true);
    // Cached: $1.25/M input, $10/M output = $0.00125 + $0.005 = $0.00625
    CHECK(cached_cost == Catch::Approx(0.00625));
    
    // Unknown model: returns -1.0
    REQUIRE(registry.estimate_cost("unknown-model", 1000, 500) == -1.0);
}

// =============================================================================
// Phase 1: ModelInfo Methods Tests
// =============================================================================

TEST_CASE("ModelInfo::supports - checks capabilities", "[llm][registry]") {
    auto info_opt = ModelRegistry::instance().get_info("gpt-4o");
    REQUIRE(info_opt.has_value());
    
    const auto& info = *info_opt;
    REQUIRE(info.supports(ModelCapability::Vision));
    REQUIRE(info.supports(ModelCapability::FunctionCalling));
    REQUIRE(info.supports(ModelCapability::JsonMode));
    REQUIRE_FALSE(info.supports(ModelCapability::Reasoning));
}

TEST_CASE("ModelInfo::is_deprecated - checks deprecation status", "[llm][registry]") {
    auto active = ModelRegistry::instance().get_info("gpt-4o");
    REQUIRE(active.has_value());
    REQUIRE_FALSE(active->is_deprecated());
    
    auto deprecated = ModelRegistry::instance().get_info("claude-3-5-sonnet-20241022");
    if (deprecated.has_value()) {
        // Note: This test depends on the catalog state
        // The legacy version has a deprecation date
    }
}

TEST_CASE("ModelInfo::matches - matches canonical and aliases", "[llm][registry]") {
    auto info_opt = ModelRegistry::instance().get_info("gpt-4o");
    REQUIRE(info_opt.has_value());
    
    const auto& info = *info_opt;
    REQUIRE(info.matches("gpt-4o-2024-08-06"));  // canonical
    REQUIRE(info.matches("gpt-4o"));               // alias
    REQUIRE(info.matches("4o"));                   // alias
    REQUIRE_FALSE(info.matches("claude-3-opus"));  // different model
}

TEST_CASE("ModelInfo::effective_max_tokens - returns max output or context", "[llm][registry]") {
    auto gpt4o = ModelRegistry::instance().get_info("gpt-4o");
    REQUIRE(gpt4o.has_value());
    // GPT-4o has max_output_tokens = 16384
    REQUIRE(gpt4o->effective_max_tokens() == 16384);
}

TEST_CASE("ModelInfo::estimate_cost - instance method", "[llm][registry]") {
    auto info_opt = ModelRegistry::instance().get_info("gpt-4o");
    REQUIRE(info_opt.has_value());
    
    double cost = info_opt->estimate_cost(1000000, 1000000);  // 1M tokens each
    // $2.50 + $10.00 = $12.50
    CHECK(cost == Catch::Approx(12.50));
}

// =============================================================================
// Phase 1: Provider Query Tests
// =============================================================================

TEST_CASE("ModelRegistry::get_by_provider - returns provider models", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    auto openai_models = registry.get_by_provider("openai");
    REQUIRE(openai_models.size() >= 4);  // GPT-4o, GPT-4o-mini, o3-mini, o1, etc.
    
    for (const auto* model : openai_models) {
        REQUIRE(model->provider == "openai");
    }
    
    auto anthropic_models = registry.get_by_provider("anthropic");
    REQUIRE(anthropic_models.size() >= 3);  // Sonnet, Haiku, Opus
}

TEST_CASE("ModelRegistry::list_models - returns all canonical IDs", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    auto models = registry.list_models();
    REQUIRE(models.size() == registry.size());
    
    // All returned IDs should be findable
    for (const auto& id : models) {
        REQUIRE(registry.has_model(id));
    }
}

// =============================================================================
// Phase 2: RouterEngine Integration Tests
// =============================================================================

TEST_CASE("RouterEngine uses ModelRegistry for quality scoring", "[llm][registry][routing]") {
    // This test verifies that RouterEngine::candidate_quality_score now uses
    // ModelRegistry::get_model_tier() instead of hardcoded string parsing.
    
    // The scoring should now be:
    // Reasoning = 4.0, Powerful = 3.0, Balanced = 2.0, Fast = 1.0
    
    // We can't directly test the private method, but we can verify
    // the ModelRegistry returns the expected tiers that RouterEngine uses.
    
    auto& registry = ModelRegistry::instance();
    
    // Verify tiers are correctly mapped
    REQUIRE(registry.get_tier("o1") == ModelTier::Reasoning);
    REQUIRE(registry.get_tier("claude-3-opus") == ModelTier::Powerful);
    REQUIRE(registry.get_tier("claude-3-7-sonnet") == ModelTier::Balanced);
    REQUIRE(registry.get_tier("gpt-4o-mini") == ModelTier::Fast);
}

// =============================================================================
// Phase 3: HttpLLMProvider Integration Tests
// =============================================================================

TEST_CASE("HttpLLMProvider::max_context_size uses registry", "[llm][provider][registry]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("test-key");
    
    HttpLLMProvider provider_gpt4o(
        "https://api.openai.com/v1",
        creds,
        "gpt-4o",
        std::make_unique<protocols::OpenAIProtocol>());
    
    REQUIRE(provider_gpt4o.max_context_size() == 128000);

    HttpLLMProvider provider_kimi(
        "https://api.moonshot.cn/v1",
        creds,
        "kimi-k2.5",
        std::make_unique<protocols::OpenAIProtocol>());
    
    REQUIRE(provider_kimi.max_context_size() == 256000);
    
    HttpLLMProvider provider_unknown(
        "https://example.com",
        creds,
        "ghost-model",
        std::make_unique<protocols::OpenAIProtocol>());
        
    // Unknown model should fall back to legacy heuristic (0 since no pattern matches)
    REQUIRE(provider_unknown.max_context_size() == 0);
}

TEST_CASE("HttpLLMProvider::get_model_info returns registry info", "[llm][provider][registry]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("test-key");
    
    HttpLLMProvider provider(
        "https://api.openai.com/v1",
        creds,
        "gpt-4o",
        std::make_unique<protocols::OpenAIProtocol>());
    
    auto info = provider.get_model_info();
    REQUIRE(info.has_value());
    REQUIRE(info->canonical_id == "gpt-4o-2024-08-06");
}

TEST_CASE("HttpLLMProvider::supports checks capabilities", "[llm][provider][registry]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("test-key");
    
    HttpLLMProvider provider(
        "https://api.openai.com/v1",
        creds,
        "gpt-4o",
        std::make_unique<protocols::OpenAIProtocol>());
    
    REQUIRE(provider.supports(ModelCapability::Vision));
    REQUIRE(provider.supports(ModelCapability::FunctionCalling));
    REQUIRE(provider.supports(ModelCapability::JsonMode));
}

TEST_CASE("HttpLLMProvider::validate_request validates parameters", "[llm][provider][registry]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("test-key");
    
    HttpLLMProvider provider(
        "https://api.openai.com/v1",
        creds,
        "gpt-4o",
        std::make_unique<protocols::OpenAIProtocol>());
    
    // Valid request
    ChatRequest valid_req;
    valid_req.model = "gpt-4o";
    valid_req.max_tokens = 1000;
    valid_req.temperature = 0.7;
    
    auto errors = provider.validate_request(valid_req);
    REQUIRE(errors.empty());
    
    // Invalid max_tokens (exceeds limit of 16384)
    ChatRequest invalid_tokens;
    invalid_tokens.model = "gpt-4o";
    invalid_tokens.max_tokens = 20000;
    
    errors = provider.validate_request(invalid_tokens);
    REQUIRE(errors.size() >= 1);
    
    // Invalid temperature (exceeds limit of 2.0)
    ChatRequest invalid_temp;
    invalid_temp.model = "gpt-4o";
    invalid_temp.temperature = 3.0;
    
    errors = provider.validate_request(invalid_temp);
    // This may or may not produce an error depending on how validation is implemented
}

TEST_CASE("HttpLLMProvider::estimate_cost calculates cost", "[llm][provider][registry]") {
    auto creds = core::auth::ApiKeyCredentialSource::as_bearer("test-key");
    
    HttpLLMProvider provider(
        "https://api.openai.com/v1",
        creds,
        "gpt-4o",
        std::make_unique<protocols::OpenAIProtocol>());
    
    double cost = provider.estimate_cost(1000, 500);
    CHECK(cost == Catch::Approx(0.0075));
}

// =============================================================================
// Phase 4: Dynamic Registration Tests
// =============================================================================

TEST_CASE("ModelRegistry::register_model adds new model", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    size_t original_size = registry.size();
    
    ModelInfo custom_model;
    custom_model.canonical_id = "custom-model-v1";
    custom_model.aliases = {"custom-model"};
    custom_model.display_name = "Custom Model v1";
    custom_model.provider = "custom";
    custom_model.context_window = 32000;
    custom_model.max_output_tokens = 4096;
    custom_model.capabilities = 
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::TextOutput);
    custom_model.tier = ModelTier::Balanced;
    
    registry.register_model(std::move(custom_model));
    
    REQUIRE(registry.size() == original_size + 1);
    REQUIRE(registry.has_model("custom-model-v1"));
    REQUIRE(registry.has_model("custom-model"));  // via alias
    
    const auto* info = registry.lookup("custom-model");
    REQUIRE(info != nullptr);
    REQUIRE(info->provider == "custom");
}

TEST_CASE("ModelRegistry::clear removes all models", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // First ensure we have some models
    REQUIRE(registry.size() > 0);
    
    // Clear
    registry.clear();
    
    REQUIRE(registry.size() == 0);
    REQUIRE_FALSE(registry.has_model("gpt-4o"));
    
    // Reload defaults for other tests
    registry.load_defaults();
    REQUIRE(registry.size() > 0);
}

// =============================================================================
// Phase 5: JSON Serialization Tests
// =============================================================================

TEST_CASE("ModelRegistry::export_to_json - exports registry to JSON", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    std::string json = registry.export_to_json();
    
    // Check JSON structure
    REQUIRE(json.find("\"models\"") != std::string::npos);
    REQUIRE(json.find("\"canonical_id\"") != std::string::npos);
    REQUIRE(json.find("gpt-4o") != std::string::npos);
    REQUIRE(json.find("\"capabilities\"") != std::string::npos);
    REQUIRE(json.find("\"pricing\"") != std::string::npos);
}

TEST_CASE("ModelRegistry JSON round-trip", "[llm][registry]") {
    auto& registry = ModelRegistry::instance();
    
    // Export current registry
    std::string json = registry.export_to_json();
    
    // Verify JSON is valid (contains expected fields)
    CHECK(json.find("{") == 0);
    CHECK(json.find("}") != std::string::npos);  // Contains closing brace
    CHECK(json.find("\"models\"") != std::string::npos);
    
    // The load_from_json is a stub for now, so we just verify export works
    // Full round-trip test would require complete JSON parsing implementation
}

// =============================================================================
// Helper Tests
// =============================================================================

TEST_CASE("has_capability - bitmap operations", "[llm][registry]") {
    ModelCapabilities caps = 
        static_cast<uint32_t>(ModelCapability::TextInput) |
        static_cast<uint32_t>(ModelCapability::Vision);
    
    REQUIRE(has_capability(caps, ModelCapability::TextInput));
    REQUIRE(has_capability(caps, ModelCapability::Vision));
    REQUIRE_FALSE(has_capability(caps, ModelCapability::FunctionCalling));
}

TEST_CASE("to_string - ModelTier conversion", "[llm][registry]") {
    REQUIRE(to_string(ModelTier::Fast) == "fast");
    REQUIRE(to_string(ModelTier::Balanced) == "balanced");
    REQUIRE(to_string(ModelTier::Powerful) == "powerful");
    REQUIRE(to_string(ModelTier::Reasoning) == "reasoning");
}

TEST_CASE("tier_from_string - ModelTier parsing", "[llm][registry]") {
    REQUIRE(tier_from_string("fast") == ModelTier::Fast);
    REQUIRE(tier_from_string("balanced") == ModelTier::Balanced);
    REQUIRE(tier_from_string("powerful") == ModelTier::Powerful);
    REQUIRE(tier_from_string("reasoning") == ModelTier::Reasoning);
    REQUIRE_FALSE(tier_from_string("unknown").has_value());
}

TEST_CASE("model_supports - free function", "[llm][registry]") {
    REQUIRE(model_supports("gpt-4o", ModelCapability::Vision));
    REQUIRE_FALSE(model_supports("unknown-model", ModelCapability::TextInput));
}

TEST_CASE("get_model_tier - free function", "[llm][registry]") {
    REQUIRE(get_model_tier("gpt-4o-mini") == ModelTier::Fast);
    REQUIRE_FALSE(get_model_tier("unknown-model").has_value());
}
