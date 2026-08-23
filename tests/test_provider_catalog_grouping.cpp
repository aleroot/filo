#include <catch2/catch_test_macros.hpp>

#include "core/llm/ProviderCatalogGrouping.hpp"
#include "core/llm/ProviderDefinition.hpp"
#include "core/llm/KimiModelTraits.hpp"

#include <string>
#include <vector>

TEST_CASE("Provider catalog grouping keeps Z.ai categories under one provider",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{"openai", "zai", "zai-coding"};

    const auto groups = core::llm::provider_catalog_groups(providers);

    REQUIRE(groups.size() == 2);
    REQUIRE(groups[0].provider_name == "openai");
    REQUIRE(groups[0].sources.size() == 1);
    REQUIRE(groups[0].sources[0].provider_name == "openai");
    REQUIRE(groups[0].sources[0].includes_registry_model("gpt-5"));

    REQUIRE(groups[1].provider_name == "zai");
    REQUIRE(groups[1].sources.size() == 2);
    REQUIRE(groups[1].sources[0].provider_name == "zai");
    REQUIRE(groups[1].sources[0].category_label == "General API.");
    REQUIRE(groups[1].sources[0].includes_registry_model("glm-5.1"));
    REQUIRE_FALSE(groups[1].sources[0].includes_registry_model("glm-5.3"));
    REQUIRE_FALSE(groups[1].sources[0].includes_registry_model("glm-5.2"));

    REQUIRE(groups[1].sources[1].provider_name == "zai-coding");
    REQUIRE(groups[1].sources[1].category_label == "Coding endpoint.");
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-5.3"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-5.2"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-5-turbo"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-4.7"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-4.5-air"));
    REQUIRE_FALSE(groups[1].sources[1].includes_registry_model("glm-5.1"));
    REQUIRE(groups[1].sources[1].includes_api_model("glm-future-live"));
    REQUIRE_FALSE(groups[1].sources[1].includes_api_model("embedding-4"));

    const auto general_key = core::llm::provider_catalog_selection_key(
        groups[1].sources[0].service_id, "GLM-5.3");
    const auto coding_key = core::llm::provider_catalog_selection_key(
        groups[1].sources[1].service_id, "glm-5.3");
    REQUIRE(general_key != coding_key);
    REQUIRE(general_key == core::llm::provider_catalog_selection_key(
        "ZAI", "glm-5.3"));
}

TEST_CASE("Provider catalog group lookup maps category source names to visible provider",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{"zai", "zai-coding"};

    const auto group = core::llm::provider_catalog_group_for("zai-coding", providers);

    REQUIRE(group.provider_name == "zai");
    REQUIRE(group.contains_source_provider("zai"));
    REQUIRE(group.contains_source_provider("zai-coding"));
    REQUIRE(group.find_source("zai-coding") != nullptr);
    REQUIRE(group.find_source("missing") == nullptr);
}

TEST_CASE("Provider catalog grouping collapses Grok presets under Grok",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{
        "grok",
        "grok-4-5",
        "grok-4",
        "grok-4-fast",
        "grok-fast",
        "grok-mini",
        "grok-mini-fast",
        "grok-reasoning",
    };

    const auto groups = core::llm::provider_catalog_groups(providers);

    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].provider_name == "grok");
    REQUIRE(groups[0].sources.size() == 1);
    CHECK(groups[0].sources.front().provider_name == "grok");
    CHECK(groups[0].sources.front().service_id == "grok");
    for (const auto& provider : providers) {
        REQUIRE(groups[0].contains_source_provider(provider));
    }

    const auto alias_group = core::llm::provider_catalog_group_for("grok-mini", providers);
    REQUIRE(alias_group.provider_name == "grok");
}

TEST_CASE("Provider catalog grouping falls back to one configured Grok alias",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{
        "grok-fast",
        "grok-mini",
        "grok-mini-fast",
    };

    const auto group =
        core::llm::provider_catalog_group_for("grok-mini", providers);

    REQUIRE(group.provider_name == "grok");
    REQUIRE(group.sources.size() == 1);
    CHECK(group.sources.front().provider_name == "grok-fast");
    for (const auto& provider : providers) {
        CHECK(group.contains_source_provider(provider));
    }
}

TEST_CASE("Provider catalog grouping exposes only Kimi API and Kimi Code",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{
        "kimi",
        "kimi-code",
        "kimi-code-fast",
        "kimi-for-coding",
        "kimi-k2-6",
        "kimi-k2-5",
        "kimi-32k",
        "kimi-128k",
    };

    const auto groups = core::llm::provider_catalog_groups(providers);

    REQUIRE(groups.size() == 1);
    REQUIRE(groups[0].provider_name == "kimi");
    REQUIRE(groups[0].sources.size() == 2);
    for (const auto& provider : providers) {
        REQUIRE(groups[0].contains_source_provider(provider));
    }
    CHECK(groups[0].find_source("kimi-code-fast") == nullptr);
    CHECK(groups[0].find_source("kimi-for-coding") == nullptr);
    CHECK(groups[0].find_source("kimi-k2-6") == nullptr);

    const auto* public_api = groups[0].find_source("kimi");
    REQUIRE(public_api != nullptr);
    REQUIRE(public_api->category_label == "Kimi API.");
    CHECK(public_api->service_id
          == core::llm::kimi_service_id(
              core::llm::KimiService::PublicApi));
    REQUIRE(public_api->includes_registry_model("kimi-k3"));
    REQUIRE_FALSE(public_api->includes_registry_model("k3"));
    REQUIRE_FALSE(public_api->includes_registry_model("k3-256k"));
    REQUIRE_FALSE(public_api->includes_registry_model("kimi-for-coding"));

    const auto* code_api = groups[0].find_source("kimi-code");
    REQUIRE(code_api != nullptr);
    REQUIRE(code_api->category_label == "Kimi Code subscription.");
    CHECK(code_api->service_id
          == core::llm::kimi_service_id(
              core::llm::KimiService::Code));
    CHECK(groups[0].find_source_by_service_id(code_api->service_id)
          == code_api);
    REQUIRE_FALSE(code_api->includes_registry_model("k3"));
    REQUIRE_FALSE(code_api->includes_registry_model("kimi-for-coding"));
    REQUIRE_FALSE(code_api->includes_registry_model("kimi-for-coding-highspeed"));
    REQUIRE_FALSE(code_api->includes_registry_model("kimi-k3"));

    const auto alias_group = core::llm::provider_catalog_group_for("kimi-code", providers);
    REQUIRE(alias_group.provider_name == "kimi");
}

TEST_CASE("Kimi service routing is host-safe and independent of catalog DTOs",
          "[llm][provider-catalog][kimi]") {
    using core::llm::KimiService;
    using core::llm::is_kimi_code_endpoint_path;
    using core::llm::kimi_service_for_endpoint;

    CHECK(kimi_service_for_endpoint("https://api.moonshot.cn/v1")
          == KimiService::PublicApi);
    CHECK(kimi_service_for_endpoint("https://api.moonshot.ai/v1")
          == KimiService::PublicApi);
    CHECK(kimi_service_for_endpoint(
              "HTTPS://API.KIMI.COM:443/coding/v1/")
          == KimiService::Code);
    CHECK(kimi_service_for_endpoint(
              "https://api.kimi.com.evil.example/coding/v1")
          == KimiService::Unknown);
    CHECK(kimi_service_for_endpoint(
              "https://api.kimi.com/not-coding/v1")
          == KimiService::Unknown);
    CHECK(is_kimi_code_endpoint_path(
        "http://127.0.0.1:1234/CODING/v1?test=true"));
    CHECK_FALSE(is_kimi_code_endpoint_path(
        "https://api.kimi.com/not-coding/v1"));
}

TEST_CASE("Provider catalog grouping chooses one Kimi source per endpoint without canonical presets",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{
        "kimi-k2-6",
        "kimi-code-fast",
        "kimi-for-coding",
    };

    const auto group = core::llm::provider_catalog_group_for("kimi-code-fast", providers);

    REQUIRE(group.provider_name == "kimi");
    REQUIRE(group.sources.size() == 2);
    CHECK(group.sources[0].provider_name == "kimi-k2-6");
    CHECK(group.sources[0].category_label == "Kimi API.");
    CHECK(group.sources[1].provider_name == "kimi-code-fast");
    CHECK(group.sources[1].category_label == "Kimi Code subscription.");
    CHECK(group.contains_source_provider("kimi-for-coding"));
}

TEST_CASE("Provider catalog grouping gives Qwen Token Plan an exact registry fallback",
          "[llm][provider-catalog][qwen][token-plan]") {
    const std::vector<std::string> providers{"qwen", "qwen-token-plan"};
    const auto group = core::llm::provider_catalog_group_for("qwen-token-plan", providers);

    REQUIRE(group.provider_name == "qwen");
    REQUIRE(group.sources.size() == 2);
    const auto* public_api = group.find_source("qwen");
    const auto* token_plan = group.find_source("qwen-token-plan");
    REQUIRE(public_api != nullptr);
    REQUIRE(token_plan != nullptr);
    REQUIRE(token_plan->category_label == "Token Plan endpoint.");
    for (const auto model : {"qwen3.8-max", "qwen3.7-max",
                             "qwen3.7-plus", "qwen3.6-flash"}) {
        CHECK(token_plan->includes_registry_model(model));
        CHECK(public_api->includes_registry_model(model));
    }
    CHECK_FALSE(token_plan->includes_registry_model("qwen4.0-max"));
    CHECK_FALSE(token_plan->includes_registry_model("qwen3-coder-plus"));
    CHECK(public_api->includes_registry_model("qwen3-coder-plus"));
    CHECK(token_plan->includes_api_model("qwen4.0-max"));
    CHECK(token_plan->includes_api_model("glm-5.2"));
    CHECK(token_plan->includes_api_model("deepseek-v5"));
    CHECK_FALSE(token_plan->includes_api_model("wan2.7-image"));
    CHECK_FALSE(token_plan->includes_api_model("qwen-audio-2"));
    CHECK(public_api->includes_api_model("qwen-image-2.0"));
}

TEST_CASE("Provider catalog family matching respects provider-name boundaries",
          "[llm][provider-catalog]") {
    const std::vector<std::string> providers{
        "grok",
        "grokker",
        "kimi",
        "kimiko",
        "zai",
        "zaire",
    };

    const auto groups = core::llm::provider_catalog_groups(providers);

    REQUIRE(groups.size() == providers.size());
    REQUIRE(core::llm::provider_catalog_group_name("grok-fast") == "grok");
    REQUIRE(core::llm::provider_catalog_group_name("grokker") == "grokker");
    REQUIRE(core::llm::provider_catalog_group_name("kimi-code") == "kimi");
    REQUIRE(core::llm::provider_catalog_group_name("kimiko") == "kimiko");
    REQUIRE(core::llm::provider_catalog_group_name("zai-coding") == "zai");
    REQUIRE(core::llm::provider_catalog_group_name("zaire") == "zaire");
}

TEST_CASE("Built-in provider definitions are ordered, boundary-aware data",
          "[llm][provider-catalog][provider-definition]") {
    using core::llm::find_builtin_provider_definition;

    const auto* zai_coding =
        find_builtin_provider_definition("zai-coding-opus");
    REQUIRE(zai_coding != nullptr);
    CHECK(zai_coding->prefix == "zai-coding");
    CHECK(zai_coding->registry_provider == "zai");
    CHECK(zai_coding->catalog_group == "zai");

    const auto* token_plan =
        find_builtin_provider_definition("qwen-token-plan-team");
    REQUIRE(token_plan != nullptr);
    CHECK(token_plan->prefix == "qwen-token-plan");
    CHECK(token_plan->registry_provider == "qwen");
    CHECK(token_plan->default_wire_api == "chat_completions");

    CHECK(find_builtin_provider_definition("grokker") == nullptr);
    CHECK(find_builtin_provider_definition("kimiko") == nullptr);
    CHECK(find_builtin_provider_definition("zaire") == nullptr);
}
