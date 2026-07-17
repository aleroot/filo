#include <catch2/catch_test_macros.hpp>

#include "core/llm/ProviderCatalogGrouping.hpp"

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
    REQUIRE(groups[1].sources[0].includes_registry_model("glm-5.1"));
    REQUIRE_FALSE(groups[1].sources[0].includes_registry_model("glm-5.2"));

    REQUIRE(groups[1].sources[1].provider_name == "zai-coding");
    REQUIRE(groups[1].sources[1].category_label == "Coding endpoint.");
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-5.2"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-5-turbo"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-4.7"));
    REQUIRE(groups[1].sources[1].includes_registry_model("glm-4.5-air"));
    REQUIRE_FALSE(groups[1].sources[1].includes_registry_model("glm-5.1"));
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
    REQUIRE(groups[0].sources.size() == providers.size());
    for (const auto& provider : providers) {
        REQUIRE(groups[0].contains_source_provider(provider));
    }

    const auto alias_group = core::llm::provider_catalog_group_for("grok-mini", providers);
    REQUIRE(alias_group.provider_name == "grok");
}

TEST_CASE("Provider catalog grouping collapses Kimi presets and keeps endpoint models separate",
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
    REQUIRE(groups[0].sources.size() == providers.size());
    for (const auto& provider : providers) {
        REQUIRE(groups[0].contains_source_provider(provider));
    }

    const auto* public_api = groups[0].find_source("kimi");
    REQUIRE(public_api != nullptr);
    REQUIRE(public_api->includes_registry_model("kimi-k3"));
    REQUIRE_FALSE(public_api->includes_registry_model("k3"));
    REQUIRE_FALSE(public_api->includes_registry_model("kimi-for-coding"));

    const auto* code_api = groups[0].find_source("kimi-code");
    REQUIRE(code_api != nullptr);
    REQUIRE(code_api->category_label == "Kimi Code endpoint.");
    REQUIRE(code_api->includes_registry_model("k3"));
    REQUIRE(code_api->includes_registry_model("kimi-for-coding"));
    REQUIRE(code_api->includes_registry_model("kimi-for-coding-highspeed"));
    REQUIRE_FALSE(code_api->includes_registry_model("kimi-k3"));

    const auto alias_group = core::llm::provider_catalog_group_for("kimi-code", providers);
    REQUIRE(alias_group.provider_name == "kimi");
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
