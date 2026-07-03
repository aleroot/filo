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
