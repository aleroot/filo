#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"

using namespace core::llm;
using namespace core::llm::protocols;

TEST_CASE("ProviderFactory creates Z.ai regular API provider", "[zai][factory]") {
    core::config::ProviderConfig config;
    config.model = "glm-5.1";

    auto provider = core::llm::ProviderFactory::create_provider("zai", config);

    REQUIRE(provider != nullptr);
    REQUIRE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 200000);
}

TEST_CASE("ProviderFactory creates Z.ai Coding Plan provider", "[zai][factory][coding]") {
    core::config::ProviderConfig config;
    config.model = "glm-4.7";

    auto provider = core::llm::ProviderFactory::create_provider("zai-coding", config);

    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 200000);
}

TEST_CASE("ProviderFactory matches Z.ai Coding Plan before generic Z.ai prefix",
          "[zai][factory][coding]") {
    core::config::ProviderConfig config;
    config.model = "glm-5.2";

    auto provider = core::llm::ProviderFactory::create_provider("zai-coding-opus", config);

    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 1000000);
}

TEST_CASE("Z.ai OpenAI-compatible protocol appends chat completions path",
          "[zai][protocol]") {
    OpenAIProtocol protocol;

    REQUIRE(protocol.build_url("https://api.z.ai/api/paas/v4", "glm-5.1")
            == "https://api.z.ai/api/paas/v4/chat/completions");
    REQUIRE(protocol.build_url("https://api.z.ai/api/coding/paas/v4", "glm-4.7")
            == "https://api.z.ai/api/coding/paas/v4/chat/completions");
}

TEST_CASE("ModelRegistry includes Z.ai GLM coding models", "[zai][registry]") {
    auto& registry = ModelRegistry::instance();

    REQUIRE(registry.has_model("glm-5.2"));
    REQUIRE(registry.has_model("glm-5-turbo"));
    REQUIRE(registry.has_model("glm-4.7"));
    REQUIRE(registry.has_model("glm-4.5-air"));
    REQUIRE(get_max_context_size("glm-5.2") == 1000000);
    REQUIRE(get_max_context_size("glm-5-turbo") == 200000);
    REQUIRE(get_max_context_size("glm-4.7") == 200000);
    REQUIRE(get_max_context_size("glm-4.5-air") == 200000);
}

TEST_CASE("Z.ai Coding Plan protocol rejects non-plan models before HTTP",
          "[zai][validation][coding]") {
    ZaiCodingProtocol protocol;

    ChatRequest valid;
    valid.model = "GLM-4.7";
    valid.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_NOTHROW(protocol.prepare_request(valid));

    ChatRequest invalid;
    invalid.model = "glm-5.1";
    invalid.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_THROWS_WITH(
        protocol.prepare_request(invalid),
        Catch::Matchers::ContainsSubstring("supports only glm-5.2"));
}
