#include <catch2/catch_test_macros.hpp>

#include "core/config/SessionModelOverride.hpp"

namespace {

core::config::AppConfig test_config() {
    core::config::AppConfig config;
    config.default_provider = "openai";
    config.default_model_selection = "manual";
    config.providers["openai"].model = "gpt-default";
    config.providers["claude"].model = "claude-default";
    return config;
}

} // namespace

TEST_CASE("Session model override supports model and provider selectors", "[config][model][session]") {
    SECTION("bare model keeps the default provider") {
        auto config = test_config();
        REQUIRE(core::config::apply_session_model_override(config, " gpt-session "));
        CHECK(config.default_provider == "openai");
        CHECK(config.default_model_selection == "manual");
        CHECK(config.providers.at("openai").model == "gpt-session");
    }

    SECTION("provider uses its configured default model") {
        auto config = test_config();
        REQUIRE(core::config::apply_session_model_override(config, "claude"));
        CHECK(config.default_provider == "claude");
        CHECK(config.providers.at("claude").model == "claude-default");
    }

    SECTION("provider and model selects both") {
        auto config = test_config();
        REQUIRE(core::config::apply_session_model_override(config, "claude/sonnet"));
        CHECK(config.default_provider == "claude");
        CHECK(config.providers.at("claude").model == "sonnet");
    }
}

TEST_CASE("Session model override supports routing without persistence", "[config][model][session]") {
    auto config = test_config();

    REQUIRE(core::config::apply_session_model_override(config, "router"));
    CHECK(config.default_model_selection == "router");

    REQUIRE(core::config::apply_session_model_override(config, "auto"));
    CHECK(config.default_model_selection == "auto");
}

TEST_CASE("Session model override rejects malformed selectors", "[config][model][session]") {
    auto config = test_config();

    auto empty = core::config::apply_session_model_override(config, "  ");
    REQUIRE_FALSE(empty);
    CHECK(empty.error().contains("non-empty"));

    auto missing_model = core::config::apply_session_model_override(config, "claude/");
    REQUIRE_FALSE(missing_model);
    CHECK(missing_model.error().contains("requires a model"));
}
