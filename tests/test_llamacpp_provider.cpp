#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/tools/Tool.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Factory (both build configurations)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory handles llama.cpp providers", "[llamacpp][factory]") {
    core::config::ProviderConfig config;
    config.api_type = core::config::ApiType::LlamaCppLocal;
    config.model    = "local-code";
    config.local    = core::config::LocalModelConfig{.model_path = "/tmp/model.gguf"};

    auto provider = core::llm::ProviderFactory::create_provider("local", config);

#ifdef FILO_ENABLE_LLAMACPP
    REQUIRE(provider != nullptr);
    REQUIRE(provider->capabilities().supports_tool_calls);
    REQUIRE(provider->capabilities().is_local);
    REQUIRE_FALSE(provider->should_estimate_cost());
#else
    REQUIRE(provider == nullptr);
#endif
}

#ifdef FILO_ENABLE_LLAMACPP

#include "core/llm/providers/LlamaCppProvider.hpp"

using namespace core::llm;
using namespace core::llm::providers;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static core::llm::Tool make_tool(std::string name,
                                 std::string description,
                                 std::vector<core::tools::ToolParameter> params = {}) {
    core::tools::ToolDefinition def;
    def.name = std::move(name);
    def.description = std::move(description);
    def.parameters = std::move(params);
    return core::llm::Tool{.function = std::move(def)};
}

static core::tools::ToolParameter make_param(std::string name,
                                             std::string type,
                                             std::string description,
                                             bool required = true) {
    return core::tools::ToolParameter{
        .name = std::move(name),
        .type = std::move(type),
        .description = std::move(description),
        .required = required,
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// build_tool_parameters_schema
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("build_tool_parameters_schema - no parameters", "[llamacpp][schema]") {
    auto tool = make_tool("ping", "Check connectivity");
    const auto schema = LlamaCppProvider::build_tool_parameters_schema(tool);

    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("type":"object")"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("properties":{})"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("required":[])"));
}

TEST_CASE("build_tool_parameters_schema - single required parameter", "[llamacpp][schema]") {
    auto tool = make_tool("bash", "Run a shell command",
                          {make_param("command", "string", "Shell command to execute", true)});
    const auto schema = LlamaCppProvider::build_tool_parameters_schema(tool);

    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("command")"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("type":"string")"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring("Shell command to execute"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("required":["command"])"));
}

TEST_CASE("build_tool_parameters_schema - optional parameter not in required array", "[llamacpp][schema]") {
    auto tool = make_tool("read_file", "Read a file",
                          {make_param("path", "string", "File path", true),
                           make_param("encoding", "string", "File encoding", false)});
    const auto schema = LlamaCppProvider::build_tool_parameters_schema(tool);

    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("path")"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("encoding")"));
    // Only "path" should appear in the required array
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("required":["path"])"));
    // "encoding" must NOT appear in the required array
    REQUIRE_THAT(schema, !Catch::Matchers::ContainsSubstring(R"("required":["path","encoding"])"));
}

TEST_CASE("build_tool_parameters_schema - multiple required parameters", "[llamacpp][schema]") {
    auto tool = make_tool("write_file", "Write content to a file",
                          {make_param("path", "string", "Destination path", true),
                           make_param("content", "string", "Content to write", true)});
    const auto schema = LlamaCppProvider::build_tool_parameters_schema(tool);

    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("path")"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("content")"));
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("required":["path","content"])"));
}

TEST_CASE("build_tool_parameters_schema - special characters are JSON-escaped", "[llamacpp][schema]") {
    auto tool = make_tool("quote_tool", "A tool with special chars",
                          {make_param("msg", "string", R"(Say "hello" & goodbye)", true)});
    const auto schema = LlamaCppProvider::build_tool_parameters_schema(tool);

    // Quotes inside a description must be escaped
    REQUIRE_THAT(schema, !Catch::Matchers::ContainsSubstring(R"(Say "hello")"));
    // The escaped form must be present
    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"(Say \"hello\")"));
}

TEST_CASE("build_tool_parameters_schema - integer type is preserved", "[llamacpp][schema]") {
    auto tool = make_tool("sleep", "Wait N seconds",
                          {make_param("seconds", "integer", "Number of seconds", true)});
    const auto schema = LlamaCppProvider::build_tool_parameters_schema(tool);

    REQUIRE_THAT(schema, Catch::Matchers::ContainsSubstring(R"("type":"integer")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Provider capabilities
// ─────────────────────────────────────────────────────────────────────────────

namespace {
core::config::ProviderConfig make_local_config(std::string model_path = "/nonexistent/model.gguf") {
    core::config::ProviderConfig cfg;
    cfg.api_type = core::config::ApiType::LlamaCppLocal;
    cfg.local    = core::config::LocalModelConfig{.model_path = std::move(model_path)};
    return cfg;
}
} // namespace

TEST_CASE("LlamaCppProvider capabilities flags", "[llamacpp][capabilities]") {
    auto provider = core::llm::ProviderFactory::create_provider("local", make_local_config());
    REQUIRE(provider != nullptr);

    const auto caps = provider->capabilities();
    REQUIRE(caps.supports_tool_calls);
    REQUIRE(caps.is_local);
    REQUIRE_FALSE(provider->should_estimate_cost());
}

TEST_CASE("LlamaCppProvider default model label is empty before inference", "[llamacpp]") {
    auto provider = core::llm::ProviderFactory::create_provider("local", make_local_config());
    REQUIRE(provider != nullptr);
    // No inference has run yet; model label should be empty
    REQUIRE(provider->get_last_model().empty());
}

TEST_CASE("LlamaCppProvider initial token usage is zero", "[llamacpp]") {
    auto provider = core::llm::ProviderFactory::create_provider("local", make_local_config());
    REQUIRE(provider != nullptr);

    const auto usage = provider->get_last_usage();
    REQUIRE(usage.prompt_tokens == 0);
    REQUIRE(usage.completion_tokens == 0);
    REQUIRE(usage.total_tokens == 0);
}

TEST_CASE("LlamaCppProvider rate limit info is zeroed", "[llamacpp]") {
    auto provider = core::llm::ProviderFactory::create_provider("local", make_local_config());
    REQUIRE(provider != nullptr);

    const auto rate_limit = provider->get_last_rate_limit_info();
    REQUIRE(rate_limit.requests_remaining == 0);
    REQUIRE(rate_limit.tokens_remaining == 0);
    REQUIRE(rate_limit.usage_windows.empty());
}

TEST_CASE("ProviderFactory creates LlamaCpp provider without base_url", "[llamacpp][factory][regression]") {
    // This is a regression test for the bug where ProviderFactory required
    // base_url for LlamaCppLocal providers even though they don't use HTTP.
    core::config::ProviderConfig config;
    config.api_type = core::config::ApiType::LlamaCppLocal;
    config.model    = "local-test-model";
    config.local    = core::config::LocalModelConfig{.model_path = "/tmp/test-model.gguf"};
    // Note: base_url is intentionally left empty

    auto provider = core::llm::ProviderFactory::create_provider("local", config);

#ifdef FILO_ENABLE_LLAMACPP
    REQUIRE(provider != nullptr);
    // Verify it's actually a LlamaCppProvider by checking capabilities
    REQUIRE(provider->capabilities().is_local);
    REQUIRE_FALSE(provider->should_estimate_cost());
#else
    REQUIRE(provider == nullptr);
#endif
}

TEST_CASE("LlamaCppProvider handles various model paths", "[llamacpp]") {
    // Test with absolute path
    {
        auto provider = core::llm::ProviderFactory::create_provider(
            "local", make_local_config("/home/user/models/model.gguf"));
        REQUIRE(provider != nullptr);
    }
    // Test with relative path
    {
        auto provider = core::llm::ProviderFactory::create_provider(
            "local", make_local_config("./models/model.gguf"));
        REQUIRE(provider != nullptr);
    }
    // Test with path containing spaces
    {
        auto provider = core::llm::ProviderFactory::create_provider(
            "local", make_local_config("/home/user/my models/model.gguf"));
        REQUIRE(provider != nullptr);
    }
}

TEST_CASE("ProviderFactory distinguishes LlamaCpp from HTTP providers", "[llamacpp][factory]") {
    // LlamaCppLocal should not require base_url
    core::config::ProviderConfig llama_config;
    llama_config.api_type = core::config::ApiType::LlamaCppLocal;
    llama_config.model    = "llama-local";
    llama_config.local    = core::config::LocalModelConfig{.model_path = "/tmp/model.gguf"};
    
    auto llama_provider = core::llm::ProviderFactory::create_provider("llama-local", llama_config);
#ifdef FILO_ENABLE_LLAMACPP
    REQUIRE(llama_provider != nullptr);
#else
    REQUIRE(llama_provider == nullptr);
#endif

    // OpenAI provider without base_url should fail
    core::config::ProviderConfig openai_config;
    openai_config.api_type = core::config::ApiType::OpenAI;
    openai_config.model    = "gpt-4";
    // No base_url provided - should fail
    
    auto openai_provider = core::llm::ProviderFactory::create_provider("custom-openai", openai_config);
    REQUIRE(openai_provider == nullptr);  // Should fail without base_url
}

#endif // FILO_ENABLE_LLAMACPP
