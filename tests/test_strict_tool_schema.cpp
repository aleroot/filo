#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/StrictToolPolicy.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/tools/ReadTool.hpp"
#include "core/tools/SearchReplaceTool.hpp"
#include "core/tools/StrictToolSchema.hpp"
#include "core/tools/ToolSchema.hpp"

#include <simdjson.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>

using Catch::Matchers::ContainsSubstring;
using core::tools::schema::StrictDialect;

namespace {

core::tools::ToolDefinition edit_tool() {
    return {
        .name = "edit",
        .description = "Apply edits",
        .parameters = {
            {.name = "file_path", .type = "string", .required = true},
            {
                .name = "edits",
                .type = "array",
                .required = true,
                .items_schema =
                    R"({"type":"object","properties":{"old_string":{"type":"string"},)"
                    R"("new_string":{"type":"string"}},)"
                    R"("required":["old_string","new_string"],"additionalProperties":false})",
            },
        },
    };
}

/// Optional scalars of every kind a projection has to keep expressible.
core::tools::ToolDefinition query_tool() {
    return {
        .name = "query",
        .description = "Query something",
        .input_schema =
            R"({"type":"object","properties":{)"
            R"("pattern":{"type":"string","minLength":1,"maxLength":512},)"
            R"("limit":{"type":"integer","minimum":1,"maximum":100},)"
            R"("mode":{"type":"string","enum":["fast","deep"]},)"
            R"("target":{"oneOf":[{"type":"string"},{"type":"array","items":{"type":"string"},"minItems":1}]})"
            R"(},"required":["pattern"],"additionalProperties":false})",
    };
}

/// Loads a configuration that enables strict schemas and restores the process
/// singleton afterwards: ConfigManager is global, so a sandboxed load would
/// otherwise leak into every later test in the binary.
class ScopedStrictConfig {
public:
    ScopedStrictConfig() {
        if (const char* existing = std::getenv("XDG_CONFIG_HOME")) {
            previous_ = std::string(existing);
        }
        root_ = std::filesystem::temp_directory_path()
            / std::format("filo-strict-config-{}",
                          std::chrono::steady_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(root_ / "filo");
        std::ofstream(root_ / "filo" / "config.json")
            << R"({"strict_tool_schemas": true})";
        setenv("XDG_CONFIG_HOME", root_.c_str(), 1);
        core::config::ConfigManager::get_instance().load();
    }

    ScopedStrictConfig(const ScopedStrictConfig&) = delete;
    ScopedStrictConfig& operator=(const ScopedStrictConfig&) = delete;

    ~ScopedStrictConfig() {
        if (previous_.has_value()) {
            setenv("XDG_CONFIG_HOME", previous_->c_str(), 1);
        } else {
            unsetenv("XDG_CONFIG_HOME");
        }
        core::config::ConfigManager::get_instance().load();
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

private:
    std::optional<std::string> previous_;
    std::filesystem::path root_;
};

[[nodiscard]] simdjson::dom::element parse(simdjson::dom::parser& parser,
                                           simdjson::padded_string& buffer,
                                           std::string_view json) {
    buffer = simdjson::padded_string(json);
    simdjson::dom::element root;
    REQUIRE(parser.parse(buffer).get(root) == simdjson::SUCCESS);
    return root;
}

} // namespace

TEST_CASE("the OpenAI projection makes every property required and nullable",
          "[tools][strict]") {
    const auto projected = core::tools::schema::strict_input_schema(
        query_tool(), StrictDialect::OpenAI);
    REQUIRE(projected.has_value());

    simdjson::dom::parser parser;
    simdjson::padded_string buffer;
    const auto root = parse(parser, buffer, *projected);

    simdjson::dom::array required;
    REQUIRE(root["required"].get(required) == simdjson::SUCCESS);
    CHECK(required.size() == 4);

    // The one genuinely required property keeps its bare type.
    std::string_view pattern_type;
    CHECK(root["properties"]["pattern"]["type"].get(pattern_type) == simdjson::SUCCESS);
    CHECK(pattern_type == "string");

    // Optional scalars express optionality by admitting null instead.
    CHECK_THAT(*projected, ContainsSubstring(R"("limit":{"type":["integer","null"]})"));
    CHECK_THAT(*projected, ContainsSubstring(R"("enum":["fast","deep",null])"));
    CHECK_THAT(*projected, ContainsSubstring(R"("type":["string","null"])"));
}

TEST_CASE("the projection drops keywords a grammar cannot enforce",
          "[tools][strict]") {
    const auto projected = core::tools::schema::strict_input_schema(
        query_tool(), StrictDialect::OpenAI);
    REQUIRE(projected.has_value());

    // Bounds live on in the canonical schema, which still gates every call.
    const auto canonical = core::tools::schema::canonical_input_schema(query_tool());
    CHECK_THAT(canonical, ContainsSubstring("maxLength"));

    for (const auto* keyword : {"minLength", "maxLength", "minimum", "maximum", "minItems"}) {
        CHECK_THAT(*projected, !ContainsSubstring(keyword));
    }
    // A union is rewritten to the spelling both subsets accept.
    CHECK_THAT(*projected, !ContainsSubstring("oneOf"));
    CHECK_THAT(*projected, ContainsSubstring("anyOf"));
}

TEST_CASE("the Anthropic projection leaves optionality alone",
          "[tools][strict]") {
    const auto projected = core::tools::schema::strict_input_schema(
        query_tool(), StrictDialect::Anthropic);
    REQUIRE(projected.has_value());

    simdjson::dom::parser parser;
    simdjson::padded_string buffer;
    const auto root = parse(parser, buffer, *projected);

    simdjson::dom::array required;
    REQUIRE(root["required"].get(required) == simdjson::SUCCESS);
    CHECK(required.size() == 1);
    CHECK_THAT(*projected, !ContainsSubstring("null"));
    CHECK_THAT(*projected, !ContainsSubstring("maxLength"));
}

TEST_CASE("nested item contracts survive the projection", "[tools][strict]") {
    const auto projected = core::tools::schema::strict_input_schema(
        edit_tool(), StrictDialect::OpenAI);
    REQUIRE(projected.has_value());

    CHECK_THAT(*projected,
               ContainsSubstring(R"("required":["new_string","old_string"])"));
    CHECK_THAT(*projected, ContainsSubstring(R"("additionalProperties":false)"));
    // An array-typed argument can only begin with '[' under the grammar, which
    // is the entire point of the exercise.
    CHECK_THAT(*projected, ContainsSubstring(R"("edits":{"items":)"));
}

TEST_CASE("a contract that cannot be expressed is refused, not weakened",
          "[tools][strict]") {
    const core::tools::ToolDefinition recursive{
        .name = "recursive",
        .description = "Refers to a definition",
        .input_schema =
            R"({"type":"object","properties":{"node":{"$ref":"#/$defs/node"}},)"
            R"("$defs":{"node":{"type":"object"}},"additionalProperties":false})",
    };
    CHECK_FALSE(core::tools::schema::strict_input_schema(
                    recursive, StrictDialect::OpenAI).has_value());

    // A fixed value cannot also admit null, so an optional const is refused
    // rather than silently widened.
    const core::tools::ToolDefinition constant{
        .name = "constant",
        .description = "Carries a const",
        .input_schema =
            R"({"type":"object","properties":{"kind":{"const":"fixed"}},)"
            R"("required":[],"additionalProperties":false})",
    };
    CHECK_FALSE(core::tools::schema::strict_input_schema(
                    constant, StrictDialect::OpenAI).has_value());
}

TEST_CASE("the shipped tool contracts project cleanly", "[tools][strict]") {
    const core::tools::SearchReplaceTool search_replace;
    const core::tools::ReadTool read;

    for (const auto dialect : {StrictDialect::OpenAI, StrictDialect::Anthropic}) {
        INFO("dialect " << core::tools::schema::strict_dialect_name(dialect));
        CHECK(core::tools::schema::strict_input_schema(
                  search_replace.get_definition(), dialect).has_value());
        CHECK(core::tools::schema::strict_input_schema(
                  read.get_definition(), dialect).has_value());
    }
}

TEST_CASE("a payload shaped by the strict schema still validates",
          "[tools][strict]") {
    // Under the OpenAI projection a model cannot omit an unused optional; it
    // sends null instead. Those nulls must reduce to omission at every depth,
    // or strict mode would trade a shape error for a type error.
    const core::tools::ReadTool read;
    const auto normalized = core::tools::schema::normalize_arguments(
        read.get_definition(),
        R"({"path":"a.cpp","view":null,"question":null,"offset_line":null,)"
        R"("limit_lines":null,"expected_digest":null,"select":{"cell":null}})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized == R"({"path":"a.cpp","select":{}})");
}

TEST_CASE("strict tool support is claimed only for documented generations",
          "[tools][strict][policy]") {
    using core::llm::model_supports_strict_tools;
    using core::llm::ToolSchemaWire;

    CHECK(model_supports_strict_tools(ToolSchemaWire::Anthropic, "claude-opus-5"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::Anthropic, "claude-sonnet-4-5"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::Anthropic,
                                      "claude-opus-4-5-20251101"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::Anthropic, "claude-haiku-4.5"));
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::Anthropic, "claude-opus-4-1"));
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::Anthropic,
                                            "claude-3-5-sonnet-20241022"));

    CHECK(model_supports_strict_tools(ToolSchemaWire::OpenAI, "gpt-5.6-sol"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::OpenAI, "gpt-4.1-mini"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::OpenAI, "gpt-4o"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::OpenAI, "gpt-4o-2024-11-20"));
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::OpenAI, "gpt-4o-2024-05-13"));
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::OpenAI, "gpt-4-turbo"));

    // A model reached through an OpenAI-compatible endpoint is not assumed to
    // inherit the guarantee.
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::OpenAI, "grok-4.6"));
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::OpenAI, "glm-5.2"));
}

TEST_CASE("the strict switch gates the whole decision", "[tools][strict][policy]") {
    using core::llm::strict_tool_dialect;
    using core::llm::ToolSchemaWire;

    CHECK_FALSE(strict_tool_dialect(
        ToolSchemaWire::Anthropic, "claude-opus-5", /*enabled=*/false).has_value());
    CHECK(strict_tool_dialect(ToolSchemaWire::Anthropic, "claude-opus-5", true)
          == StrictDialect::Anthropic);
    CHECK(strict_tool_dialect(ToolSchemaWire::OpenAI, "gpt-5.6-sol", true)
          == StrictDialect::OpenAI);
    CHECK_FALSE(strict_tool_dialect(ToolSchemaWire::OpenAI, "grok-4.6", true)
                    .has_value());
}

TEST_CASE("the Anthropic wire claims strictness only where it is documented",
          "[tools][strict][wire]") {
    const ScopedStrictConfig strict_config;

    core::llm::ChatRequest request;
    request.tools.push_back(core::llm::Tool{.function = edit_tool()});

    core::llm::protocols::AnthropicProtocol protocol;

    request.model = "claude-opus-5";
    const std::string supported = protocol.serialize(request);
    CHECK_THAT(supported, ContainsSubstring(R"("strict":true)"));
    CHECK_THAT(supported, ContainsSubstring(R"("input_schema":)"));

    request.model = "claude-3-5-sonnet-20241022";
    CHECK_THAT(protocol.serialize(request), !ContainsSubstring(R"("strict")"));
}

TEST_CASE("strict schemas stay off until the switch is set",
          "[tools][strict][wire]") {
    core::llm::ChatRequest request;
    request.model = "claude-opus-5";
    request.tools.push_back(core::llm::Tool{.function = edit_tool()});

    core::llm::protocols::AnthropicProtocol protocol;
    CHECK_THAT(protocol.serialize(request), !ContainsSubstring(R"("strict")"));
}

TEST_CASE("the OpenAI-compatible serializer flags projected tools strict",
          "[tools][strict][wire]") {
    core::llm::ChatRequest request;
    request.model = "gpt-5.6-sol";
    request.tools.push_back(core::llm::Tool{.function = edit_tool()});

    const std::string permissive = core::llm::Serializer::serialize(request);
    CHECK_THAT(permissive, !ContainsSubstring(R"("strict")"));

    core::llm::Serializer::Options options;
    options.strict_tools = StrictDialect::OpenAI;
    const std::string strict = core::llm::Serializer::serialize(request, options);
    CHECK_THAT(strict, ContainsSubstring(R"("strict":true)"));
    CHECK_THAT(strict, ContainsSubstring(R"("required":["edits","file_path"])"));

    simdjson::dom::parser parser;
    simdjson::padded_string buffer(strict);
    simdjson::dom::element root;
    CHECK(parser.parse(buffer).get(root) == simdjson::SUCCESS);
}
