#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/ToolSchema.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {

core::tools::ToolDefinition complex_tool() {
    return {
        .name = "edit",
        .description = "Edit a file",
        .input_schema = R"({
            "required":["path","operation"],
            "properties":{
                "operation":{"oneOf":[
                    {"type":"object","additionalProperties":false,"required":["replace"],"properties":{"replace":{"type":"string"}}},
                    {"type":"object","additionalProperties":false,"required":["append"],"properties":{"append":{"type":"string"}}}
                ]},
                "path":{"description":"Target path","type":"string"},
                "encoding":{"enum":["utf-8","ascii"],"type":"string"}
            },
            "additionalProperties":false,
            "type":"object"
        })",
    };
}

} // namespace

TEST_CASE("canonical tool schema preserves full JSON Schema constraints",
          "[tools][schema]") {
    const std::string schema =
        core::tools::schema::canonical_input_schema(complex_tool());

    CHECK_THAT(schema, ContainsSubstring(R"("additionalProperties":false)"));
    CHECK_THAT(schema, ContainsSubstring(R"("enum":["utf-8","ascii"])"));
    CHECK_THAT(schema, ContainsSubstring(R"("oneOf":[)"));
    CHECK(schema == core::tools::schema::canonical_input_schema(complex_tool()));
    CHECK(schema.find("additionalProperties") < schema.find("properties"));
}

TEST_CASE("canonical tool schema derives arrays and rejects unknown properties",
          "[tools][schema]") {
    const core::tools::ToolDefinition definition{
        .name = "batch",
        .parameters = {
            {
                .name = "paths",
                .type = "array",
                .description = "Files to inspect",
                .required = true,
                .items_schema = R"({"type":"string"})",
            },
            {
                .name = "limit",
                .type = "integer",
                .description = "Optional limit",
            },
        },
    };

    const std::string schema = core::tools::schema::canonical_input_schema(definition);
    CHECK_THAT(schema, ContainsSubstring(R"("items":{"type":"string"})"));
    CHECK_THAT(schema, ContainsSubstring(R"("required":["paths"])"));
    CHECK_THAT(schema, ContainsSubstring(R"("additionalProperties":false)"));
}

TEST_CASE("tool arguments are normalized before validation", "[tools][schema]") {
    auto definition = complex_tool();

    const auto normalized = core::tools::schema::normalize_arguments(
        definition,
        R"({"operation":{"append":"hello"},"encoding":null,"path":"a.cpp"})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized == R"({"operation":{"append":"hello"},"path":"a.cpp"})");
}

TEST_CASE("tool argument validation catches structural model mistakes",
          "[tools][schema]") {
    const auto definition = complex_tool();

    const auto missing = core::tools::schema::normalize_arguments(
        definition, R"({"operation":{"append":"hello"}})");
    REQUIRE_FALSE(missing.has_value());
    CHECK_THAT(missing.error(), ContainsSubstring("missing required property 'path'"));

    const auto wrong_union = core::tools::schema::normalize_arguments(
        definition, R"({"path":"a.cpp","operation":{"delete":true}})");
    REQUIRE_FALSE(wrong_union.has_value());
    CHECK_THAT(wrong_union.error(), ContainsSubstring("does not satisfy oneOf"));

    const auto unknown = core::tools::schema::normalize_arguments(
        definition,
        R"({"path":"a.cpp","operation":{"append":"hello"},"surprise":1})");
    REQUIRE_FALSE(unknown.has_value());
    CHECK_THAT(unknown.error(), ContainsSubstring("Unknown argument 'surprise'"));
}
