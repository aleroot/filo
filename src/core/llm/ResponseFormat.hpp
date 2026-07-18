#pragma once

#include <string>
#include <string_view>

namespace core::llm {

struct ResponseFormat {
    enum class Type { Text, JsonObject, JsonSchema };

    Type type = Type::Text;
    std::string schema;

    [[nodiscard]] constexpr std::string_view to_string() const noexcept {
        switch (type) {
        case Type::JsonObject: return "json_object";
        case Type::JsonSchema: return "json_schema";
        case Type::Text:       return "text";
        }
        return "text";
    }

    [[nodiscard]] constexpr bool is_structured() const noexcept {
        return type != Type::Text;
    }
};

} // namespace core::llm
