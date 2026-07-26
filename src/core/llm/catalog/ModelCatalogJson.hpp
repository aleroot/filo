#pragma once

#include "../ModelCatalogProvider.hpp"

#include <simdjson.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm::catalog {

/**
 * Small read-only adapter around simdjson's provider-owned object view.
 *
 * Provider catalog decoders use this class instead of duplicating JSON type
 * checks, integer clamping, optional-field handling, and the capability
 * encodings shared by several APIs.
 */
class JsonObjectView {
public:
    explicit JsonObjectView(simdjson::dom::object object) noexcept
        : object_(object) {}

    [[nodiscard]] bool string(std::string_view key, std::string& out) const;
    [[nodiscard]] int32_t integer(std::string_view key,
                                  int32_t fallback = 0) const;
    [[nodiscard]] int32_t first_integer(
        std::initializer_list<std::string_view> keys,
        int32_t fallback = 0) const;
    [[nodiscard]] bool boolean(std::string_view key,
                               bool fallback = false) const;
    [[nodiscard]] std::optional<bool> optional_boolean(
        std::string_view key) const;
    [[nodiscard]] bool string_array_contains(
        std::string_view key,
        std::string_view needle) const;
    [[nodiscard]] std::vector<std::string> string_array(
        std::string_view key) const;
    [[nodiscard]] bool object_array_contains(
        std::string_view key,
        std::string_view field,
        std::string_view needle) const;

    /**
     * Accept the capability representations used by current provider APIs:
     * an object containing booleans or {supported}, or an array of names.
     */
    [[nodiscard]] bool capability_supported(std::string_view name) const;

    /**
     * Read nested capability details such as
     * capabilities.thinking.types.adaptive.supported.
     */
    [[nodiscard]] bool nested_capability_supported(
        std::string_view capability,
        std::initializer_list<std::string_view> path) const;

    [[nodiscard]] bool has_capability_catalog() const;
    [[nodiscard]] simdjson::dom::object raw() const noexcept { return object_; }

private:
    simdjson::dom::object object_;
};

[[nodiscard]] ModelCatalogResult parse_catalog_json(
    std::string_view body,
    simdjson::dom::element& document,
    simdjson::dom::parser& parser);

[[nodiscard]] std::string copy_string(std::string_view value);

} // namespace core::llm::catalog
