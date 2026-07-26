#include "ModelCatalogJson.hpp"

#include <limits>

namespace core::llm::catalog {
namespace {

[[nodiscard]] int32_t clamp_i32(int64_t value) noexcept {
    if (value < 0) return 0;
    if (value > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(value);
}

[[nodiscard]] bool support_value(simdjson::dom::element element) {
    bool supported = false;
    if (element.get(supported) == simdjson::SUCCESS) {
        return supported;
    }

    simdjson::dom::object object;
    return element.get(object) == simdjson::SUCCESS
        && object["supported"].get(supported) == simdjson::SUCCESS
        && supported;
}

} // namespace

std::string copy_string(std::string_view value) {
    return std::string(value.data(), value.size());
}

bool JsonObjectView::string(std::string_view key, std::string& out) const {
    std::string_view value;
    if (object_[key].get(value) != simdjson::SUCCESS) {
        return false;
    }
    out = copy_string(value);
    return true;
}

int32_t JsonObjectView::integer(std::string_view key, int32_t fallback) const {
    int64_t value = 0;
    return object_[key].get(value) == simdjson::SUCCESS
        ? clamp_i32(value)
        : fallback;
}

int32_t JsonObjectView::first_integer(
    std::initializer_list<std::string_view> keys,
    int32_t fallback) const {
    for (const std::string_view key : keys) {
        int64_t value = 0;
        if (object_[key].get(value) == simdjson::SUCCESS) {
            return clamp_i32(value);
        }
    }
    return fallback;
}

bool JsonObjectView::boolean(std::string_view key, bool fallback) const {
    bool value = false;
    return object_[key].get(value) == simdjson::SUCCESS ? value : fallback;
}

std::optional<bool> JsonObjectView::optional_boolean(
    std::string_view key) const {
    bool value = false;
    if (object_[key].get(value) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return value;
}

bool JsonObjectView::string_array_contains(
    std::string_view key,
    std::string_view needle) const {
    simdjson::dom::array array;
    if (object_[key].get(array) != simdjson::SUCCESS) {
        return false;
    }

    for (simdjson::dom::element item : array) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && value == needle) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> JsonObjectView::string_array(
    std::string_view key) const {
    std::vector<std::string> values;
    simdjson::dom::array array;
    if (object_[key].get(array) != simdjson::SUCCESS) {
        return values;
    }

    for (simdjson::dom::element item : array) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && !value.empty()) {
            values.push_back(copy_string(value));
        }
    }
    return values;
}

bool JsonObjectView::object_array_contains(
    std::string_view key,
    std::string_view field,
    std::string_view needle) const {
    simdjson::dom::array array;
    if (object_[key].get(array) != simdjson::SUCCESS) {
        return false;
    }

    for (simdjson::dom::element item : array) {
        simdjson::dom::object nested;
        if (item.get(nested) != simdjson::SUCCESS) continue;
        std::string_view value;
        if (nested[field].get(value) == simdjson::SUCCESS && value == needle) {
            return true;
        }
    }
    return false;
}

bool JsonObjectView::capability_supported(std::string_view name) const {
    simdjson::dom::object capabilities;
    if (object_["capabilities"].get(capabilities) == simdjson::SUCCESS) {
        simdjson::dom::element capability;
        return capabilities[name].get(capability) == simdjson::SUCCESS
            && support_value(capability);
    }

    simdjson::dom::array capability_names;
    if (object_["capabilities"].get(capability_names) != simdjson::SUCCESS) {
        return false;
    }
    for (simdjson::dom::element item : capability_names) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && value == name) {
            return true;
        }
    }
    return false;
}

bool JsonObjectView::nested_capability_supported(
    std::string_view capability,
    std::initializer_list<std::string_view> path) const {
    simdjson::dom::object capabilities;
    if (object_["capabilities"].get(capabilities) != simdjson::SUCCESS) {
        return false;
    }

    simdjson::dom::element current;
    if (capabilities[capability].get(current) != simdjson::SUCCESS) {
        return false;
    }
    for (const std::string_view key : path) {
        simdjson::dom::object object;
        if (current.get(object) != simdjson::SUCCESS) {
            return false;
        }
        simdjson::dom::element next;
        if (object[key].get(next) != simdjson::SUCCESS) {
            return false;
        }
        current = next;
    }
    return support_value(current);
}

bool JsonObjectView::has_capability_catalog() const {
    simdjson::dom::object capabilities_object;
    if (object_["capabilities"].get(capabilities_object) == simdjson::SUCCESS) {
        return true;
    }
    simdjson::dom::array capabilities_array;
    return object_["capabilities"].get(capabilities_array) == simdjson::SUCCESS;
}

ModelCatalogResult parse_catalog_json(
    std::string_view body,
    simdjson::dom::element& document,
    simdjson::dom::parser& parser) {
    ModelCatalogResult result;
    if (body.empty()) {
        result.error = "empty model catalog response";
        return result;
    }
    if (parser.parse(body.data(), body.size()).get(document) != simdjson::SUCCESS) {
        result.error = "invalid model catalog JSON";
    }
    return result;
}

} // namespace core::llm::catalog
