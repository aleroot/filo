#pragma once

#include "../utils/JsonUtils.hpp"
#include "../workspace/Workspace.hpp"
#include <simdjson.h>

#include <algorithm>
#include <format>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <filesystem>

namespace core::tools::detail {

inline std::optional<std::string> check_workspace_access(const std::filesystem::path& path, const std::string& path_str) {
    if (!core::workspace::Workspace::get_instance().is_path_allowed(path)) {
        return std::format(
            R"({{"error": "Access denied: Path '{}' is outside the allowed workspace scope."}})",
            core::utils::escape_json_string(path_str));
    }
    return std::nullopt;
}

inline std::string join_allowed_keys(std::initializer_list<std::string_view> allowed) {
    std::string joined;
    bool first = true;
    for (const auto key : allowed) {
        if (!first) {
            joined += ", ";
        }
        joined += key;
        first = false;
    }
    return joined;
}

inline std::optional<std::string> validate_object_arguments(
    const simdjson::dom::element& document,
    std::string_view tool_name,
    std::initializer_list<std::string_view> allowed_keys
) {
    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        return std::format(
            "{{\"error\":\"Invalid JSON arguments provided to {}. Expected an object.\"}}",
            core::utils::escape_json_string(tool_name));
    }

    for (const auto field : object) {
        const std::string_view key = std::string_view(field.key);
        const bool allowed = std::find(allowed_keys.begin(), allowed_keys.end(), key)
            != allowed_keys.end();
        if (allowed) {
            continue;
        }

        return std::format(
            "{{\"error\":\"Unknown argument '{}' provided to {}. Allowed arguments: {}.\"}}",
            core::utils::escape_json_string(key),
            core::utils::escape_json_string(tool_name),
            core::utils::escape_json_string(join_allowed_keys(allowed_keys)));
    }

    return std::nullopt;
}

} // namespace core::tools::detail
