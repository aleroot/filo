#pragma once

#include "../context/SessionContext.hpp"
#include "../workspace/PathVisibility.hpp"
#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>

#include <algorithm>
#include <format>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <filesystem>

namespace core::tools::detail {

inline std::filesystem::path resolve_workspace_path(
    const std::filesystem::path& path,
    const core::context::SessionContext& context)
{
    return context.resolve_path(path);
}

/**
 * Single authorization gate for every path-taking tool.
 *
 * The order is uniform across tools: workspace scope, then path visibility
 * (agent-ignore and sensitive-path rules), then configured tool policy.
 *
 * There is deliberately no per-tool escape hatch. Scope is a property of the
 * session (see core::workspace::FileAccessScope, carried in WorkspaceSnapshot),
 * not of which tool happens to be asking, so it is not possible to wire one
 * tool with a weaker or stronger view of the filesystem than its peers -- which
 * is precisely the failure mode of a defaulted, opt-in capability parameter.
 */
inline std::optional<std::string> check_workspace_access(
    const std::filesystem::path& path,
    const std::string& path_str,
    const core::context::SessionContext& context,
    std::filesystem::path* resolved_out = nullptr,
    std::string_view tool_name = {})
{
    const auto resolved = context.resolve_path(path);
    if (resolved_out) {
        *resolved_out = resolved;
    }

    // Mutating tools must clear the writable scope; everything else needs only
    // read. The two differ for scratch directories under a read-only sandbox,
    // where the shell may read /tmp but not write it.
    const bool mutates = names::is_file_modification_tool(tool_name);
    const bool permitted = mutates
        ? context.allows_write(resolved)
        : context.allows_read(resolved);
    if (!permitted) {
        return std::format(
            R"({{"error": "Access denied: Path '{}' is outside the allowed workspace scope."}})",
            core::utils::escape_json_string(path_str));
    }
    if (names::is_path_visibility_constrained_tool(tool_name)) {
        const auto* visibility = context.path_visibility.get();
        if (visibility != nullptr) {
            if (const auto hidden_reason =
                    visibility->hidden_reason(path_str, resolved)) {
                return std::format(
                    R"({{"error":"{}."}})",
                    *hidden_reason);
            }
        }
    }
    if (!tool_name.empty()) {
        if (const auto policy_error =
                core::tools::policy::enforce_path_policy(tool_name, resolved, context)) {
            return std::format(
                R"({{"error":"Tool policy blocked path '{}': {}."}})",
                core::utils::escape_json_string(path_str),
                core::utils::escape_json_string(*policy_error));
        }
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
