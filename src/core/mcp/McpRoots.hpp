#pragma once

#include "../workspace/Workspace.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace core::mcp {

struct RootsCapability {
    bool supported = false;
    bool list_changed = false;
};

[[nodiscard]] RootsCapability extract_roots_capability_from_initialize(std::string_view json_body);

[[nodiscard]] bool is_workspace_sensitive_request(std::string_view method);

[[nodiscard]] std::string build_roots_list_request_body(std::string_view request_id);

[[nodiscard]] std::optional<core::workspace::WorkspaceSnapshot>
parse_roots_list_response(std::string_view response_body,
                          std::string_view expected_request_id,
                          std::uint64_t workspace_version,
                          std::string& error_out);

} // namespace core::mcp
