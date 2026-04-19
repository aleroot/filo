#pragma once

#include <string>
#include <vector>

namespace core::cli {

struct TrustFlagResolution {
    bool trust_all_tools = false;
    std::vector<std::string> trusted_tool_names;
    std::vector<std::string> session_allow_rules;
};

[[nodiscard]] TrustFlagResolution resolve_trust_flags(bool yolo_mode, const std::vector<std::string>& trusted_tools);

} // namespace core::cli
