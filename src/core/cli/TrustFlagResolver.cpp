#include "TrustFlagResolver.hpp"

#include "../permissions/PermissionSystem.hpp"
#include "../tools/ToolPolicy.hpp"
#include "../utils/StringUtils.hpp"

#include <unordered_set>

namespace core::cli {

TrustFlagResolution resolve_trust_flags(bool yolo_mode, const std::vector<std::string>& trusted_tools) {
    TrustFlagResolution resolution;
    resolution.trust_all_tools = yolo_mode;

    std::unordered_set<std::string> seen_tools;
    std::unordered_set<std::string> seen_rules;

    for (const auto& raw_tool : trusted_tools) {
        const std::string trimmed = core::utils::str::trim_ascii_copy(raw_tool);
        if (trimmed.empty()) {
            continue;
        }

        const std::string lowered = core::utils::str::to_lower_ascii_copy(trimmed);
        if (lowered == "*" || lowered == "all") {
            resolution.trust_all_tools = true;
            continue;
        }

        const std::string canonical = core::tools::policy::canonical_tool_name(trimmed);
        if (canonical.empty()) {
            continue;
        }

        if (seen_tools.insert(canonical).second) {
            resolution.trusted_tool_names.push_back(canonical);
        }

        const auto normalized_rule =
            core::permissions::normalize_session_allow_rule("tool:" + canonical);
        if (!normalized_rule.empty() && seen_rules.insert(normalized_rule).second) {
            resolution.session_allow_rules.push_back(normalized_rule);
        }
    }

    return resolution;
}

} // namespace core::cli
