#pragma once

#include "ui/AuthUI.hpp"
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth::google_code_assist {

struct TierInfo {
    std::string id;
    bool        user_defined_project = false;
    bool        is_default = false;
};

struct LoadCodeAssistResponseData {
    std::optional<TierInfo> current_tier;
    std::vector<TierInfo>   allowed_tiers;
    std::string             project_id;
};

struct OnboardUserOperation {
    bool        done = false;
    std::string name;         ///< Long-running operation to poll (`operations/...`).
    std::string project_id;
};

/// `CODE_ASSIST_ENDPOINT` override, else the daily host for the "ANTIGRAVITY"
/// ide type and the production host for everything else.
[[nodiscard]] std::string code_assist_endpoint(std::string_view ide_type = "IDE_UNSPECIFIED");
[[nodiscard]] std::optional<std::string> configured_project_override();
/// Request-body builders, exposed for tests. Each mirrors the metadata shape
/// of the matching first-party client: "ANTIGRAVITY" sends `metadata` with
/// only `ideType` and no `cloudaicompanionProject`; other ide types follow the
/// gemini-cli shape.
[[nodiscard]] std::string client_metadata_json(std::string_view project_id,
                                               std::string_view ide_type);
[[nodiscard]] std::string load_code_assist_payload(std::string_view project_id,
                                                   std::string_view ide_type);
[[nodiscard]] std::string onboard_user_payload(std::string_view tier_id,
                                               std::string_view project_id,
                                               std::string_view ide_type);
[[nodiscard]] LoadCodeAssistResponseData parse_load_code_assist_response(std::string_view json);
[[nodiscard]] OnboardUserOperation parse_onboard_user_response(std::string_view json);
[[nodiscard]] TierInfo select_onboard_tier(const LoadCodeAssistResponseData& response);
[[nodiscard]] std::string setup_user(std::string_view access_token,
                                     std::shared_ptr<ui::AuthUI> ui = nullptr,
                                     std::string_view ide_type = "IDE_UNSPECIFIED");

} // namespace core::auth::google_code_assist
