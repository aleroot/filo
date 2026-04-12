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
    std::string project_id;
};

[[nodiscard]] std::string code_assist_endpoint();
[[nodiscard]] std::optional<std::string> configured_project_override();
[[nodiscard]] LoadCodeAssistResponseData parse_load_code_assist_response(std::string_view json);
[[nodiscard]] OnboardUserOperation parse_onboard_user_response(std::string_view json);
[[nodiscard]] TierInfo select_onboard_tier(const LoadCodeAssistResponseData& response);
[[nodiscard]] std::string setup_user(std::string_view access_token,
                                     std::shared_ptr<ui::AuthUI> ui = nullptr);

} // namespace core::auth::google_code_assist
