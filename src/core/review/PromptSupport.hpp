#pragma once

#include "Types.hpp"

#include <string>

namespace core::review::prompt_detail {

void append_steering_context(std::string& prompt,
                             const CampaignInput& input,
                             const ReviewGroup* group = nullptr);

} // namespace core::review::prompt_detail
