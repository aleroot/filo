#pragma once

#include "Types.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::review {

[[nodiscard]] std::string uncommitted_task();
[[nodiscard]] std::string staged_task();
[[nodiscard]] std::string base_branch_task(std::string_view branch,
                                           std::string_view merge_base_sha);
[[nodiscard]] std::string base_branch_backup_task(std::string_view branch);
[[nodiscard]] std::string commit_task(std::string_view sha, std::string_view title);

[[nodiscard]] std::string build_plan_prompt(const CampaignInput& input,
                                            const ReviewGroup& group);

[[nodiscard]] std::string build_review_prompt(const CampaignInput& input,
                                              const ReviewGroup& group,
                                              std::span<const RiskItem> risks,
                                              bool tools_available = false);

[[nodiscard]] std::string_view review_rubric() noexcept;

} // namespace core::review
