#pragma once

#include "Types.hpp"

#include "../context/SteeringLoader.hpp"

#include <filesystem>
#include <cstddef>
#include <string_view>
#include <vector>

namespace core::review {

[[nodiscard]] ReviewSteeringContext load_review_steering_guidance(
    const GitSnapshot& snapshot,
    const std::vector<std::filesystem::path>& workspace_roots,
    const core::context::SteeringPolicy& policy);

/// Keep only model-supplied steering citations whose source applies to the
/// finding path and whose whitespace-normalized quote exists in the loaded source.
[[nodiscard]] std::size_t validate_steering_references(
    Report& report,
    const ReviewGroup& group,
    const ReviewSteeringContext& steering,
    std::string_view worktree_root);

} // namespace core::review
