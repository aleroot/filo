#pragma once

#include "MergePolicy.hpp"
#include "Types.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {
class Agent;
}

namespace core::review {

// Result of reviewing one git checkout. An empty diff never calls the model.
// `ok` is false when the campaign did not finish; `findings` may still hold
// whatever units completed, and `failed_groups` names the units that did not.
struct WorkspaceReview {
    bool ok = false;
    bool empty = false;
    ClassifiedFindings findings;
    std::vector<FailedGroup> failed_groups;
    std::string repository;
    /// The reviewing model when it is not the conversation's own; empty otherwise.
    std::string reviewer;
    std::string error;
};

// `target` is "uncommitted", "staged", or "against" (`against_ref` required).
// The isolated reviewer is pointed at `repository`, so its read tools open
// that tree rather than whichever checkout the session was using.
[[nodiscard]] WorkspaceReview review_workspace(
    const std::shared_ptr<core::agent::Agent>& session_agent,
    const std::filesystem::path& repository,
    std::string_view target,
    std::string_view against_ref = {},
    const std::function<void(const Progress&)>& on_progress = {},
    const std::function<bool()>& cancellation_requested = {});

[[nodiscard]] std::string render_review_json(const WorkspaceReview& review);

} // namespace core::review
