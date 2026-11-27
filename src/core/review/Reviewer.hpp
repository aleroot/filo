#pragma once

#include "../config/ConfigManager.hpp"

#include <expected>
#include <memory>
#include <string>
#include <string_view>

namespace core::agent {
class Agent;
}

namespace core::llm {
class LLMProvider;
}

namespace core::review {

/// The subagent profile whose provider/model also run /review and the
/// `review` tool, so one setting decides which model reviews.
inline constexpr std::string_view kReviewerProfile = "code-review";

struct ReviewerSelection {
    std::string provider;
    std::string model;
};

/// Pure resolution, free of provider state:
///   - `subagents.code-review.provider` / `.model` win when set;
///   - a provider without a model uses that provider's configured model;
///   - anything unset falls back to the conversation's own provider/model.
/// Fails only when a different provider is named and no model can be found
/// for it; without configuration it always returns the conversation's own.
[[nodiscard]] std::expected<ReviewerSelection, std::string>
select_reviewer(const core::config::AppConfig& config,
                std::string_view session_provider,
                std::string_view session_model);

/// What the review UI shows: empty when the conversation's own model reviews,
/// the model when only the model differs, "provider · model" otherwise.
[[nodiscard]] std::string reviewer_label(const ReviewerSelection& reviewer,
                                         std::string_view session_provider,
                                         std::string_view session_model);

struct Reviewer {
    std::shared_ptr<core::llm::LLMProvider> provider;
    ReviewerSelection selection;
    std::string label;
};

/// Resolves the reviewer from the loaded configuration and acquires its
/// provider (the conversation's own when the provider is the same).
[[nodiscard]] std::expected<Reviewer, std::string>
resolve_reviewer(const core::agent::Agent& session_agent);

/// An isolated agent of `session_agent` that runs on the reviewer's
/// provider and model. Null when the reviewer has no provider.
[[nodiscard]] std::shared_ptr<core::agent::Agent>
make_review_agent(const core::agent::Agent& session_agent, const Reviewer& reviewer);

} // namespace core::review
