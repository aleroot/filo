#pragma once

// Wording of the prompt-cache questions and notices, and the mapping from a
// chosen label back to an action. Kept apart from PromptCacheWatch so the
// text is written once, read once, and testable without threads or a TUI.

#include "QuestionDialogController.hpp"
#include "core/agent/Agent.hpp"
#include "core/agent/PromptCacheGuard.hpp"
#include "core/tools/AskUserQuestionTool.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace tui::prompt_cache {

/// What to do with a prompt whose heavy cache has expired.
enum class SendChoice {
    Send,
    CompactThenSend,
    Cancel,
};

/// What to do with an idle heavy cache that is about to expire.
enum class ExpiryChoice {
    KeepWarm,
    CompactNow,
    LetExpire,
};

inline constexpr std::string_view kCompactThenSendLabel = "Compact, then send";
inline constexpr std::string_view kSendAnywayLabel = "Send anyway";
inline constexpr std::string_view kCancelLabel = "Cancel";
inline constexpr std::string_view kKeepWarmLabel = "Keep warm";
inline constexpr std::string_view kCompactNowLabel = "Compact now";
inline constexpr std::string_view kLetExpireLabel = "Let it expire";

/// Asked before a prompt re-writes an expired heavy cache.
[[nodiscard]] core::tools::QuestionItem send_question(
    const core::agent::PromptCacheAssessment& assessment);

/// Dismissing the question cancels: nothing is sent without an answer.
[[nodiscard]] SendChoice send_choice(
    const std::optional<QuestionDialogAnswers>& answers);

/// Offered shortly before an idle heavy cache expires. "Keep warm" is listed
/// only when @p can_keep_warm.
[[nodiscard]] core::tools::QuestionItem expiry_question(
    const core::agent::PromptCacheAssessment& assessment,
    bool can_keep_warm);

/// Dismissing the notice lets the cache expire.
[[nodiscard]] ExpiryChoice expiry_choice(
    const std::optional<QuestionDialogAnswers>& answers);

/// Transcript line when an unanswered notice outlived the cache.
[[nodiscard]] std::string expired_notice(
    const core::agent::PromptCacheAssessment& assessment);

/// Which automatic renewal a keep-warm request was, out of how many.
struct AutoRenewal {
    int number = 0;
    int budget = 0;
};

/// Transcript line reporting a keep-warm request; @p renewal is set when it
/// ran without asking.
[[nodiscard]] std::string keep_warm_notice(
    const core::agent::Agent::PromptCacheRefresh& refresh,
    const core::llm::PromptCacheProfile& profile,
    std::optional<AutoRenewal> renewal = std::nullopt);

/// Transcript line when "Compact, then send" could not compact.
[[nodiscard]] std::string compaction_skipped_notice();

} // namespace tui::prompt_cache
