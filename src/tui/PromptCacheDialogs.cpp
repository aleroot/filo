#include "PromptCacheDialogs.hpp"

#include "ActivityTimer.hpp"
#include "core/budget/TokenUsageFormatters.hpp"

#include <format>

namespace tui::prompt_cache {

namespace {

[[nodiscard]] std::string tokens(std::int64_t count) {
    return core::budget::formatters::CompactTokenCountFormatter{}.format(count);
}

[[nodiscard]] std::string duration(std::chrono::seconds value) {
    return format_elapsed_compact(value, ElapsedFormat::humanized);
}

/// Only a documented lifetime is stated as fact; an estimated or observed
/// one is what the cache usually does, not what it promises.
[[nodiscard]] bool documented(const core::llm::PromptCacheProfile& profile) {
    return profile.basis == core::llm::PromptCacheBasis::Documented;
}

/// @p value as stated for @p profile: exact when documented, approximate
/// otherwise.
[[nodiscard]] std::string duration(const core::llm::PromptCacheProfile& profile,
                                   std::chrono::seconds value) {
    return documented(profile) ? duration(value) : std::format("about {}", duration(value));
}

[[nodiscard]] std::string lifetime(const core::llm::PromptCacheProfile& profile) {
    return duration(profile, profile.ttl);
}

[[nodiscard]] std::string has_expired(const core::llm::PromptCacheProfile& profile) {
    return documented(profile) ? "has expired" : "has likely expired";
}

[[nodiscard]] core::tools::QuestionOption option(std::string_view label,
                                                 std::string description) {
    return {.label = std::string(label), .description = std::move(description)};
}

[[nodiscard]] std::optional<std::string_view> chosen_label(
    const std::optional<QuestionDialogAnswers>& answers) {
    if (!answers.has_value() || answers->empty()) {
        return std::nullopt;
    }
    return answers->front().second;
}

} // namespace

core::tools::QuestionItem send_question(
    const core::agent::PromptCacheAssessment& assessment) {
    const std::string size = tokens(assessment.context_tokens);
    return core::tools::QuestionItem{
        .question = std::format(
            "This {}-token session sat idle {}, so its prompt cache {}. Send this prompt?",
            size, duration(assessment.idle), has_expired(assessment.profile)),
        .header = "Cold cache",
        .options = {
            option(kCompactThenSendLabel,
                   "Summarize the conversation first, so this and every later prompt re-read far less"),
            option(kSendAnywayLabel,
                   std::format("Re-send all {} tokens uncached; they stay cached for {} after each request",
                               size, lifetime(assessment.profile))),
            option(kCancelLabel, "Send nothing and put the prompt back in the composer"),
        },
    };
}

SendChoice send_choice(const std::optional<QuestionDialogAnswers>& answers) {
    const auto label = chosen_label(answers);
    if (label == kCompactThenSendLabel) return SendChoice::CompactThenSend;
    if (label == kSendAnywayLabel) return SendChoice::Send;
    return SendChoice::Cancel;
}

core::tools::QuestionItem expiry_question(
    const core::agent::PromptCacheAssessment& assessment,
    bool can_keep_warm) {
    const std::string size = tokens(assessment.context_tokens);
    core::tools::QuestionItem item{
        .question = std::format(
            "The prompt cache of this {}-token session {} in {}. Keep it?",
            size,
            documented(assessment.profile) ? "expires" : "likely expires",
            duration(assessment.profile, assessment.remaining)),
        .header = "Cache expiring",
    };
    if (can_keep_warm) {
        item.options.push_back(option(
            kKeepWarmLabel,
            std::format("Re-read the context from the cache now, renewing it for another {}",
                        lifetime(assessment.profile))));
    }
    item.options.push_back(option(
        kCompactNowLabel, "Summarize the conversation now, so the next prompt starts small"));
    item.options.push_back(option(
        kLetExpireLabel,
        std::format("The next prompt will ask before re-sending all {} tokens", size)));
    return item;
}

ExpiryChoice expiry_choice(const std::optional<QuestionDialogAnswers>& answers) {
    const auto label = chosen_label(answers);
    if (label == kKeepWarmLabel) return ExpiryChoice::KeepWarm;
    if (label == kCompactNowLabel) return ExpiryChoice::CompactNow;
    return ExpiryChoice::LetExpire;
}

std::string expired_notice(const core::agent::PromptCacheAssessment& assessment) {
    return std::format(
        "\n\xe2\x84\xb9  The prompt cache of this {}-token session {}. "
        "The next prompt will ask before re-sending it.\n",
        tokens(assessment.context_tokens), has_expired(assessment.profile));
}

std::string keep_warm_notice(const core::agent::Agent::PromptCacheRefresh& refresh,
                             const core::llm::PromptCacheProfile& profile,
                             std::optional<AutoRenewal> renewal) {
    if (!refresh.refreshed) {
        return std::format(
            "\n\xe2\x9c\x97  Could not keep the prompt cache warm: {}{}\n",
            refresh.error.empty() ? std::string("the provider returned an error") : refresh.error,
            renewal ? "; Filo will ask before renewing it again." : "");
    }
    // An automatic renewal says which one it was, and the last one says that
    // the next decision is the user's again.
    const std::string automatic =
        !renewal ? std::string{}
                 : std::format(" (automatic, {} of {}{})", renewal->number, renewal->budget,
                               renewal->number >= renewal->budget ? "; next time Filo asks" : "");
    const auto& usage = refresh.usage;
    if (usage.prompt_tokens > 0 && usage.cached_prompt_tokens == 0) {
        // The provider answered without reading its cache: it was already
        // gone, and this request paid to write it again.
        return std::format(
            "\n\xe2\x84\xb9  The prompt cache had already expired; {} tokens were re-sent "
            "and stay cached for {}{}.\n",
            tokens(usage.prompt_tokens), lifetime(profile), automatic);
    }
    if (usage.cached_prompt_tokens > 0) {
        return std::format(
            "\n\xe2\x9c\x93  Prompt cache kept warm: {} tokens read from the cache, "
            "renewed for {}{}.\n",
            tokens(usage.cached_prompt_tokens), lifetime(profile), automatic);
    }
    return std::format("\n\xe2\x9c\x93  Prompt cache kept warm for {}{}.\n",
                       lifetime(profile), automatic);
}

std::string compaction_skipped_notice() {
    return "\n\xe2\x84\xb9  The conversation was not compacted; sending the prompt as it is.\n";
}

} // namespace tui::prompt_cache
