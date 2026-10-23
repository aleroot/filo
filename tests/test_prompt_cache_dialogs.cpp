#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/PromptCacheDialogs.hpp"

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;
using core::agent::PromptCacheAssessment;
using core::agent::PromptCacheWarmth;
using core::llm::PromptCacheProfile;
namespace prompt_cache = tui::prompt_cache;

namespace {

PromptCacheAssessment expired(PromptCacheProfile profile) {
    return {
        .warmth = PromptCacheWarmth::Expired,
        .idle = 2h + 5min,
        .context_tokens = 486'000,
        .profile = profile,
    };
}

PromptCacheAssessment expiring(PromptCacheProfile profile) {
    return {
        .warmth = PromptCacheWarmth::Warm,
        .idle = 4min,
        .remaining = 1min,
        .context_tokens = 486'000,
        .profile = profile,
    };
}

std::vector<std::string> labels(const core::tools::QuestionItem& item) {
    std::vector<std::string> out;
    for (const auto& option : item.options) out.push_back(option.label);
    return out;
}

std::optional<tui::QuestionDialogAnswers> answer(const core::tools::QuestionItem& item,
                                                 std::string_view label) {
    return tui::QuestionDialogAnswers{{item.question, std::string(label)}};
}

} // namespace

TEST_CASE("The cold-send question states the size, the idle time and the evidence",
          "[prompt-cache][dialogs]") {
    const auto documented = prompt_cache::send_question(
        expired(PromptCacheProfile::documented(5min, true)));
    CHECK_THAT(documented.question, ContainsSubstring("486k-token"));
    CHECK_THAT(documented.question, ContainsSubstring("2h 5m"));
    CHECK_THAT(documented.question, ContainsSubstring("has expired"));
    CHECK(labels(documented) == std::vector<std::string>{
        std::string(prompt_cache::kCompactThenSendLabel),
        std::string(prompt_cache::kSendAnywayLabel),
        std::string(prompt_cache::kCancelLabel)});

    const auto estimated = prompt_cache::send_question(
        expired(PromptCacheProfile::estimated(5min)));
    CHECK_THAT(estimated.question, ContainsSubstring("has likely expired"));
    CHECK_THAT(estimated.options[1].description, ContainsSubstring("about 5m"));
}

TEST_CASE("Every cold-send answer maps back to its action, and silence cancels",
          "[prompt-cache][dialogs]") {
    const auto item = prompt_cache::send_question(expired(PromptCacheProfile::documented(5min, true)));
    CHECK(prompt_cache::send_choice(answer(item, prompt_cache::kCompactThenSendLabel))
          == prompt_cache::SendChoice::CompactThenSend);
    CHECK(prompt_cache::send_choice(answer(item, prompt_cache::kSendAnywayLabel))
          == prompt_cache::SendChoice::Send);
    CHECK(prompt_cache::send_choice(answer(item, prompt_cache::kCancelLabel))
          == prompt_cache::SendChoice::Cancel);
    CHECK(prompt_cache::send_choice(std::nullopt) == prompt_cache::SendChoice::Cancel);
}

TEST_CASE("The expiry notice offers keep-warm only where a replay renews the cache",
          "[prompt-cache][dialogs]") {
    const auto renewable = prompt_cache::expiry_question(
        expiring(PromptCacheProfile::documented(5min, true)), true);
    CHECK_THAT(renewable.question, ContainsSubstring("expires in 1m"));
    CHECK(labels(renewable).front() == prompt_cache::kKeepWarmLabel);
    CHECK(prompt_cache::expiry_choice(answer(renewable, prompt_cache::kKeepWarmLabel))
          == prompt_cache::ExpiryChoice::KeepWarm);

    const auto chained = prompt_cache::expiry_question(
        expiring(PromptCacheProfile::estimated(5min)), false);
    CHECK_THAT(chained.question, ContainsSubstring("likely expires in about 1m"));
    const auto chained_labels = labels(chained);
    CHECK(std::ranges::find(chained_labels, std::string(prompt_cache::kKeepWarmLabel))
          == chained_labels.end());
    CHECK(prompt_cache::expiry_choice(answer(chained, prompt_cache::kCompactNowLabel))
          == prompt_cache::ExpiryChoice::CompactNow);
    CHECK(prompt_cache::expiry_choice(std::nullopt) == prompt_cache::ExpiryChoice::LetExpire);
}

TEST_CASE("Keep-warm results say what the provider actually did", "[prompt-cache][dialogs]") {
    const auto profile = PromptCacheProfile::documented(5min, true);
    using Refresh = core::agent::Agent::PromptCacheRefresh;

    CHECK_THAT(prompt_cache::keep_warm_notice(
                   Refresh{.refreshed = true,
                           .usage = {.prompt_tokens = 486'000, .cached_prompt_tokens = 485'000}},
                   profile),
               ContainsSubstring("485k tokens read from the cache"));
    CHECK_THAT(prompt_cache::keep_warm_notice(
                   Refresh{.refreshed = true, .usage = {.prompt_tokens = 486'000}}, profile),
               ContainsSubstring("had already expired"));
    CHECK_THAT(prompt_cache::keep_warm_notice(Refresh{.error = "HTTP 529"}, profile),
               ContainsSubstring("HTTP 529"));
}

TEST_CASE("Automatic keep-warm results count down to the next question",
          "[prompt-cache][dialogs]") {
    const auto profile = PromptCacheProfile::documented(5min, true);
    using Refresh = core::agent::Agent::PromptCacheRefresh;
    const Refresh hit{.refreshed = true,
                      .usage = {.prompt_tokens = 200'000, .cached_prompt_tokens = 199'000}};

    CHECK_THAT(prompt_cache::keep_warm_notice(hit, profile, prompt_cache::AutoRenewal{3, 11}),
               ContainsSubstring("(automatic, 3 of 11)."));
    CHECK_THAT(prompt_cache::keep_warm_notice(hit, profile, prompt_cache::AutoRenewal{11, 11}),
               ContainsSubstring("11 of 11; next time Filo asks"));
    CHECK_THAT(prompt_cache::keep_warm_notice(Refresh{.error = "HTTP 529"}, profile,
                                              prompt_cache::AutoRenewal{1, 11}),
               ContainsSubstring("Filo will ask before renewing it again"));
    CHECK_THAT(prompt_cache::keep_warm_notice(hit, profile),
               !ContainsSubstring("automatic"));
}
