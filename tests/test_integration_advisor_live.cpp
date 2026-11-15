#include <catch2/catch_test_macros.hpp>

#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ClaudeBilling.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/config/ConfigManager.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace core::llm;

namespace {

/// The provider's own utilization windows, as percentages per label.
using Windows = std::map<std::string, float>;

/// One request's outcome: the answer, any error, the advisor events, and the
/// utilization windows the response itself reported.
struct ProbeOutcome {
    std::string answer;
    std::string error;
    std::vector<AdvisorEvent> advisor_events;
    Windows response_windows;
    std::string stop_reason;
    /// The assistant turn's signed reasoning, as the provider returned it.
    /// Replaying it is what makes the next request's prefix check run.
    std::vector<ContinuationItem> continuation_items;
};

[[nodiscard]] std::string usage_url(std::string_view base_url) {
    std::string url{base_url};
    while (!url.empty() && url.back() == '/') url.pop_back();
    return url + "/api/oauth/usage";
}

/// Reads the subscription's utilization windows. Only labels and percentages
/// are reported; nothing about the account is printed.
[[nodiscard]] Windows read_usage_windows(const core::auth::AuthInfo& auth,
                                         std::string_view base_url) {
    Windows windows;
    cpr::Header headers;
    for (const auto& [key, value] : auth.headers) headers[key] = value;
    headers["Accept"] = "application/json";
    const cpr::Response response = cpr::Get(cpr::Url{usage_url(base_url)}, headers,
                                            cpr::Timeout{5000});
    if (response.status_code != 200 || response.text.empty()) return windows;

    simdjson::dom::parser parser;
    simdjson::padded_string padded{response.text};
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return windows;

    simdjson::dom::object object;
    if (document.get_object().get(object) != simdjson::SUCCESS) return windows;
    for (const auto field : object) {
        simdjson::dom::element window;
        if (field.value.get(window) != simdjson::SUCCESS) continue;
        double utilization = 0.0;
        if (window["utilization"].get(utilization) != simdjson::SUCCESS) continue;
        windows.emplace(std::string(field.key), static_cast<float>(utilization * 100.0));
    }
    return windows;
}

/// The names of the windows the usage endpoint reports. Names only: no
/// account data leaves the process.
[[nodiscard]] std::vector<std::string> read_usage_window_names(
    const core::auth::AuthInfo& auth, std::string_view base_url) {
    std::vector<std::string> names;
    cpr::Header headers;
    for (const auto& [key, value] : auth.headers) headers[key] = value;
    headers["Accept"] = "application/json";
    const cpr::Response response = cpr::Get(cpr::Url{usage_url(base_url)}, headers,
                                            cpr::Timeout{5000});
    if (response.status_code != 200 || response.text.empty()) return names;

    simdjson::dom::parser parser;
    simdjson::padded_string padded{response.text};
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return names;
    simdjson::dom::object object;
    if (document.get_object().get(object) != simdjson::SUCCESS) return names;
    for (const auto field : object) names.emplace_back(field.key);
    return names;
}

/// The members of the usage endpoint's billing sections, with the percentages
/// they carry. Names and percentages only.
[[nodiscard]] std::string read_usage_billing_sections(
    const core::auth::AuthInfo& auth, std::string_view base_url) {
    static constexpr std::array<std::string_view, 5> kSections{
        "extra_usage", "spend", "limits", "seven_day_breakdown", "weekly_scoped_shares"};

    cpr::Header headers;
    for (const auto& [key, value] : auth.headers) headers[key] = value;
    headers["Accept"] = "application/json";
    const cpr::Response response = cpr::Get(cpr::Url{usage_url(base_url)}, headers,
                                            cpr::Timeout{5000});
    if (response.status_code != 200 || response.text.empty()) return {};

    simdjson::dom::parser parser;
    simdjson::padded_string padded{response.text};
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return {};
    simdjson::dom::object object;
    if (document.get_object().get(object) != simdjson::SUCCESS) return {};

    std::string report;
    for (const auto section : kSections) {
        simdjson::dom::element value;
        if (object[section].get(value) != simdjson::SUCCESS) continue;
        std::string members;
        simdjson::dom::object members_object;
        if (value.get_object().get(members_object) == simdjson::SUCCESS) {
            for (const auto member : members_object) {
                if (!members.empty()) members += ", ";
                members += member.key;
                double utilization = 0.0;
                if (member.value["utilization"].get(utilization) == simdjson::SUCCESS) {
                    members += std::format("={:.2f}%", utilization * 100.0);
                }
            }
        }
        if (!report.empty()) report += "; ";
        report += std::format("{}[{}]", section, members);
    }
    return report;
}

[[nodiscard]] std::string format_windows(const Windows& windows) {
    std::string text;
    for (const auto& [label, percent] : windows) {
        if (!text.empty()) text += ' ';
        text += std::format("{}={:.2f}%", label, percent);
    }
    return text;
}

/// What one window moved between two readings.
[[nodiscard]] std::string format_window_deltas(const Windows& before,
                                               const Windows& after) {
    std::string text;
    for (const auto& [label, percent] : after) {
        const auto it = before.find(label);
        if (it == before.end()) continue;
        const double delta = percent - it->second;
        if (std::abs(delta) < 0.005) continue;
        if (!text.empty()) text += ' ';
        text += std::format("{}={:+.2f}%", label, delta);
    }
    if (text.empty()) return "no window moved";
    return text;
}

/// Runs one request and reports what came back.
[[nodiscard]] ProbeOutcome run_request(const std::shared_ptr<LLMProvider>& provider,
                                       ChatRequest request) {
    ProbeOutcome outcome;
    std::mutex mutex;
    std::condition_variable finished_signal;
    bool finished = false;
    provider->stream_response(request, [&](const StreamChunk& chunk) {
        std::lock_guard lock(mutex);
        if (chunk.advisor_event) outcome.advisor_events.push_back(*chunk.advisor_event);
        for (const auto& item : chunk.continuation_items) {
            outcome.continuation_items.push_back(item);
        }
        if (!chunk.content.empty()) {
            if (chunk.is_error) outcome.error += chunk.content;
            else outcome.answer += chunk.content;
        }
        if (chunk.is_final) {
            outcome.stop_reason = chunk.stop_reason;
            finished = true;
            finished_signal.notify_all();
        }
    });
    std::unique_lock lock(mutex);
    finished_signal.wait_for(lock, std::chrono::minutes(3), [&] { return finished; });
    lock.unlock();

    for (const auto& window : provider->get_last_rate_limit_info().usage_windows) {
        outcome.response_windows.emplace(window.label, window.utilization * 100.0f);
    }
    return outcome;
}

/// The provider's answer to the same request, with the setting that reports
/// a prefix mismatch and with the one that drops the invalidated reasoning.
struct DropCounterfactual {
    long with_drop_status = 0;
    long with_error_status = 0;
    long without_setting_status = 0;
    std::string with_error_body = {};
};

/// Sends the same request three times: once as the wire writes it, once with
/// the setting that reports a prefix mismatch, and once with no setting at
/// all.
///
/// The provider rejects the second one, which proves the request's prefix
/// really moved, and accepts the first one, which proves the setting that
/// drops the invalidated reasoning is what lets the switch through. The third
/// shows what a wire that asks for nothing gets: for a model whose mismatch
/// is an error the request fails, for a model whose mismatch is not the
/// provider may answer from reasoning about content that has changed.
[[nodiscard]] DropCounterfactual run_drop_counterfactual(
    std::string_view base_url,
    const core::auth::AuthInfo& auth,
    const ChatRequest& request) {
    DropCounterfactual result;

    core::llm::protocols::ClaudeProtocol wire;
    auto prepared = request;
    wire.prepare_request(prepared);
    const auto advisor_json = wire.advisor_support()->tool_definition_json(
        *request.advisor);
    const auto body = core::llm::protocols::ClaudeSerializer::serialize_unsigned(
        prepared,
        core::llm::protocols::kAnthropicDefaultMaxTokens,
        {},
        core::llm::PromptCacheTtl::Standard,
        nullptr,
        advisor_json);

    cpr::Header headers;
    for (const auto& [key, value] : wire.build_headers(auth)) headers[key] = value;

    std::string url{base_url};
    while (!url.empty() && url.back() == '/') url.pop_back();
    url += "/v1/messages";

    const auto send = [&](std::string payload) {
        core::llm::claude::sign_body(payload);
        const auto response = cpr::Post(cpr::Url{url}, headers,
                                        cpr::Body{payload},
                                        cpr::Timeout{120000});
        return std::pair{response.status_code, response.text};
    };

    // The request as the wire writes it: the provider drops the reasoning
    // the changed tool list invalidated and answers.
    const auto with_drop = send(body.payload);
    result.with_drop_status = with_drop.first;

    // The same request with the setting that reports the mismatch. The
    // provider rejects it: the request's prefix really moved.
    std::string with_error = body.payload;
    const auto setting_at = with_error.find(R"("prefix_mismatch_behavior":"drop_block")");
    if (setting_at != std::string::npos) {
        with_error.replace(setting_at, std::string(R"("prefix_mismatch_behavior":"drop_block")").size(),
                           R"("prefix_mismatch_behavior":"error")");
    }
    const auto rejected = send(with_error);
    result.with_error_status = rejected.first;
    result.with_error_body = rejected.second;

    // The same request with no setting at all.
    std::string without_setting = body.payload;
    const std::string setting = R"(,"block_binding":{"prefix_mismatch_behavior":"drop_block"})";
    const auto at = without_setting.find(setting);
    if (at != std::string::npos) without_setting.erase(at, setting.size());
    const auto plain = send(without_setting);
    result.without_setting_status = plain.first;

    return result;
}

} // namespace

/**
 * @brief A live probe of Anthropic's advisor server tool.
 *
 * Skipped unless FILO_LIVE_ADVISOR=1. It answers the three questions the
 * offline tests cannot:
 *
 *  1. Does a Claude subscription account accept the advisor tool at all?
 *  2. Does Anthropic accept Filo's Claude Code-shaped request (signed body,
 *     `x-claude-code-request-class`) together with the advisor declaration?
 *  3. Does a Fable advisor move the plan's weekly window, or is it billed to
 *     usage credits (the behaviour the "advisor does not eat my weekly Fable
 *     limit" report describes)?
 *
 * The probe sends one short request and reads the utilization windows before
 * and after, so it costs a few thousand tokens.
 */
TEST_CASE("A live Anthropic advisor call runs and reports its outcome",
          "[advisor][anthropic][live]") {
    if (std::getenv("FILO_LIVE_ADVISOR") == nullptr) {
        SKIP("Set FILO_LIVE_ADVISOR=1 to run the live advisor probe.");
    }

    core::config::ProviderConfig config;
    config.api_type = core::config::ApiType::Anthropic;
    config.auth_type = "oauth_claude";
    config.model = "claude-opus-5-5";
    auto provider = ProviderFactory::create_provider("claude", config);
    REQUIRE(provider != nullptr);

    const auto metadata = provider->metadata();
    REQUIRE(metadata.has_value());
    REQUIRE(metadata->credential_source != nullptr);
    const auto auth = metadata->credential_source->get_auth();

    const auto before = read_usage_windows(auth, metadata->base_url);
    std::printf("[probe] windows before: %s\n", format_windows(before).c_str());
    {
        std::string names;
        for (const auto& name : read_usage_window_names(auth, metadata->base_url)) {
            if (!names.empty()) names += ", ";
            names += name;
        }
        std::printf("[probe] usage endpoint reports windows: %s\n", names.c_str());
    }
    std::printf("[probe] usage billing sections: %s\n",
                read_usage_billing_sections(auth, metadata->base_url).c_str());

    ChatRequest request;
    request.model = "claude-opus-5-5";
    request.messages.push_back(Message{
        .role = "user",
        .content = "Consult the advisor before you continue: what is the one design "
                   "decision in a two-worker work queue that is easiest to get wrong?",
    });

    // Control: the same request without an advisor, so the advisor's own
    // contribution to the plan's windows can be told apart from the executor's.
    const auto control = run_request(provider, request);
    const auto after_control = read_usage_windows(auth, metadata->base_url);
    std::printf("[probe] control answer bytes=%zu response windows: %s\n",
                control.answer.size(), format_windows(control.response_windows).c_str());
    std::printf("[probe] windows after control: %s\n",
                format_windows(after_control).c_str());

    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};
    const auto with_advisor = run_request(provider, request);
    const auto after_advisor = read_usage_windows(auth, metadata->base_url);

    for (const auto& event : with_advisor.advisor_events) {
        std::printf("[probe] advisor state=%s model=%s error=%s readable=%d\n",
                    std::string(to_string(event.state)).c_str(),
                    event.advisor_model.c_str(), event.error_code.c_str(),
                    static_cast<int>(event.guidance_readable));
    }
    if (!with_advisor.error.empty()) {
        std::printf("[probe] error: %s\n", with_advisor.error.c_str());
    }
    std::printf("[probe] advisor request response windows: %s\n",
                format_windows(with_advisor.response_windows).c_str());
    // The advisor's own contribution, as far as the response reports it: the
    // windows it moved relative to the control request.
    std::printf("[probe] advisor vs control windows: %s\n",
                format_window_deltas(control.response_windows,
                                     with_advisor.response_windows).c_str());
    std::printf("[probe] windows after advisor: %s\n",
                format_windows(after_advisor).c_str());
    std::printf("[probe] answer: %s\n", with_advisor.answer.substr(0, 400).c_str());

    // The mid-session switch: the same conversation now declares a
    // different advisor. That edits the request's tool list, which
    // invalidates the reasoning the provider bound to the request prefix,
    // so the wire has to tell the provider to drop it.
    auto switched = request;
    switched.advisor = AdvisorSpec{.model = "claude-opus-5-5"};
    switched.messages.push_back(Message{
        .role = "assistant",
        .content = with_advisor.answer,
    });
    switched.messages.push_back(Message{
        .role = "user",
        .content = "Now consult the other advisor before you answer.",
    });
    const auto after_switch = run_request(provider, switched);
    for (const auto& event : after_switch.advisor_events) {
        std::printf("[probe] switched advisor state=%s model=%s error=%s readable=%d\n",
                    std::string(to_string(event.state)).c_str(),
                    event.advisor_model.c_str(), event.error_code.c_str(),
                    static_cast<int>(event.guidance_readable));
    }
    if (!after_switch.error.empty()) {
        std::printf("[probe] switch error: %s\n", after_switch.error.c_str());
    }
    std::printf("[probe] switch answer bytes=%zu\n", after_switch.answer.size());

    REQUIRE(control.error.empty());
    REQUIRE(with_advisor.error.empty());
    REQUIRE_FALSE(with_advisor.advisor_events.empty());
    // The advisor ran: a pending call and its result.
    bool saw_pending = false;
    bool saw_reviewed = false;
    for (const auto& event : with_advisor.advisor_events) {
        saw_pending |= event.state == AdvisorCallState::Pending;
        saw_reviewed |= event.state == AdvisorCallState::Reviewed;
    }
    CHECK(saw_pending);
    CHECK(saw_reviewed);

    // The switch is accepted: the conversation continues with the other
    // advisor, and the provider did not reject the edited request.
    CHECK(after_switch.error.empty());
    bool switch_saw_advisor = false;
    for (const auto& event : after_switch.advisor_events) {
        switch_saw_advisor |= event.state == AdvisorCallState::Pending
            || event.state == AdvisorCallState::Reviewed;
    }
    CHECK(switch_saw_advisor);
}

/**
 * @brief The live proof that a switch survives the reasoning it invalidates.
 *
 * A conversation that replays the provider's signed thinking runs a prefix
 * check: the advisor switch edits the request's tool list, which moves the
 * prefix the reasoning was bound to. The Claude Code wire tells the provider
 * to drop the invalidated reasoning, so the switch goes through. A wire that
 * does not asks the provider to reject the request.
 *
 * The probe sends the same switched request three times: as the wire writes
 * it, with the setting that reports a prefix mismatch, and with no setting at
 * all. The provider's own answers are the proof: the mismatch is reported
 * (the tools list really moved), and the drop setting is what makes the
 * request go through.
 */
TEST_CASE("A live advisor switch survives the reasoning it invalidates",
          "[advisor][anthropic][live][switch]") {
    if (std::getenv("FILO_LIVE_ADVISOR") == nullptr) {
        SKIP("Set FILO_LIVE_ADVISOR=1 to run the live advisor probe.");
    }

    core::config::ProviderConfig config;
    config.api_type = core::config::ApiType::Anthropic;
    config.auth_type = "oauth_claude";
    config.model = "claude-opus-5-5";
    auto provider = ProviderFactory::create_provider("claude", config);
    REQUIRE(provider != nullptr);

    const auto metadata = provider->metadata();
    REQUIRE(metadata.has_value());
    REQUIRE(metadata->credential_source != nullptr);

    // The conversation: one turn that produces reasoning, then a turn that
    // changes the advisor.
    ChatRequest request;
    request.model = "claude-opus-5-5";
    request.messages.push_back(Message{
        .role = "user",
        .content = "Two workers pop items from one queue. Worker A pops an item "
                   "and reads it; worker B pops the next one. Give the invariant "
                   "that stops both from processing the same item, and prove it.",
    });
    request.advisor = AdvisorSpec{.model = "claude-fable-5-2"};
    // The turn has to produce signed reasoning: the prefix check runs on the
    // thinking blocks the provider signs, and a turn that does not think
    // leaves nothing to invalidate. The model chooses when to think, so the
    // probe asks again until it does.
    ProbeOutcome first;
    bool has_signed_reasoning = false;
    for (int attempt = 0; attempt < 3 && !has_signed_reasoning; ++attempt) {
        first = run_request(provider, request);
        REQUIRE(first.error.empty());
        for (const auto& item : first.continuation_items) {
            has_signed_reasoning |= item.kind == "thinking";
        }
        if (!has_signed_reasoning) {
            std::printf("[probe] attempt %d produced no signed reasoning; asking again\n",
                        attempt + 1);
        }
    }
    REQUIRE(has_signed_reasoning);
    std::printf("[probe] replayed reasoning blocks: %zu\n",
                first.continuation_items.size());

    // The switch: the same conversation, the other advisor. The tool list
    // changed, so the reasoning the provider bound to the old prefix is
    // invalidated.
    auto switched = request;
    switched.advisor = AdvisorSpec{.model = "claude-opus-5-5"};
    Message assistant{.role = "assistant", .content = first.answer};
    assistant.continuation_items = first.continuation_items;
    switched.messages.push_back(assistant);
    switched.messages.push_back(Message{
        .role = "user",
        .content = "Now consult the other advisor before you answer.",
    });

    const auto after_switch = run_request(provider, switched);
    for (const auto& event : after_switch.advisor_events) {
        std::printf("[probe] switched advisor state=%s model=%s error=%s readable=%d\n",
                    std::string(to_string(event.state)).c_str(),
                    event.advisor_model.c_str(),
                    event.error_code.c_str(),
                    static_cast<int>(event.guidance_readable));
    }
    if (!after_switch.error.empty()) {
        std::printf("[probe] switch error: %s\n", after_switch.error.c_str());
    }

    // The counterfactual, on the same wire: the request the provider accepts
    // with the drop setting, and the same request with the setting that
    // reports the mismatch. The provider rejects the second one, so the
    // changed tool list really moved the prefix the reasoning was bound to.
    const auto counterfactual = run_drop_counterfactual(
        metadata->base_url, metadata->credential_source->get_auth(), switched);
    std::printf("[probe] with drop: HTTP %ld, with error: HTTP %ld, without setting: HTTP %ld\n",
                counterfactual.with_drop_status, counterfactual.with_error_status,
                counterfactual.without_setting_status);
    std::printf("[probe] with error body: %s\n",
                counterfactual.with_error_body.substr(0, 400).c_str());
    CHECK(counterfactual.with_drop_status == 200);
    CHECK(counterfactual.with_error_status == 400);
    // The provider names the reason: the tools list is what moved, and the
    // tools list is what the advisor declaration changed.
    CHECK(counterfactual.with_error_body.find("tools") != std::string::npos);

    // The switch goes through: the wire told the provider to drop the
    // reasoning the changed tool list invalidated.
    CHECK(after_switch.error.empty());
    bool switch_saw_advisor = false;
    for (const auto& event : after_switch.advisor_events) {
        switch_saw_advisor |= event.state == AdvisorCallState::Pending
            || event.state == AdvisorCallState::Reviewed;
    }
    CHECK(switch_saw_advisor);
}

