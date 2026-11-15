#pragma once

#include "../Advisor.hpp"
#include "../AdvisorSupport.hpp"
#include "../ClaudeModelTraits.hpp"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::llm::protocols::anthropic {

/// Anthropic's advisor server tool: the executor asks the provider to run a
/// second model inside the same turn. The tool is declared in the request's
/// `tools` array; the provider answers with a `server_tool_use` block and an
/// `advisor_tool_result` block.
inline constexpr std::string_view kAdvisorToolType = "advisor_20260301";
inline constexpr std::string_view kAdvisorToolName = "advisor";
/// The beta flag the declaration requires.
inline constexpr std::string_view kAdvisorBeta = "advisor-tool-2026-03-01";
/// The smallest advisor output cap the API accepts.
inline constexpr int kAdvisorMinOutputTokens = 1024;

/// The concrete model id behind a Claude alias, in the Messages API's namespace.
[[nodiscard]] std::string resolve_advisor_model(std::string_view model);

/// The advisor's declaration, as the Messages API expects it. Empty when the
/// advisor names no model.
[[nodiscard]] std::string advisor_tool_definition(const AdvisorSpec& advisor);

/// The feature flags the capability needs. Anthropic names them beta flags.
[[nodiscard]] std::vector<std::string_view> advisor_feature_flags();

/// Whether @p advisor_model may advise @p executor_model.
[[nodiscard]] bool advisor_pairing_supported(std::string_view executor_model,
                                             std::string_view advisor_model);

/// Why the pairing is refused; empty when it is supported.
[[nodiscard]] std::string advisor_pairing_refusal(std::string_view executor_model,
                                                   std::string_view advisor_model);

/// The advisors @p executor_model accepts, as concrete model ids.
[[nodiscard]] std::vector<std::string> advisor_choices_for(std::string_view executor_model);

/// Whether a rejected response refuses the advisor rather than the request.
///
/// Anthropic validates the pairing again on its side, and an account that
/// cannot bill the advisor (for example a Fable advisor without the consent
/// to bill Fable to usage credits) is refused there. The client cannot know
/// that in advance, so it reads the answer.
[[nodiscard]] bool advisor_refused_by_response(int status_code, std::string_view body);

/// Whether the conversation carries reasoning a prefix check would reject.
///
/// Anthropic binds a thinking block to the request prefix that produced it,
/// so a later change to the request's tools invalidates that reasoning. A
/// conversation with none of it can change its tools freely.
[[nodiscard]] bool conversation_carries_bound_reasoning(
    std::span<const core::llm::Message> conversation);

/// The reason the provider gives for refusing the advisor, taken from the
/// error body it sent back. Empty when it gave none.
[[nodiscard]] std::string advisor_refusal_detail(std::string_view body);

/// The provider-neutral event an advisor result block carries. Empty when the
/// block is not an advisor result.
[[nodiscard]] std::optional<core::llm::AdvisorEvent> advisor_result_event(
    std::string_view block_json);

/// Tells whether the request a wire sends asks the provider to drop the
/// reasoning that a changed request prefix invalidated.
///
/// The wire owns the shape of its requests, so the wire answers this. A
/// request that carries `thinking.block_binding.prefix_mismatch_behavior`
/// can absorb a change to its tools; a wire that sends no thinking
/// configuration never can, and neither can a model whose requests the wire
/// leaves unbound.
using DropsInvalidatedReasoning = std::function<bool(std::string_view model)>;

/// Anthropic's advisor capability. Every Anthropic-wire provider exposes it;
/// the policy (which model may advise which) is Anthropic's own.
///
/// @p drops_invalidated_reasoning: the wire tells the capability which of
/// its requests ask the provider to drop the reasoning a changed request
/// prefix invalidated. A wire that does not cannot absorb a change to the
/// request's tools, so the conversation's advisor declaration stays stable
/// there.
class AnthropicAdvisorSupport final : public core::llm::IAdvisorSupport {
public:
    explicit AnthropicAdvisorSupport(DropsInvalidatedReasoning drops_invalidated_reasoning = {})
        : drops_invalidated_reasoning_(std::move(drops_invalidated_reasoning)) {}

    [[nodiscard]] bool supports(std::string_view executor_model,
                                const AdvisorSpec& advisor) const override {
        return advisor_pairing_supported(executor_model, advisor.model);
    }

    [[nodiscard]] std::string unsupported_reason(
        std::string_view executor_model,
        const AdvisorSpec& advisor) const override {
        return advisor_pairing_refusal(executor_model, advisor.model);
    }

    [[nodiscard]] std::string tool_definition_json(
        const AdvisorSpec& advisor) const override {
        return advisor_tool_definition(advisor);
    }

    [[nodiscard]] std::vector<std::string> advisor_choices(
        std::string_view executor_model) const override {
        return advisor_choices_for(executor_model);
    }

    [[nodiscard]] std::vector<std::string_view> feature_flags() const override {
        return advisor_feature_flags();
    }

    [[nodiscard]] bool absorbs_advisor_changes(
        std::string_view executor_model,
        std::span<const core::llm::Message> conversation) const noexcept override {
        // A model that runs no prefix check is not affected by a change to
        // the request's tools.
        if (!claude::model_runs_prefix_check(executor_model)) return true;
        // A request that asks the provider to drop the invalidated reasoning
        // can absorb the change at any point.
        if (drops_invalidated_reasoning_ && drops_invalidated_reasoning_(executor_model)) {
            return true;
        }
        // Otherwise only a conversation that produced no bound reasoning yet
        // can change its tools: there is nothing for the change to
        // invalidate.
        return !conversation_carries_bound_reasoning(conversation);
    }

    /// Anthropic ends a turn with a pending advisor call and finishes it when
    /// the client resumes the turn.
    [[nodiscard]] bool can_pause_turn() const noexcept override { return true; }

private:
    DropsInvalidatedReasoning drops_invalidated_reasoning_;
};

} // namespace core::llm::protocols::anthropic
