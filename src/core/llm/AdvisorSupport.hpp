#pragma once

#include "Advisor.hpp"
#include "Models.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm {

/**
 * @brief Optional provider capability: run a second model inside one turn.
 *
 * A provider whose wire cannot do this exposes no support at all, so the core
 * never offers an advisor for it and the null object stands in for "no
 * advisor". An implementation owns every vendor detail: which model may
 * advise which, how the advisor is declared on the wire, and which feature
 * flags that declaration needs.
 *
 * The port is deliberately narrow. A provider that needs more surface grows a
 * second, separate interface rather than widening this one.
 */
class IAdvisorSupport {
public:
    virtual ~IAdvisorSupport() = default;

    /// May @p advisor advise while @p executor_model is the main model?
    [[nodiscard]] virtual bool supports(std::string_view executor_model,
                                        const AdvisorSpec& advisor) const = 0;

    /// Why the pairing is refused; empty when it is supported.
    [[nodiscard]] virtual std::string unsupported_reason(
        std::string_view executor_model,
        const AdvisorSpec& advisor) const = 0;

    /// The wire entry that declares the advisor: one JSON object in the
    /// provider's own tool shape. Empty when the provider declares none.
    [[nodiscard]] virtual std::string tool_definition_json(
        const AdvisorSpec& advisor) const = 0;

    /// The advisors @p executor_model accepts, named the way the provider
    /// names them. Empty when the provider cannot advise that executor, or
    /// when it does not publish the list.
    ///
    /// The list is what a picker offers; `supports` stays the authority on
    /// whether one of them may actually advise.
    [[nodiscard]] virtual std::vector<std::string> advisor_choices(
        [[maybe_unused]] std::string_view executor_model) const {
        return {};
    }

    /// The feature flags the capability requires. A wire protocol maps them
    /// to whatever its endpoint calls them.
    ///
    /// The flags belong to the capability, not to one advisor: a provider
    /// that needs them to accept an advisor's answer still needs them when a
    /// later request carries that answer without running a new advisor.
    [[nodiscard]] virtual std::vector<std::string_view> feature_flags() const {
        return {};
    }

    /// May the conversation's advisor declaration change now?
    ///
    /// Changing the advisor edits the request's tool list, and a provider
    /// that binds its reasoning to the request prefix invalidates that
    /// reasoning when the tools change. A provider answers for the
    /// conversation as it stands: a conversation that produced no bound
    /// reasoning yet can change its tools freely, and one whose wire tells
    /// the provider to drop the invalidated reasoning can always.
    [[nodiscard]] virtual bool absorbs_advisor_changes(
        [[maybe_unused]] std::string_view executor_model,
        [[maybe_unused]] std::span<const core::llm::Message> conversation) const noexcept {
        return false;
    }

    /// May a turn end with the consultation still pending, so the client has
    /// to resume it?
    [[nodiscard]] virtual bool can_pause_turn() const noexcept { return false; }
};

} // namespace core::llm
