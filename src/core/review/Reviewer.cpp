#include "Reviewer.hpp"

#include "../agent/Agent.hpp"
#include "../llm/ProviderFactory.hpp"
#include "../llm/ProviderManager.hpp"

#include <exception>
#include <format>
#include <utility>

namespace core::review {

std::expected<ReviewerSelection, std::string>
select_reviewer(const core::config::AppConfig& config,
                std::string_view session_provider,
                std::string_view session_model) {
    std::string provider;
    std::string model;
    if (const auto it = config.subagents.find(std::string(kReviewerProfile));
        it != config.subagents.end()) {
        provider = it->second.provider;
        model = it->second.model;
    }
    const bool other_provider = !provider.empty() && provider != session_provider;
    if (provider.empty()) provider = std::string(session_provider);
    if (model.empty()) {
        if (!other_provider) {
            model = std::string(session_model);
        } else if (const auto it = config.providers.find(provider);
                   it != config.providers.end()) {
            model = it->second.model;
        }
    }
    if (model.empty() && other_provider) {
        return std::unexpected(std::format(
            "Review provider '{}' has no model; set subagents.{}.model.",
            provider, kReviewerProfile));
    }
    return ReviewerSelection{std::move(provider), std::move(model)};
}

std::string reviewer_label(const ReviewerSelection& reviewer,
                           std::string_view session_provider,
                           std::string_view session_model) {
    if (reviewer.provider != session_provider) {
        return std::format("{} · {}", reviewer.provider, reviewer.model);
    }
    if (reviewer.model != session_model) return reviewer.model;
    return {};
}

std::expected<Reviewer, std::string> resolve_reviewer(const core::agent::Agent& session_agent) {
    const auto& config = core::config::ConfigManager::get_instance().get_config();
    const std::string session_provider = session_agent.get_active_provider_name();
    const std::string session_model = session_agent.get_active_model_name();
    auto selection = select_reviewer(config, session_provider, session_model);
    if (!selection) return std::unexpected(selection.error());

    Reviewer reviewer{
        .provider = {},
        .selection = std::move(*selection),
        .label = {},
    };
    reviewer.label = reviewer_label(reviewer.selection, session_provider, session_model);
    if (reviewer.selection.provider == session_provider) {
        reviewer.provider = session_agent.get_provider();
        return reviewer;
    }

    // A different provider gets its own instance, so the review's requests
    // never share a connection or turn state with the conversation's.
    const std::string& name = reviewer.selection.provider;
    try {
        auto shared = core::llm::ProviderManager::get_instance().get_provider(name);
        if (shared) {
            reviewer.provider = shared->fork_for_parallel_request();
            if (!reviewer.provider) reviewer.provider = std::move(shared);
        }
    } catch (const std::exception&) {
    }
    if (!reviewer.provider) {
        if (const auto it = config.providers.find(name); it != config.providers.end()) {
            reviewer.provider = core::llm::ProviderFactory::create_provider(name, it->second);
        }
    }
    if (!reviewer.provider) {
        return std::unexpected(std::format("Review provider '{}' is not available.", name));
    }
    return reviewer;
}

std::shared_ptr<core::agent::Agent>
make_review_agent(const core::agent::Agent& session_agent, const Reviewer& reviewer) {
    auto agent = session_agent.make_isolated_agent(reviewer.provider);
    if (agent && !reviewer.label.empty()) {
        agent->set_active_provider_name(reviewer.selection.provider);
        agent->set_active_model(reviewer.selection.model);
    }
    return agent;
}

} // namespace core::review
