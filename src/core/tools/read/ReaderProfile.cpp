#include "ReaderProfile.hpp"
#include "ReadTypes.hpp"

#include <optional>
#include <string>

namespace core::tools::read {
namespace {

using core::llm::routing::PolicyDefinition;
using core::llm::routing::RouteCandidate;
using core::llm::routing::RouterConfig;

[[nodiscard]] bool usable(const RouteCandidate& candidate) noexcept {
    return !candidate.provider.empty() && !candidate.model.empty();
}

/// Mirrors RouterEngine's default-policy selection ("named default, else any
/// policy") but deterministically: unordered_map iteration order is
/// unspecified, so ties resolve to the lexicographically smallest name.
[[nodiscard]] std::optional<std::string> active_policy_name(const RouterConfig& router) {
    if (!router.default_policy.empty() && router.policies.contains(router.default_policy))
        return router.default_policy;
    std::optional<std::string> smallest;
    for (const auto& [name, policy] : router.policies) {
        static_cast<void>(policy);
        if (!smallest || name < *smallest) smallest = name;
    }
    return smallest;
}

/// First fast-tier candidate across policy defaults and rules in declaration
/// order; falls back to the first usable default candidate when nothing is
/// pinned "fast".
[[nodiscard]] const RouteCandidate* fast_tier_candidate(const PolicyDefinition& policy) noexcept {
    const RouteCandidate* first_default = nullptr;
    for (const auto& candidate : policy.defaults) {
        if (!usable(candidate)) continue;
        if (!first_default) first_default = &candidate;
        if (candidate.tier == "fast") return &candidate;
    }
    for (const auto& rule : policy.rules)
        for (const auto& candidate : rule.candidates)
            if (usable(candidate) && candidate.tier == "fast") return &candidate;
    return first_default;
}

} // namespace

std::expected<ReaderProfileSelection, std::string>
resolve_reader_profile(const core::config::AppConfig& config) {
    if (const auto it = config.subagents.find(std::string(kReaderProfile));
        it != config.subagents.end()) {
        const auto& profile = it->second;
        if (profile.enabled.value_or(true)
            && !profile.provider.empty() && !profile.model.empty()) {
            return ReaderProfileSelection{profile.provider, profile.model};
        }
    }
    if (const auto name = active_policy_name(config.router)) {
        const auto& policy = config.router.policies.at(*name);
        if (const RouteCandidate* candidate = fast_tier_candidate(policy))
            return ReaderProfileSelection{candidate->provider, candidate->model};
    }
    return std::unexpected(
        "Configure subagents.reader.provider and model, or pin a fast-tier "
        "candidate in a routing policy, to enable question answering.");
}

} // namespace core::tools::read
