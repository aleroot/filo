#pragma once

#include "AutoClassifier.hpp"
#include "PolicyConfig.hpp"
#include "../ProviderDescriptor.hpp"

#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::llm::routing {

// RouterEngine is the hot-path decision engine that lives entirely inside the
// process.  Design goals:
//
//  - Zero network round-trips:  all routing decisions are in-process.
//  - Read concurrency:          route() acquires a shared lock so many agent
//    threads can make decisions concurrently.  Only record_latency() and
//    set_active_policy() take an exclusive write lock.
//  - O(1) rule iteration:       rules are pre-sorted by priority in the
//    constructor so select_matching_rule never allocates or sorts at runtime.
//  - Deterministic load-balance: uses prompt hash so identical prompts always
//    resolve to the same provider within a session.
//
class RouterEngine {
public:
    RouterEngine(RouterConfig config,
                 core::llm::ProviderDescriptorSet providers = {});

    [[nodiscard]] bool has_policy(std::string_view policy_name) const;
    [[nodiscard]] std::vector<std::string> list_policies() const;
    [[nodiscard]] std::string active_policy() const;
    [[nodiscard]] std::optional<RouterGuardrails> guardrails() const;

    [[nodiscard]] bool set_active_policy(std::string policy_name);

    // Single best-candidate decision (original behaviour, used by tests and callers
    // that only want the top pick without the retry/fallback machinery).
    RouteDecision route(const RouteContext& context);

    // Full ordered fallback chain for a request.  The first entry is the preferred
    // candidate (chosen by the active strategy); subsequent entries are the remaining
    // available candidates in priority/quality order so that RouterProvider can try
    // them sequentially if the earlier ones fail.  Each entry carries its own
    // `retries` count from the policy so the caller knows how many times to retry
    // that specific candidate before moving on.
    [[nodiscard]] std::vector<RouteDecision> route_chain(const RouteContext& context);

    void record_latency(const RouteDecision& decision, int latency_ms);

private:
    struct LatencyStat {
        double ema_ms = 0.0;
        std::uint64_t samples = 0;
    };

    [[nodiscard]] const PolicyDefinition* get_active_policy_shared() const;
    [[nodiscard]] const RouteRule* select_matching_rule(const PolicyDefinition& policy,
                                                        std::string_view prompt_lower,
                                                        const RouteContext& context) const;
    [[nodiscard]] bool matches_condition(std::string_view prompt_lower,
                                         const RouteContext& context,
                                         const RuleCondition& condition) const;
    [[nodiscard]] const RouteCandidate* choose_candidate(const std::vector<RouteCandidate>& candidates,
                                                         Strategy strategy,
                                                         std::string_view prompt_lower,
                                                         const RouteContext& context,
                                                         std::string& reason);
    [[nodiscard]] const RouteCandidate* choose_fallback(const std::vector<RouteCandidate>& candidates,
                                                        std::string& reason) const;
    [[nodiscard]] const RouteCandidate* choose_load_balance(const std::vector<RouteCandidate>& candidates,
                                                            std::string_view prompt_lower,
                                                            std::string& reason);
    [[nodiscard]] const RouteCandidate* choose_latency(const std::vector<RouteCandidate>& candidates,
                                                       std::string& reason);
    [[nodiscard]] const RouteCandidate* choose_smart(const std::vector<RouteCandidate>& candidates,
                                                     std::string_view prompt_lower,
                                                     const RouteContext& context,
                                                     std::string& reason);

    [[nodiscard]] bool candidate_available(const RouteCandidate& candidate) const;
    [[nodiscard]] bool candidate_is_local(const RouteCandidate& candidate) const;
    [[nodiscard]] static std::string candidate_key(const RouteCandidate& candidate);
    [[nodiscard]] static std::string decision_key(const RouteDecision& decision);
    [[nodiscard]] double candidate_quality_score(std::string_view model_name) const;

    // shared_mutex: route() = shared lock (many concurrent readers OK)
    //               record_latency() / set_active_policy() = exclusive lock
    mutable std::shared_mutex rwmutex_;

    AutoClassifier auto_classifier_;                // stateless — no lock required; init before config_ move
    RouterConfig config_;                            // rules pre-sorted by priority in ctor
    core::llm::ProviderDescriptorSet providers_;
    std::string active_policy_;
    std::unordered_map<std::string, LatencyStat> latency_stats_;

    // Atomic so it can be incremented under a shared lock without UB.
    std::atomic<std::uint64_t> routing_counter_{0};
};

} // namespace core::llm::routing
