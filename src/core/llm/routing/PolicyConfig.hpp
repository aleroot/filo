#pragma once

#include "AutoClassifier.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::llm::routing {

enum class Strategy {
    Smart,
    Fallback,
    LoadBalance,
    Latency,
};

[[nodiscard]] std::string_view to_string(Strategy strategy) noexcept;
[[nodiscard]] std::optional<Strategy> strategy_from_string(std::string_view value) noexcept;

struct RouteCandidate {
    std::string provider = {};
    std::string model = {};
    int weight = 1;
    int retries = 0;
    int latency_bias_ms = 0;
    // Optional explicit tier pin ("fast", "balanced", "powerful").
    // When set, Smart strategy only picks this candidate if the classified
    // tier matches (or the candidate is the only available one).
    std::string tier = {}; // empty = any tier
};

struct RuleCondition {
    int min_prompt_chars = -1;
    int max_prompt_chars = -1;
    std::vector<std::string> any_keywords;
    std::vector<std::string> all_keywords;
    bool needs_tool_history = false;
};

struct RouteRule {
    std::string name;
    int priority = 100;
    Strategy strategy = Strategy::Fallback;
    RuleCondition when;
    std::vector<RouteCandidate> candidates;
};

struct PolicyDefinition {
    std::string name;
    std::string description;
    Strategy strategy = Strategy::Fallback;
    std::vector<RouteCandidate> defaults;
    std::vector<RouteRule> rules; // kept sorted by (priority, name) after construction
};

// Router-level guardrails evaluated by RouterProvider before dispatching a
// candidate.  These make fallback deterministic when remote quotas/spend hit a
// configured reserve threshold.
struct RouterGuardrails {
    // Hard cap on estimated session spend (USD). 0 = disabled.
    double max_session_cost_usd = 0.0;

    // Reserve thresholds represented as minimum remaining fraction in [0,1].
    // Example: 0.20 means keep 20% reserve and stop using that provider below it.
    float min_requests_remaining_ratio = 0.0f;
    float min_tokens_remaining_ratio = 0.0f;
    float min_window_remaining_ratio = 0.0f; // for unified windows (e.g. 5h / 7d)

    // Local providers are exempt by default.
    bool enforce_on_local = false;

    [[nodiscard]] bool enabled() const noexcept {
        return max_session_cost_usd > 0.0
            || min_requests_remaining_ratio > 0.0f
            || min_tokens_remaining_ratio > 0.0f
            || min_window_remaining_ratio > 0.0f;
    }
};

// Opt-in behaviour when every candidate in the fallback chain is exhausted
// (failed or unavailable).  Mirrors the "wait for the printed reset and
// continue" workflow popularised by rate-limit auto-retry helpers, but driven
// by structured provider data and with cross-provider failover first.
struct RouterFailover {
    // false = surface an error and end the turn (historical behaviour).
    // true  = when every candidate is rate-limited, park the turn until the
    //         soonest provider reset and automatically re-route.
    bool wait_on_exhaustion = false;

    // Total wait budget per request when waiting is enabled. 0 = unlimited
    // (overnight jobs).  When the soonest reset exceeds the budget the turn
    // fails with the usual router error instead of parking.
    int max_wait_seconds = 0;

    // Extra pause after a reset before re-routing, absorbing clock skew and
    // providers that reset slightly after the announced timestamp.
    int reset_margin_seconds = 60;
};

struct RouterConfig {
    bool enabled = false;
    std::string default_policy;
    std::unordered_map<std::string, PolicyDefinition> policies;
    std::optional<RouterGuardrails> guardrails;

    // Failover policy applied when the whole chain is exhausted.
    RouterFailover failover;

    // True when this config instance explicitly set the failover section.
    // Needed so merge_router_config can avoid clobbering an existing failover
    // policy when an overlay omits the section.
    bool has_failover_overrides = false;

    // Auto-classifier configuration shared across all policies.
    // The Smart strategy uses this to choose the right tier per request.
    AutoClassifierConfig auto_classifier;

    // True when this config instance explicitly set auto_classifier fields.
    // Needed so merge_router_config can avoid clobbering an existing classifier
    // config when an overlay omits the section.
    bool has_auto_classifier_overrides = false;

    // True when this config instance explicitly set router.scoring.  Kept
    // separate from auto_classifier so scoring-only overlays do not reset the
    // rest of the classifier config to defaults.
    bool has_scoring_overrides = false;
};

struct RouteContext {
    std::string prompt;
    bool has_tool_messages = false;

    // Enriched signals forwarded by RouterProvider.
    int         turn_count     = 0;  // number of conversation turns so far
    std::size_t history_tokens = 0;  // rough token count of prior messages
};

struct RouteDecision {
    std::string policy = {};
    std::string rule = {};
    std::string provider = {};
    std::string model = {};
    Strategy strategy = Strategy::Fallback;
    std::string reason = {};
    // How many times RouterProvider should retry THIS candidate before moving on.
    int retries = 0;
};

} // namespace core::llm::routing
