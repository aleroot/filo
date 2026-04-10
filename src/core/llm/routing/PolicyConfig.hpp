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

struct RouterConfig {
    bool enabled = false;
    std::string default_policy;
    std::unordered_map<std::string, PolicyDefinition> policies;
    std::optional<RouterGuardrails> guardrails;

    // Auto-classifier configuration shared across all policies.
    // The Smart strategy uses this to choose the right tier per request.
    AutoClassifierConfig auto_classifier;

    // True when this config instance explicitly set auto_classifier fields.
    // Needed so merge_router_config can avoid clobbering an existing classifier
    // config when an overlay omits the section.
    bool has_auto_classifier_overrides = false;
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
