#include "RouterEngine.hpp"

#include "../ModelRegistry.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace core::llm::routing {

namespace {

[[nodiscard]] std::string lower_copy(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

[[nodiscard]] bool contains_keyword(std::string_view haystack,
                                    std::string_view keyword) {
    if (keyword.empty()) return false;
    const auto lower_keyword = lower_copy(keyword);
    return haystack.find(lower_keyword) != std::string_view::npos;
}

// Sort policy rules by (priority asc, name asc) once at construction so that
// select_matching_rule is a plain O(n) scan with no allocation or sorting.
void presort_policy_rules(PolicyDefinition& policy) {
    std::ranges::sort(policy.rules, [](const RouteRule& a, const RouteRule& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        return a.name < b.name;
    });
}

} // namespace

RouterEngine::RouterEngine(RouterConfig config,
                           std::unordered_set<std::string> available_providers)
    : auto_classifier_(config.auto_classifier)   // read before move below
    , config_(std::move(config))
    , available_providers_(std::move(available_providers)) {
    // Pre-sort rules in every policy so route() never allocates during matching.
    for (auto& [_, policy] : config_.policies) {
        presort_policy_rules(policy);
    }

    if (!config_.default_policy.empty() && config_.policies.contains(config_.default_policy)) {
        active_policy_ = config_.default_policy;
    } else if (!config_.policies.empty()) {
        active_policy_ = config_.policies.begin()->first;
    }
}

bool RouterEngine::has_policy(std::string_view policy_name) const {
    std::shared_lock lock(rwmutex_);
    return config_.policies.contains(std::string(policy_name));
}

std::vector<std::string> RouterEngine::list_policies() const {
    std::shared_lock lock(rwmutex_);
    std::vector<std::string> names;
    names.reserve(config_.policies.size());
    for (const auto& [name, _] : config_.policies) {
        names.push_back(name);
    }
    std::ranges::sort(names);
    return names;
}

std::string RouterEngine::active_policy() const {
    std::shared_lock lock(rwmutex_);
    return active_policy_;
}

std::optional<RouterGuardrails> RouterEngine::guardrails() const {
    std::shared_lock lock(rwmutex_);
    return config_.guardrails;
}

bool RouterEngine::set_active_policy(std::string policy_name) {
    std::unique_lock lock(rwmutex_);
    if (!config_.policies.contains(policy_name)) {
        return false;
    }
    active_policy_ = std::move(policy_name);
    return true;
}

RouteDecision RouterEngine::route(const RouteContext& context) {
    // Shared lock: multiple agent threads can route concurrently.
    std::shared_lock lock(rwmutex_);

    RouteDecision decision;

    const PolicyDefinition* policy = get_active_policy_shared();
    if (policy == nullptr) {
        decision.reason = "Router has no policies configured";
        return decision;
    }

    decision.policy = active_policy_;

    const std::string prompt_lower = lower_copy(context.prompt);
    // Rules are pre-sorted by priority; just scan linearly.
    const RouteRule* matched_rule = select_matching_rule(*policy, prompt_lower, context);

    Strategy selected_strategy = policy->strategy;
    const std::vector<RouteCandidate>* candidates = &policy->defaults;

    if (matched_rule != nullptr) {
        selected_strategy = matched_rule->strategy;
        candidates = &matched_rule->candidates;
        decision.rule = matched_rule->name;
    }

    std::string reason;
    const RouteCandidate* selected_candidate = choose_candidate(
        *candidates, selected_strategy, prompt_lower, context, reason);

    if (selected_candidate == nullptr && matched_rule != nullptr) {
        // Rule matched but all candidates unavailable: fall back to policy defaults.
        selected_strategy = policy->strategy;
        candidates = &policy->defaults;
        selected_candidate = choose_candidate(
            *candidates, selected_strategy, prompt_lower, context, reason);
        decision.rule = matched_rule->name + " (fallback to defaults)";
    }

    if (selected_candidate == nullptr) {
        // Last resort: first available candidate from any rule/default list.
        std::vector<RouteCandidate> last_resort = policy->defaults;
        for (const auto& rule : policy->rules) {
            last_resort.insert(last_resort.end(), rule.candidates.begin(), rule.candidates.end());
        }
        selected_candidate = choose_fallback(last_resort, reason);
        selected_strategy = Strategy::Fallback;
        if (decision.rule.empty()) {
            decision.rule = "last-resort";
        }
    }

    if (selected_candidate == nullptr) {
        decision.reason = "No available providers match router policy";
        return decision;
    }

    decision.provider = selected_candidate->provider;
    decision.model = selected_candidate->model;
    decision.strategy = selected_strategy;
    decision.reason = reason.empty()
        ? std::format("policy '{}' selected provider '{}'", decision.policy, decision.provider)
        : std::move(reason);

    return decision;
}

std::vector<RouteDecision> RouterEngine::route_chain(const RouteContext& context) {
    std::shared_lock lock(rwmutex_);

    const PolicyDefinition* policy = get_active_policy_shared();
    if (policy == nullptr) return {};

    const std::string prompt_lower = lower_copy(context.prompt);
    const RouteRule* matched_rule = select_matching_rule(*policy, prompt_lower, context);

    const Strategy strategy = matched_rule ? matched_rule->strategy : policy->strategy;
    const std::vector<RouteCandidate>* primary = matched_rule ? &matched_rule->candidates : &policy->defaults;
    const std::string rule_name = matched_rule ? matched_rule->name : std::string{};

    // Helper: build a RouteDecision from a candidate.
    auto make_decision = [&](const RouteCandidate& c,
                              Strategy s,
                              const std::string& rule,
                              std::string reason) -> RouteDecision {
        RouteDecision d;
        d.policy   = active_policy_;
        d.rule     = rule;
        d.provider = c.provider;
        d.model    = c.model;
        d.strategy = s;
        d.retries  = std::max(0, c.retries);
        d.reason   = std::move(reason);
        return d;
    };

    std::vector<RouteDecision> chain;

    if (strategy == Strategy::Fallback) {
        // Fallback: candidates list IS the ordered chain, no selection needed.
        for (const auto& c : *primary) {
            if (!candidate_available(c)) continue;
            chain.push_back(make_decision(c, Strategy::Fallback, rule_name,
                std::format("fallback chain picked '{}'", c.provider)));
        }
    } else {
        // Other strategies: pick the "best" candidate first, then append the rest
        // in list order as a fallback tail so callers have something to try on failure.
        std::string reason;
        const RouteCandidate* selected = choose_candidate(*primary, strategy, prompt_lower, context, reason);

        if (selected) {
            chain.push_back(make_decision(*selected, strategy, rule_name, std::move(reason)));
        }

        for (const auto& c : *primary) {
            if (!candidate_available(c)) continue;
            if (selected && c.provider == selected->provider && c.model == selected->model) continue;
            chain.push_back(make_decision(c, Strategy::Fallback, rule_name,
                std::format("fallback tail candidate '{}'", c.provider)));
        }
    }

    // Append policy defaults as last-resort when the primary list came from a rule.
    if (matched_rule) {
        for (const auto& c : policy->defaults) {
            if (!candidate_available(c)) continue;
            bool dup = false;
            for (const auto& d : chain) {
                if (d.provider == c.provider && d.model == c.model) { dup = true; break; }
            }
            if (!dup) {
                chain.push_back(make_decision(c, Strategy::Fallback, "defaults",
                    std::format("last-resort default '{}'", c.provider)));
            }
        }
    }

    return chain;
}

void RouterEngine::record_latency(const RouteDecision& decision, int latency_ms) {
    if (latency_ms <= 0 || decision.provider.empty()) {
        return;
    }

    std::unique_lock lock(rwmutex_);
    auto& stat = latency_stats_[decision_key(decision)];
    const double sample = static_cast<double>(latency_ms);
    if (stat.samples == 0) {
        stat.ema_ms = sample;
    } else {
        stat.ema_ms = (stat.ema_ms * 0.75) + (sample * 0.25);
    }
    ++stat.samples;
}

const PolicyDefinition* RouterEngine::get_active_policy_shared() const {
    if (!active_policy_.empty()) {
        if (const auto it = config_.policies.find(active_policy_); it != config_.policies.end()) {
            return &it->second;
        }
    }
    if (!config_.policies.empty()) {
        return &config_.policies.begin()->second;
    }
    return nullptr;
}

const RouteRule* RouterEngine::select_matching_rule(const PolicyDefinition& policy,
                                                     std::string_view prompt_lower,
                                                     const RouteContext& context) const {
    // Rules are already sorted by (priority asc, name asc) — just scan.
    for (const auto& rule : policy.rules) {
        if (matches_condition(prompt_lower, context, rule.when)) {
            return &rule;
        }
    }
    return nullptr;
}

bool RouterEngine::matches_condition(std::string_view prompt_lower,
                                     const RouteContext& context,
                                     const RuleCondition& condition) const {
    const int chars = static_cast<int>(prompt_lower.size());

    if (condition.min_prompt_chars >= 0 && chars < condition.min_prompt_chars) {
        return false;
    }
    if (condition.max_prompt_chars >= 0 && chars > condition.max_prompt_chars) {
        return false;
    }

    if (condition.needs_tool_history && !context.has_tool_messages) {
        return false;
    }

    if (!condition.any_keywords.empty()) {
        bool found = false;
        for (const auto& keyword : condition.any_keywords) {
            if (contains_keyword(prompt_lower, keyword)) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }

    for (const auto& keyword : condition.all_keywords) {
        if (!contains_keyword(prompt_lower, keyword)) {
            return false;
        }
    }

    return true;
}

const RouteCandidate* RouterEngine::choose_candidate(const std::vector<RouteCandidate>& candidates,
                                                      Strategy strategy,
                                                      std::string_view prompt_lower,
                                                      const RouteContext& context,
                                                      std::string& reason) {
    if (candidates.empty()) {
        reason = "candidate list is empty";
        return nullptr;
    }

    switch (strategy) {
        case Strategy::Fallback:
            return choose_fallback(candidates, reason);
        case Strategy::LoadBalance:
            return choose_load_balance(candidates, prompt_lower, reason);
        case Strategy::Latency:
            return choose_latency(candidates, reason);
        case Strategy::Smart:
            return choose_smart(candidates, prompt_lower, context, reason);
    }

    reason = "unsupported strategy";
    return nullptr;
}

const RouteCandidate* RouterEngine::choose_fallback(const std::vector<RouteCandidate>& candidates,
                                                     std::string& reason) const {
    for (const auto& candidate : candidates) {
        if (candidate_available(candidate)) {
            reason = std::format("fallback picked '{}' first", candidate.provider);
            return &candidate;
        }
    }

    reason = "fallback found no available providers";
    return nullptr;
}

const RouteCandidate* RouterEngine::choose_load_balance(const std::vector<RouteCandidate>& candidates,
                                                         std::string_view prompt_lower,
                                                         std::string& reason) {
    std::vector<const RouteCandidate*> available;
    available.reserve(candidates.size());

    int total_weight = 0;
    for (const auto& candidate : candidates) {
        if (!candidate_available(candidate)) continue;
        available.push_back(&candidate);
        total_weight += std::max(1, candidate.weight);
    }

    if (available.empty() || total_weight <= 0) {
        reason = "load_balance found no available providers";
        return nullptr;
    }

    // Deterministic: derive slot from prompt hash so identical prompts always
    // resolve to the same provider in a session; fall back to counter for empty prompts.
    const std::uint64_t counter = routing_counter_.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t seed = prompt_lower.empty()
        ? counter
        : std::hash<std::string_view>{}(prompt_lower);

    const int slot = static_cast<int>(seed % static_cast<std::uint64_t>(total_weight));
    int cumulative = 0;
    for (const auto* candidate : available) {
        cumulative += std::max(1, candidate->weight);
        if (slot < cumulative) {
            reason = std::format(
                "load_balance picked '{}' using weighted slot {}/{}",
                candidate->provider,
                slot,
                total_weight);
            return candidate;
        }
    }

    const auto* fallback = available.back();
    reason = std::format("load_balance fallback picked '{}'", fallback->provider);
    return fallback;
}

const RouteCandidate* RouterEngine::choose_latency(const std::vector<RouteCandidate>& candidates,
                                                    std::string& reason) {
    std::vector<const RouteCandidate*> available;
    available.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        if (candidate_available(candidate)) {
            available.push_back(&candidate);
        }
    }

    if (available.empty()) {
        reason = "latency strategy found no available providers";
        return nullptr;
    }

    const std::uint64_t counter = routing_counter_.fetch_add(1, std::memory_order_relaxed);

    std::vector<const RouteCandidate*> unknown;
    unknown.reserve(available.size());
    for (const auto* candidate : available) {
        if (!latency_stats_.contains(candidate_key(*candidate))) {
            unknown.push_back(candidate);
        }
    }

    // Occasional exploration of unknown providers (every 10 calls).
    if (!unknown.empty() && counter % 10 == 0) {
        const auto* candidate = unknown[static_cast<std::size_t>(counter % unknown.size())];
        reason = std::format("latency exploration sampled '{}' (no historical latency)", candidate->provider);
        return candidate;
    }

    const RouteCandidate* best = nullptr;
    double best_score = std::numeric_limits<double>::infinity();

    for (const auto* candidate : available) {
        const auto it = latency_stats_.find(candidate_key(*candidate));
        if (it == latency_stats_.end()) {
            continue;
        }

        const double score = it->second.ema_ms + static_cast<double>(candidate->latency_bias_ms);
        if (score < best_score) {
            best_score = score;
            best = candidate;
        }
    }

    if (best != nullptr) {
        reason = std::format("latency picked '{}' with EMA {:.1f}ms", best->provider, best_score);
        return best;
    }

    // No latency data yet: fall back to weighted load balancing.
    return choose_load_balance(candidates, "", reason);
}

const RouteCandidate* RouterEngine::choose_smart(const std::vector<RouteCandidate>& candidates,
                                                  std::string_view prompt_lower,
                                                  const RouteContext& context,
                                                  std::string& reason) {
    // ── Classify the request ──────────────────────────────────────────────────
    ClassificationInput input = AutoClassifier::extract_signals(
        context.prompt,
        context.history_tokens,
        context.turn_count,
        context.has_tool_messages);

    const ClassificationResult cr = auto_classifier_.classify(input);

    // ── Tier-aware candidate selection ───────────────────────────────────────
    //
    // If ANY candidate has an explicit tier pin we use tier-based selection:
    //   • The classified tier restricts the candidate pool.
    //   • Within the pool, Powerful tier → quality-score order;
    //     Fast/Balanced tier → latency order (cheaper/faster providers preferred).
    //
    // If NO candidate has a tier pin, we use the simpler heuristic from the
    // original choose_smart: quality-score for complex requests, latency for
    // everything else.  This preserves backwards compatibility when the policy
    // author hasn't opted into explicit tier pinning.

    const bool any_pinned = std::ranges::any_of(candidates, [](const RouteCandidate& c) {
        return !c.tier.empty();
    });

    auto tier_pin_matches = [](const RouteCandidate& c, Tier target) {
        // An unpinned candidate (empty tier) accepts any tier.
        if (c.tier.empty()) return true;
        if (c.tier == "fast"     && target == Tier::Fast)     return true;
        if (c.tier == "balanced" && target == Tier::Balanced) return true;
        if (c.tier == "powerful" && target == Tier::Powerful) return true;
        return false;
    };

    if (any_pinned) {
        // ── Tier-pinned path ─────────────────────────────────────────────────
        // Build a pool of candidates that match the classified tier.
        std::vector<const RouteCandidate*> pool;
        for (const auto& c : candidates) {
            if (candidate_available(c) && tier_pin_matches(c, cr.tier)) {
                pool.push_back(&c);
            }
        }

        // If pool is empty, relax and include all available candidates.
        if (pool.empty()) {
            for (const auto& c : candidates) {
                if (candidate_available(c)) pool.push_back(&c);
            }
        }

        if (!pool.empty()) {
            if (cr.tier == Tier::Powerful) {
                // Quality-first within the pool.
                const RouteCandidate* best = nullptr;
                double best_score = -std::numeric_limits<double>::infinity();
                for (const auto* c : pool) {
                    const double score = candidate_quality_score(
                        c->model.empty() ? c->provider : c->model);
                    if (score > best_score) { best_score = score; best = c; }
                }
                if (best) {
                    reason = std::format("smart [{}] → powerful quality ({})", cr.reason, best->provider);
                    return best;
                }
            } else {
                // Fast/Balanced: latency-first within the pool.
                // Build a temporary vector of RouteCandidate for choose_latency.
                std::vector<RouteCandidate> sub;
                sub.reserve(pool.size());
                for (const auto* c : pool) sub.push_back(*c);
                const RouteCandidate* picked = choose_latency(sub, reason);
                if (picked) {
                    // The returned pointer points into `sub` — find the original.
                    for (const auto& c : candidates) {
                        if (c.provider == picked->provider && c.model == picked->model) {
                            reason = std::format("smart [{}] → {} latency ({})",
                                                 cr.reason, to_string(cr.tier), c.provider);
                            return &c;
                        }
                    }
                }
            }
        }
    } else {
        // ── Unpinned / backwards-compatible path ─────────────────────────────
        // Complex requests → quality-score selection.
        // Simple/fast requests → latency-based selection.
        if (cr.tier == Tier::Powerful) {
            const RouteCandidate* best = nullptr;
            double best_score = -std::numeric_limits<double>::infinity();

            for (const auto& c : candidates) {
                if (!candidate_available(c)) continue;
                const double score = candidate_quality_score(
                    c.model.empty() ? c.provider : c.model);
                if (score > best_score) {
                    best_score = score;
                    best = &c;
                }
            }

            if (best != nullptr) {
                reason = std::format("smart [{}] → quality pick ({})", cr.reason, best->provider);
                return best;
            }
        }
    }

    // Latency-based fallback for all remaining cases.
    const RouteCandidate* candidate = choose_latency(candidates, reason);
    if (candidate != nullptr) {
        reason = std::format("smart [{}] → latency; {}", cr.reason, reason);
        return candidate;
    }

    return choose_load_balance(candidates, prompt_lower, reason);
}

bool RouterEngine::candidate_available(const RouteCandidate& candidate) const {
    if (candidate.provider.empty()) return false;
    if (available_providers_.empty()) return true;
    return available_providers_.contains(candidate.provider);
}

std::string RouterEngine::candidate_key(const RouteCandidate& candidate) {
    return candidate.provider + "::" + candidate.model;
}

std::string RouterEngine::decision_key(const RouteDecision& decision) {
    return decision.provider + "::" + decision.model;
}

double RouterEngine::candidate_quality_score(std::string_view model_name) const {
    // Use ModelRegistry for proper tier-based quality scoring
    // This replaces fragile string heuristics with structured registry data
    
    auto tier = core::llm::get_model_tier(model_name);
    
    if (tier) {
        switch (*tier) {
            case core::llm::ModelTier::Reasoning:  return 4.0;  // Highest quality
            case core::llm::ModelTier::Powerful:   return 3.0;
            case core::llm::ModelTier::Balanced:   return 2.0;
            case core::llm::ModelTier::Fast:       return 1.0;  // Lowest quality
        }
    }
    
    // Fallback to legacy heuristic matching for unknown models
    // This maintains backward compatibility during migration
    const std::string lower = lower_copy(model_name);
    double score = 0.0;

    if (lower.find("reasoning") != std::string::npos) score += 3.0;
    if (lower.find("pro") != std::string::npos) score += 2.0;
    if (lower.find("sonnet") != std::string::npos) score += 2.0;
    if (lower.find("opus") != std::string::npos) score += 2.5;
    if (lower.find("gpt-5") != std::string::npos) score += 2.5;

    if (lower.find("mini") != std::string::npos) score -= 1.5;
    if (lower.find("flash") != std::string::npos) score -= 1.0;
    if (lower.find("lite") != std::string::npos) score -= 1.0;
    if (lower.find("fast") != std::string::npos) score -= 0.5;

    return score;
}

} // namespace core::llm::routing
