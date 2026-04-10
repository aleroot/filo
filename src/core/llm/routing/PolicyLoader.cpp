#include "PolicyLoader.hpp"

#include <algorithm>
#include <format>
#include <string_view>
#include <tuple>

namespace core::llm::routing {

namespace {

template <typename Class, typename Member>
struct FieldSpec {
    std::string_view name;
    Member Class::* member;
};

template <typename Class, typename Member>
consteval auto field(std::string_view name, Member Class::* member)
    -> FieldSpec<Class, Member> {
    return FieldSpec<Class, Member>{name, member};
}

template <typename T>
struct Reflection;

template <>
struct Reflection<RouteCandidate> {
    static constexpr auto fields() {
        return std::tuple{
            field("provider", &RouteCandidate::provider),
            field("model", &RouteCandidate::model),
            field("weight", &RouteCandidate::weight),
            field("retries", &RouteCandidate::retries),
            field("latency_bias_ms", &RouteCandidate::latency_bias_ms),
            field("tier", &RouteCandidate::tier),
        };
    }
};

template <>
struct Reflection<RuleCondition> {
    static constexpr auto fields() {
        return std::tuple{
            field("min_prompt_chars", &RuleCondition::min_prompt_chars),
            field("max_prompt_chars", &RuleCondition::max_prompt_chars),
            field("any_keywords", &RuleCondition::any_keywords),
            field("all_keywords", &RuleCondition::all_keywords),
            field("needs_tool_history", &RuleCondition::needs_tool_history),
        };
    }
};

template <>
struct Reflection<RouteRule> {
    static constexpr auto fields() {
        return std::tuple{
            field("name", &RouteRule::name),
            field("priority", &RouteRule::priority),
            field("strategy", &RouteRule::strategy),
            field("when", &RouteRule::when),
            field("candidates", &RouteRule::candidates),
        };
    }
};

template <>
struct Reflection<PolicyDefinition> {
    static constexpr auto fields() {
        return std::tuple{
            field("name", &PolicyDefinition::name),
            field("description", &PolicyDefinition::description),
            field("strategy", &PolicyDefinition::strategy),
            field("defaults", &PolicyDefinition::defaults),
            field("rules", &PolicyDefinition::rules),
        };
    }
};

template <typename T>
concept Reflectable = requires {
    Reflection<T>::fields();
};

template <typename T>
[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               T& out,
                               std::string& error);

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               std::string& out,
                               std::string& error);

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               int& out,
                               std::string& error);

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               bool& out,
                               std::string& error);

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               Strategy& out,
                               std::string& error);

template <typename T>
[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               std::vector<T>& out,
                               std::string& error);

template <typename T, typename Member>
[[nodiscard]] bool parse_field(simdjson::dom::object obj,
                               T& out,
                               const FieldSpec<T, Member>& spec,
                               std::string& error) {
    simdjson::dom::element value;
    const auto ec = obj[spec.name].get(value);
    if (ec == simdjson::NO_SUCH_FIELD) {
        return true;
    }
    if (ec != simdjson::SUCCESS) {
        error = std::format("failed to read field '{}'", spec.name);
        return false;
    }

    if (!parse_value(value, out.*(spec.member), error)) {
        error = std::format("field '{}': {}", spec.name, error);
        return false;
    }

    return true;
}

template <Reflectable T>
[[nodiscard]] bool parse_reflectable(simdjson::dom::element element,
                                     T& out,
                                     std::string& error) {
    simdjson::dom::object obj;
    if (const auto ec = element.get(obj); ec != simdjson::SUCCESS) {
        error = "expected object";
        return false;
    }

    bool ok = true;
    std::apply([&](const auto&... specs) {
        ((ok = ok && parse_field(obj, out, specs, error)), ...);
    }, Reflection<T>::fields());
    return ok;
}

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               std::string& out,
                               std::string& error) {
    std::string_view value;
    if (const auto ec = element.get(value); ec != simdjson::SUCCESS) {
        error = "expected string";
        return false;
    }
    out = std::string(value);
    return true;
}

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               int& out,
                               std::string& error) {
    int64_t value = 0;
    if (const auto ec = element.get(value); ec != simdjson::SUCCESS) {
        error = "expected integer";
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               bool& out,
                               std::string& error) {
    if (const auto ec = element.get(out); ec != simdjson::SUCCESS) {
        error = "expected boolean";
        return false;
    }
    return true;
}

[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               Strategy& out,
                               std::string& error) {
    std::string value;
    if (!parse_value(element, value, error)) {
        return false;
    }

    const auto strategy = strategy_from_string(value);
    if (!strategy.has_value()) {
        error = std::format("unknown strategy '{}'", value);
        return false;
    }

    out = *strategy;
    return true;
}

template <typename T>
[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               std::vector<T>& out,
                               std::string& error) {
    simdjson::dom::array arr;
    if (const auto ec = element.get(arr); ec != simdjson::SUCCESS) {
        error = "expected array";
        return false;
    }

    out.clear();
    for (const auto item : arr) {
        T parsed{};
        if (!parse_value(item, parsed, error)) {
            return false;
        }
        out.push_back(std::move(parsed));
    }

    return true;
}

template <typename T>
[[nodiscard]] bool parse_value(simdjson::dom::element element,
                               T& out,
                               std::string& error) {
    if constexpr (Reflectable<T>) {
        return parse_reflectable(element, out, error);
    } else {
        static_assert(sizeof(T) == 0, "Unsupported field type for reflected parser");
    }
}

[[nodiscard]] std::string normalize_token(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        if (std::isspace(ch) || ch == '_' || ch == '-') continue;
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] std::string normalize_tier_pin(std::string_view value) {
    const std::string token = normalize_token(value);
    if (token == "fast") return "fast";
    if (token == "balanced" || token == "balance") return "balanced";
    if (token == "powerful" || token == "power") return "powerful";
    return {};
}

void normalize_candidates(std::vector<RouteCandidate>& candidates) {
    for (auto& candidate : candidates) {
        if (candidate.weight <= 0) candidate.weight = 1;
        if (candidate.retries < 0) candidate.retries = 0;
        if (candidate.latency_bias_ms < 0) candidate.latency_bias_ms = 0;
        if (!candidate.tier.empty()) {
            candidate.tier = normalize_tier_pin(candidate.tier);
        }
    }
}

void normalize_policy(PolicyDefinition& policy) {
    normalize_candidates(policy.defaults);

    for (std::size_t i = 0; i < policy.rules.size(); ++i) {
        auto& rule = policy.rules[i];
        if (rule.name.empty()) {
            rule.name = std::format("rule-{}", i + 1);
        }
        normalize_candidates(rule.candidates);
    }
}

[[nodiscard]] bool parse_guardrails_object(simdjson::dom::object guardrails_obj,
                                           RouterGuardrails& out,
                                           std::string& error) {
    out = RouterGuardrails{};

    auto read_non_negative_double = [&](std::string_view field_name,
                                        double& target) -> bool {
        simdjson::dom::element el;
        const auto ec = guardrails_obj[field_name].get(el);
        if (ec == simdjson::NO_SUCH_FIELD) return true;
        if (ec != simdjson::SUCCESS) {
            error = std::format("failed to read field '{}'", field_name);
            return false;
        }

        double value = 0.0;
        if (el.get(value) != simdjson::SUCCESS) {
            error = std::format("field '{}': expected number", field_name);
            return false;
        }

        target = std::max(0.0, value);
        return true;
    };

    auto read_ratio = [&](std::string_view field_name, float& target) -> bool {
        simdjson::dom::element el;
        const auto ec = guardrails_obj[field_name].get(el);
        if (ec == simdjson::NO_SUCH_FIELD) return true;
        if (ec != simdjson::SUCCESS) {
            error = std::format("failed to read field '{}'", field_name);
            return false;
        }

        double value = 0.0;
        if (el.get(value) != simdjson::SUCCESS) {
            error = std::format("field '{}': expected number", field_name);
            return false;
        }

        value = std::clamp(value, 0.0, 1.0);
        target = static_cast<float>(value);
        return true;
    };

    if (!read_non_negative_double("max_session_cost_usd", out.max_session_cost_usd)) {
        return false;
    }
    if (!read_ratio("min_requests_remaining_ratio", out.min_requests_remaining_ratio)) {
        return false;
    }
    if (!read_ratio("min_tokens_remaining_ratio", out.min_tokens_remaining_ratio)) {
        return false;
    }
    if (!read_ratio("min_window_remaining_ratio", out.min_window_remaining_ratio)) {
        return false;
    }

    {
        simdjson::dom::element enforce_on_local_el;
        const auto ec = guardrails_obj["enforce_on_local"].get(enforce_on_local_el);
        if (ec == simdjson::SUCCESS) {
            if (!parse_value(enforce_on_local_el, out.enforce_on_local, error)) {
                error = std::format("field 'enforce_on_local': {}", error);
                return false;
            }
        } else if (ec != simdjson::NO_SUCH_FIELD) {
            error = "failed to read field 'enforce_on_local'";
            return false;
        }
    }

    return true;
}

[[nodiscard]] bool parse_auto_classifier_object(simdjson::dom::object classifier_obj,
                                                AutoClassifierConfig& out,
                                                std::string& error) {
    auto read_ratio = [&](std::string_view field_name, double& target) -> bool {
        simdjson::dom::element el;
        const auto ec = classifier_obj[field_name].get(el);
        if (ec == simdjson::NO_SUCH_FIELD) return true;
        if (ec != simdjson::SUCCESS) {
            error = std::format("failed to read field '{}'", field_name);
            return false;
        }

        double value = 0.0;
        if (el.get(value) != simdjson::SUCCESS) {
            error = std::format("field '{}': expected number", field_name);
            return false;
        }

        target = std::clamp(value, 0.0, 1.0);
        return true;
    };

    auto read_non_negative_size = [&](std::string_view field_name,
                                      std::size_t& target) -> bool {
        simdjson::dom::element el;
        const auto ec = classifier_obj[field_name].get(el);
        if (ec == simdjson::NO_SUCH_FIELD) return true;
        if (ec != simdjson::SUCCESS) {
            error = std::format("failed to read field '{}'", field_name);
            return false;
        }

        uint64_t value = 0;
        if (el.get(value) != simdjson::SUCCESS) {
            error = std::format("field '{}': expected non-negative integer", field_name);
            return false;
        }

        target = static_cast<std::size_t>(value);
        return true;
    };

    auto read_non_negative_int = [&](std::string_view field_name, int& target) -> bool {
        simdjson::dom::element el;
        const auto ec = classifier_obj[field_name].get(el);
        if (ec == simdjson::NO_SUCH_FIELD) return true;
        if (ec != simdjson::SUCCESS) {
            error = std::format("failed to read field '{}'", field_name);
            return false;
        }

        int64_t value = 0;
        if (el.get(value) != simdjson::SUCCESS) {
            error = std::format("field '{}': expected integer", field_name);
            return false;
        }

        target = static_cast<int>(std::max<int64_t>(0, value));
        return true;
    };

    if (!read_ratio("quality_bias", out.quality_bias)) return false;
    if (!read_non_negative_size("fast_token_threshold", out.fast_token_threshold)) return false;
    if (!read_non_negative_size("powerful_token_threshold", out.powerful_token_threshold)) return false;
    if (!read_non_negative_int("escalation_turn_threshold", out.escalation_turn_threshold)) return false;

    if (out.powerful_token_threshold < out.fast_token_threshold) {
        out.powerful_token_threshold = out.fast_token_threshold;
    }

    return true;
}

} // namespace

RouterConfig make_default_router_config() {
    RouterConfig config;
    config.enabled = true;
    config.default_policy = "smart-code";

    PolicyDefinition policy;
    policy.name = "smart-code";
    policy.description = "Embedded smart routing inspired by Requesty policy composition.";
    policy.strategy = Strategy::Smart;

    policy.defaults = {
        RouteCandidate{.provider = "grok", .model = "grok-code-fast-1", .weight = 5},
        RouteCandidate{.provider = "grok-mini-fast", .model = "grok-3-mini-fast", .weight = 3},
        RouteCandidate{.provider = "openai", .model = "gpt-5.4", .weight = 2},
    };

    RouteRule deep_reasoning;
    deep_reasoning.name = "deep-reasoning";
    deep_reasoning.priority = 10;
    deep_reasoning.strategy = Strategy::Fallback;
    deep_reasoning.when.min_prompt_chars = 260;
    deep_reasoning.when.any_keywords = {
        "debug", "root cause", "architecture", "design", "migration", "refactor", "reasoning"
    };
    deep_reasoning.candidates = {
        RouteCandidate{.provider = "claude", .model = "claude-sonnet-4-6", .weight = 1, .retries = 1},
        RouteCandidate{.provider = "grok-reasoning", .model = "grok-4.20-reasoning", .weight = 1, .retries = 1},
        RouteCandidate{.provider = "openai", .model = "gpt-5.4", .weight = 1, .retries = 1},
    };

    RouteRule quick_iteration;
    quick_iteration.name = "quick-iteration";
    quick_iteration.priority = 40;
    quick_iteration.strategy = Strategy::LoadBalance;
    quick_iteration.when.max_prompt_chars = 220;
    quick_iteration.candidates = {
        RouteCandidate{.provider = "grok-mini-fast", .model = "grok-3-mini-fast", .weight = 6},
        RouteCandidate{.provider = "gemini", .model = "gemini-2.5-flash", .weight = 3},
        RouteCandidate{.provider = "grok", .model = "grok-code-fast-1", .weight = 1},
    };

    RouteRule tool_heavy;
    tool_heavy.name = "tool-heavy";
    tool_heavy.priority = 25;
    tool_heavy.strategy = Strategy::Latency;
    tool_heavy.when.needs_tool_history = true;
    tool_heavy.candidates = {
        RouteCandidate{.provider = "grok", .model = "grok-code-fast-1", .weight = 1, .latency_bias_ms = 15},
        RouteCandidate{.provider = "gemini", .model = "gemini-2.5-flash", .weight = 1, .latency_bias_ms = 10},
        RouteCandidate{.provider = "openai", .model = "gpt-5.4", .weight = 1, .latency_bias_ms = 20},
    };

    policy.rules = {deep_reasoning, tool_heavy, quick_iteration};
    config.policies[policy.name] = std::move(policy);
    return config;
}

bool parse_router_config(simdjson::dom::object router_obj,
                         RouterConfig& out,
                         std::string& error) {
    out = RouterConfig{};

    // Reflection parser for scalar router fields.
    {
        simdjson::dom::element enabled_el;
        if (router_obj["enabled"].get(enabled_el) == simdjson::SUCCESS) {
            if (!parse_value(enabled_el, out.enabled, error)) {
                error = std::format("router.enabled: {}", error);
                return false;
            }
        }

        simdjson::dom::element default_policy_el;
        if (router_obj["default_policy"].get(default_policy_el) == simdjson::SUCCESS) {
            if (!parse_value(default_policy_el, out.default_policy, error)) {
                error = std::format("router.default_policy: {}", error);
                return false;
            }
        }
    }

    {
        simdjson::dom::object classifier_obj;
        const auto ec = router_obj["auto_classifier"].get(classifier_obj);
        if (ec == simdjson::SUCCESS) {
            if (!parse_auto_classifier_object(classifier_obj, out.auto_classifier, error)) {
                error = std::format("router.auto_classifier: {}", error);
                return false;
            }
            out.has_auto_classifier_overrides = true;
        } else if (ec != simdjson::NO_SUCH_FIELD) {
            error = "router.auto_classifier must be an object";
            return false;
        }
    }

    for (const auto section_name : {"guardrails", "spend_limits", "limits"}) {
        simdjson::dom::object guardrails_obj;
        const auto ec = router_obj[section_name].get(guardrails_obj);
        if (ec == simdjson::NO_SUCH_FIELD) {
            continue;
        }
        if (ec != simdjson::SUCCESS) {
            error = std::format("router.{} must be an object", section_name);
            return false;
        }

        RouterGuardrails parsed_guardrails;
        if (!parse_guardrails_object(guardrails_obj, parsed_guardrails, error)) {
            error = std::format("router.{}: {}", section_name, error);
            return false;
        }
        out.guardrails = parsed_guardrails;
        break;
    }

    simdjson::dom::object policies_obj;
    if (const auto ec = router_obj["policies"].get(policies_obj);
        ec != simdjson::NO_SUCH_FIELD) {
        if (ec != simdjson::SUCCESS) {
            error = "router.policies must be an object";
            return false;
        }

        for (const auto entry : policies_obj) {
            simdjson::dom::object policy_obj;
            if (const auto policy_ec = entry.value.get(policy_obj); policy_ec != simdjson::SUCCESS) {
                error = std::format("router.policies.{} must be an object", entry.key);
                return false;
            }

            PolicyDefinition policy;
            simdjson::dom::element policy_el = entry.value;
            if (!parse_value(policy_el, policy, error)) {
                error = std::format("router.policies.{}: {}", entry.key, error);
                return false;
            }

            // Compatibility alias: "models" can be used instead of "defaults".
            if (policy.defaults.empty()) {
                simdjson::dom::element models_el;
                if (policy_obj["models"].get(models_el) == simdjson::SUCCESS) {
                    if (!parse_value(models_el, policy.defaults, error)) {
                        error = std::format("router.policies.{}.models: {}", entry.key, error);
                        return false;
                    }
                }
            }

            if (policy.name.empty()) {
                policy.name = std::string(entry.key);
            }
            normalize_policy(policy);
            out.policies[std::string(entry.key)] = std::move(policy);
        }
    }

    if (out.default_policy.empty() && !out.policies.empty()) {
        out.default_policy = out.policies.begin()->first;
    }

    if (out.enabled && out.policies.empty()) {
        error = "router.enabled=true requires at least one policy definition";
        return false;
    }

    return true;
}

void merge_router_config(RouterConfig& base, const RouterConfig& overlay) {
    base.enabled = overlay.enabled;

    if (!overlay.default_policy.empty()) {
        base.default_policy = overlay.default_policy;
    }
    if (overlay.guardrails.has_value()) {
        base.guardrails = overlay.guardrails;
    }
    if (overlay.has_auto_classifier_overrides) {
        base.auto_classifier = overlay.auto_classifier;
        base.has_auto_classifier_overrides = true;
    }

    for (const auto& [name, policy] : overlay.policies) {
        base.policies[name] = policy;
    }

    if (base.default_policy.empty() && !base.policies.empty()) {
        base.default_policy = base.policies.begin()->first;
    }
}

} // namespace core::llm::routing
