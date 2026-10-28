#include "PolicySource.hpp"

#include <simdjson.h>

#include <array>
#include <cstddef>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace core::policy {
namespace {

// The words an enumeration is written with in a policy document.
template <typename Enum, std::size_t N>
using ChoiceTable = std::array<std::pair<std::string_view, Enum>, N>;

constexpr ChoiceTable<Effect, 3> kEffects{{
    {"allow", Effect::allow},
    {"deny", Effect::deny},
    {"ask", Effect::ask},
}};

constexpr ChoiceTable<PrincipalKind, 4> kPrincipalKinds{{
    {"web_tool", PrincipalKind::web_tool},
    {"child_process", PrincipalKind::child_process},
    {"mcp_server", PrincipalKind::mcp_server},
    {"model_provider", PrincipalKind::model_provider},
}};

constexpr ChoiceTable<OperationKind, 5> kOperationKinds{{
    {"read", OperationKind::read},
    {"write", OperationKind::write},
    {"connect", OperationKind::connect},
    {"invoke", OperationKind::invoke},
    {"infer", OperationKind::infer},
}};

constexpr ChoiceTable<DataFlow, 4> kDataFlows{{
    {"none", DataFlow::none},
    {"inbound", DataFlow::inbound},
    {"outbound", DataFlow::outbound},
    {"bidirectional", DataFlow::bidirectional},
}};

// Reads one of the words of an enumeration.
template <typename Enum, std::size_t N>
[[nodiscard]] std::expected<Enum, std::string> parse_choice(
    std::string_view text,
    const ChoiceTable<Enum, N>& choices,
    std::string_view kind) {
    for (const auto& [name, value] : choices) {
        if (text == name) return value;
    }
    return std::unexpected(std::format("unknown {} \"{}\"", kind, text));
}

[[nodiscard]] bool is_known_rule_key(std::string_view key) {
    return key == "id" || key == "effect" || key == "principal" || key == "operation"
        || key == "flow" || key == "host" || key == "port";
}

// A missing field is the wildcard. A present field of the wrong type is not:
// `"host": 123` must not become an allow rule for every host.
[[nodiscard]] std::expected<std::optional<std::string>, std::string>
optional_string_field(simdjson::dom::object object, std::string_view key) {
    const auto field = object[key];
    if (field.error() == simdjson::NO_SUCH_FIELD) return std::optional<std::string>{};
    if (field.error() != simdjson::SUCCESS)
        return std::unexpected(std::format("policy rule field \"{}\" is invalid", key));
    std::string_view text;
    if (field.get(text) != simdjson::SUCCESS)
        return std::unexpected(std::format("policy rule field \"{}\" must be a string", key));
    return std::string(text);
}

// An omitted field and the wildcard both mean "any".
template <typename Enum, std::size_t N>
[[nodiscard]] std::expected<std::optional<Enum>, std::string> optional_enum_field(
    simdjson::dom::object object,
    std::string_view key,
    const ChoiceTable<Enum, N>& choices,
    std::string_view kind) {
    const auto text = optional_string_field(object, key);
    if (!text) return std::unexpected(text.error());
    if (!text->has_value() || **text == "*") return std::optional<Enum>{};
    auto parsed = parse_choice(**text, choices, kind);
    if (!parsed) return std::unexpected(parsed.error());
    return std::optional<Enum>{*parsed};
}

[[nodiscard]] std::expected<std::string, std::string>
required_string_field(simdjson::dom::object object, std::string_view key) {
    auto value = optional_string_field(object, key);
    if (!value) return std::unexpected(value.error());
    if (!value->has_value())
        return std::unexpected(std::format("policy rule is missing \"{}\"", key));
    return std::move(**value);
}

[[nodiscard]] std::expected<PolicyRule, std::string> parse_rule(simdjson::dom::object object) {
    for (const auto field : object) {
        if (!is_known_rule_key(field.key)) {
            return std::unexpected(std::format("unknown policy rule field \"{}\"",
                                               field.key));
        }
    }

    PolicyRule rule;

    const auto id = optional_string_field(object, "id");
    if (!id) return std::unexpected(id.error());
    if (id->has_value()) rule.id = std::move(**id);

    const auto effect_text = required_string_field(object, "effect");
    if (!effect_text) return std::unexpected(effect_text.error());
    const auto effect = parse_choice(*effect_text, kEffects, "effect");
    if (!effect) return std::unexpected(effect.error());
    rule.effect = *effect;

    const auto principal = optional_string_field(object, "principal");
    if (!principal) return std::unexpected(principal.error());
    if (principal->has_value()) {
        const std::string_view text = **principal;
        if (text == "*") {
            rule.principal_id = "*";
        } else if (const auto colon = text.find(':'); colon != std::string_view::npos) {
            const auto kind = parse_choice(text.substr(0, colon), kPrincipalKinds, "principal kind");
            if (!kind) return std::unexpected(kind.error());
            rule.principal = *kind;
            rule.principal_id = std::string(text.substr(colon + 1));
            if (rule.principal_id.empty())
                return std::unexpected("policy rule has an empty principal identifier");
        } else {
            const auto kind = parse_choice(text, kPrincipalKinds, "principal kind");
            if (!kind) return std::unexpected(kind.error());
            rule.principal = *kind;
        }
    }

    const auto operation = optional_enum_field(object, "operation", kOperationKinds, "operation");
    if (!operation) return std::unexpected(operation.error());
    rule.operation = *operation;

    const auto flow = optional_enum_field(object, "flow", kDataFlows, "data flow");
    if (!flow) return std::unexpected(flow.error());
    rule.flow = *flow;

    const auto host = optional_string_field(object, "host");
    if (!host) return std::unexpected(host.error());
    if (host->has_value()) {
        const auto pattern = HostPattern::parse(**host);
        if (!pattern) return std::unexpected(pattern.error());
        rule.host = *pattern;
    }

    const auto port_field = object["port"];
    if (port_field.error() == simdjson::NO_SUCH_FIELD) {
        // An omitted port matches every port.
    } else if (port_field.error() != simdjson::SUCCESS) {
        return std::unexpected("policy rule field \"port\" is invalid");
    } else if (std::string_view port_text; port_field.get(port_text) == simdjson::SUCCESS) {
        const auto range = PortRange::parse(port_text);
        if (!range) return std::unexpected(range.error());
        rule.port = *range;
    } else if (int64_t port_number = 0; port_field.get(port_number) == simdjson::SUCCESS) {
        if (port_number <= 0 || port_number > 65535)
            return std::unexpected("policy rule has an out-of-range port");
        rule.port = PortRange::exact(static_cast<std::uint16_t>(port_number));
    } else {
        return std::unexpected("policy rule field \"port\" must be an integer or a string");
    }

    return rule;
}

} // namespace

std::expected<PolicySet, std::string> parse_policy_document(std::string_view json_text) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    const auto padded = simdjson::padded_string(json_text);
    if (parser.parse(padded).get(document) != simdjson::SUCCESS)
        return std::unexpected("network policy is not valid JSON");

    simdjson::dom::object root;
    if (document.get(root) != simdjson::SUCCESS)
        return std::unexpected("network policy must be a JSON object");

    for (const auto field : root) {
        const std::string_view key = field.key;
        if (key != "version" && key != "default" && key != "rules")
            return std::unexpected(std::format("unknown network policy field \"{}\"", key));
    }

    const auto version = root["version"];
    if (version.error() == simdjson::NO_SUCH_FIELD) {
        // Version 1 is the only document this parser understands, and an
        // omitted version is that document.
    } else if (version.error() != simdjson::SUCCESS) {
        return std::unexpected("network policy \"version\" is invalid");
    } else if (int64_t version_number = 0; version.get(version_number) == simdjson::SUCCESS) {
        if (version_number != 1)
            return std::unexpected("network policy version is not supported");
    } else {
        return std::unexpected("network policy \"version\" must be an integer");
    }

    Effect default_effect = Effect::deny;
    const auto default_field = root["default"];
    if (default_field.error() == simdjson::NO_SUCH_FIELD) {
        // The default is deny when the document does not name one.
    } else if (default_field.error() != simdjson::SUCCESS) {
        return std::unexpected("network policy \"default\" is invalid");
    } else if (std::string_view default_text; default_field.get(default_text) == simdjson::SUCCESS) {
        const auto effect = parse_choice(default_text, kEffects, "effect");
        if (!effect) return std::unexpected(effect.error());
        default_effect = *effect;
    } else {
        return std::unexpected("network policy \"default\" must be a string");
    }

    std::vector<PolicyRule> rules;
    simdjson::dom::array rule_array;
    if (root["rules"].get(rule_array) == simdjson::SUCCESS) {
        for (const auto element : rule_array) {
            simdjson::dom::object object;
            if (element.get(object) != simdjson::SUCCESS)
                return std::unexpected("network policy rule must be a JSON object");
            const auto rule = parse_rule(object);
            if (!rule) return std::unexpected(rule.error());
            rules.push_back(*rule);
        }
    } else if (!root["rules"].error()) {
        return std::unexpected("network policy \"rules\" must be an array");
    }

    PolicySet policy(std::move(rules), default_effect);
    if (!policy.valid())
        return std::unexpected("network policy is invalid: " + policy.invalid_reason());
    return policy;
}

std::expected<PolicySet, std::string> load_policy_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return std::unexpected("could not open network policy file " + path.string());
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const auto text = buffer.str();
    auto policy = parse_policy_document(text);
    if (!policy)
        return std::unexpected(path.string() + ": " + policy.error());
    return policy;
}

} // namespace core::policy
