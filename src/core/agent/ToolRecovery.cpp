#include "ToolRecovery.hpp"

#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <utility>

namespace core::agent::recovery {
namespace {

using Element = simdjson::dom::element;
using Object = simdjson::dom::object;

[[nodiscard]] std::uint64_t fnv1a64(std::string_view data) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : data) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::string to_hex(std::uint64_t value) {
    return std::format("{:016x}", value);
}

/// Classic edit distance over small identifier strings.
[[nodiscard]] std::size_t levenshtein(std::string_view left,
                                      std::string_view right) noexcept {
    if (left.empty()) return right.size();
    if (right.empty()) return left.size();

    std::vector<std::size_t> previous(right.size() + 1);
    std::vector<std::size_t> current(right.size() + 1);
    for (std::size_t j = 0; j <= right.size(); ++j) {
        previous[j] = j;
    }
    for (std::size_t i = 1; i <= left.size(); ++i) {
        current[0] = i;
        for (std::size_t j = 1; j <= right.size(); ++j) {
            const std::size_t substitution =
                previous[j - 1] + (left[i - 1] == right[j - 1] ? 0 : 1);
            current[j] = std::min({previous[j] + 1,
                                   current[j - 1] + 1,
                                   substitution});
        }
        previous.swap(current);
    }
    return previous[right.size()];
}

/// A rename is plausible when the edit distance is small relative to the
/// longer identifier ("query"~"query_text", "path"~"patch").
[[nodiscard]] bool plausibly_renamed(std::string_view candidate,
                                     std::string_view offending) {
    if (candidate.empty() || offending.empty()) return false;
    const std::size_t longest = std::max(candidate.size(), offending.size());
    return levenshtein(candidate, offending) <= longest / 3;
}

[[nodiscard]] std::string join_names(const std::vector<std::string>& names,
                                     std::size_t limit = kMaxListedCandidates) {
    std::string joined;
    for (std::size_t i = 0; i < names.size() && i < limit; ++i) {
        if (i > 0) joined += ", ";
        joined += names[i];
    }
    if (names.size() > limit) {
        joined += ", …";
    }
    return joined;
}

[[nodiscard]] std::vector<std::string> candidates_for(
    const core::tools::schema::ArgumentIssue& issue,
    const core::tools::ToolDefinition& definition) {
    if (!issue.allowed.empty()) {
        return issue.allowed;
    }
    std::vector<std::string> names;
    names.reserve(definition.parameters.size());
    for (const auto& parameter : definition.parameters) {
        names.push_back(parameter.name);
    }
    std::ranges::sort(names);
    return names;
}

struct ParsedArguments {
    simdjson::dom::parser parser;
    Object object;
    bool valid = false;
};

[[nodiscard]] ParsedArguments parse_arguments(std::string_view arguments) {
    ParsedArguments parsed;
    parsed.parser.number_as_string(true);
    simdjson::padded_string padded(arguments.empty() ? "{}" : arguments);
    Element root;
    if (parsed.parser.parse(padded).get(root) != simdjson::SUCCESS
        || root.get(parsed.object) != simdjson::SUCCESS) {
        return parsed;
    }
    parsed.valid = true;
    return parsed;
}

[[nodiscard]] std::vector<std::pair<std::string, Element>> fields_of(
    Object object) {
    std::vector<std::pair<std::string, Element>> fields;
    for (const auto field : object) {
        fields.emplace_back(std::string(field.key), field.value);
    }
    std::ranges::sort(fields, {}, &std::pair<std::string, Element>::first);
    return fields;
}

[[nodiscard]] bool has_field(Object object, std::string_view name) {
    Element ignored;
    return object[name].get(ignored) == simdjson::SUCCESS;
}

/// Verbatim serialization of one argument value. Comparison across documents
/// is intentionally strict: differently ordered object keys count as unequal,
/// which errs toward precision over recall in runtime lesson derivation.
[[nodiscard]] std::string value_image(Element value) {
    return simdjson::to_string(value);
}

[[nodiscard]] RecoveryLesson lesson_for(RecoveryKey key, std::string hint) {
    return RecoveryLesson{.key = std::move(key), .hint = clamp_hint(std::move(hint))};
}

[[nodiscard]] RecoveryHint deduced(std::string instruction) {
    return RecoveryHint{clamp_hint(std::move(instruction)), false};
}

} // namespace

std::string schema_fingerprint(const core::tools::ToolDefinition& definition) {
    return to_hex(fnv1a64(core::tools::schema::structural_input_schema(definition)));
}

RecoveryKey make_key(std::string_view tool,
                     const core::tools::ToolDefinition& definition,
                     const core::tools::schema::ArgumentIssue& issue) {
    return RecoveryKey{
        .tool = std::string(tool),
        .schema_fingerprint = schema_fingerprint(definition),
        .issue_code = std::string(
            core::tools::schema::issue_code_name(issue.code)),
        .parameter = issue.parameter,
    };
}

std::optional<RecoveryHint> advise(
    const core::tools::schema::ArgumentIssue& issue,
    const core::tools::ToolDefinition& definition) {
    using core::tools::schema::ArgumentIssueCode;
    switch (issue.code) {
        case ArgumentIssueCode::UnknownArgument: {
            if (issue.parameter.empty()) return std::nullopt;
            const std::vector<std::string> candidates =
                candidates_for(issue, definition);
            if (candidates.empty()) {
                return deduced(
                    std::format("Remove the parameter '{}'.", issue.parameter));
            }
            const std::string* best = nullptr;
            std::size_t best_distance = 0;
            for (const auto& candidate : candidates) {
                if (candidate == issue.parameter) continue;
                const std::size_t distance =
                    levenshtein(candidate, issue.parameter);
                if (!best || distance < best_distance
                    || (distance == best_distance && candidate < *best)) {
                    best = &candidate;
                    best_distance = distance;
                }
            }
            if (best != nullptr && plausibly_renamed(*best, issue.parameter)) {
                return deduced(
                    std::format("Replace '{}' with '{}'.",
                                issue.parameter, *best));
            }
            return deduced(
                std::format("Remove '{}'; this tool accepts: {}.",
                            issue.parameter, join_names(candidates)));
        }
        case ArgumentIssueCode::MissingRequired: {
            if (issue.parameter.empty()) return std::nullopt;
            return deduced(
                std::format("Add the required parameter '{}'.", issue.parameter));
        }
        case ArgumentIssueCode::EnumMismatch: {
            if (issue.allowed.empty()) return std::nullopt;
            return deduced(
                std::format("'{}' accepts one of: {}.",
                            issue.parameter, join_names(issue.allowed)));
        }
        case ArgumentIssueCode::TypeMismatch: {
            if (issue.allowed.empty()) return std::nullopt;
            return deduced(
                std::format("'{}' must be of type {}.",
                            issue.parameter, join_names(issue.allowed)));
        }
        case ArgumentIssueCode::InvalidJson:
        case ArgumentIssueCode::NotAnObject:
        case ArgumentIssueCode::SchemaInvalid:
        case ArgumentIssueCode::NormalizeFailed:
        case ArgumentIssueCode::ConstMismatch:
        case ArgumentIssueCode::CombinatorMismatch:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<RecoveryLesson> derive_validation_lesson(
    std::string_view tool,
    const core::tools::ToolDefinition& definition,
    const core::tools::schema::ArgumentIssue& issue,
    std::string_view failed_arguments,
    std::string_view successful_arguments) {
    using core::tools::schema::ArgumentIssueCode;
    if (issue.code != ArgumentIssueCode::UnknownArgument
        || issue.parameter.empty()) {
        return std::nullopt;
    }

    const ParsedArguments failed = parse_arguments(failed_arguments);
    const ParsedArguments successful = parse_arguments(successful_arguments);
    if (!failed.valid || !successful.valid
        || !has_field(failed.object, issue.parameter)
        || has_field(successful.object, issue.parameter)) {
        return std::nullopt;
    }

    std::vector<std::string> replacements;
    for (const auto& [name, value] : fields_of(successful.object)) {
        static_cast<void>(value);
        if (!has_field(failed.object, name)) {
            replacements.push_back(name);
        }
    }

    const RecoveryKey key = make_key(tool, definition, issue);
    if (replacements.size() == 1) {
        return lesson_for(
            key,
            std::format("Replace '{}' with '{}'.",
                        issue.parameter, replacements.front()));
    }
    return lesson_for(
        key,
        std::format("Remove the unsupported '{}' parameter.", issue.parameter));
}

std::optional<RecoveryLesson> derive_runtime_lesson(
    std::string_view tool,
    const core::tools::ToolDefinition& definition,
    std::string_view failed_arguments,
    std::string_view successful_arguments) {
    const ParsedArguments failed = parse_arguments(failed_arguments);
    const ParsedArguments successful = parse_arguments(successful_arguments);
    if (!failed.valid || !successful.valid) {
        return std::nullopt;
    }

    const auto failed_fields = fields_of(failed.object);
    const auto successful_fields = fields_of(successful.object);

    std::vector<std::pair<std::string, Element>> removed;
    std::vector<std::pair<std::string, Element>> added;
    std::vector<std::pair<std::string, std::string>> shared_values;
    for (const auto& [name, value] : failed_fields) {
        Element counterpart;
        if (successful.object[name].get(counterpart) == simdjson::SUCCESS) {
            shared_values.emplace_back(name, value_image(value));
        } else {
            removed.emplace_back(name, value);
        }
    }
    for (const auto& [name, value] : successful_fields) {
        if (!has_field(failed.object, name)) {
            added.emplace_back(name, value);
        }
    }

    // Every shared parameter must be byte-identical: with more differences the
    // failure cause is not attributable to the structural delta alone.
    for (const auto& [name, image] : shared_values) {
        Element counterpart;
        if (successful.object[name].get(counterpart) != simdjson::SUCCESS
            || value_image(counterpart) != image) {
            return std::nullopt;
        }
    }

    const RecoveryKey key{
        .tool = std::string(tool),
        .schema_fingerprint = schema_fingerprint(definition),
        .issue_code = std::string(kRuntimeFailureCode),
        .parameter = removed.empty() ? std::string{}
                                     : removed.front().first,
    };

    if (removed.size() == 1 && added.empty()) {
        return lesson_for(
            key,
            std::format("Remove the '{}' parameter.", removed.front().first));
    }
    if (removed.size() == 1 && added.size() == 1
        && value_image(removed.front().second)
            == value_image(added.front().second)) {
        return lesson_for(
            key,
            std::format("Replace '{}' with '{}'.",
                        removed.front().first, added.front().first));
    }
    return std::nullopt;
}

bool result_indicates_error(std::string_view tool_result) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(tool_result);
    Element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) {
        return false;
    }
    Object object;
    if (root.get(object) != simdjson::SUCCESS) {
        return false;
    }
    if (core::utils::json::bool_field(object, "isError", false)) {
        return true;
    }
    Element error;
    return object["error"].get(error) == simdjson::SUCCESS;
}

std::string augment_error_payload(std::string_view tool_result,
                                  std::string_view hint) {
    if (hint.empty()) {
        return std::string(tool_result);
    }
    simdjson::dom::parser parser;
    simdjson::padded_string padded(tool_result);
    Element root;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS) {
        return std::string(tool_result);
    }
    Object object;
    if (root.get(object) != simdjson::SUCCESS) {
        return std::string(tool_result);
    }

    std::vector<std::pair<std::string, std::string>> members;
    bool has_recovery_hint = false;
    for (const auto field : object) {
        if (field.key == "recovery_hint") {
            has_recovery_hint = true;
        }
        members.emplace_back(std::string(field.key),
                             simdjson::to_string(field.value));
    }
    if (has_recovery_hint) {
        return std::string(tool_result);
    }
    members.emplace_back(
        "recovery_hint",
        "\"" + core::utils::escape_json_string(hint) + "\"");
    std::ranges::sort(members, {}, &std::pair<std::string, std::string>::first);

    core::utils::JsonWriter writer(tool_result.size() + hint.size() + 32);
    {
        auto scope = writer.object();
        for (std::size_t i = 0; i < members.size(); ++i) {
            if (i > 0) writer.comma();
            writer.key(members[i].first);
            writer.raw(members[i].second);
        }
    }
    return std::move(writer).take();
}

} // namespace core::agent::recovery
