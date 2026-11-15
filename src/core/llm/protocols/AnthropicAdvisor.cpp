#include "AnthropicAdvisor.hpp"

#include "../ClaudeModelTraits.hpp"
#include "../../utils/JsonWriter.hpp"
#include "../../utils/StringUtils.hpp"

#include <simdjson.h>

#include <array>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace core::llm::protocols::anthropic {

namespace {

/// A model generation: the major version and, when the id names a point
/// release, the minor one.
struct Generation {
    int major = 0;
    int minor = 0;
};

/// The family and generation of a Claude model id.
struct FamilyGeneration {
    std::string_view family;
    Generation generation;
};

/// The Claude families Anthropic's advisor table knows.
inline constexpr std::array<std::string_view, 4> kAdvisorFamilies{
    "fable", "opus", "sonnet", "haiku"};

/// The advisors one executor generation accepts: a family and the versions of
/// it that qualify. A negative maximum means "no upper bound".
struct AdvisorRange {
    std::string_view family;
    int min_major = 0;
    int min_minor = 0;
    int max_major = -1;
    int max_minor = -1;
};

/// One executor generation and the advisors it accepts.
struct PairingRow {
    std::string_view executor_family;
    int major = 0;
    int minor = 0;
    int advisor_count = 0;
    std::array<AdvisorRange, 4> advisors{};
};

[[nodiscard]] constexpr AdvisorRange any_of(std::string_view family) {
    return AdvisorRange{family, 0, 0, -1, -1};
}

[[nodiscard]] constexpr AdvisorRange since(std::string_view family, int major, int minor) {
    return AdvisorRange{family, major, minor, -1, -1};
}

[[nodiscard]] constexpr AdvisorRange between(std::string_view family,
                                             int min_major, int min_minor,
                                             int max_major, int max_minor) {
    return AdvisorRange{family, min_major, min_minor, max_major, max_minor};
}

/**
 * Anthropic's advisor pairing table.
 *
 * Each row names an executor generation and the advisors it accepts. A newer
 * executor within the same major takes the newest row that does not exceed
 * it, the same rule Filo uses for Claude wire policy; a model Anthropic does
 * not document has no row and is refused.
 *
 * "X or later" is read as a version floor inside the family, so it spans
 * generations: Opus 4.7 or later includes Opus 5. The API validates the
 * pairing again, and a client that is refused falls back to no advisor.
 */
inline constexpr std::array<PairingRow, 11> kPairingRows{{
    // Haiku
    {.executor_family = "haiku", .major = 4, .minor = 5, .advisor_count = 4,
     .advisors = {any_of("fable"), any_of("opus"), any_of("sonnet"), since("haiku", 5, 5)}},
    {.executor_family = "haiku", .major = 5, .minor = 5, .advisor_count = 4,
     .advisors = {any_of("fable"), since("opus", 4, 7), since("sonnet", 5, 0), since("haiku", 5, 5)}},
    // Sonnet
    {.executor_family = "sonnet", .major = 4, .minor = 6, .advisor_count = 4,
     .advisors = {any_of("fable"), any_of("opus"), any_of("sonnet"), since("haiku", 5, 5)}},
    {.executor_family = "sonnet", .major = 5, .minor = 0, .advisor_count = 4,
     .advisors = {any_of("fable"), since("opus", 4, 7), since("sonnet", 5, 0), since("haiku", 5, 5)}},
    {.executor_family = "sonnet", .major = 5, .minor = 5, .advisor_count = 3,
     .advisors = {any_of("fable"), since("opus", 5, 0), since("sonnet", 5, 5)}},
    // Opus
    // Opus 4.6 accepts every Opus, so the range has no floor.
    {.executor_family = "opus", .major = 4, .minor = 6, .advisor_count = 4,
     .advisors = {any_of("fable"), any_of("opus"), since("sonnet", 5, 0), since("haiku", 5, 5)}},
    {.executor_family = "opus", .major = 4, .minor = 7, .advisor_count = 3,
     .advisors = {any_of("fable"), since("opus", 4, 7), since("sonnet", 5, 5)}},
    {.executor_family = "opus", .major = 5, .minor = 0, .advisor_count = 2,
     .advisors = {any_of("fable"), since("opus", 5, 0)}},
    // Fable
    {.executor_family = "fable", .major = 5, .minor = 0, .advisor_count = 1,
     .advisors = {between("fable", 5, 0, 5, 2)}},
    {.executor_family = "fable", .major = 5, .minor = 1, .advisor_count = 1,
     .advisors = {between("fable", 5, 1, 5, 2)}},
    {.executor_family = "fable", .major = 5, .minor = 2, .advisor_count = 1,
     .advisors = {between("fable", 5, 2, 5, 2)}},
}};

[[nodiscard]] std::optional<FamilyGeneration> parse_claude_generation(
    std::string_view model) {
    if (model.empty()) return std::nullopt;
    constexpr std::string_view kClaudePrefix = "claude-";
    // The table speaks canonical Claude ids. A family alias the user typed
    // ("opus", "fable") resolves to the generation it names, and an id that
    // already carries its family ("opus-5-5") keeps it: the family is what
    // the table keys on, and the prefix is only how a canonical id spells it.
    const std::string canonical = resolve_advisor_model(model);
    if (canonical.empty()) return std::nullopt;
    const std::string_view canonical_view{canonical};
    const auto tail = canonical_view.starts_with(kClaudePrefix)
        ? canonical_view.substr(kClaudePrefix.size())
        : canonical_view;
    for (const auto family : kAdvisorFamilies) {
        if (!tail.starts_with(family)) continue;
        const auto version = claude::detail::version_after(canonical, family);
        if (!version) continue;
        return FamilyGeneration{
            .family = family,
            .generation = Generation{
                .major = static_cast<int>(version->parts[0]),
                .minor = version->size > 1 ? static_cast<int>(version->parts[1]) : 0,
            },
        };
    }
    return std::nullopt;
}

/// The advisors one row accepts.
[[nodiscard]] std::span<const AdvisorRange> accepted_advisors(const PairingRow& row) {
    return {row.advisors.data(), static_cast<std::size_t>(row.advisor_count)};
}

/// The row that governs an executor: the newest row of its family whose
/// generation does not exceed the executor's.
[[nodiscard]] const PairingRow* pairing_row(const FamilyGeneration& executor) {
    const PairingRow* best = nullptr;
    for (const auto& row : kPairingRows) {
        if (row.executor_family != executor.family) continue;
        if (row.major != executor.generation.major) continue;
        if (row.minor > executor.generation.minor) continue;
        if (!best || best->minor < row.minor) best = &row;
    }
    return best;
}

/// True when @p generation falls inside @p range.
[[nodiscard]] constexpr bool within(const Generation& generation,
                                    const AdvisorRange& range) {
    const auto value = std::pair{generation.major, generation.minor};
    if (value < std::pair{range.min_major, range.min_minor}) return false;
    if (range.max_major < 0) return true;
    return value <= std::pair{range.max_major, range.max_minor};
}

[[nodiscard]] std::string_view family_label(std::string_view family) {
    if (family == "fable") return "Fable";
    if (family == "opus") return "Opus";
    if (family == "sonnet") return "Sonnet";
    if (family == "haiku") return "Haiku";
    return family;
}

[[nodiscard]] std::string range_label(const AdvisorRange& range) {
    const auto family = family_label(range.family);
    if (range.max_major < 0 && range.min_major == 0) return std::string(family);
    const auto min_label = range.min_minor == 0
        ? std::to_string(range.min_major)
        : std::format("{}.{}", range.min_major, range.min_minor);
    if (range.max_major < 0) return std::format("{} {} or later", family, min_label);
    const auto max_label = range.max_minor == 0
        ? std::to_string(range.max_major)
        : std::format("{}.{}", range.max_major, range.max_minor);
    return std::format("{} {} to {}", family, min_label, max_label);
}

} // namespace

std::string resolve_advisor_model(std::string_view model) {
    const auto normalized = claude::normalized_claude_id(model);
    if (normalized.empty()) return {};
    if (normalized == "haiku" || normalized == "claude-haiku") {
        return std::string(claude::kDefaultHaiku);
    }
    if (normalized == "sonnet" || normalized == "claude-sonnet") {
        return std::string(claude::kDefaultSonnet);
    }
    if (normalized == "opus" || normalized == "claude-opus") {
        return std::string(claude::kDefaultOpus);
    }
    if (claude::is_opus_55_alias(normalized)) return std::string(claude::kDefaultOpus55);
    if (claude::is_fable_alias(normalized)) return std::string(claude::kDefaultFable);
    return std::string(normalized);
}

std::string advisor_tool_definition(const AdvisorSpec& advisor) {
    const auto model = resolve_advisor_model(advisor.model);
    if (model.empty()) return {};

    core::utils::JsonWriter writer(256);
    {
        auto _object = writer.object();
        writer.kv_str("type", kAdvisorToolType)
            .comma().kv_str("name", kAdvisorToolName)
            .comma().kv_str("model", model);
        if (advisor.max_uses > 0) {
            writer.comma().kv_num("max_uses", advisor.max_uses);
        }
        if (advisor.max_output_tokens > 0) {
            writer.comma().kv_num(
                "max_tokens",
                std::max(advisor.max_output_tokens, kAdvisorMinOutputTokens));
        }
    }
    return std::move(writer).take();
}

std::vector<std::string_view> advisor_feature_flags() {
    return {kAdvisorBeta};
}

bool advisor_pairing_supported(std::string_view executor_model,
                               std::string_view advisor_model) {
    const auto executor = parse_claude_generation(executor_model);
    if (!executor) return false;
    const auto row = pairing_row(*executor);
    if (!row) return false;
    const auto advisor = parse_claude_generation(resolve_advisor_model(advisor_model));
    if (!advisor) return false;
    for (const auto& range : accepted_advisors(*row)) {
        if (range.family != advisor->family) continue;
        if (within(advisor->generation, range)) return true;
    }
    return false;
}

std::string advisor_pairing_refusal(std::string_view executor_model,
                                    std::string_view advisor_model) {
    // A supported pairing has no refusal to report.
    if (advisor_pairing_supported(executor_model, advisor_model)) return {};
    if (advisor_model.empty()) return "No advisor model was selected.";
    const auto executor = parse_claude_generation(executor_model);
    if (!executor) {
        return std::format("{} is not a model Anthropic can advise with.", executor_model);
    }
    const auto row = pairing_row(*executor);
    if (!row) {
        return std::format("Anthropic documents no advisor for {}.", executor_model);
    }
    const auto advisor = parse_claude_generation(resolve_advisor_model(advisor_model));
    if (!advisor) {
        return std::format("{} is not a model Anthropic can advise with.", advisor_model);
    }

    std::string accepted;
    for (const auto& range : accepted_advisors(*row)) {
        if (!accepted.empty()) accepted += ", ";
        accepted += range_label(range);
    }
    return std::format("{} does not accept {} as an advisor. Accepted advisors: {}.",
                       executor_model, advisor_model, accepted);
}

std::vector<std::string> advisor_choices_for(std::string_view executor_model) {
    std::vector<std::string> choices;
    const auto executor = parse_claude_generation(executor_model);
    if (!executor) return choices;
    const auto* row = pairing_row(*executor);
    if (!row) return choices;

    // One entry per accepted family, named the way the user names it. A
    // family is offered only when the version its alias resolves to falls
    // inside the accepted range, so the picker never offers a pairing the
    // provider would refuse.
    for (const auto& range : accepted_advisors(*row)) {
        const auto alias = resolve_advisor_model(range.family);
        if (alias.empty()) continue;
        const auto advisor = parse_claude_generation(alias);
        if (!advisor) continue;
        if (!within(advisor->generation, range)) continue;
        choices.emplace_back(range.family);
    }
    return choices;
}

bool advisor_refused_by_response(int status_code, std::string_view body) {
    if (status_code < 400 || status_code >= 500) return false;
    // Anthropic names the advisor in the refusal it sends back.
    static const core::utils::str::CaseInsensitiveAsciiSearcher kAdvisorWord{"advisor"};
    return kAdvisorWord.find(body) != std::string_view::npos;
}

bool conversation_carries_bound_reasoning(
    std::span<const core::llm::Message> conversation) {
    for (const auto& message : conversation) {
        for (const auto& item : message.continuation_items) {
            if (item.provider != "anthropic") continue;
            if (item.kind != "thinking" && item.kind != "redacted_thinking") continue;
            return true;
        }
    }
    return false;
}

std::string advisor_refusal_detail(std::string_view body) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return {};
    std::string_view message;
    if (doc["error"]["message"].get(message) != simdjson::SUCCESS) return {};
    return std::string(message).substr(0, 500);
}

std::optional<core::llm::AdvisorEvent> advisor_result_event(std::string_view block_json) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(block_json);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return std::nullopt;

    std::string_view type;
    if (doc["type"].get(type) != simdjson::SUCCESS) return std::nullopt;
    if (type != "advisor_tool_result") return std::nullopt;

    simdjson::dom::element content;
    if (doc["content"].get(content) != simdjson::SUCCESS) return std::nullopt;
    std::string_view content_type;
    if (content["type"].get(content_type) != simdjson::SUCCESS) return std::nullopt;

    core::llm::AdvisorEvent event;
    if (content_type == "advisor_result") {
        // The advisor answered in the clear.
        event.state = core::llm::AdvisorCallState::Reviewed;
        event.guidance_readable = true;
        return event;
    }
    if (content_type == "advisor_redacted_result") {
        // The newest advisors answer encrypted: the executor still receives the
        // guidance, but the client cannot read it.
        event.state = core::llm::AdvisorCallState::Reviewed;
        event.guidance_readable = false;
        return event;
    }
    if (content_type == "advisor_tool_result_error") {
        event.state = core::llm::AdvisorCallState::Unavailable;
        std::string_view error_code;
        if (content["error_code"].get(error_code) == simdjson::SUCCESS) {
            event.error_code = std::string(error_code);
        }
        return event;
    }
    return std::nullopt;
}

} // namespace core::llm::protocols::anthropic
