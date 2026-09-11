#include "StrictToolPolicy.hpp"

#include "../config/ConfigManager.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm {
namespace {

/// Anthropic documents strict tool use from the 4.5 generation onward.
constexpr double kMinimumClaudeGeneration = 4.5;
/// OpenAI's first snapshot with Structured Outputs for function calling.
constexpr int kMinimumGpt4oSnapshot = 2024'08'06;

[[nodiscard]] std::vector<std::string_view> split_tokens(std::string_view text) {
    std::vector<std::string_view> tokens;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = std::min(text.find_first_of("-_", start), text.size());
        if (end > start) tokens.emplace_back(text.substr(start, end - start));
        start = end + 1;
    }
    return tokens;
}

[[nodiscard]] std::optional<double> parse_number(std::string_view token) noexcept {
    double value = 0.0;
    const auto* end = token.data() + token.size();
    if (std::from_chars(token.data(), end, value).ptr != end) return std::nullopt;
    return value;
}

/// True for an 8-digit release stamp such as 20251101, which is version-shaped
/// but is not a generation number.
[[nodiscard]] bool is_release_stamp(std::string_view token) noexcept {
    return token.size() == 8
        && std::ranges::all_of(token, [](char ch) { return ch >= '0' && ch <= '9'; });
}

/**
 * Generation number of a model identifier, tolerating both spellings vendors
 * use: "claude-opus-4-5-20251101" and "claude-sonnet-4.5" both read as 4.5,
 * while "claude-3-5-sonnet-20241022" reads as 3.5.
 */
[[nodiscard]] std::optional<double> model_generation(std::string_view model) {
    std::optional<double> major;
    for (const auto token : split_tokens(model)) {
        if (is_release_stamp(token)) break;
        const auto number = parse_number(token);
        if (!number.has_value()) {
            if (major.has_value()) break;  // the version run has ended
            continue;
        }
        if (!major.has_value()) {
            major = *number;
            // A dotted token already carries the full generation.
            if (token.find('.') != std::string_view::npos) return major;
            continue;
        }
        if (*number >= 0.0 && *number < 10.0) {
            return *major + *number / 10.0;
        }
        break;
    }
    return major;
}

/// Trailing YYYY-MM-DD snapshot of an OpenAI model id, as a comparable integer.
[[nodiscard]] std::optional<int> openai_snapshot(std::string_view model) {
    const auto tokens = split_tokens(model);
    if (tokens.size() < 3) return std::nullopt;
    const auto year = parse_number(tokens[tokens.size() - 3]);
    const auto month = parse_number(tokens[tokens.size() - 2]);
    const auto day = parse_number(tokens[tokens.size() - 1]);
    if (!year || !month || !day) return std::nullopt;
    if (*year < 2020 || *year > 2100 || *month < 1 || *month > 12 || *day < 1 || *day > 31) {
        return std::nullopt;
    }
    return static_cast<int>(*year) * 10000
        + static_cast<int>(*month) * 100
        + static_cast<int>(*day);
}

[[nodiscard]] bool openai_model_supports_strict(std::string_view model) {
    // Families that support Structured Outputs across every snapshot.
    constexpr std::array kFamilies = std::to_array<std::string_view>({
        "gpt-4.1", "gpt-4o-mini", "gpt-5", "o3", "o4", "o5",
    });
    if (std::ranges::any_of(kFamilies, [model](std::string_view family) {
            return model.starts_with(family);
        })) {
        return true;
    }
    if (model.starts_with("gpt-4o")) {
        // Undated aliases resolve to a current snapshot; dated ones must be at
        // or after the snapshot that introduced the feature.
        const auto snapshot = openai_snapshot(model);
        return !snapshot.has_value() || *snapshot >= kMinimumGpt4oSnapshot;
    }
    return false;
}

[[nodiscard]] bool anthropic_model_supports_strict(std::string_view model) {
    if (!model.starts_with("claude")) return false;
    const auto generation = model_generation(model.substr(std::string_view("claude").size()));
    return generation.has_value() && *generation >= kMinimumClaudeGeneration;
}

} // namespace

bool model_supports_strict_tools(ToolSchemaWire wire, std::string_view model) noexcept {
    const std::string normalized = core::utils::str::to_lower_ascii_copy(model);
    switch (wire) {
        case ToolSchemaWire::OpenAI: return openai_model_supports_strict(normalized);
        case ToolSchemaWire::Anthropic: return anthropic_model_supports_strict(normalized);
    }
    return false;
}

std::optional<core::tools::schema::StrictDialect> strict_tool_dialect(
    ToolSchemaWire wire,
    std::string_view model,
    bool enabled) noexcept {
    if (!enabled || !model_supports_strict_tools(wire, model)) return std::nullopt;
    switch (wire) {
        case ToolSchemaWire::OpenAI:
            return core::tools::schema::StrictDialect::OpenAI;
        case ToolSchemaWire::Anthropic:
            return core::tools::schema::StrictDialect::Anthropic;
    }
    return std::nullopt;
}

std::optional<core::tools::schema::StrictDialect> configured_strict_tool_dialect(
    ToolSchemaWire wire,
    std::string_view model) {
    const auto& config = core::config::ConfigManager::get_instance().get_config();
    return strict_tool_dialect(wire, model, config.strict_tool_schemas);
}

} // namespace core::llm
