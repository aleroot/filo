#pragma once

#include "../utils/StringUtils.hpp"

#include <string_view>

namespace core::llm::anthropic {

inline constexpr std::string_view kBillingHeader = "cc_version=2.1.255.13b; cc_entrypoint=cli; cch=0;";
inline constexpr std::string_view kDefaultFable = "claude-fable-5-1";
inline constexpr std::string_view kThinkingBindingBeta =
    "thinking-binding-controls-2026-08-01";

[[nodiscard]] inline bool is_fable_alias(std::string_view model) {
    return model == "fable" || model == "best" || model == "claude-fable"
        || model == "fable-5-1";
}

[[nodiscard]] inline bool is_model_version(std::string_view model,
                                           std::string_view version) {
    return model == version
        || (model.starts_with(version) && model.size() > version.size()
            && (model[version.size()] == '-' || model[version.size()] == '['));
}

[[nodiscard]] inline bool uses_bound_thinking(std::string_view model) {
    auto normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_copy(model));
    if (normalized.ends_with("[1m]")) normalized.resize(normalized.size() - 4);
    return is_fable_alias(normalized) || is_model_version(normalized, kDefaultFable);
}

[[nodiscard]] inline bool rejects_forced_tool_choice(std::string_view model) {
    if (uses_bound_thinking(model)) return true;
    const auto normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_copy(model));
    return is_model_version(normalized, "claude-mythos-5-1");
}

} // namespace core::llm::anthropic
