#include "AntigravityModelTraits.hpp"

#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>

namespace core::llm::antigravity {

namespace ascii = core::utils::ascii;

namespace {

/// Position of `level` in kThinkingLevels; `nullopt` for any other token.
[[nodiscard]] std::optional<std::size_t> level_rank(std::string_view level) {
    const auto it = std::ranges::find_if(kThinkingLevels, [&](std::string_view known) {
        return ascii::iequals(level, known);
    });
    if (it == kThinkingLevels.end()) return std::nullopt;
    return static_cast<std::size_t>(it - kThinkingLevels.begin());
}

[[nodiscard]] bool offers(const ModelReasoningProfile& family, std::string_view level) {
    return std::ranges::find(family.supported_efforts, level) != family.supported_efforts.end();
}

} // namespace

std::optional<ThinkingVariant> split_thinking_variant(std::string_view sku) {
    const std::size_t dash = sku.rfind('-');
    if (dash == std::string_view::npos || dash == 0) return std::nullopt;
    const auto rank = level_rank(sku.substr(dash + 1));
    if (!rank) return std::nullopt;
    return ThinkingVariant{sku.substr(0, dash), kThinkingLevels[*rank]};
}

std::string thinking_variant_id(std::string_view family, std::string_view level) {
    std::string id(family);
    id += '-';
    id += level;
    return id;
}

ModelReasoningProfile thinking_family_profile(std::span<const std::string_view> levels) {
    ModelReasoningProfile profile;
    profile.complete = true;
    for (const std::string_view level : kThinkingLevels) {
        if (std::ranges::find(levels, level) != levels.end()) {
            profile.supported_efforts.emplace_back(level);
        }
    }
    if (profile.supported_efforts.empty()) return profile;

    // A plain ladder; the optional rungs follow from what the family is sold
    // at, and a missing `minimal` resolves to the weakest rung offered.
    profile.effort = ReasoningCapabilities{ReasoningCapability::Effort}
        | (offers(profile, "minimal") ? ReasoningCapability::MinimalEffort
                                      : ReasoningCapability::MapsMinimalToLow);
    if (offers(profile, "xhigh")) profile.effort = profile.effort | ReasoningCapability::XHighEffort;
    if (offers(profile, "max")) profile.effort = profile.effort | ReasoningCapability::MaxEffort;
    profile.catalog_default_effort = profile.supported_efforts.back();
    return profile;
}

std::string select_thinking_level(const ModelReasoningProfile& family,
                                  std::string_view requested,
                                  std::string_view pinned) {
    const auto& offered = family.supported_efforts;
    if (offered.empty()) return {};

    const std::string effort = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(requested));
    if (effort == "ultra") return offered.back();
    if (const auto wanted = level_rank(effort)) {
        // The weakest level that satisfies the request; above the top of the
        // ladder, the strongest there is.
        const auto it = std::ranges::find_if(offered, [&](const std::string& level) {
            return level_rank(level) >= wanted;
        });
        return it == offered.end() ? offered.back() : *it;
    }

    // "off" and unknown spellings carry no preference: there is no SKU
    // without thinking to route them to.
    const std::string pinned_level = core::utils::str::to_lower_ascii_copy(pinned);
    if (offers(family, pinned_level)) return pinned_level;
    if (family.catalog_default_effort && offers(family, *family.catalog_default_effort)) {
        return *family.catalog_default_effort;
    }
    return offered.back();
}

bool takes_thinking_config(std::string_view model) {
    if (ascii::iends_with(model, "-image")) return false;
    return ascii::istarts_with(model, "gemini-") || ascii::istarts_with(model, "claude-");
}

std::string family_display_name(std::string_view sku_display_name) {
    const std::string_view trimmed = core::utils::str::trim_ascii_view(sku_display_name);
    const std::size_t open = trimmed.rfind(" (");
    if (open == std::string_view::npos || !trimmed.ends_with(')')) return std::string(trimmed);
    const std::string_view level = trimmed.substr(open + 2, trimmed.size() - open - 3);
    return level_rank(level) ? std::string(trimmed.substr(0, open)) : std::string(trimmed);
}

} // namespace core::llm::antigravity
