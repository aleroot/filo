#pragma once

#include "ReasoningCapabilities.hpp"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::llm::antigravity {

/// Catalog/service identity of every Gemini provider signed in through
/// Antigravity (`auth_type: oauth_google`), whatever its configured name: it
/// serves the Cloud Code Assist catalog, not the public Gemini API one.
inline constexpr std::string_view kServiceId = "gemini-antigravity";

/// Effort levels a SKU id may end in, weakest first.
///
/// The Antigravity backend sells some families once per thinking level:
/// `fetchAvailableModels` lists `claude-opus-5-5-low`, `-medium` and `-high`,
/// each with its own `model_enum`, so the level is part of the wire id rather
/// than a request field. Filo folds such a family into one model and turns the
/// session's effort back into the SKU id when serializing. Nothing here names
/// a vendor or a model: any id the catalog lists at two or more of these
/// levels folds, so a new family (or a new `-minimal` / `-max` tier) needs no
/// code change.
inline constexpr std::array<std::string_view, 6> kThinkingLevels{
    "minimal", "low", "medium", "high", "xhigh", "max"};

/// A SKU id split into its family and level.
struct ThinkingVariant {
    std::string_view family;
    std::string_view level; ///< One of kThinkingLevels.
};

/// Splits `<family>-<level>` when `sku` ends in one of kThinkingLevels;
/// `nullopt` otherwise. The views point into `sku`.
[[nodiscard]] std::optional<ThinkingVariant> split_thinking_variant(std::string_view sku);

/// The wire id of `family` sold at `level`.
[[nodiscard]] std::string thinking_variant_id(std::string_view family, std::string_view level);

/// The reasoning profile of a family sold at `levels` (any order): the levels
/// ordered weakest first as `supported_efforts`, an effort ladder covering
/// them, and the strongest level as the default, matching both vendors' API
/// default (Gemini's `thinkingLevel` and Anthropic's effort default to high).
[[nodiscard]] ModelReasoningProfile thinking_family_profile(std::span<const std::string_view> levels);

/// The level of `family` (a thinking_family_profile) a request runs at.
///
/// `requested` is the session effort in any Filo spelling: empty or "off"
/// means no preference and "ultra" means the strongest level; any other ladder
/// value picks the weakest level at least as strong, else the strongest.
/// Without a preference, `pinned` (the level of a configured variant id such
/// as `claude-opus-5-5-high`) wins, then the profile default.
[[nodiscard]] std::string select_thinking_level(const ModelReasoningProfile& family,
                                                std::string_view requested,
                                                std::string_view pinned);

/// Whether the backend takes a `thinkingConfig` for this id: Gemini and Claude
/// SKUs, except the image-generation ones, which cannot think.
[[nodiscard]] bool takes_thinking_config(std::string_view model);

/// A family's display name from one SKU's: drops the trailing level
/// parenthetical the backend appends ("Claude Opus 5.5 (High)").
[[nodiscard]] std::string family_display_name(std::string_view sku_display_name);

} // namespace core::llm::antigravity
