#pragma once
#include "ModelRegistry.hpp"
#include "MiniMaxEndpoints.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace core::llm {
inline constexpr std::string_view kMiniMaxDefaultModel = "MiniMax-M3.1-Flash";
// MiniMax documents a 1M context window for both curated models. Discovery
// keeps the tighter of this ceiling and any API-reported `max_input_tokens`;
// an explicit `context_window` config narrows it the same way.
inline constexpr int kMiniMaxContextWindow = 1000000;
// MiniMax Code's conservative working window (also M3's cheaper pricing-tier
// boundary); the only other value the context config accepts.
inline constexpr int kMiniMaxConservativeContextWindow = 512000;
inline constexpr int kMiniMaxOutputLimit = 128000;

[[nodiscard]] std::string canonical_minimax_model(std::string_view model);
[[nodiscard]] std::optional<ModelInfo> minimax_model_info(std::string_view model);
[[nodiscard]] bool is_minimax_plan_model(std::string_view model);

/// MiniMax counts thinking toward `max_tokens`; a short output budget at a
/// deep effort ends with `stop_reason: max_tokens` and no text. Returns the
/// deepest effort that still leaves such a request room for its answer, or
/// empty when the budget imposes no ceiling.
[[nodiscard]] std::string_view minimax_effort_ceiling(int max_tokens) noexcept;
} // namespace core::llm
