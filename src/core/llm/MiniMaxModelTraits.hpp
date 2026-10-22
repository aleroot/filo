#pragma once
#include "ModelRegistry.hpp"
#include "MiniMaxEndpoints.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace core::llm {
inline constexpr std::string_view kMiniMaxDefaultModel = "MiniMax-M3.1-Flash";
inline constexpr int kMiniMaxContextWindow = 512000;
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
