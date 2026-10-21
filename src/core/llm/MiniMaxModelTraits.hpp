#pragma once
#include "ModelRegistry.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace core::llm {
inline constexpr std::string_view kMiniMaxDefaultModel = "MiniMax-M3.1-Flash";
inline constexpr std::string_view kMiniMaxBaseUrl = "https://api.minimax.io/anthropic";
inline constexpr int kMiniMaxContextWindow = 512000;
inline constexpr int kMiniMaxOutputLimit = 128000;

[[nodiscard]] std::string canonical_minimax_model(std::string_view model);
[[nodiscard]] std::optional<ModelInfo> minimax_model_info(std::string_view model);
[[nodiscard]] bool is_minimax_plan_model(std::string_view model);
} // namespace core::llm
