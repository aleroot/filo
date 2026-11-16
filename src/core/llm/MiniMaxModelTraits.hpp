#pragma once
#include "ModelRegistry.hpp"
#include "MiniMaxEndpoints.hpp"
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace core::llm {
inline constexpr std::string_view kMiniMaxDefaultModel = "MiniMax-M3.1-Flash";
// Documented context ceiling for both curated models. Discovery keeps the
// tighter of this and any API-reported `max_input_tokens`.
inline constexpr int kMiniMaxContextWindow = 1000000;
// Working window MiniMax Code uses unless the user opts into 1M
// (`higher_usage`). For pay-as-you-go M3 this is also the price boundary:
// input above 512k tokens is billed at twice the rate. Flash's list price is
// flat, so its working default stays at the ceiling.
inline constexpr int kMiniMaxConservativeContextWindow = 512000;
inline constexpr int kMiniMaxOutputLimit = 128000;
// MiniMax Code's `max_request_body_bytes` for the Messages endpoint.
inline constexpr std::size_t kMiniMaxRequestBodyLimit = 64 * 1024 * 1024;

[[nodiscard]] std::string canonical_minimax_model(std::string_view model);
[[nodiscard]] std::optional<ModelInfo> minimax_model_info(std::string_view model);
[[nodiscard]] bool is_minimax_plan_model(std::string_view model);
/// Working context for admission and compaction. `configured` 0 keeps the
/// model default (M3 at 512k, Flash at 1M). An explicit 512000 or 1000000
/// replaces that default and is still capped at the 1M ceiling.
[[nodiscard]] int minimax_working_context_window(std::string_view model,
                                                 int configured = 0) noexcept;

/// MiniMax counts thinking toward `max_tokens`; a short output budget at a
/// deep effort ends with `stop_reason: max_tokens` and no text. Returns the
/// deepest effort that still leaves such a request room for its answer, or
/// empty when the budget imposes no ceiling.
[[nodiscard]] std::string_view minimax_effort_ceiling(int max_tokens) noexcept;
} // namespace core::llm
