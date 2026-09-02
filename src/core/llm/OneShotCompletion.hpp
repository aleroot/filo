#pragma once

#include "LLMProvider.hpp"

#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace core::llm {

inline constexpr std::string_view kCancelledMessage = "cancelled";

[[nodiscard]] inline bool
is_cancelled_error(std::string_view error) noexcept {
  return error == kCancelledMessage;
}

// History-free completion used by harness-owned planning and judging. The
// adapter centralizes the stream contract so orchestration layers never copy
// provider lifecycle logic or accidentally append internal prompts to chat.
// When `cancellation_requested` becomes true, the forked provider is cancelled
// and the function returns `kCancelledMessage` instead of a fallback answer.
[[nodiscard]] std::expected<std::string, std::string>
complete_once(const std::shared_ptr<LLMProvider> &provider,
              std::string_view model, std::string_view prompt,
              std::function<bool()> cancellation_requested = {});

} // namespace core::llm
