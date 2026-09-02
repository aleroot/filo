#pragma once

#include <string_view>
#include <vector>

namespace core::llm {

[[nodiscard]] std::vector<int> qwen_model_generation(std::string_view model_id);
[[nodiscard]] bool is_qwen_text_model(std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_preserve_thinking(
    std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_token_plan_hosted_tools(
    std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_tiered_effort(
    std::string_view model_id);

} // namespace core::llm
