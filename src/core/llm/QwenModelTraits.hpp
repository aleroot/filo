#pragma once

#include <string_view>
#include <vector>

namespace core::llm {

enum class QwenTokenPlanWireApi {
    ChatCompletions,
    Responses,
};

[[nodiscard]] std::vector<int> qwen_model_generation(std::string_view model_id);
[[nodiscard]] bool is_qwen_text_model(std::string_view model_id);
[[nodiscard]] QwenTokenPlanWireApi qwen_token_plan_wire_api(
    std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_preserve_thinking(
    std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_token_plan_hosted_tools(
    std::string_view model_id);

} // namespace core::llm
