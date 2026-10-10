#pragma once

#include <string_view>
#include <vector>

namespace core::llm {

struct ModelInfo;

[[nodiscard]] std::vector<int> qwen_model_generation(std::string_view model_id);
[[nodiscard]] bool is_qwen_text_model(std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_preserve_thinking(
    std::string_view model_id);
/// DashScope's OpenAI-compatible Qwen models require `content:""` when an
/// assistant tool-call turn has no visible text. Third-party models hosted on
/// DashScope retain their own OpenAI-compatible null-content contract.
[[nodiscard]] bool qwen_model_requires_nonnull_assistant_content(
    std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_vision(std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_token_plan_hosted_tools(
    std::string_view model_id);
[[nodiscard]] bool qwen_model_supports_tiered_effort(
    std::string_view model_id);
[[nodiscard]] std::vector<ModelInfo> qwen_coding_plan_fallback_models();
[[nodiscard]] int qwen_endpoint_context_window(std::string_view model);

} // namespace core::llm
