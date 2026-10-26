#pragma once

#include "ModelRegistry.hpp"

#include <optional>
#include <string_view>
#include <vector>

namespace core::llm {

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
/// Endpoint-local cards; never publish these over the native vendor registry.
[[nodiscard]] std::optional<ModelInfo> qwen_hosted_model_info(std::string_view model);
[[nodiscard]] int qwen_endpoint_context_window(std::string_view model);
[[nodiscard]] int qwen_endpoint_output_limit(std::string_view model);

/// Match official URL hosts exactly, including regional and QwenCloud hosts.
[[nodiscard]] bool is_qwen_token_plan_endpoint(std::string_view base_url);
[[nodiscard]] bool is_qwen_coding_plan_endpoint(std::string_view base_url);
[[nodiscard]] bool is_qwencloud_endpoint(std::string_view base_url);
[[nodiscard]] bool is_qwen_standard_endpoint(std::string_view base_url);

} // namespace core::llm
