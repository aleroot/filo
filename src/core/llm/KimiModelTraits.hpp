#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace core::llm {

enum class KimiService {
    Unknown,
    PublicApi,
    Code,
};

[[nodiscard]] std::span<const std::string_view> kimi_code_model_ids() noexcept;
[[nodiscard]] bool is_kimi_code_provider_name(
    std::string_view provider_name) noexcept;
[[nodiscard]] bool is_kimi_code_model(std::string_view model_id) noexcept;
[[nodiscard]] bool is_kimi_k3_model(std::string_view model_id) noexcept;
[[nodiscard]] bool is_kimi_k3_256k_model(std::string_view model_id) noexcept;
[[nodiscard]] bool is_kimi_public_k3_model(
    std::string_view model_id) noexcept;
[[nodiscard]] int32_t kimi_model_context_window(
    std::string_view model_id) noexcept;
[[nodiscard]] int32_t kimi_model_max_output_tokens(
    std::string_view model_id) noexcept;

[[nodiscard]] bool is_kimi_code_endpoint_path(
    std::string_view base_url) noexcept;
[[nodiscard]] KimiService kimi_service_for_endpoint(
    std::string_view base_url) noexcept;
[[nodiscard]] std::string_view kimi_service_id(KimiService service) noexcept;

} // namespace core::llm
