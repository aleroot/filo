#pragma once

#include "../ModelRegistry.hpp"

#include <string>
#include <string_view>

namespace core::llm::catalog {

inline constexpr ModelCapabilities kTextCapabilities =
    static_cast<uint32_t>(ModelCapability::TextInput) |
    static_cast<uint32_t>(ModelCapability::TextOutput) |
    static_cast<uint32_t>(ModelCapability::Streaming) |
    static_cast<uint32_t>(ModelCapability::SystemPrompts);

inline constexpr ModelCapabilities kToolCapabilities =
    kTextCapabilities |
    static_cast<uint32_t>(ModelCapability::FunctionCalling) |
    static_cast<uint32_t>(ModelCapability::ParallelToolCalls);

inline constexpr ModelCapabilities kFullCapabilities =
    kToolCapabilities |
    static_cast<uint32_t>(ModelCapability::JsonMode) |
    static_cast<uint32_t>(ModelCapability::Vision);

[[nodiscard]] bool contains_ascii(std::string_view haystack,
                                  std::string_view needle);
[[nodiscard]] std::string strip_prefix(std::string_view value,
                                       std::string_view prefix);
[[nodiscard]] bool is_embedding_model(std::string_view model_id);
[[nodiscard]] ModelTier infer_tier(std::string_view model_id,
                                   bool reasoning = false);
[[nodiscard]] ParameterConstraints standard_constraints(
    double max_temperature = 2.0);

} // namespace core::llm::catalog
