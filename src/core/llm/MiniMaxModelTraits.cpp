#include "MiniMaxModelTraits.hpp"
#include "core/utils/StringUtils.hpp"

namespace core::llm {
std::string canonical_minimax_model(std::string_view model) {
    const auto id = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_copy(model));
    if (id == "minimax-m3.1-flash" || id == "minimax-m3.1-flash-preview") return std::string(kMiniMaxDefaultModel);
    if (id == "minimax-m3") return "MiniMax-M3";
    return std::string(model);
}

bool is_minimax_plan_model(std::string_view model) {
    return canonical_minimax_model(model) == kMiniMaxDefaultModel;
}

std::optional<ModelInfo> minimax_model_info(std::string_view model) {
    const auto id = canonical_minimax_model(model);
    if (id != kMiniMaxDefaultModel && id != "MiniMax-M3") return std::nullopt;
    const bool flash = id == kMiniMaxDefaultModel;
    ModelInfo info;
    info.canonical_id = id;
    info.display_name = id;
    info.provider = "minimax";
    if (flash) info.aliases = {"MiniMax-M3.1-Flash-Preview"};
    info.context_window = kMiniMaxContextWindow;
    info.max_output_tokens = kMiniMaxOutputLimit;
    info.capabilities = static_cast<uint32_t>(ModelCapability::TextInput)
        | static_cast<uint32_t>(ModelCapability::TextOutput)
        | static_cast<uint32_t>(ModelCapability::SystemPrompts)
        | static_cast<uint32_t>(ModelCapability::Vision)
        | static_cast<uint32_t>(ModelCapability::FunctionCalling)
        | static_cast<uint32_t>(ModelCapability::Streaming)
        | static_cast<uint32_t>(ModelCapability::Reasoning)
        | static_cast<uint32_t>(ModelCapability::PromptCaching)
        | static_cast<uint32_t>(ModelCapability::TokenCounting);
    info.capabilities_complete = true;
    info.reasoning.complete = true;
    info.reasoning.adaptive_thinking = true;
    info.reasoning.effort = flash
        ? ReasoningCapability::Effort | ReasoningCapability::MaxEffort
            | ReasoningCapability::XHighEffort | ReasoningCapability::Required
        : ReasoningCapabilities(ReasoningCapability::ToggleOnly) | ReasoningCapability::Disable;
    info.reasoning.catalog_default_effort = flash ? "max" : "off";
    if (flash) info.reasoning.supported_efforts = {"low", "medium", "high", "xhigh", "max"};
    info.wire.thinking_always_on = flash;
    info.wire.fixed_sampling = true;
    info.wire.forced_tool_choice_rejected = true;
    info.tier = flash ? ModelTier::Fast : ModelTier::Powerful;
    return info;
}
} // namespace core::llm
