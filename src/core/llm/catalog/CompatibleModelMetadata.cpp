#include "CompatibleModelMetadata.hpp"

#include "ModelCatalogTraits.hpp"

namespace core::llm::catalog {
namespace {

[[nodiscard]] bool any_capability(
    const JsonObjectView& model,
    std::initializer_list<std::string_view> names) {
    for (const std::string_view name : names) {
        if (model.capability_supported(name)) return true;
    }
    return false;
}

} // namespace

ModelCapabilities compatible_advertised_capabilities(
    std::string_view model_id,
    const JsonObjectView& model) {
    if (is_embedding_model(model_id)) {
        return static_cast<uint32_t>(ModelCapability::Embeddings);
    }

    ModelCapabilities capabilities = 0;
    const auto add = [&](ModelCapability capability, bool supported) {
        if (supported) {
            capabilities |= static_cast<uint32_t>(capability);
        }
    };

    add(
        ModelCapability::TextInput,
        any_capability(model, {"text_input", "text"})
            || model.string_array_contains("input_modalities", "text"));
    add(
        ModelCapability::TextOutput,
        any_capability(model, {"text_output", "text"})
            || model.string_array_contains("output_modalities", "text"));
    add(
        ModelCapability::Streaming,
        any_capability(model, {"streaming", "stream"})
            || model.boolean("supports_streaming"));
    add(
        ModelCapability::SystemPrompts,
        any_capability(model, {"system_prompts", "system"})
            || model.boolean("supports_system_prompts"));
    add(
        ModelCapability::FunctionCalling,
        any_capability(model, {"function_calling", "tools"})
            || model.boolean("supports_function_calling")
            || model.boolean("supports_tools"));
    add(
        ModelCapability::ParallelToolCalls,
        model.capability_supported("parallel_tool_calls")
            || model.boolean("supports_parallel_tool_calls"));
    add(
        ModelCapability::JsonMode,
        any_capability(model, {"json_mode", "structured_outputs"})
            || model.boolean("supports_json_mode")
            || model.boolean("supports_structured_outputs"));
    add(
        ModelCapability::Vision,
        any_capability(model, {"vision", "image_input", "image"})
            || model.boolean("supports_image_in")
            || model.string_array_contains("input_modalities", "image"));
    add(
        ModelCapability::VideoInput,
        any_capability(model, {"video_input", "video"})
            || model.boolean("supports_video_in")
            || model.string_array_contains("input_modalities", "video"));
    add(
        ModelCapability::Reasoning,
        any_capability(model, {"reasoning", "thinking"})
            || model.boolean("supports_reasoning"));
    add(
        ModelCapability::TokenCounting,
        model.capability_supported("token_counting")
            || model.boolean("supports_token_counting"));
    add(
        ModelCapability::PromptCaching,
        model.capability_supported("prompt_caching")
            || model.boolean("supports_prompt_caching"));
    add(
        ModelCapability::Logprobs,
        model.capability_supported("logprobs")
            || model.boolean("supports_logprobs"));
    add(
        ModelCapability::PdfInput,
        any_capability(model, {"pdf_input", "pdf"})
            || model.boolean("supports_pdf_in"));
    add(
        ModelCapability::Citations,
        model.capability_supported("citations")
            || model.boolean("supports_citations"));
    add(
        ModelCapability::CodeExecution,
        model.capability_supported("code_execution")
            || model.boolean("supports_code_execution"));
    add(
        ModelCapability::Batch,
        model.capability_supported("batch")
            || model.boolean("supports_batch"));
    add(
        ModelCapability::ContextManagement,
        model.capability_supported("context_management")
            || model.boolean("supports_context_management"));
    return capabilities;
}

ModelReasoningProfile compatible_reasoning_profile(
    const JsonObjectView& model) {
    ModelReasoningProfile profile;
    const bool has_effort =
        model.capability_supported("effort")
        || model.capability_supported("reasoning")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "minimal")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "low")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "medium")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "high")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "xhigh")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "ultra");
    if (has_effort) {
        profile.effort = ReasoningCapabilities{ReasoningCapability::Effort};
    }
    if (model.object_array_contains(
            "supported_reasoning_levels", "effort", "max")) {
        profile.effort = profile.effort | ReasoningCapability::MaxEffort;
    }
    if (model.object_array_contains(
            "supported_reasoning_levels", "effort", "xhigh")) {
        profile.effort = profile.effort | ReasoningCapability::XHighEffort;
    }
    if (model.object_array_contains(
            "supported_reasoning_levels", "effort", "ultra")) {
        profile.effort = profile.effort | ReasoningCapability::UltraEffort;
    }
    return profile;
}

} // namespace core::llm::catalog
