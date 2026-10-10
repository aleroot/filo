#include "QwenModelTraits.hpp"
#include "ModelRegistry.hpp"

#include "core/utils/StringUtils.hpp"

#include <array>
#include <limits>

namespace core::llm {

namespace {

struct Entry { std::string_view id; int context; bool vision; };
constexpr std::array kCodingPlanModels{
    Entry{"qwen3.5-plus", 1000000, true},
    Entry{"qwen3.6-plus", 1000000, true},
    Entry{"qwen3.7-plus", 1000000, true},
    Entry{"qwen3-coder-plus", 1000000, false},
    Entry{"qwen3-coder-next", 262144, false},
    Entry{"qwen3-max-2026-01-23", 262144, false},
    Entry{"glm-5", 202752, false},
    Entry{"glm-4.7", 202752, false},
    Entry{"kimi-k2.5", 262144, true},
    Entry{"MiniMax-M2.5", 196608, false},
};

[[nodiscard]] bool is_qwen_chat_model(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    return is_qwen_text_model(model_id) || lowered == "coder-model";
}

} // namespace

std::vector<int> qwen_model_generation(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    if (!lowered.starts_with("qwen")) return {};

    std::size_t pos = 4;
    std::vector<int> generation;
    while (pos < lowered.size()) {
        if (lowered[pos] < '0' || lowered[pos] > '9') break;
        int component = 0;
        while (pos < lowered.size() && lowered[pos] >= '0' && lowered[pos] <= '9') {
            if (component <= (std::numeric_limits<int>::max() - 9) / 10) {
                component = component * 10 + (lowered[pos] - '0');
            }
            ++pos;
        }
        generation.push_back(component);
        if (pos >= lowered.size() || lowered[pos] != '.') break;
        ++pos;
    }
    return generation;
}

bool is_qwen_text_model(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    return lowered.starts_with("qwen")
        && lowered.find("image") == std::string::npos
        && lowered.find("audio") == std::string::npos
        && lowered.find("embedding") == std::string::npos
        && lowered.find("rerank") == std::string::npos;
}

std::vector<ModelInfo> qwen_coding_plan_fallback_models() {
    // Endpoint-local cards: do not register these under another vendor's
    // global model identity. Live endpoint discovery always wins.
    std::vector<ModelInfo> models;
    for (const auto& entry : kCodingPlanModels) {
        ModelInfo model;
        if (const auto existing = ModelRegistry::instance().lookup(entry.id)) model = *existing;
        model.canonical_id = entry.id;
        model.display_name = entry.id;
        model.provider = "qwen";
        model.context_window = entry.context;
        if (model.max_output_tokens <= 0) model.max_output_tokens = 32000;
        model.reasoning = {};
        model.wire = {};
        model.capabilities = static_cast<uint32_t>(ModelCapability::TextInput)
            | static_cast<uint32_t>(ModelCapability::TextOutput)
            | static_cast<uint32_t>(ModelCapability::Streaming)
            | static_cast<uint32_t>(ModelCapability::SystemPrompts)
            | static_cast<uint32_t>(ModelCapability::FunctionCalling);
        if (entry.vision) {
            model.capabilities |= static_cast<uint32_t>(ModelCapability::Vision)
                | static_cast<uint32_t>(ModelCapability::VideoInput);
        }
        if (!entry.id.starts_with("qwen3-coder")) {
            model.capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
        }
        models.push_back(std::move(model));
    }
    return models;
}

int qwen_endpoint_context_window(std::string_view model) {
    std::string id = core::utils::str::to_lower_ascii_copy(model);
    if (const auto slash = id.rfind('/'); slash != std::string::npos) id.erase(0, slash + 1);
    for (const auto& entry : kCodingPlanModels) {
        const auto canonical = core::utils::str::to_lower_ascii_copy(entry.id);
        if (id == canonical || (id.starts_with(canonical) && id.size() > canonical.size()
                                && id[canonical.size()] == '-')) return entry.context;
    }
    return id.starts_with("qwen") || id == "coder-model" ? get_max_context_size(id) : 0;
}

bool qwen_model_supports_preserve_thinking(std::string_view model_id) {
    // Qwen Code stamps preserve_thinking on every DashScope chat request for
    // the Qwen family (including coder-model). Restricting it to 3.6+/3.7+
    // left qwen3-coder-plus and qwen3.5-plus unable to keep reasoning across
    // tool turns, which is the common Coding Plan / public DashScope path.
    return is_qwen_chat_model(model_id);
}

bool qwen_model_requires_nonnull_assistant_content(
    std::string_view model_id) {
    // Qwen's documented Chat Completions function-calling transcript uses
    // `content:""` alongside assistant `tool_calls`. The Qwen chat family
    // includes Qwen Code's `coder-model` alias and excludes third-party
    // models served by DashScope.
    return is_qwen_chat_model(model_id);
}

bool qwen_model_supports_vision(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    if (lowered == "coder-model") return true;
    static constexpr std::array<std::string_view, 8> kPrefixes{
        "qwen-vl",
        "qwen3-vl",
        "qwen3.5-plus",
        "qwen3.6-plus",
        "qwen3.7-plus",
        "qwen3.8-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
    };
    for (const auto prefix : kPrefixes) {
        if (lowered.starts_with(prefix)) return true;
    }
    return false;
}

bool qwen_model_supports_token_plan_hosted_tools(
    std::string_view model_id) {
    const std::vector<int> generation = qwen_model_generation(model_id);
    if (generation >= std::vector<int>{3, 7}) return true;
    if (generation != std::vector<int>{3, 6}) return false;

    const std::string lowered =
        core::utils::str::to_lower_ascii_copy(model_id);
    return lowered.find("-max") != std::string::npos
        || lowered.find("-plus") != std::string::npos
        || lowered.find("-flash") != std::string::npos;
}

bool qwen_model_supports_tiered_effort(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    return lowered.starts_with("qwen3.8-max")
        || lowered.starts_with("qwen3.8-flash");
}

} // namespace core::llm
