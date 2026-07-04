#include "ModelRegistry.hpp"

#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <limits>
#include <mutex>
#include <utility>

namespace core::llm {

struct ModelRegistry::RegistryState {
    std::unordered_map<std::string, ModelInfo> models;
    std::unordered_map<std::string, std::string> aliases;
};

// ============================================================================
// ModelInfo Implementation
// ============================================================================

bool ModelInfo::matches(std::string_view model_id) const noexcept {
    if (canonical_id == model_id) return true;
    for (const auto& alias : aliases) {
        if (alias == model_id) return true;
    }
    return false;
}

bool ModelInfo::validate_parameter(std::string_view param_name, double value) const noexcept {
    if (param_name == "temperature") {
        if (!constraints.temperature) return true;
        return constraints.temperature->contains(value);
    }
    if (param_name == "top_p" || param_name == "topP") {
        if (!constraints.top_p) return true;
        return constraints.top_p->contains(value);
    }
    if (param_name == "frequency_penalty" || param_name == "frequencyPenalty") {
        if (!constraints.frequency_penalty) return true;
        return constraints.frequency_penalty->contains(value);
    }
    if (param_name == "presence_penalty" || param_name == "presencePenalty") {
        if (!constraints.presence_penalty) return true;
        return constraints.presence_penalty->contains(value);
    }
    // Unknown parameters are allowed (provider-specific)
    return true;
}

double ModelInfo::estimate_cost(int input_tokens, int output_tokens, 
                                bool use_cached_input) const noexcept {
    if (context_window == 0) return 0.0;  // Unknown model, can't estimate
    
    double cost = 0.0;
    
    // Input tokens
    if (use_cached_input && pricing.has_cached_pricing()) {
        cost += (input_tokens / 1'000'000.0) * pricing.cached_input_per_mtok;
    } else {
        cost += (input_tokens / 1'000'000.0) * pricing.input_per_mtok;
    }
    
    // Output tokens
    cost += (output_tokens / 1'000'000.0) * pricing.output_per_mtok;
    
    return cost;
}

// ============================================================================
// Legacy Static Registry (for backward compatibility fallback)
// ============================================================================

namespace {

struct LegacyModelEntry {
    std::string_view pattern;
    int              max_context_tokens;
};

// This is the original static registry, preserved for fallback matching
constexpr LegacyModelEntry kLegacyRegistry[] = {
    // -----------------------------------------------------------------------
    // Qwen / DashScope Models (Alibaba Cloud)
    // -----------------------------------------------------------------------
    { "qwen3-coder-",      1000000 },
    { "qwen3-vl-",         1000000 },
    { "qwen3.5-",          1000000 },
    { "qwen3-turbo",       1000000 },
    { "qwen3-max",          256000 },
    { "qwen3-plus",         256000 },
    { "qwen3-",             256000 },
    { "qwen-",              128000 },

    // -----------------------------------------------------------------------
    // Kimi Models (Moonshot AI)
    // -----------------------------------------------------------------------
    { "kimi-k2.6",         256000 },
    { "kimi-k2-6",         256000 },
    { "kimi-k2.5",         256000 },
    { "kimi-for-coding",   256000 },
    { "kimi-k2",           256000 },
    { "kimi-for-university", 256000 },
    { "kimi-k1.5",         256000 },
    { "kimi-",             128000 },

    // -----------------------------------------------------------------------
    // Z.ai GLM Models
    // -----------------------------------------------------------------------
    { "glm-5.2",          1000000 },
    { "glm-5-turbo",     200000 },
    { "glm-5.1",         200000 },
    { "glm-5",           200000 },
    { "glm-4.7",         200000 },
    { "glm-4.5-air",     200000 },
    { "glm-",            200000 },

    // -----------------------------------------------------------------------
    // Anthropic Claude Models
    // -----------------------------------------------------------------------
    { "claude-fable-5",     1000000 },
    { "claude-sonnet-5",    1000000 },
    { "claude-haiku-4-5",   200000 },
    { "claude-opus-4-8",    200000 },
    { "claude-sonnet-4-6", 1000000 },
    { "claude-",           200000 },

    // -----------------------------------------------------------------------
    // OpenAI GPT Models
    // -----------------------------------------------------------------------
    { "gpt-5.4",           200000 },
    { "gpt-5",             200000 },
    { "gpt-4o",            128000 },
    { "gpt-4-turbo",       128000 },
    { "gpt-4-32k",         32000 },
    { "gpt-4",             8192 },
    { "gpt-3.5-turbo-16k", 16000 },
    { "gpt-3.5-turbo",     4096 },
    { "gpt-3.5",           4096 },

    // -----------------------------------------------------------------------
    // Google Gemini Models
    // -----------------------------------------------------------------------
    { "gemini-1.5",        2097152 },
    { "gemini-2.0",        1048576 },
    { "gemini-2.5",        1048576 },
    { "gemini-3.1",        1048576 },
    { "gemini-3",          1048576 },
    { "gemini-",           1048576 },

    // -----------------------------------------------------------------------
    // xAI Grok Models
    // -----------------------------------------------------------------------
    { "grok-",             128000 },

    // -----------------------------------------------------------------------
    // Local / Open Source Models
    // -----------------------------------------------------------------------
    { "llama3",            128000 },
    { "llama-3",           128000 },
    { "mixtral",           32000 },
    { "mistral",           32000 }
};

int legacy_get_max_context_size(std::string_view model_name) noexcept {
    if (model_name.empty()) return 0;
    
    for (const auto& entry : kLegacyRegistry) {
        if (model_name.find(entry.pattern) != std::string_view::npos) {
            return entry.max_context_tokens;
        }
    }
    return 0;
}

// ============================================================================
// Built-in Model Catalog
// ============================================================================

// Helper for building capabilities
constexpr ModelCapabilities CAP_ALL = 
    static_cast<uint32_t>(ModelCapability::TextInput) |
    static_cast<uint32_t>(ModelCapability::TextOutput) |
    static_cast<uint32_t>(ModelCapability::Streaming) |
    static_cast<uint32_t>(ModelCapability::SystemPrompts);

constexpr ModelCapabilities CAP_TOOLS = CAP_ALL |
    static_cast<uint32_t>(ModelCapability::FunctionCalling) |
    static_cast<uint32_t>(ModelCapability::ParallelToolCalls);

constexpr ModelCapabilities CAP_JSON = CAP_ALL |
    static_cast<uint32_t>(ModelCapability::JsonMode);

constexpr ModelCapabilities CAP_FULL = CAP_TOOLS | CAP_JSON |
    static_cast<uint32_t>(ModelCapability::Vision);

// Common parameter constraints
const ParameterConstraints kStandardConstraints = [] {
    ParameterConstraints c;
    c.temperature = {0.0, 2.0};
    c.top_p = {0.0, 1.0};
    c.frequency_penalty = {-2.0, 2.0};
    c.presence_penalty = {-2.0, 2.0};
    return c;
}();

const ParameterConstraints kClaudeConstraints = [] {
    ParameterConstraints c;
    c.temperature = {0.0, 1.0};
    c.top_p = {0.0, 1.0};
    return c;
}();

// Anthropic models
std::vector<ModelInfo> build_anthropic_catalog() {
    return {
        // Claude Fable 5
        {
            .canonical_id = "claude-fable-5",
            .aliases = {"fable", "claude-fable", "fable-5"},
            .display_name = "Claude Fable 5",
            .provider = "anthropic",
            .context_window = 1'000'000,
            .max_output_tokens = 128'000,
            .max_reasoning_tokens = 0,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching) |
                static_cast<uint32_t>(ModelCapability::TokenCounting) |
                static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Powerful,
            .pricing = {10.0, 50.0, 1.0, 12.5},
            .knowledge_cutoff = "2026-01",
            .constraints = kClaudeConstraints,
            .max_tool_calls = 32
        },
        // Claude Sonnet 5
        {
            .canonical_id = "claude-sonnet-5",
            .aliases = {"sonnet", "claude-sonnet", "sonnet-5"},
            .display_name = "Claude Sonnet 5",
            .provider = "anthropic",
            .context_window = 1'000'000,
            .max_output_tokens = 128'000,
            .max_reasoning_tokens = 0,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching) |
                static_cast<uint32_t>(ModelCapability::TokenCounting) |
                static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Balanced,
            // Introductory launch pricing is effective through 2026-08-31.
            .pricing = {2.0, 10.0, 0.20, 2.5},
            .knowledge_cutoff = "2026-01",
            .constraints = kClaudeConstraints,
            .max_tool_calls = 32
        },
        // Claude Haiku 4.5
        {
            .canonical_id = "claude-haiku-4-5",
            .aliases = {"haiku", "claude-haiku", "haiku-4.5", "haiku-4-5"},
            .display_name = "Claude Haiku 4.5",
            .provider = "anthropic",
            .context_window = 200000,
            .max_output_tokens = 64000,
            .max_reasoning_tokens = 0,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching) |
                static_cast<uint32_t>(ModelCapability::TokenCounting) |
                static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Fast,
            .pricing = {1.0, 5.0, 0.10, 1.25},
            .knowledge_cutoff = "2025-02",
            .constraints = kClaudeConstraints,
            .max_tool_calls = 32
        },
        // Claude Opus 4.8
        {
            .canonical_id = "claude-opus-4-8",
            .aliases = {"opus", "claude-opus", "opus-4.8", "opus-4-8"},
            .display_name = "Claude Opus 4.8",
            .provider = "anthropic",
            .context_window = 200000,
            .max_output_tokens = 32000,
            .max_reasoning_tokens = 0,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching) |
                static_cast<uint32_t>(ModelCapability::TokenCounting) |
                static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Powerful,
            .pricing = {5.0, 25.0, 0.50, 6.25},
            .knowledge_cutoff = "2026-03",
            .constraints = kClaudeConstraints,
            .max_tool_calls = 32
        },
    };
}

// OpenAI models (updated March 2026)
std::vector<ModelInfo> build_openai_catalog() {
    return {
        // GPT-5.4 (recommended Codex default model)
        {
            .canonical_id = "gpt-5.4",
            .aliases = {"gpt-5.4-latest", "gpt-5", "gpt5"},
            .display_name = "GPT-5.4",
            .provider = "openai",
            .context_window = 200000,
            .max_output_tokens = 100000,
            .max_reasoning_tokens = 25000,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::Reasoning) |
                static_cast<uint32_t>(ModelCapability::Logprobs),
            .tier = ModelTier::Reasoning,
            .pricing = {2.50, 10.0, 1.25, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        // GPT-4o
        {
            .canonical_id = "gpt-4o-2024-08-06",
            .aliases = {"gpt-4o", "4o"},
            .display_name = "GPT-4o",
            .provider = "openai",
            .context_window = 128000,
            .max_output_tokens = 16384,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::Logprobs),
            .tier = ModelTier::Powerful,
            .pricing = {2.50, 10.0, 1.25, -1.0},  // cached input at half price
            .knowledge_cutoff = "2023-10",
            .constraints = kStandardConstraints,
        },
        // GPT-4o Mini
        {
            .canonical_id = "gpt-4o-mini-2024-07-18",
            .aliases = {"gpt-4o-mini", "4o-mini", "mini"},
            .display_name = "GPT-4o Mini",
            .provider = "openai",
            .context_window = 128000,
            .max_output_tokens = 16384,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::Logprobs),
            .tier = ModelTier::Fast,
            .pricing = {0.15, 0.60, 0.075, -1.0},
            .knowledge_cutoff = "2023-10",
            .constraints = kStandardConstraints,
        },
        // o3 Mini (reasoning model)
        {
            .canonical_id = "o3-mini-2025-01-31",
            .aliases = {"o3-mini", "o3mini"},
            .display_name = "o3 Mini",
            .provider = "openai",
            .context_window = 200000,
            .max_output_tokens = 100000,
            .max_reasoning_tokens = 25000,
            .capabilities = CAP_TOOLS | CAP_JSON |
                static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Reasoning,
            .pricing = {1.10, 4.40, 0.55, -1.0},
            .knowledge_cutoff = "2023-10",
            .constraints = kStandardConstraints,
        },
        // o1 (reasoning model)
        {
            .canonical_id = "o1-2024-12-17",
            .aliases = {"o1"},
            .display_name = "o1",
            .provider = "openai",
            .context_window = 200000,
            .max_output_tokens = 100000,
            .max_reasoning_tokens = 25000,
            .capabilities = CAP_TOOLS | CAP_JSON |
                static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Reasoning,
            .pricing = {15.0, 60.0, 7.50, -1.0},
            .knowledge_cutoff = "2023-10",
            .constraints = [] {
                ParameterConstraints c;
                c.temperature = {0.0, 2.0};
                return c;
            }(),
        },
        // GPT-4 Turbo
        {
            .canonical_id = "gpt-4-turbo-2024-04-09",
            .aliases = {"gpt-4-turbo", "gpt-4-turbo-preview"},
            .display_name = "GPT-4 Turbo",
            .provider = "openai",
            .context_window = 128000,
            .max_output_tokens = 4096,
            .capabilities = CAP_FULL,
            .tier = ModelTier::Powerful,
            .pricing = {10.0, 30.0, -1.0, -1.0},
            .knowledge_cutoff = "2023-12",
            .constraints = kStandardConstraints,
        },
        // GPT-3.5 Turbo
        {
            .canonical_id = "gpt-3.5-turbo-0125",
            .aliases = {"gpt-3.5-turbo", "gpt-3.5"},
            .display_name = "GPT-3.5 Turbo",
            .provider = "openai",
            .context_window = 16385,
            .max_output_tokens = 4096,
            .capabilities = CAP_TOOLS | CAP_JSON |
                static_cast<uint32_t>(ModelCapability::SystemPrompts),
            .tier = ModelTier::Fast,
            .pricing = {0.50, 1.50, -1.0, -1.0},
            .knowledge_cutoff = "2021-09",
            .constraints = kStandardConstraints,
        },
    };
}

// Kimi models (Moonshot AI)
std::vector<ModelInfo> build_kimi_catalog() {
    constexpr ModelCapabilities CAP_KIMI_MULTIMODAL =
        CAP_FULL | static_cast<uint32_t>(ModelCapability::VideoInput);
    return {
        {
            .canonical_id = "kimi-k2.7-code",
            .aliases = {"k2.7-code"},
            .display_name = "Kimi K2.7 Code",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_KIMI_MULTIMODAL | static_cast<uint32_t>(ModelCapability::Reasoning),
            .tier = ModelTier::Balanced,
            .pricing = {0.95, 4.0, 0.19, -1.0},
            .knowledge_cutoff = "2026-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "kimi-k2.6",
            .aliases = {"kimi-k2-6", "k2.6"},
            .display_name = "Kimi K2.6",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_KIMI_MULTIMODAL,
            .tier = ModelTier::Balanced,
            .pricing = {5.0, 20.0, -1.0, -1.0},
            .knowledge_cutoff = "2026-04",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "kimi-k2.5",
            .aliases = {"kimi-k2-5", "k2.5"},
            .display_name = "Kimi K2.5",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_KIMI_MULTIMODAL,
            .tier = ModelTier::Balanced,
            .pricing = {5.0, 20.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "kimi-for-coding",
            .aliases = {},
            .display_name = "Kimi for Coding",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_KIMI_MULTIMODAL,
            .tier = ModelTier::Balanced,
            .pricing = {5.0, 20.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "kimi-k2",
            .aliases = {},
            .display_name = "Kimi K2",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_KIMI_MULTIMODAL,
            .tier = ModelTier::Balanced,
            .pricing = {5.0, 20.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        // Generic fallback for other kimi models
        {
            .canonical_id = "kimi-default",
            .aliases = {"kimi-"},
            .display_name = "Kimi (Generic)",
            .provider = "kimi",
            .context_window = 128000,
            .max_output_tokens = 4096,
            .capabilities = CAP_KIMI_MULTIMODAL,
            .tier = ModelTier::Balanced,
            .pricing = {3.0, 12.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-01",
            .constraints = kStandardConstraints,
        },
    };
}

// Z.ai GLM models.
std::vector<ModelInfo> build_zai_catalog() {
    constexpr ModelCapabilities CAP_GLM =
        CAP_TOOLS | CAP_JSON |
        static_cast<uint32_t>(ModelCapability::Reasoning);

    return {
        {
            .canonical_id = "glm-5.2",
            .aliases = {},
            .display_name = "GLM-5.2",
            .provider = "zai",
            .context_window = 1'000'000,
            .max_output_tokens = 128'000,
            .max_reasoning_tokens = 64'000,
            .capabilities = CAP_GLM,
            .tier = ModelTier::Reasoning,
            .pricing = {0.0, 0.0, -1.0, -1.0},
            .knowledge_cutoff = "2026-04",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "glm-5-turbo",
            .aliases = {"glm-5-turbo-latest"},
            .display_name = "GLM-5 Turbo",
            .provider = "zai",
            .context_window = 200'000,
            .max_output_tokens = 64'000,
            .max_reasoning_tokens = 64'000,
            .capabilities = CAP_GLM,
            .tier = ModelTier::Reasoning,
            .pricing = {0.0, 0.0, -1.0, -1.0},
            .knowledge_cutoff = "2026-04",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "glm-5.1",
            .aliases = {},
            .display_name = "GLM-5.1",
            .provider = "zai",
            .context_window = 200'000,
            .max_output_tokens = 64'000,
            .max_reasoning_tokens = 64'000,
            .capabilities = CAP_GLM,
            .tier = ModelTier::Reasoning,
            .pricing = {0.0, 0.0, -1.0, -1.0},
            .knowledge_cutoff = "2026-04",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "glm-4.7",
            .aliases = {},
            .display_name = "GLM-4.7",
            .provider = "zai",
            .context_window = 200'000,
            .max_output_tokens = 64'000,
            .capabilities = CAP_TOOLS | CAP_JSON,
            .tier = ModelTier::Balanced,
            .pricing = {0.0, 0.0, -1.0, -1.0},
            .knowledge_cutoff = "2025-12",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "glm-4.5-air",
            .aliases = {"glm-4.5-air-latest"},
            .display_name = "GLM-4.5 Air",
            .provider = "zai",
            .context_window = 200'000,
            .max_output_tokens = 64'000,
            .capabilities = CAP_TOOLS | CAP_JSON,
            .tier = ModelTier::Fast,
            .pricing = {0.0, 0.0, -1.0, -1.0},
            .knowledge_cutoff = "2025-07",
            .constraints = kStandardConstraints,
        },
    };
}

// Gemini models (Google)
std::vector<ModelInfo> build_gemini_catalog() {
    constexpr ModelCapabilities CAP_GEMINI =
        CAP_FULL |
        static_cast<uint32_t>(ModelCapability::PromptCaching) |
        static_cast<uint32_t>(ModelCapability::TokenCounting);

    return {
        {
            .canonical_id = "gemini-3.1-pro-preview",
            .aliases = {"auto-gemini-3"},
            .display_name = "Gemini 3.1 Pro Preview",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Powerful,
            .pricing = {1.25, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = [] {
                ParameterConstraints c;
                c.temperature = {0.0, 2.0};
                return c;
            }(),
        },
        {
            .canonical_id = "gemini-3.1-pro-preview-customtools",
            .aliases = {},
            .display_name = "Gemini 3.1 Pro Preview (Custom Tools)",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Powerful,
            .pricing = {1.25, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-3-pro-preview",
            .aliases = {},
            .display_name = "Gemini 3 Pro Preview",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Powerful,
            .pricing = {1.25, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-3-flash-preview",
            .aliases = {},
            .display_name = "Gemini 3 Flash Preview",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Fast,
            .pricing = {0.30, 2.50, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-3.1-flash-lite-preview",
            .aliases = {},
            .display_name = "Gemini 3.1 Flash-Lite Preview",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Fast,
            .pricing = {0.10, 0.40, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-2.5-pro",
            .aliases = {"gemini-2.5", "gemini-pro-2.5", "gemini-pro-latest", "auto-gemini-2.5"},
            .display_name = "Gemini 2.5 Pro",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Powerful,
            .pricing = {1.25, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = [] {
                ParameterConstraints c;
                c.temperature = {0.0, 2.0};
                return c;
            }(),
        },
        {
            .canonical_id = "gemini-2.5-flash",
            .aliases = {"gemini-flash-latest"},
            .display_name = "Gemini 2.5 Flash",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Fast,
            .pricing = {0.30, 2.50, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-2.5-flash-lite",
            .aliases = {"gemini-flash-lite-latest"},
            .display_name = "Gemini 2.5 Flash-Lite",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 65'536,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Fast,
            .pricing = {0.10, 0.40, -1.0, -1.0},
            .knowledge_cutoff = "2025-01",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-2.0-flash",
            .aliases = {"gemini-2.0-flash-001", "gemini-2.0"},
            .display_name = "Gemini 2.0 Flash",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 8'192,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Fast,
            .pricing = {0.10, 0.40, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-2.0-flash-lite",
            .aliases = {"gemini-2.0-flash-lite-001"},
            .display_name = "Gemini 2.0 Flash-Lite",
            .provider = "gemini",
            .context_window = 1'048'576,
            .max_output_tokens = 8'192,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Fast,
            .pricing = {0.075, 0.30, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-1.5-pro-002",
            .aliases = {"gemini-1.5-pro", "gemini-1.5"},
            .display_name = "Gemini 1.5 Pro",
            .provider = "gemini",
            .context_window = 2'097'152,
            .max_output_tokens = 8'192,
            .capabilities = CAP_GEMINI,
            .tier = ModelTier::Balanced,
            .pricing = {1.25, 5.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-05",
            .constraints = kStandardConstraints,
        },
    };
}

// Grok models (xAI)
std::vector<ModelInfo> build_grok_catalog() {
    return {
        {
            .canonical_id = "grok-2-1212",
            .aliases = {"grok-2", "grok2"},
            .display_name = "Grok 2",
            .provider = "grok",
            .context_window = 128000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL,
            .tier = ModelTier::Balanced,
            .pricing = {2.0, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-07",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "grok-2-vision-1212",
            .aliases = {"grok-2-vision", "grok-vision"},
            .display_name = "Grok 2 Vision",
            .provider = "grok",
            .context_window = 128000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL,
            .tier = ModelTier::Balanced,
            .pricing = {2.0, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-07",
            .constraints = kStandardConstraints,
        },
        // Generic fallback
        {
            .canonical_id = "grok-default",
            .aliases = {"grok-"},
            .display_name = "Grok (Generic)",
            .provider = "grok",
            .context_window = 128000,
            .max_output_tokens = 4096,
            .capabilities = CAP_FULL,
            .tier = ModelTier::Balanced,
            .pricing = {2.0, 10.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-01",
            .constraints = kStandardConstraints,
        },
    };
}

// Qwen / DashScope models (Alibaba Cloud — verified 2026-03-27)
//
// Context windows and output limits follow the values documented by DashScope
// and cross-checked against the qwen-code project's token limit tables.
//
// Pricing is approximate (USD / 1M tokens) sourced from the DashScope pricing
// page; actual costs may vary by region and promotional tier.
//
// Thinking mode (enable_thinking=true) is supported on all Qwen3 models.
// It is activated at the protocol level via DashScopeProtocol::thinking_budget_.
std::vector<ModelInfo> build_qwen_catalog() {
    // Capabilities shared by all Qwen text models that support tool calling.
    // Note: reasoning flag marks support for enable_thinking mode, not that
    // the model always thinks — it's opt-in via the provider's thinking_budget.
    constexpr ModelCapabilities CAP_QWEN_TEXT =
        CAP_TOOLS | CAP_JSON |
        static_cast<uint32_t>(ModelCapability::Reasoning) |
        static_cast<uint32_t>(ModelCapability::PromptCaching);

    // Vision-capable models (qwen3-vl, qwen3.5-plus).
    constexpr ModelCapabilities CAP_QWEN_VL =
        CAP_FULL |
        static_cast<uint32_t>(ModelCapability::Reasoning) |
        static_cast<uint32_t>(ModelCapability::PromptCaching);

    // Coder models: tools + reasoning, no vision (VL is a separate model line).
    constexpr ModelCapabilities CAP_QWEN_CODER =
        CAP_TOOLS |
        static_cast<uint32_t>(ModelCapability::Reasoning) |
        static_cast<uint32_t>(ModelCapability::PromptCaching);

    return {
        // ── Qwen3-Coder family ────────────────────────────────────────────
        // Specialist coding models; 1M-token context window.
        {
            .canonical_id      = "qwen3-coder-plus",
            .aliases           = {"qwen3-coder"},
            .display_name      = "Qwen3 Coder Plus",
            .provider          = "qwen",
            .context_window    = 1'000'000,
            .max_output_tokens = 64'000,
            .capabilities      = CAP_QWEN_CODER,
            .tier              = ModelTier::Balanced,
            .pricing           = {0.43, 1.77, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
        {
            .canonical_id      = "qwen3-coder-flash",
            .aliases           = {},
            .display_name      = "Qwen3 Coder Flash",
            .provider          = "qwen",
            .context_window    = 1'000'000,
            .max_output_tokens = 64'000,
            .capabilities      = CAP_QWEN_CODER,
            .tier              = ModelTier::Fast,
            .pricing           = {0.043, 0.18, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
        // ── Qwen3 general-purpose family ─────────────────────────────────
        {
            .canonical_id      = "qwen3-max",
            .aliases           = {"qwen3-max-latest"},
            .display_name      = "Qwen3 Max",
            .provider          = "qwen",
            .context_window    = 256'000,
            .max_output_tokens = 64'000,
            .capabilities      = CAP_QWEN_TEXT,
            .tier              = ModelTier::Powerful,
            .pricing           = {1.07, 3.18, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
        {
            .canonical_id      = "qwen3-plus",
            .aliases           = {"qwen3-plus-latest", "qwen-plus-latest"},
            .display_name      = "Qwen3 Plus",
            .provider          = "qwen",
            .context_window    = 256'000,
            .max_output_tokens = 32'000,
            .capabilities      = CAP_QWEN_TEXT,
            .tier              = ModelTier::Balanced,
            .pricing           = {0.32, 0.96, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
        {
            .canonical_id      = "qwen3-turbo",
            .aliases           = {"qwen3-turbo-latest", "qwen-turbo-latest"},
            .display_name      = "Qwen3 Turbo",
            .provider          = "qwen",
            .context_window    = 1'000'000,
            .max_output_tokens = 8'000,
            .capabilities      = CAP_QWEN_TEXT,
            .tier              = ModelTier::Fast,
            .pricing           = {0.05, 0.20, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
        // ── Qwen3.5 family ────────────────────────────────────────────────
        // Vision-capable; 1M-token context window.
        {
            .canonical_id      = "qwen3.5-plus",
            .aliases           = {"qwen3.5-plus-latest"},
            .display_name      = "Qwen3.5 Plus",
            .provider          = "qwen",
            .context_window    = 1'000'000,
            .max_output_tokens = 64'000,
            .capabilities      = CAP_QWEN_VL,
            .tier              = ModelTier::Balanced,
            .pricing           = {0.50, 2.00, -1.0, -1.0},
            .knowledge_cutoff  = "2025-03",
            .constraints       = kStandardConstraints,
        },
        // ── Qwen3 VL family ───────────────────────────────────────────────
        // Multimodal vision-language models.
        {
            .canonical_id      = "qwen3-vl-plus",
            .aliases           = {"qwen3-vl", "qwen3-vl-plus-latest"},
            .display_name      = "Qwen3 VL Plus",
            .provider          = "qwen",
            .context_window    = 1'000'000,
            .max_output_tokens = 64'000,
            .capabilities      = CAP_QWEN_VL,
            .tier              = ModelTier::Balanced,
            .pricing           = {0.50, 2.00, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
        {
            .canonical_id      = "qwen3-vl-flash",
            .aliases           = {"qwen3-vl-flash-latest"},
            .display_name      = "Qwen3 VL Flash",
            .provider          = "qwen",
            .context_window    = 1'000'000,
            .max_output_tokens = 64'000,
            .capabilities      = CAP_QWEN_VL,
            .tier              = ModelTier::Fast,
            .pricing           = {0.05, 0.20, -1.0, -1.0},
            .knowledge_cutoff  = "2024-12",
            .constraints       = kStandardConstraints,
        },
    };
}

// Local/Open source models
std::vector<ModelInfo> build_local_catalog() {
    return {
        {
            .canonical_id = "llama-3.3-70b",
            .aliases = {"llama-3.3", "llama3.3", "llama-3-3-70b"},
            .display_name = "Llama 3.3 70B",
            .provider = "local",
            .context_window = 128000,
            .max_output_tokens = 4096,
            .capabilities = CAP_TOOLS | CAP_JSON,
            .tier = ModelTier::Balanced,
            .pricing = {0.0, 0.0, 0.0, 0.0},  // Local = free
            .knowledge_cutoff = "2023-12",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "llama-3.1-8b",
            .aliases = {"llama-3.1", "llama3.1", "llama3", "llama-3"},
            .display_name = "Llama 3.1 8B",
            .provider = "local",
            .context_window = 128000,
            .max_output_tokens = 4096,
            .capabilities = CAP_TOOLS | CAP_JSON,
            .tier = ModelTier::Fast,
            .pricing = {0.0, 0.0, 0.0, 0.0},
            .knowledge_cutoff = "2023-12",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "mixtral-8x7b",
            .aliases = {"mixtral", "mixtral-8x7b-instruct"},
            .display_name = "Mixtral 8x7B",
            .provider = "local",
            .context_window = 32000,
            .max_output_tokens = 4096,
            .capabilities = CAP_TOOLS | CAP_JSON,
            .tier = ModelTier::Balanced,
            .pricing = {0.0, 0.0, 0.0, 0.0},
            .knowledge_cutoff = "2023-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "mistral-large-2407",
            .aliases = {"mistral-large", "mistral"},
            .display_name = "Mistral Large",
            .provider = "local",
            .context_window = 128000,
            .max_output_tokens = 4096,
            .capabilities = CAP_TOOLS | CAP_JSON,
            .tier = ModelTier::Balanced,
            .pricing = {2.0, 6.0, 0.0, 0.0},  // API pricing if using Mistral API
            .knowledge_cutoff = "2024-07",
            .constraints = kStandardConstraints,
        },
    };
}

} // anonymous namespace

// ============================================================================
// ModelRegistry Implementation
// ============================================================================

ModelRegistry& ModelRegistry::instance() {
    // Use function-local static with proper initialization sequence
    static ModelRegistry* instance = []() {
        auto* reg = new ModelRegistry();
        reg->load_defaults();
        reg->defaults_loaded_ = true;
        return reg;
    }();
    return *instance;
}

ModelRegistry::ModelRegistry()
    : state_(std::make_shared<RegistryState>()) {}

ModelRegistry::~ModelRegistry() = default;

void ModelRegistry::load_defaults() {
    // Load all built-in catalogs
    auto register_all = [this](std::vector<ModelInfo> models) {
        for (auto& info : models) {
            register_model(std::move(info));
        }
    };
    
    register_all(build_anthropic_catalog());
    register_all(build_openai_catalog());
    register_all(build_kimi_catalog());
    register_all(build_zai_catalog());
    register_all(build_gemini_catalog());
    register_all(build_grok_catalog());
    register_all(build_qwen_catalog());
    register_all(build_local_catalog());
}

void ModelRegistry::register_model(ModelInfo info) {
    std::lock_guard lock(write_mutex_);

    if (info.canonical_id.empty()) return;

    auto next = std::make_shared<RegistryState>(
        *std::atomic_load_explicit(&state_, std::memory_order_acquire));
    const std::string canonical = info.canonical_id;
    if (auto existing = next->models.find(canonical); existing != next->models.end()) {
        for (const auto& alias : existing->second.aliases) {
            if (const auto alias_it = next->aliases.find(alias);
                alias_it != next->aliases.end() && alias_it->second == canonical) {
                next->aliases.erase(alias_it);
            }
        }
    }

    // Register the canonical entry
    next->models[canonical] = std::move(info);

    // Register aliases
    register_aliases(*next, next->models.at(canonical));
    std::shared_ptr<const RegistryState> published = std::move(next);
    std::atomic_store_explicit(&state_, std::move(published), std::memory_order_release);
}

bool ModelRegistry::merge_model(ModelInfo info) {
    std::lock_guard lock(write_mutex_);

    if (info.canonical_id.empty()) return false;

    auto next = std::make_shared<RegistryState>(
        *std::atomic_load_explicit(&state_, std::memory_order_acquire));
    const std::string discovered_id = info.canonical_id;
    auto existing = next->models.find(discovered_id);
    if (existing == next->models.end()) {
        if (const auto alias = next->aliases.find(discovered_id);
            alias != next->aliases.end()) {
            existing = next->models.find(alias->second);
        }
    }
    if (existing == next->models.end()) {
        next->models[discovered_id] = std::move(info);
        register_aliases(*next, next->models.at(discovered_id));
        std::shared_ptr<const RegistryState> published = std::move(next);
        std::atomic_store_explicit(&state_, std::move(published), std::memory_order_release);
        return true;
    }

    const std::string canonical = existing->second.canonical_id;
    ModelInfo merged = existing->second;
    if (!info.display_name.empty()
        && (merged.display_name.empty() || info.display_name != info.canonical_id)) {
        merged.display_name = std::move(info.display_name);
    }
    if (!info.provider.empty() && merged.provider.empty()) {
        merged.provider = std::move(info.provider);
    }
    if (info.context_window > 0) merged.context_window = info.context_window;
    if (info.max_output_tokens > 0) merged.max_output_tokens = info.max_output_tokens;
    if (info.max_reasoning_tokens > 0) merged.max_reasoning_tokens = info.max_reasoning_tokens;
    if (info.capabilities != 0) merged.capabilities |= info.capabilities;
    if (info.tier != ModelTier::Balanced || merged.tier == ModelTier::Balanced) {
        merged.tier = info.tier;
    }
    if (info.pricing.input_per_mtok > 0.0) merged.pricing.input_per_mtok = info.pricing.input_per_mtok;
    if (info.pricing.output_per_mtok > 0.0) merged.pricing.output_per_mtok = info.pricing.output_per_mtok;
    if (info.pricing.cached_input_per_mtok >= 0.0) {
        merged.pricing.cached_input_per_mtok = info.pricing.cached_input_per_mtok;
    }
    if (info.pricing.prompt_caching_write_per_mtok >= 0.0) {
        merged.pricing.prompt_caching_write_per_mtok = info.pricing.prompt_caching_write_per_mtok;
    }
    if (!info.knowledge_cutoff.empty()) merged.knowledge_cutoff = std::move(info.knowledge_cutoff);
    if (!info.deprecation_date.empty()) merged.deprecation_date = std::move(info.deprecation_date);
    if (!info.expected_completion_date.empty()) {
        merged.expected_completion_date = std::move(info.expected_completion_date);
    }
    if (info.constraints.temperature) merged.constraints.temperature = info.constraints.temperature;
    if (info.constraints.top_p) merged.constraints.top_p = info.constraints.top_p;
    if (info.constraints.frequency_penalty) {
        merged.constraints.frequency_penalty = info.constraints.frequency_penalty;
    }
    if (info.constraints.presence_penalty) {
        merged.constraints.presence_penalty = info.constraints.presence_penalty;
    }
    if (info.constraints.max_tokens_min != 1) {
        merged.constraints.max_tokens_min = info.constraints.max_tokens_min;
    }
    if (info.constraints.max_tokens_max > 0) {
        merged.constraints.max_tokens_max = info.constraints.max_tokens_max;
    }
    if (info.max_tool_calls != 32) merged.max_tool_calls = info.max_tool_calls;
    for (auto& alias : info.aliases) {
        if (std::ranges::find(merged.aliases, alias) == merged.aliases.end()) {
            merged.aliases.push_back(std::move(alias));
        }
    }

    for (const auto& alias : existing->second.aliases) {
        if (const auto alias_it = next->aliases.find(alias);
            alias_it != next->aliases.end() && alias_it->second == canonical) {
            next->aliases.erase(alias_it);
        }
    }
    existing->second = std::move(merged);
    register_aliases(*next, existing->second);
    std::shared_ptr<const RegistryState> published = std::move(next);
    std::atomic_store_explicit(&state_, std::move(published), std::memory_order_release);
    return false;
}

void ModelRegistry::register_aliases(RegistryState& state, const ModelInfo& info) {
    for (const auto& alias : info.aliases) {
        state.aliases[alias] = info.canonical_id;
    }
}

void ModelRegistry::register_models(std::span<const ModelInfo> models) {
    for (const auto& info : models) {
        register_model(info);
    }
}

std::shared_ptr<const ModelInfo> ModelRegistry::lookup(std::string_view model_id) const {
    auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);
    
    // Try canonical ID first
    auto it = state->models.find(std::string(model_id));
    if (it != state->models.end()) {
        return {std::move(state), &it->second};
    }
    
    // Try alias lookup
    auto alias_it = state->aliases.find(std::string(model_id));
    if (alias_it != state->aliases.end()) {
        auto canon_it = state->models.find(alias_it->second);
        if (canon_it != state->models.end()) {
            return {std::move(state), &canon_it->second};
        }
    }
    
    return nullptr;
}

bool ModelRegistry::has_model(std::string_view model_id) const {
    return lookup(model_id) != nullptr;
}

int ModelRegistry::get_max_context_size(std::string_view model_id) const {
    if (const auto info = lookup(model_id)) {
        return info->context_window;
    }
    return 0;
}

int ModelRegistry::get_max_context_size_legacy(std::string_view model_name) noexcept {
    return legacy_get_max_context_size(model_name);
}

ModelCapabilities ModelRegistry::get_capabilities(std::string_view model_id) const {
    if (const auto info = lookup(model_id)) {
        return info->capabilities;
    }
    return 0;
}

bool ModelRegistry::supports(std::string_view model_id, ModelCapability cap) const {
    return has_capability(get_capabilities(model_id), cap);
}

std::optional<ModelTier> ModelRegistry::get_tier(std::string_view model_id) const {
    if (const auto info = lookup(model_id)) {
        return info->tier;
    }
    return std::nullopt;
}

std::optional<ModelInfo> ModelRegistry::get_info(std::string_view model_id) const {
    if (const auto info = lookup(model_id)) {
        return *info;
    }
    return std::nullopt;
}

bool ModelRegistry::validate_parameter(std::string_view model_id,
                                       std::string_view param_name,
                                       double value) const {
    if (const auto info = lookup(model_id)) {
        return info->validate_parameter(param_name, value);
    }
    return false;  // Unknown model
}

bool ModelRegistry::validate_max_tokens(std::string_view model_id, int max_tokens) const {
    if (const auto info = lookup(model_id)) {
        int limit = info->effective_max_tokens();
        if (limit > 0 && max_tokens > limit) return false;
        if (max_tokens < info->constraints.max_tokens_min) return false;
        return true;
    }
    return false;
}

std::vector<ModelInfo> ModelRegistry::filter_by_tier(ModelTier tier) const {
    auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);
    
    std::vector<ModelInfo> result;
    for (const auto& [_, info] : state->models) {
        if (info.tier == tier) {
            result.push_back(info);
        }
    }
    return result;
}

std::vector<ModelInfo> ModelRegistry::filter_by_capability(ModelCapability cap) const {
    auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);
    
    std::vector<ModelInfo> result;
    for (const auto& [_, info] : state->models) {
        if (info.supports(cap)) {
            result.push_back(info);
        }
    }
    return result;
}

std::vector<std::string> ModelRegistry::list_models() const {
    auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);
    
    std::vector<std::string> result;
    result.reserve(state->models.size());
    for (const auto& [id, _] : state->models) {
        result.push_back(id);
    }
    return result;
}

std::vector<ModelInfo> ModelRegistry::get_by_provider(std::string_view provider) const {
    auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);
    
    std::vector<ModelInfo> result;
    for (const auto& [_, info] : state->models) {
        if (info.provider == provider) {
            result.push_back(info);
        }
    }
    return result;
}

double ModelRegistry::estimate_cost(std::string_view model_id,
                                    int input_tokens,
                                    int output_tokens,
                                    bool use_cached_input) const {
    if (const auto info = lookup(model_id)) {
        return info->estimate_cost(input_tokens, output_tokens, use_cached_input);
    }
    return -1.0;  // Unknown model
}

void ModelRegistry::clear() {
    std::lock_guard lock(write_mutex_);
    std::shared_ptr<const RegistryState> empty = std::make_shared<RegistryState>();
    std::atomic_store_explicit(&state_, std::move(empty), std::memory_order_release);
    defaults_loaded_ = false;
}

size_t ModelRegistry::size() const {
    return std::atomic_load_explicit(&state_, std::memory_order_acquire)->models.size();
}

// ============================================================================
// JSON Serialization
// ============================================================================

static bool get_json_string(simdjson::dom::object object,
                            const char* key,
                            std::string& out) {
    std::string_view value;
    if (object[key].get(value) != simdjson::SUCCESS) {
        return false;
    }
    out.assign(value.data(), value.size());
    return true;
}

[[nodiscard]] static int32_t clamp_json_i32(int64_t value) {
    if (value < 0) return 0;
    if (value > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(value);
}

static bool get_json_int(simdjson::dom::object object,
                         const char* key,
                         int32_t& out) {
    int64_t value = 0;
    if (object[key].get(value) != simdjson::SUCCESS) {
        return false;
    }
    out = clamp_json_i32(value);
    return true;
}

static bool get_json_double(simdjson::dom::object object,
                            const char* key,
                            double& out) {
    double value = 0.0;
    if (object[key].get(value) != simdjson::SUCCESS) {
        return false;
    }
    out = value;
    return true;
}

static void parse_constraints_range(simdjson::dom::object constraints_obj,
                                    const char* key,
                                    std::optional<ParameterConstraints::Range>& out) {
    simdjson::dom::array range;
    if (constraints_obj[key].get(range) != simdjson::SUCCESS || range.size() != 2) {
        return;
    }

    auto it = range.begin();
    double min = 0.0;
    double max = 0.0;
    if ((*it).get(min) != simdjson::SUCCESS) return;
    ++it;
    if ((*it).get(max) != simdjson::SUCCESS) return;
    out = ParameterConstraints::Range{min, max};
}

int ModelRegistry::load_from_json(std::string_view json_data) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_data.data(), json_data.size()).get(doc) != simdjson::SUCCESS) {
        return -1;
    }

    simdjson::dom::array models;
    if (doc["models"].get(models) != simdjson::SUCCESS) {
        if (doc.get(models) != simdjson::SUCCESS) {
            return -1;
        }
    }

    int loaded = 0;
    for (simdjson::dom::element element : models) {
        simdjson::dom::object model_obj;
        if (element.get(model_obj) != simdjson::SUCCESS) {
            continue;
        }

        std::string canonical_id;
        if (!get_json_string(model_obj, "canonical_id", canonical_id)) {
            get_json_string(model_obj, "id", canonical_id);
        }
        if (canonical_id.empty()) {
            continue;
        }

        ModelInfo info;
        if (const auto existing = lookup(canonical_id)) {
            info = *existing;
        }
        info.canonical_id = canonical_id;

        std::string display_name;
        if (get_json_string(model_obj, "display_name", display_name)) {
            info.display_name = display_name;
        }
        if (info.display_name.empty()) {
            info.display_name = info.canonical_id;
        }
        get_json_string(model_obj, "provider", info.provider);
        get_json_int(model_obj, "context_window", info.context_window);
        get_json_int(model_obj, "max_output_tokens", info.max_output_tokens);
        get_json_int(model_obj, "max_reasoning_tokens", info.max_reasoning_tokens);
        get_json_string(model_obj, "knowledge_cutoff", info.knowledge_cutoff);
        get_json_string(model_obj, "deprecation_date", info.deprecation_date);
        get_json_string(model_obj, "expected_completion_date", info.expected_completion_date);
        get_json_int(model_obj, "max_tool_calls", info.max_tool_calls);

        simdjson::dom::array aliases;
        if (model_obj["aliases"].get(aliases) == simdjson::SUCCESS) {
            info.aliases.clear();
            for (simdjson::dom::element alias_el : aliases) {
                std::string_view alias;
                if (alias_el.get(alias) == simdjson::SUCCESS && !alias.empty()) {
                    info.aliases.emplace_back(alias.data(), alias.size());
                }
            }
        }

        std::string tier;
        if (get_json_string(model_obj, "tier", tier)) {
            if (auto parsed = tier_from_string(tier)) {
                info.tier = *parsed;
            }
        }

        simdjson::dom::array capabilities;
        if (model_obj["capabilities"].get(capabilities) == simdjson::SUCCESS) {
            info.capabilities = 0;
            for (simdjson::dom::element cap_el : capabilities) {
                std::string_view cap;
                if (cap_el.get(cap) == simdjson::SUCCESS) {
                    if (auto parsed = model_capability_from_string(cap)) {
                        info.capabilities |= static_cast<uint32_t>(*parsed);
                    }
                }
            }
        }

        simdjson::dom::object pricing_obj;
        if (model_obj["pricing"].get(pricing_obj) == simdjson::SUCCESS) {
            get_json_double(pricing_obj, "input", info.pricing.input_per_mtok);
            get_json_double(pricing_obj, "input_per_mtok", info.pricing.input_per_mtok);
            get_json_double(pricing_obj, "output", info.pricing.output_per_mtok);
            get_json_double(pricing_obj, "output_per_mtok", info.pricing.output_per_mtok);
            get_json_double(pricing_obj, "cached_input", info.pricing.cached_input_per_mtok);
            get_json_double(pricing_obj, "cached_input_per_mtok", info.pricing.cached_input_per_mtok);
            get_json_double(pricing_obj, "prompt_caching_write", info.pricing.prompt_caching_write_per_mtok);
            get_json_double(pricing_obj, "prompt_caching_write_per_mtok", info.pricing.prompt_caching_write_per_mtok);
        }

        simdjson::dom::object constraints_obj;
        if (model_obj["constraints"].get(constraints_obj) == simdjson::SUCCESS) {
            parse_constraints_range(constraints_obj, "temperature", info.constraints.temperature);
            parse_constraints_range(constraints_obj, "top_p", info.constraints.top_p);
            parse_constraints_range(constraints_obj, "frequency_penalty", info.constraints.frequency_penalty);
            parse_constraints_range(constraints_obj, "presence_penalty", info.constraints.presence_penalty);
            get_json_int(constraints_obj, "max_tokens_min", info.constraints.max_tokens_min);
            get_json_int(constraints_obj, "max_tokens_max", info.constraints.max_tokens_max);
        }

        register_model(std::move(info));
        ++loaded;
    }

    return loaded;
}

int ModelRegistry::load_from_json_file(const std::string& file_path) {
    // Read file contents
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return -1;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    
    return load_from_json(content);
}

std::string ModelRegistry::export_to_json() const {
    auto state = std::atomic_load_explicit(&state_, std::memory_order_acquire);

    core::utils::JsonWriter writer;
    {
        auto root = writer.object();
        writer.key("models");
        {
            auto models = writer.array();
            bool first_model = true;

            for (const auto& [id, info] : state->models) {
                if (!first_model) writer.comma();
                first_model = false;

                auto model = writer.object();
                writer.kv_str("canonical_id", id).comma();
                writer.kv_str("display_name", info.display_name).comma();
                writer.kv_str("provider", info.provider).comma();

                writer.key("aliases");
                {
                    auto aliases = writer.array();
                    for (size_t i = 0; i < info.aliases.size(); ++i) {
                        if (i > 0) writer.comma();
                        writer.str(info.aliases[i]);
                    }
                }
                writer.comma();

                writer.kv_num("context_window", info.context_window).comma();
                writer.kv_num("max_output_tokens", info.max_output_tokens).comma();
                if (info.max_reasoning_tokens > 0) {
                    writer.kv_num("max_reasoning_tokens", info.max_reasoning_tokens).comma();
                }

                writer.kv_str("tier", to_string(info.tier)).comma();

                writer.key("capabilities");
                {
                    auto capabilities = writer.array();
                    bool first_cap = true;
                    for (const auto& [cap, name] : kModelCapabilityNames) {
                        if (!has_capability(info.capabilities, cap)) continue;
                        if (!first_cap) writer.comma();
                        first_cap = false;
                        writer.str(name);
                    }
                }
                writer.comma();

                writer.key("pricing");
                {
                    auto pricing = writer.object();
                    writer.kv_float("input", info.pricing.input_per_mtok, 3).comma();
                    writer.kv_float("output", info.pricing.output_per_mtok, 3);
                    if (info.pricing.has_cached_pricing()) {
                        writer.comma().kv_float("cached_input", info.pricing.cached_input_per_mtok, 3);
                    }
                }

                if (!info.knowledge_cutoff.empty()) {
                    writer.comma().kv_str("knowledge_cutoff", info.knowledge_cutoff);
                }
                if (!info.deprecation_date.empty()) {
                    writer.comma().kv_str("deprecation_date", info.deprecation_date);
                }
            }
        }
    }

    return std::move(writer).take();
}

// ============================================================================
// Free Functions (Backward Compatible API)
// ============================================================================

namespace {

bool has_1m_context_suffix(std::string_view model_id) {
    const std::string trimmed = core::utils::str::trim_ascii_copy(model_id);
    if (trimmed.size() < 4) return false;

    const std::size_t tail = trimmed.size() - 4;
    return trimmed[tail] == '['
        && trimmed[tail + 1] == '1'
        && (trimmed[tail + 2] == 'm' || trimmed[tail + 2] == 'M')
        && trimmed[tail + 3] == ']';
}

std::string normalize_context_lookup_model(std::string_view model_id) {
    std::string normalized = core::utils::str::trim_ascii_copy(model_id);
    if (normalized.empty()) return normalized;

    if (has_1m_context_suffix(normalized)) {
        normalized = core::utils::str::trim_ascii_copy(
            std::string_view(normalized).substr(0, normalized.size() - 4));
    }

    const std::string lowered = core::utils::str::to_lower_ascii_copy(normalized);
    if (lowered == "sonnet") {
        return "claude-sonnet-5";
    }
    if (lowered == "fable" || lowered == "best") {
        return "claude-fable-5";
    }
    if (lowered == "haiku") {
        return "claude-haiku-4-5";
    }
    if (lowered == "opusplan") {
        return "claude-sonnet-5";
    }
    if (lowered == "opus") {
        return "claude-opus-4-8";
    }
    return normalized;
}

} // namespace

int get_max_context_size(std::string_view model_id) {
    if (model_id.empty()) return 0;
    if (has_1m_context_suffix(model_id)) {
        return 1'000'000;
    }

    const std::string original = core::utils::str::trim_ascii_copy(model_id);
    const std::string normalized = normalize_context_lookup_model(model_id);

    // Try new registry first
    auto& registry = ModelRegistry::instance();
    int size = registry.get_max_context_size(normalized);
    if (size > 0) {
        return size;
    }

    // Fall back to legacy heuristic matching on the normalized model.
    size = ModelRegistry::get_max_context_size_legacy(normalized);
    if (size > 0) {
        return size;
    }

    // If normalization changed the model (aliases like "sonnet"/"opus"), stop here.
    // This prevents older alias registrations from overriding newer Claude mappings.
    if (normalized != original) {
        return 0;
    }

    // Try the original model string too (in case external code passed a
    // canonical model we do not normalize specially).
    size = registry.get_max_context_size(model_id);
    if (size > 0) {
        return size;
    }

    return ModelRegistry::get_max_context_size_legacy(model_id);
}

bool model_supports(std::string_view model_id, ModelCapability cap) {
    return ModelRegistry::instance().supports(model_id, cap);
}

std::optional<ModelTier> get_model_tier(std::string_view model_id) {
    return ModelRegistry::instance().get_tier(model_id);
}

} // namespace core::llm
