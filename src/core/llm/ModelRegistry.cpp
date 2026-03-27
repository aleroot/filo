#include "ModelRegistry.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>

namespace core::llm {

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
    { "kimi-k2.5",         256000 },
    { "kimi-for-coding",   256000 },
    { "kimi-k2",           256000 },
    { "kimi-for-university", 256000 },
    { "kimi-k1.5",         256000 },
    { "kimi-",             128000 },

    // -----------------------------------------------------------------------
    // Anthropic Claude Models (Updated March 2026)
    // -----------------------------------------------------------------------
    { "claude-3-opus",     200000 },
    { "claude-3-sonnet",   200000 },
    { "claude-3-5-sonnet", 200000 },
    { "claude-3-haiku",    200000 },
    { "claude-",           200000 },

    // -----------------------------------------------------------------------
    // OpenAI GPT Models
    // -----------------------------------------------------------------------
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
    { "gemini-1.5",        1000000 },
    { "gemini-2.0",        1000000 },
    { "gemini-2.5",        1000000 },
    { "gemini-",           32000 },

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

// Anthropic models (updated March 2026)
std::vector<ModelInfo> build_anthropic_catalog() {
    return {
        // Claude 3.7 Sonnet (latest as of March 2026)
        {
            .canonical_id = "claude-3-7-sonnet-20250219",
            .aliases = {"claude-3-7-sonnet", "claude-3.7-sonnet", "sonnet-3.7"},
            .display_name = "Claude 3.7 Sonnet",
            .provider = "anthropic",
            .context_window = 200000,
            .max_output_tokens = 8192,
            .max_reasoning_tokens = 0,
            .capabilities = CAP_FULL | 
                static_cast<uint32_t>(ModelCapability::PromptCaching) |
                static_cast<uint32_t>(ModelCapability::TokenCounting),
            .tier = ModelTier::Balanced,
            .pricing = {3.0, 15.0, 0.30, 3.75},  // $3/Mtok input, $15/Mtok output
            .knowledge_cutoff = "2025-01",
            .constraints = kClaudeConstraints,
            .max_tool_calls = 32
        },
        // Claude 3.5 Haiku
        {
            .canonical_id = "claude-3-5-haiku-20241022",
            .aliases = {"claude-3-5-haiku", "claude-3.5-haiku", "haiku-3.5"},
            .display_name = "Claude 3.5 Haiku",
            .provider = "anthropic",
            .context_window = 200000,
            .max_output_tokens = 4096,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching),
            .tier = ModelTier::Fast,
            .pricing = {0.80, 4.0, 0.08, 1.0},
            .knowledge_cutoff = "2024-07",
            .constraints = kClaudeConstraints,
        },
        // Claude 3 Opus
        {
            .canonical_id = "claude-3-opus-20240229",
            .aliases = {"claude-3-opus", "opus", "claude-opus"},
            .display_name = "Claude 3 Opus",
            .provider = "anthropic",
            .context_window = 200000,
            .max_output_tokens = 4096,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching),
            .tier = ModelTier::Powerful,
            .pricing = {15.0, 75.0, 1.50, 18.75},
            .knowledge_cutoff = "2024-02",
            .constraints = kClaudeConstraints,
        },
        // Claude 3.5 Sonnet (legacy version)
        {
            .canonical_id = "claude-3-5-sonnet-20241022",
            .aliases = {"claude-3-5-sonnet-legacy"},
            .display_name = "Claude 3.5 Sonnet (Legacy)",
            .provider = "anthropic",
            .context_window = 200000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching),
            .tier = ModelTier::Balanced,
            .pricing = {3.0, 15.0, 0.30, 3.75},
            .knowledge_cutoff = "2024-04",
            .deprecation_date = "2025-06-01",
            .constraints = kClaudeConstraints,
        },
    };
}

// OpenAI models (updated March 2026)
std::vector<ModelInfo> build_openai_catalog() {
    return {
        // GPT-4o (latest)
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
    return {
        {
            .canonical_id = "kimi-k2.5",
            .aliases = {"kimi-k2-5", "k2.5"},
            .display_name = "Kimi K2.5",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL,
            .tier = ModelTier::Balanced,
            .pricing = {5.0, 20.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "kimi-for-coding",
            .aliases = {"kimi-coding"},
            .display_name = "Kimi for Coding",
            .provider = "kimi",
            .context_window = 256000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL,
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
            .capabilities = CAP_FULL,
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
            .capabilities = CAP_FULL,
            .tier = ModelTier::Balanced,
            .pricing = {3.0, 12.0, -1.0, -1.0},
            .knowledge_cutoff = "2024-01",
            .constraints = kStandardConstraints,
        },
    };
}

// Gemini models (Google)
std::vector<ModelInfo> build_gemini_catalog() {
    return {
        {
            .canonical_id = "gemini-2.5-pro-exp-03-25",
            .aliases = {"gemini-2.5-pro", "gemini-2.5", "gemini-pro-2.5"},
            .display_name = "Gemini 2.5 Pro",
            .provider = "gemini",
            .context_window = 1000000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching),
            .tier = ModelTier::Powerful,
            .pricing = {1.25, 10.0, -1.0, -1.0},  // Context caching: $4.50/Mtok
            .knowledge_cutoff = "2025-01",
            .constraints = [] {
                ParameterConstraints c;
                c.temperature = {0.0, 2.0};
                return c;
            }(),
        },
        {
            .canonical_id = "gemini-2.0-flash-001",
            .aliases = {"gemini-2.0-flash", "gemini-flash", "gemini-2.0"},
            .display_name = "Gemini 2.0 Flash",
            .provider = "gemini",
            .context_window = 1000000,
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching),
            .tier = ModelTier::Fast,
            .pricing = {0.10, 0.40, -1.0, -1.0},
            .knowledge_cutoff = "2024-06",
            .constraints = kStandardConstraints,
        },
        {
            .canonical_id = "gemini-1.5-pro-002",
            .aliases = {"gemini-1.5-pro", "gemini-1.5"},
            .display_name = "Gemini 1.5 Pro",
            .provider = "gemini",
            .context_window = 2000000,  // 2M tokens!
            .max_output_tokens = 8192,
            .capabilities = CAP_FULL |
                static_cast<uint32_t>(ModelCapability::PromptCaching),
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

ModelRegistry::ModelRegistry() = default;

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
    register_all(build_gemini_catalog());
    register_all(build_grok_catalog());
    register_all(build_qwen_catalog());
    register_all(build_local_catalog());
}

void ModelRegistry::register_model(ModelInfo info) {
    std::unique_lock lock(mutex_);
    
    std::string canonical = info.canonical_id;
    
    // Register the canonical entry
    models_[canonical] = info;
    
    // Register aliases
    register_aliases(info);
}

void ModelRegistry::register_aliases(const ModelInfo& info) {
    // This is called with lock already held
    for (const auto& alias : info.aliases) {
        alias_map_[alias] = info.canonical_id;
    }
}

void ModelRegistry::register_models(std::span<const ModelInfo> models) {
    for (const auto& info : models) {
        register_model(info);
    }
}

const ModelInfo* ModelRegistry::lookup(std::string_view model_id) const {
    std::shared_lock lock(mutex_);
    
    // Try canonical ID first
    auto it = models_.find(std::string(model_id));
    if (it != models_.end()) {
        return &it->second;
    }
    
    // Try alias lookup
    auto alias_it = alias_map_.find(std::string(model_id));
    if (alias_it != alias_map_.end()) {
        auto canon_it = models_.find(alias_it->second);
        if (canon_it != models_.end()) {
            return &canon_it->second;
        }
    }
    
    return nullptr;
}

bool ModelRegistry::has_model(std::string_view model_id) const {
    return lookup(model_id) != nullptr;
}

int ModelRegistry::get_max_context_size(std::string_view model_id) const {
    if (const auto* info = lookup(model_id)) {
        return info->context_window;
    }
    return 0;
}

int ModelRegistry::get_max_context_size_legacy(std::string_view model_name) noexcept {
    return legacy_get_max_context_size(model_name);
}

ModelCapabilities ModelRegistry::get_capabilities(std::string_view model_id) const {
    if (const auto* info = lookup(model_id)) {
        return info->capabilities;
    }
    return 0;
}

bool ModelRegistry::supports(std::string_view model_id, ModelCapability cap) const {
    return has_capability(get_capabilities(model_id), cap);
}

std::optional<ModelTier> ModelRegistry::get_tier(std::string_view model_id) const {
    if (const auto* info = lookup(model_id)) {
        return info->tier;
    }
    return std::nullopt;
}

std::optional<ModelInfo> ModelRegistry::get_info(std::string_view model_id) const {
    if (const auto* info = lookup(model_id)) {
        return *info;
    }
    return std::nullopt;
}

bool ModelRegistry::validate_parameter(std::string_view model_id,
                                       std::string_view param_name,
                                       double value) const {
    if (const auto* info = lookup(model_id)) {
        return info->validate_parameter(param_name, value);
    }
    return false;  // Unknown model
}

bool ModelRegistry::validate_max_tokens(std::string_view model_id, int max_tokens) const {
    if (const auto* info = lookup(model_id)) {
        int limit = info->effective_max_tokens();
        if (limit > 0 && max_tokens > limit) return false;
        if (max_tokens < info->constraints.max_tokens_min) return false;
        return true;
    }
    return false;
}

std::vector<const ModelInfo*> ModelRegistry::filter_by_tier(ModelTier tier) const {
    std::shared_lock lock(mutex_);
    
    std::vector<const ModelInfo*> result;
    for (const auto& [_, info] : models_) {
        if (info.tier == tier) {
            result.push_back(&info);
        }
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::filter_by_capability(ModelCapability cap) const {
    std::shared_lock lock(mutex_);
    
    std::vector<const ModelInfo*> result;
    for (const auto& [_, info] : models_) {
        if (info.supports(cap)) {
            result.push_back(&info);
        }
    }
    return result;
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::shared_lock lock(mutex_);
    
    std::vector<std::string> result;
    result.reserve(models_.size());
    for (const auto& [id, _] : models_) {
        result.push_back(id);
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::get_by_provider(std::string_view provider) const {
    std::shared_lock lock(mutex_);
    
    std::vector<const ModelInfo*> result;
    for (const auto& [_, info] : models_) {
        if (info.provider == provider) {
            result.push_back(&info);
        }
    }
    return result;
}

double ModelRegistry::estimate_cost(std::string_view model_id,
                                    int input_tokens,
                                    int output_tokens,
                                    bool use_cached_input) const {
    if (const auto* info = lookup(model_id)) {
        return info->estimate_cost(input_tokens, output_tokens, use_cached_input);
    }
    return -1.0;  // Unknown model
}

void ModelRegistry::clear() {
    std::unique_lock lock(mutex_);
    models_.clear();
    alias_map_.clear();
    defaults_loaded_ = false;
}

size_t ModelRegistry::size() const {
    std::shared_lock lock(mutex_);
    return models_.size();
}

// ============================================================================
// JSON Serialization
// ============================================================================

// Helper to convert capability enum to string
[[nodiscard]] static const char* capability_to_string(ModelCapability cap) {
    switch (cap) {
        case ModelCapability::TextInput: return "text_input";
        case ModelCapability::TextOutput: return "text_output";
        case ModelCapability::Vision: return "vision";
        case ModelCapability::FunctionCalling: return "function_calling";
        case ModelCapability::JsonMode: return "json_mode";
        case ModelCapability::Streaming: return "streaming";
        case ModelCapability::Reasoning: return "reasoning";
        case ModelCapability::SystemPrompts: return "system_prompts";
        case ModelCapability::ParallelToolCalls: return "parallel_tool_calls";
        case ModelCapability::TokenCounting: return "token_counting";
        case ModelCapability::PromptCaching: return "prompt_caching";
        case ModelCapability::Logprobs: return "logprobs";
        case ModelCapability::Embeddings: return "embeddings";
        default: return "unknown";
    }
}

int ModelRegistry::load_from_json(std::string_view json_data) {
    // Simple JSON parsing without external library dependency
    // This is a minimal implementation for demonstration
    // In production, use simdjson or nlohmann/json
    
    // For now, return 0 to indicate no models loaded (feature stub)
    // Full implementation would parse the JSON and register models
    (void)json_data;
    return 0;
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
    std::shared_lock lock(mutex_);
    
    std::string json = "{\n  \"models\": [\n";
    bool first_model = true;
    
    for (const auto& [id, info] : models_) {
        if (!first_model) json += ",\n";
        first_model = false;
        
        json += "    {\n";
        json += std::format("      \"canonical_id\": \"{}\",\n", id);
        json += std::format("      \"display_name\": \"{}\",\n", info.display_name);
        json += std::format("      \"provider\": \"{}\",\n", info.provider);
        
        // Aliases
        json += "      \"aliases\": [";
        for (size_t i = 0; i < info.aliases.size(); ++i) {
            if (i > 0) json += ", ";
            json += std::format("\"{}\"", info.aliases[i]);
        }
        json += "],\n";
        
        // Token limits
        json += std::format("      \"context_window\": {},\n", info.context_window);
        json += std::format("      \"max_output_tokens\": {},\n", info.max_output_tokens);
        if (info.max_reasoning_tokens > 0) {
            json += std::format("      \"max_reasoning_tokens\": {},\n", info.max_reasoning_tokens);
        }
        
        // Tier
        json += std::format("      \"tier\": \"{}\",\n", std::string(to_string(info.tier)));
        
        // Capabilities
        json += "      \"capabilities\": [";
        bool first_cap = true;
        for (uint32_t i = 0; i < 32; ++i) {
            ModelCapability cap = static_cast<ModelCapability>(1u << i);
            if (has_capability(info.capabilities, cap)) {
                if (!first_cap) json += ", ";
                first_cap = false;
                json += std::format("\"{}\"", capability_to_string(cap));
            }
        }
        json += "],\n";
        
        // Pricing
        json += "      \"pricing\": {\n";
        json += std::format("        \"input\": {:.2f},\n", info.pricing.input_per_mtok);
        json += std::format("        \"output\": {:.2f}", info.pricing.output_per_mtok);
        if (info.pricing.has_cached_pricing()) {
            json += std::format(",\n        \"cached_input\": {:.2f}", info.pricing.cached_input_per_mtok);
        }
        json += "\n      },\n";
        
        // Knowledge cutoff
        if (!info.knowledge_cutoff.empty()) {
            json += std::format("      \"knowledge_cutoff\": \"{}\"\n", info.knowledge_cutoff);
        }
        
        // Deprecation
        if (!info.deprecation_date.empty()) {
            json += std::format(",\n      \"deprecation_date\": \"{}\"\n", info.deprecation_date);
        }
        
        json += "    }";
    }
    
    json += "\n  ]\n}";
    return json;
}

// ============================================================================
// Free Functions (Backward Compatible API)
// ============================================================================

int get_max_context_size(std::string_view model_id) {
    // Try new registry first
    auto& registry = ModelRegistry::instance();
    int size = registry.get_max_context_size(model_id);
    if (size > 0) {
        return size;
    }
    
    // Fall back to legacy heuristic matching
    return ModelRegistry::get_max_context_size_legacy(model_id);
}

bool model_supports(std::string_view model_id, ModelCapability cap) {
    return ModelRegistry::instance().supports(model_id, cap);
}

std::optional<ModelTier> get_model_tier(std::string_view model_id) {
    return ModelRegistry::instance().get_tier(model_id);
}

} // namespace core::llm
