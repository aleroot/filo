#pragma once

#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::llm {

// ============================================================================
// Model Capabilities Bitmap (efficient feature testing)
// ============================================================================

/**
 * @brief Model capability flags for efficient feature detection.
 * 
 * These flags represent what a model can do. Use has_capability() to test.
 */
enum class ModelCapability : uint32_t {
    None                = 0,
    TextInput           = 1 << 0,   ///< Basic text input
    TextOutput          = 1 << 1,   ///< Basic text output
    Vision              = 1 << 2,   ///< Image understanding (multimodal)
    FunctionCalling     = 1 << 3,   ///< Tool/function calling
    JsonMode            = 1 << 4,   ///< Structured JSON output mode
    Streaming           = 1 << 5,   ///< SSE streaming support
    Reasoning           = 1 << 6,   ///< Extended thinking/reasoning (o1, Claude extended)
    SystemPrompts       = 1 << 7,   ///< Supports system/developer prompts
    ParallelToolCalls   = 1 << 8,   ///< Can call multiple tools at once
    TokenCounting       = 1 << 9,   ///< Native token counting endpoint
    PromptCaching       = 1 << 10,  ///< Supports prompt caching for cost reduction
    Logprobs            = 1 << 11,  ///< Can return log probabilities
    Embeddings          = 1 << 12,  ///< Has embedding variant
};

using ModelCapabilities = uint32_t;

/**
 * @brief Check if a capability set includes a specific capability.
 */
[[nodiscard]] constexpr bool has_capability(ModelCapabilities caps, ModelCapability cap) noexcept {
    return (caps & static_cast<uint32_t>(cap)) != 0;
}

[[nodiscard]] constexpr ModelCapabilities operator|(ModelCapability a, ModelCapability b) noexcept {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}

[[nodiscard]] constexpr ModelCapabilities operator|(ModelCapabilities a, ModelCapability b) noexcept {
    return a | static_cast<uint32_t>(b);
}

// ============================================================================
// Model Tier (for routing decisions)
// ============================================================================

/**
 * @brief Model quality/performance tier for intelligent routing.
 * 
 * The router uses this to match task complexity to model capability.
 * Replaces fragile string heuristics with explicit classification.
 */
enum class ModelTier {
    Fast,       ///< Low latency, cost-effective (Haiku, GPT-3.5, Flash, Mini, Nano)
    Balanced,   ///< Good balance of quality and speed (Sonnet, GPT-4o)
    Powerful,   ///< Maximum quality (Opus, GPT-4o-latest, o1)
    Reasoning,  ///< Extended thinking mode required (o1, o3, deep research)
};

[[nodiscard]] constexpr std::string_view to_string(ModelTier tier) noexcept {
    switch (tier) {
        case ModelTier::Fast: return "fast";
        case ModelTier::Balanced: return "balanced";
        case ModelTier::Powerful: return "powerful";
        case ModelTier::Reasoning: return "reasoning";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::optional<ModelTier> tier_from_string(std::string_view s) noexcept {
    if (s == "fast") return ModelTier::Fast;
    if (s == "balanced") return ModelTier::Balanced;
    if (s == "powerful") return ModelTier::Powerful;
    if (s == "reasoning") return ModelTier::Reasoning;
    return std::nullopt;
}

// ============================================================================
// Pricing Information
// ============================================================================

/**
 * @brief Pricing data for cost-aware routing and estimation.
 * 
 * All prices are in USD per 1 million tokens.
 */
struct ModelPricing {
    double input_per_mtok = 0.0;              ///< Standard input tokens
    double output_per_mtok = 0.0;             ///< Standard output tokens
    double cached_input_per_mtok = -1.0;      ///< Cached input (-1 = not supported)
    double prompt_caching_write_per_mtok = -1.0; ///< Write to cache cost
    
    [[nodiscard]] constexpr bool has_cached_pricing() const noexcept {
        return cached_input_per_mtok >= 0;
    }
    
    [[nodiscard]] constexpr bool has_prompt_caching() const noexcept {
        return prompt_caching_write_per_mtok >= 0;
    }
};

// ============================================================================
// Parameter Constraints
// ============================================================================

/**
 * @brief Valid ranges for model parameters.
 */
struct ParameterConstraints {
    struct Range {
        double min = 0.0;
        double max = 0.0;
        
        [[nodiscard]] constexpr bool contains(double value) const noexcept {
            return value >= min && value <= max;
        }
    };
    
    std::optional<Range> temperature;
    std::optional<Range> top_p;
    std::optional<Range> frequency_penalty;
    std::optional<Range> presence_penalty;
    
    // Token limits
    int32_t max_tokens_min = 1;
    int32_t max_tokens_max = 0;  ///< 0 = use max_output_tokens from model
};

// ============================================================================
// Model Information Record (the "Model Card")
// ============================================================================

/**
 * @brief Comprehensive metadata for an LLM model.
 * 
 * This is the core data structure representing a model's capabilities,
 * limits, and characteristics. Think of it as a programmatic model card.
 */
struct ModelInfo {
    // ------------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------------
    std::string canonical_id;                    ///< Full model ID, e.g., "gpt-4o-2024-08-06"
    std::vector<std::string> aliases;            ///< Alternative names, e.g., "gpt-4o"
    std::string display_name;                    ///< Human-readable name
    std::string provider;                        ///< Provider key: "openai", "anthropic", etc.
    
    // ------------------------------------------------------------------------
    // Token Limits
    // ------------------------------------------------------------------------
    int32_t context_window = 0;                  ///< Total context window (input + output)
    int32_t max_output_tokens = 0;               ///< Max tokens to generate
    int32_t max_reasoning_tokens = 0;            ///< For reasoning models (0 = not reasoning)
    
    // ------------------------------------------------------------------------
    // Capabilities & Classification
    // ------------------------------------------------------------------------
    ModelCapabilities capabilities = 0;          ///< Bitmap of ModelCapability flags
    ModelTier tier = ModelTier::Balanced;        ///< Quality tier for routing
    
    // ------------------------------------------------------------------------
    // Metadata
    // ------------------------------------------------------------------------
    ModelPricing pricing;                        ///< Cost information
    std::string knowledge_cutoff;                ///< Training data cutoff, e.g., "2024-06"
    std::string deprecation_date = "";           ///< Deprecation date (empty if active)
    std::string expected_completion_date = "";   ///< When deprecated model stops working
    
    // ------------------------------------------------------------------------
    // Constraints
    // ------------------------------------------------------------------------
    ParameterConstraints constraints;            ///< Parameter valid ranges
    int32_t max_tool_calls = 32;                 ///< Max parallel tool calls (0 = unlimited)
    
    // ------------------------------------------------------------------------
    // Query Methods
    // ------------------------------------------------------------------------
    
    /**
     * @brief Check if this model supports a specific capability.
     */
    [[nodiscard]] bool supports(ModelCapability cap) const noexcept {
        return has_capability(capabilities, cap);
    }
    
    /**
     * @brief Check if this model is deprecated.
     */
    [[nodiscard]] bool is_deprecated() const noexcept {
        return !deprecation_date.empty();
    }
    
    /**
     * @brief Check if a model ID matches this model (canonical or alias).
     */
    [[nodiscard]] bool matches(std::string_view model_id) const noexcept;
    
    /**
     * @brief Validate a parameter value for this model.
     */
    [[nodiscard]] bool validate_parameter(std::string_view param_name, double value) const noexcept;
    
    /**
     * @brief Estimate cost for a request.
     */
    [[nodiscard]] double estimate_cost(int input_tokens, int output_tokens, 
                                       bool use_cached_input = false) const noexcept;
    
    /**
     * @brief Get effective max tokens (min of max_output_tokens and context_window).
     */
    [[nodiscard]] int32_t effective_max_tokens() const noexcept {
        return max_output_tokens > 0 ? max_output_tokens : context_window;
    }
};

// ============================================================================
// Enhanced Model Registry
// ============================================================================

/**
 * @brief Central registry for LLM model metadata and capabilities.
 * 
 * This is a singleton registry that provides:
 * - Lookup of model information by ID or alias
 * - Capability queries for feature detection
 * - Cost estimation for routing decisions
 * - Parameter validation before API calls
 * 
 * Thread-safe: all public methods can be called concurrently.
 * 
 * @note This replaces the static-only API with an instance-based design
 * while maintaining backward compatibility through the static get_max_context_size().
 */
class ModelRegistry {
public:
    // ------------------------------------------------------------------------
    // Singleton Access
    // ------------------------------------------------------------------------
    
    /**
     * @brief Get the global registry instance.
     */
    [[nodiscard]] static ModelRegistry& instance();
    
    // ------------------------------------------------------------------------
    // Core Queries
    // ------------------------------------------------------------------------
    
    /**
     * @brief Look up a model by ID or alias.
     * @return Pointer to ModelInfo, or nullptr if not found.
     */
    [[nodiscard]] const ModelInfo* lookup(std::string_view model_id) const;
    
    /**
     * @brief Check if a model is registered.
     */
    [[nodiscard]] bool has_model(std::string_view model_id) const;
    
    /**
     * @brief Get the maximum context window size for a model.
     * @return Context window in tokens, or 0 if unknown.
     */
    [[nodiscard]] int get_max_context_size(std::string_view model_id) const;
    
    /**
     * @brief Static version for backward compatibility.
     * @deprecated Use instance().get_max_context_size() for new code.
     */
    [[nodiscard]] static int get_max_context_size_legacy(std::string_view model_name) noexcept;
    
    /**
     * @brief Get capabilities for a model.
     * @return Capability bitmap, or 0 if model not found.
     */
    [[nodiscard]] ModelCapabilities get_capabilities(std::string_view model_id) const;
    
    /**
     * @brief Check if a model supports a specific capability.
     */
    [[nodiscard]] bool supports(std::string_view model_id, ModelCapability cap) const;
    
    /**
     * @brief Get the tier classification for a model.
     */
    [[nodiscard]] std::optional<ModelTier> get_tier(std::string_view model_id) const;
    
    /**
     * @brief Get full model info.
     */
    [[nodiscard]] std::optional<ModelInfo> get_info(std::string_view model_id) const;
    
    // ------------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------------
    
    /**
     * @brief Validate a parameter value for a specific model.
     * @return True if valid, false if invalid or model unknown.
     */
    [[nodiscard]] bool validate_parameter(std::string_view model_id,
                                          std::string_view param_name,
                                          double value) const;
    
    /**
     * @brief Check if max_tokens value is valid for a model.
     */
    [[nodiscard]] bool validate_max_tokens(std::string_view model_id, int max_tokens) const;
    
    // ------------------------------------------------------------------------
    // Batch Queries (for routing)
    // ------------------------------------------------------------------------
    
    /**
     * @brief Get all models in a specific tier.
     */
    [[nodiscard]] std::vector<const ModelInfo*> filter_by_tier(ModelTier tier) const;
    
    /**
     * @brief Get all models with a specific capability.
     */
    [[nodiscard]] std::vector<const ModelInfo*> filter_by_capability(ModelCapability cap) const;
    
    /**
     * @brief Get all registered model IDs (canonical).
     */
    [[nodiscard]] std::vector<std::string> list_models() const;
    
    /**
     * @brief Get all models for a provider.
     */
    [[nodiscard]] std::vector<const ModelInfo*> get_by_provider(std::string_view provider) const;
    
    // ------------------------------------------------------------------------
    // Cost Estimation
    // ------------------------------------------------------------------------
    
    /**
     * @brief Estimate cost for a request.
     * @return Cost in USD, or -1.0 if model not found.
     */
    [[nodiscard]] double estimate_cost(std::string_view model_id,
                                       int input_tokens,
                                       int output_tokens,
                                       bool use_cached_input = false) const;
    
    // ------------------------------------------------------------------------
    // Registration (for dynamic updates)
    // ------------------------------------------------------------------------
    
    /**
     * @brief Register a new model or update existing.
     */
    void register_model(ModelInfo info);
    
    /**
     * @brief Register multiple models.
     */
    void register_models(std::span<const ModelInfo> models);
    
    /**
     * @brief Load built-in default models.
     * Called automatically on first instance() call.
     */
    void load_defaults();
    
    /**
     * @brief Load models from JSON configuration.
     * 
     * JSON format:
     * {
     *   "models": [
     *     {
     *       "canonical_id": "custom-model-v1",
     *       "aliases": ["custom-model"],
     *       "display_name": "Custom Model v1",
     *       "provider": "custom",
     *       "context_window": 32000,
     *       "max_output_tokens": 4096,
     *       "capabilities": ["text", "vision", "tools"],
     *       "tier": "balanced",
     *       "pricing": {"input": 1.0, "output": 3.0},
     *       "knowledge_cutoff": "2024-01"
     *     }
     *   ]
     * }
     * 
     * @param json_data JSON string containing model definitions
     * @return Number of models loaded, or -1 on parse error
     */
    int load_from_json(std::string_view json_data);
    
    /**
     * @brief Load models from JSON file.
     * @param file_path Path to JSON file
     * @return Number of models loaded, or -1 on error
     */
    int load_from_json_file(const std::string& file_path);
    
    /**
     * @brief Export current registry to JSON.
     * @return JSON string representation of all registered models
     */
    [[nodiscard]] std::string export_to_json() const;
    
    /**
     * @brief Clear all registered models.
     */
    void clear();
    
    /**
     * @brief Get count of registered models.
     */
    [[nodiscard]] size_t size() const;

private:
    ModelRegistry();
    ~ModelRegistry() = default;
    
    // Non-copyable, non-movable
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;
    
    void register_aliases(const ModelInfo& info);
    
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, ModelInfo> models_;      ///< canonical_id -> info
    std::unordered_map<std::string, std::string> alias_map_; ///< alias -> canonical_id
    bool defaults_loaded_ = false;
};

// ============================================================================
// Convenience Free Functions
// ============================================================================

/**
 * @brief Get maximum context size (backward compatible API).
 * 
 * This is the legacy static API. It delegates to the singleton registry
 * but falls back to the original heuristic matching for unknown models.
 */
[[nodiscard]] int get_max_context_size(std::string_view model_id);

/**
 * @brief Check if a model supports a capability.
 */
[[nodiscard]] bool model_supports(std::string_view model_id, ModelCapability cap);

/**
 * @brief Get model tier for routing.
 */
[[nodiscard]] std::optional<ModelTier> get_model_tier(std::string_view model_id);

} // namespace core::llm
