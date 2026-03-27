#pragma once

#include "core/llm/LLMProvider.hpp"
#include "core/config/ConfigManager.hpp"
#include <memory>
#include <string_view>

namespace core::llm {

/**
 * @brief Factory for creating LLMProvider instances from configuration.
 *
 * Built-in providers (openai, grok, claude, gemini, mistral, kimi, ollama)
 * are recognised by name prefix and have their api_type and base_url inferred
 * from an internal registry.  User-defined providers must supply an explicit
 * `api_type` and `base_url` in `ProviderConfig`.
 */
class ProviderFactory {
public:
    /**
     * @brief Create a provider for the given named entry.
     *
     * @param name    The provider map key from `settings.json` (e.g. "grok",
     *                "my-groq").  Used to look up built-in defaults and to
     *                resolve OAuth strategy matching.
     * @param config  The merged `ProviderConfig` for this entry.
     * @return        A ready-to-use `LLMProvider`, or `nullptr` on error.
     */
    static std::shared_ptr<LLMProvider> create_provider(
        std::string_view name,
        const core::config::ProviderConfig& config);
};

} // namespace core::llm
