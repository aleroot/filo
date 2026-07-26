#pragma once

#include "core/config/ConfigManager.hpp"

#include <algorithm>
#include <array>
#include <string_view>

namespace core::llm {

enum class ProviderAuthStyle {
    Bearer,
    QueryParam,
    XApiKey,
    None,
};

/**
 * Declarative identity and transport defaults for a built-in provider family.
 *
 * More-specific prefixes must precede their parent family. Matching observes a
 * provider-name boundary, so "grok-fast" belongs to grok while "grokker" is a
 * custom provider.
 */
struct BuiltinProviderDefinition {
    std::string_view prefix;
    std::string_view registry_provider;
    std::string_view catalog_group;
    config::ApiType api_type;
    std::string_view base_url;
    std::string_view env_var;
    ProviderAuthStyle auth_style;
    std::string_view default_wire_api;

    [[nodiscard]] constexpr bool matches(
        std::string_view provider_name) const noexcept {
        return provider_name == prefix
            || (provider_name.starts_with(prefix)
                && provider_name.size() > prefix.size()
                && provider_name[prefix.size()] == '-');
    }
};

inline constexpr std::array kBuiltinProviderDefinitions{
    BuiltinProviderDefinition{
        "zai-coding", "zai", "zai", config::ApiType::OpenAI,
        "https://api.z.ai/api/coding/paas/v4", "ZAI_API_KEY",
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "qwen-token-plan", "qwen", "qwen", config::ApiType::DashScope,
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "QWEN_TOKEN_PLAN_API_KEY", ProviderAuthStyle::Bearer, "responses",
    },
    BuiltinProviderDefinition{
        "grok", "grok", "grok", config::ApiType::OpenAI,
        "https://api.x.ai/v1", "XAI_API_KEY",
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "openai", "openai", {}, config::ApiType::OpenAI,
        "https://api.openai.com/v1", "OPENAI_API_KEY",
        ProviderAuthStyle::Bearer, "responses",
    },
    BuiltinProviderDefinition{
        "claude", "anthropic", {}, config::ApiType::Anthropic,
        "https://api.anthropic.com", "ANTHROPIC_API_KEY",
        ProviderAuthStyle::XApiKey, {},
    },
    BuiltinProviderDefinition{
        "gemini", "gemini", {}, config::ApiType::Gemini,
        "https://generativelanguage.googleapis.com", "GEMINI_API_KEY",
        ProviderAuthStyle::QueryParam, {},
    },
    BuiltinProviderDefinition{
        "mistral", "mistral", {}, config::ApiType::OpenAI,
        "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "kimi", "kimi", "kimi", config::ApiType::Kimi,
        "https://api.moonshot.ai/v1", "KIMI_API_KEY",
        ProviderAuthStyle::Bearer, {},
    },
    BuiltinProviderDefinition{
        "ollama", "local", {}, config::ApiType::Ollama,
        "http://localhost:11434", {}, ProviderAuthStyle::None, {},
    },
    BuiltinProviderDefinition{
        "zai", "zai", "zai", config::ApiType::OpenAI,
        "https://api.z.ai/api/paas/v4", "ZAI_API_KEY",
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "qwen", "qwen", "qwen", config::ApiType::DashScope,
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY", ProviderAuthStyle::Bearer, "chat_completions",
    },
};

[[nodiscard]] constexpr const BuiltinProviderDefinition*
find_builtin_provider_definition(std::string_view provider_name) noexcept {
    const auto it = std::ranges::find_if(
        kBuiltinProviderDefinitions,
        [provider_name](const BuiltinProviderDefinition& definition) {
            return definition.matches(provider_name);
        });
    return it == kBuiltinProviderDefinitions.end()
        ? nullptr
        : &*it;
}

} // namespace core::llm
