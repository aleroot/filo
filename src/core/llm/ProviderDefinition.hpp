#pragma once

#include "../auth/ICredentialSource.hpp"
#include "core/config/ConfigManager.hpp"

#include <algorithm>
#include <array>
#include <string_view>

namespace core::llm {

enum class ProviderAuthStyle {
    Bearer,
    QueryParam,
    XApiKey,
    ApiKey,
    None,
};

/**
 * Declarative identity and transport defaults for a built-in provider family.
 *
 * More-specific prefixes must precede their parent family. Matching observes a
 * provider-name boundary, so "grok-fast" belongs to grok while "grokker" is a
 * custom provider.
 *
 * `env_vars` is the ordered credential lookup for this provider: the first
 * non-empty name is canonical, and later names are accepted aliases (the same
 * shape as Kimi's KIMI_API_KEY / MOONSHOT_API_KEY pair).
 */
struct BuiltinProviderDefinition {
    std::string_view prefix;
    std::string_view registry_provider;
    std::string_view catalog_group;
    config::ApiType api_type;
    std::string_view base_url;
    std::array<std::string_view, 2> env_vars{};
    ProviderAuthStyle auth_style;
    std::string_view default_wire_api;
    /// Billing relationship of this provider's credentials. Subscription
    /// plans settle usage against plan quota, so per-token cost estimation
    /// is disabled for them. Providers not listed here are Metered.
    core::auth::BillingKind billing_kind = core::auth::BillingKind::Metered;
    /// Optional provider-owned page that shows account usage when the API does
    /// not return an authoritative quota snapshot. This is a navigation target,
    /// not an endpoint Filo should scrape or call with the user's API key.
    std::string_view usage_dashboard_url{};
    /// Optional user-facing name. The configured provider key remains the
    /// routing identity, while the UI can hide implementation details such as
    /// a regional gateway suffix.
    std::string_view display_name{};

    [[nodiscard]] constexpr bool matches(
        std::string_view provider_name) const noexcept {
        return provider_name == prefix
            || (provider_name.starts_with(prefix)
                && provider_name.size() > prefix.size()
                && provider_name[prefix.size()] == '-');
    }

    [[nodiscard]] constexpr std::string_view env_var() const noexcept {
        return env_vars.front();
    }

    [[nodiscard]] constexpr bool has_usage_dashboard() const noexcept {
        return !usage_dashboard_url.empty();
    }

    [[nodiscard]] constexpr std::string_view display_name_or_prefix() const noexcept {
        return display_name.empty() ? prefix : display_name;
    }
};

inline constexpr std::array kBuiltinProviderDefinitions{
    BuiltinProviderDefinition{
        "zai-coding", "zai", "zai", config::ApiType::Anthropic,
        "https://api.z.ai/api/anthropic",
        { "ZAI_CODING_API_KEY", "ZAI_API_KEY" },
        ProviderAuthStyle::XApiKey, {},
        core::auth::BillingKind::Subscription,
    },
    // MiMo Token Plan is one subscription served by three regional gateways.
    // The bare "mimo-token-plan" name is the Europe (Amsterdam) gateway, so
    // "mimo-token-plan-ams" falls through to it and resolves identically.
    BuiltinProviderDefinition{
        "mimo-token-plan-cn", "mimo", "xiaomi", config::ApiType::OpenAI,
        "https://token-plan-cn.xiaomimimo.com/v1",
        { "MIMO_TOKEN_PLAN_API_KEY", "XIAOMI_API_KEY" },
        ProviderAuthStyle::ApiKey, "chat_completions",
        core::auth::BillingKind::Subscription,
        "https://platform.xiaomimimo.com/console/plan-manage",
        "mimo-token-plan",
    },
    BuiltinProviderDefinition{
        "mimo-token-plan-sgp", "mimo", "xiaomi", config::ApiType::OpenAI,
        "https://token-plan-sgp.xiaomimimo.com/v1",
        { "MIMO_TOKEN_PLAN_API_KEY", "XIAOMI_API_KEY" },
        ProviderAuthStyle::ApiKey, "chat_completions",
        core::auth::BillingKind::Subscription,
        "https://platform.xiaomimimo.com/console/plan-manage",
        "mimo-token-plan",
    },
    BuiltinProviderDefinition{
        "mimo-token-plan", "mimo", "xiaomi", config::ApiType::OpenAI,
        "https://token-plan-ams.xiaomimimo.com/v1",
        { "MIMO_TOKEN_PLAN_API_KEY", "XIAOMI_API_KEY" },
        ProviderAuthStyle::ApiKey, "chat_completions",
        core::auth::BillingKind::Subscription,
        "https://platform.xiaomimimo.com/console/plan-manage",
        "mimo-token-plan",
    },
    BuiltinProviderDefinition{
        "qwen-token-plan", "qwen", "qwen", config::ApiType::DashScope,
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        { "QWEN_TOKEN_PLAN_API_KEY" },
        ProviderAuthStyle::Bearer, "chat_completions",
        core::auth::BillingKind::Subscription,
        "https://home.qwencloud.com/analytics/token-plan/individual",
    },
    BuiltinProviderDefinition{
        "qwen-coding", "qwen", "qwen", config::ApiType::DashScope,
        "https://coding.dashscope.aliyuncs.com/v1",
        { "QWEN_CODING_PLAN_API_KEY", "BAILIAN_CODING_PLAN_API_KEY" },
        ProviderAuthStyle::Bearer, "chat_completions",
        core::auth::BillingKind::Subscription,
    },
    BuiltinProviderDefinition{
        "grok", "grok", "grok", config::ApiType::OpenAI,
        "https://api.x.ai/v1", { "XAI_API_KEY" },
        ProviderAuthStyle::Bearer, "responses",
    },
    BuiltinProviderDefinition{
        "openai", "openai", {}, config::ApiType::OpenAI,
        "https://api.openai.com/v1", { "OPENAI_API_KEY" },
        ProviderAuthStyle::Bearer, "responses",
    },
    BuiltinProviderDefinition{
        "claude", "anthropic", {}, config::ApiType::Anthropic,
        "https://api.anthropic.com", { "ANTHROPIC_API_KEY" },
        ProviderAuthStyle::XApiKey, {},
    },
    BuiltinProviderDefinition{
        "gemini", "gemini", {}, config::ApiType::Gemini,
        "https://generativelanguage.googleapis.com", { "GEMINI_API_KEY" },
        ProviderAuthStyle::QueryParam, {},
    },
    BuiltinProviderDefinition{
        "mistral", "mistral", {}, config::ApiType::OpenAI,
        "https://api.mistral.ai/v1", { "MISTRAL_API_KEY" },
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "kimi", "kimi", "kimi", config::ApiType::Kimi,
        "https://api.moonshot.ai/v1", { "KIMI_API_KEY", "MOONSHOT_API_KEY" },
        ProviderAuthStyle::Bearer, {},
    },
    BuiltinProviderDefinition{
        "ollama", "local", {}, config::ApiType::Ollama,
        "http://localhost:11434", {}, ProviderAuthStyle::None, {},
    },
    BuiltinProviderDefinition{
        "zai", "zai", "zai", config::ApiType::OpenAI,
        "https://api.z.ai/api/paas/v4", { "ZAI_API_KEY" },
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "mimo", "mimo", "xiaomi", config::ApiType::OpenAI,
        "https://api.xiaomimimo.com/v1",
        { "XIAOMI_API_KEY", "MIMO_API_KEY" },
        ProviderAuthStyle::Bearer, "chat_completions",
    },
    BuiltinProviderDefinition{
        "qwen", "qwen", "qwen", config::ApiType::DashScope,
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        { "QWEN_API_KEY", "DASHSCOPE_API_KEY" },
        ProviderAuthStyle::Bearer, "chat_completions",
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

/// Resolves a configured provider key to its compact user-facing name.
/// Unknown and custom providers preserve the configured name unchanged.
[[nodiscard]] constexpr std::string_view provider_display_name(
    std::string_view provider_name) noexcept {
    if (const auto* definition = find_builtin_provider_definition(provider_name)) {
        return definition->display_name.empty()
            ? provider_name
            : definition->display_name;
    }
    return provider_name;
}

} // namespace core::llm
