#include "ProviderFactory.hpp"
#include "HttpLLMProvider.hpp"
#ifdef FILO_ENABLE_LLAMACPP
#include "providers/LlamaCppProvider.hpp"
#endif
#include "protocols/OpenAIProtocol.hpp"
#include "protocols/OpenAIResponsesProtocol.hpp"
#include "protocols/KimiProtocol.hpp"
#include "protocols/GrokProtocol.hpp"
#include "protocols/DashScopeProtocol.hpp"
#include "protocols/AnthropicProtocol.hpp"
#include "protocols/GeminiProtocol.hpp"
#include "protocols/GeminiCodeAssistProtocol.hpp"
#include "protocols/OllamaProtocol.hpp"
#include "OpenAIEndpointUtils.hpp"
#include "../auth/ApiKeyCredentialSource.hpp"
#include "../auth/AuthenticationManager.hpp"
#include "../auth/GoogleCodeAssist.hpp"
#include "../logging/Logger.hpp"
#include "../utils/StringUtils.hpp"
#include <algorithm>
#include <cstdlib>

namespace core::llm {

using core::config::ApiType;
using core::config::ProviderConfig;

namespace {

enum class AuthStyle { Bearer, QueryParam, XApiKey, None };
enum class OpenAIWireApi { ChatCompletions, Responses };

struct BuiltinDef {
    const char* prefix;
    ApiType     api_type;
    const char* base_url;
    const char* env_var;    ///< Environment variable for the API key; "" = none
    AuthStyle   auth_style;
    const char* default_wire_api; ///< OpenAI wire API default for OpenAI-family providers
};

// First matching prefix wins — order matters.
static constexpr BuiltinDef kBuiltins[] = {
    { "grok",    ApiType::OpenAI,     "https://api.x.ai/v1",                                            "XAI_API_KEY",         AuthStyle::Bearer,     "chat_completions" },
    { "openai",  ApiType::OpenAI,     "https://api.openai.com/v1",                                      "OPENAI_API_KEY",      AuthStyle::Bearer,     "responses" },
    { "claude",  ApiType::Anthropic,  "https://api.anthropic.com",                                      "ANTHROPIC_API_KEY",   AuthStyle::XApiKey,    "" },
    { "gemini",  ApiType::Gemini,     "https://generativelanguage.googleapis.com",                       "GEMINI_API_KEY",      AuthStyle::QueryParam, "" },
    { "mistral", ApiType::OpenAI,     "https://api.mistral.ai/v1",                                      "MISTRAL_API_KEY",     AuthStyle::Bearer,     "chat_completions" },
    { "kimi",    ApiType::Kimi,       "https://api.moonshot.cn/v1",                                     "KIMI_API_KEY",        AuthStyle::Bearer,     "" },
    { "ollama",  ApiType::Ollama,     "http://localhost:11434",                                          "",                    AuthStyle::None,       "" },
    { "qwen",    ApiType::DashScope,  "https://dashscope.aliyuncs.com/compatible-mode/v1",               "DASHSCOPE_API_KEY",   AuthStyle::Bearer,     "" },
};

const BuiltinDef* find_builtin(std::string_view name) noexcept {
    for (const auto& def : kBuiltins) {
        if (name.starts_with(def.prefix)) return &def;
    }
    return nullptr;
}

std::string resolve_key(std::string_view config_key, const char* env_var) {
    if (!config_key.empty()) return std::string(config_key);
    if (env_var && *env_var) {
        if (const char* e = std::getenv(env_var)) return e;
    }
    return {};
}

[[nodiscard]] std::string infer_openai_wire_api(std::string configured_wire_api,
                                                std::string_view canonical_type,
                                                std::string_view base_url,
                                                std::string_view builtin_default) {
    if (!configured_wire_api.empty()) return configured_wire_api;

    if (canonical_type == "openai"
        && !openai_endpoint::is_native_openai_responses_base_url(base_url)) {
        return "chat_completions";
    }

    return std::string(builtin_default.empty() ? "chat_completions" : builtin_default);
}

[[nodiscard]] OpenAIWireApi parse_openai_wire_api(std::string_view configured,
                                                  std::string_view fallback) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        configured.empty() ? fallback : configured);

    if (lowered == "responses") return OpenAIWireApi::Responses;
    if (lowered == "chat") return OpenAIWireApi::ChatCompletions;
    if (lowered == "chat_completions") return OpenAIWireApi::ChatCompletions;

    return OpenAIWireApi::ChatCompletions;
}

} // namespace

std::shared_ptr<LLMProvider> ProviderFactory::create_provider(
    std::string_view name, const ProviderConfig& config)
{
    const BuiltinDef* builtin = find_builtin(name);

    // Resolve api_type and base_url: explicit config overrides built-in defaults.
    ApiType     api_type  = config.api_type;
    std::string base_url  = config.base_url;
    AuthStyle   auth_style = AuthStyle::Bearer;
    const char* env_var   = "";
    std::string_view canonical_type = name;   // for OAuth strategy matching
    std::string wire_api  = config.wire_api;

    if (builtin) {
        if (api_type == ApiType::Unknown) api_type = builtin->api_type;
        if (base_url.empty())            base_url  = builtin->base_url;
        auth_style     = builtin->auth_style;
        env_var        = builtin->env_var;
        canonical_type = builtin->prefix;    // e.g. "grok" for "grok-reasoning"
    } else {
        // User-defined provider: api_type and base_url must be explicit.
        if (api_type == ApiType::Unknown) {
            core::logging::warn(
                "Provider '{}': no api_type specified; skipping.", name);
            return nullptr;
        }
        // LlamaCppLocal is not HTTP-based — delegate directly (skip base_url check).
        if (api_type == ApiType::LlamaCppLocal) {
#ifdef FILO_ENABLE_LLAMACPP
            return std::make_shared<providers::LlamaCppProvider>(config);
#else
            core::logging::warn(
                "Provider '{}': requires llama.cpp support "
                "(reconfigure with FILO_ENABLE_LLAMACPP=ON).", name);
            return nullptr;
#endif
        }
        if (base_url.empty()) {
            core::logging::warn(
                "Provider '{}': no base_url configured; skipping.", name);
            return nullptr;
        }
        // Infer auth_style from api_type.
        if      (api_type == ApiType::Anthropic) auth_style = AuthStyle::XApiKey;
        else if (api_type == ApiType::Gemini)    auth_style = AuthStyle::QueryParam;
        else if (api_type == ApiType::Ollama)    auth_style = AuthStyle::None;
    }

    // LlamaCppLocal is not HTTP-based — delegate directly.
    if (api_type == ApiType::LlamaCppLocal) {
#ifdef FILO_ENABLE_LLAMACPP
        return std::make_shared<providers::LlamaCppProvider>(config);
#else
        core::logging::warn(
            "Provider '{}': requires llama.cpp support "
            "(reconfigure with FILO_ENABLE_LLAMACPP=ON).", name);
        return nullptr;
#endif
    }

    // Try OAuth strategies first; fall back to API key credential.
    auto auth_manager = core::auth::AuthenticationManager::create_with_defaults(
        core::config::ConfigManager::get_instance().get_config_dir());
    std::shared_ptr<core::auth::ICredentialSource> cred =
        auth_manager.create_credential_source(canonical_type, config);
    const std::string normalized_auth_type =
        core::utils::str::to_lower_ascii_copy(config.auth_type);

    // For Kimi OAuth, use the correct OAuth endpoint instead of the API key endpoint.
    // The official kimi-cli uses https://api.kimi.com/coding/v1 for OAuth,
    // while https://api.moonshot.cn/v1 is for API key authentication.
    if (cred && canonical_type == "kimi" && base_url == "https://api.moonshot.cn/v1") {
        base_url = "https://api.kimi.com/coding/v1";
        core::logging::debug("Using Kimi OAuth endpoint: {}", base_url);
    }

    // For OpenAI ChatGPT PKCE auth, route to the ChatGPT Codex backend by default.
    if (cred && canonical_type == "openai"
        && normalized_auth_type == "oauth_openai_pkce"
        && base_url == "https://api.openai.com/v1") {
        base_url = "https://chatgpt.com/backend-api/codex";
        core::logging::debug("Using OpenAI PKCE endpoint: {}", base_url);
    }

    if (cred && canonical_type == "gemini"
        && normalized_auth_type == "oauth_google") {
        base_url = core::auth::google_code_assist::code_assist_endpoint();
        core::logging::debug("Using Gemini Code Assist endpoint: {}", base_url);
    }

    if (!cred) {
        const std::string key = resolve_key(config.api_key, env_var);
        switch (auth_style) {
        case AuthStyle::Bearer:
            if (api_type == ApiType::OpenAI
                && openai_endpoint::is_azure_openai_base_url(base_url)) {
                cred = core::auth::ApiKeyCredentialSource::as_custom_header(key, "api-key");
            } else {
                cred = core::auth::ApiKeyCredentialSource::as_bearer(key);
            }
            break;
        case AuthStyle::QueryParam:
            cred = core::auth::ApiKeyCredentialSource::as_query_param(key);
            break;
        case AuthStyle::XApiKey:
            cred = core::auth::ApiKeyCredentialSource::as_custom_header(key, "x-api-key");
            break;
        case AuthStyle::None:
            cred = core::auth::ApiKeyCredentialSource::none();
            break;
        }
    }

    // Build protocol based on api_type (and provider-specific extensions).
    std::unique_ptr<protocols::ApiProtocolBase> protocol;
    switch (api_type) {
    case ApiType::OpenAI: {
        const std::string inferred_wire_api = infer_openai_wire_api(
            wire_api,
            canonical_type,
            base_url,
            builtin ? builtin->default_wire_api : "chat_completions");

        const OpenAIWireApi wire = parse_openai_wire_api(
            inferred_wire_api,
            "chat_completions");

        if (wire == OpenAIWireApi::Responses) {
            if (canonical_type.starts_with("grok") && !config.reasoning_effort.empty()) {
                core::logging::debug(
                    "Provider '{}': reasoning_effort is ignored for wire_api='responses'.",
                    name);
            }
            protocol = std::make_unique<protocols::OpenAIResponsesProtocol>(
                /*include_reasoning_encrypted=*/false,
                config.service_tier);
        } else {
            // Use GrokProtocol for grok-prefixed providers with reasoning_effort.
            if (canonical_type.starts_with("grok") && !config.reasoning_effort.empty()) {
                protocols::GrokReasoningEffort effort = protocols::GrokReasoningEffort::None;
                if (config.reasoning_effort == "low")    effort = protocols::GrokReasoningEffort::Low;
                if (config.reasoning_effort == "medium") effort = protocols::GrokReasoningEffort::Medium;
                if (config.reasoning_effort == "high")   effort = protocols::GrokReasoningEffort::High;
                protocol = std::make_unique<protocols::GrokProtocol>(effort);
            } else {
                protocol = std::make_unique<protocols::OpenAIProtocol>(config.stream_usage);
            }
        }
        break;
    }
    case ApiType::Kimi:
        // Kimi uses OpenAI wire format but requires special X-Msh-* headers
        // for OAuth authentication to work properly
        protocol = std::make_unique<protocols::KimiProtocol>();
        break;
    case ApiType::DashScope:
        // DashScope (Qwen) uses OpenAI wire format with X-DashScope-* headers,
        // prompt caching, and optional Qwen3 thinking mode via thinking_budget.
        protocol = std::make_unique<protocols::DashScopeProtocol>(config.thinking_budget);
        break;
    case ApiType::Anthropic: {
        protocols::AnthropicThinkingConfig thinking;
        if (config.thinking_budget > 0) {
            thinking.enabled       = true;
            thinking.budget_tokens = config.thinking_budget;
        }
        protocol = std::make_unique<protocols::AnthropicProtocol>(thinking);
        break;
    }
    case ApiType::Gemini:
        if (normalized_auth_type == "oauth_google") {
            protocol = std::make_unique<protocols::GeminiCodeAssistProtocol>();
        } else {
            protocol = std::make_unique<protocols::GeminiProtocol>();
        }
        break;
    case ApiType::Ollama:
        protocol = std::make_unique<protocols::OllamaProtocol>();
        break;
    default:
        core::logging::warn(
            "Provider '{}': unhandled api_type '{}'.", name,
            core::config::to_string(api_type));
        return nullptr;
    }

    return std::make_shared<HttpLLMProvider>(
        base_url, std::move(cred), config.model, std::move(protocol));
}

} // namespace core::llm
