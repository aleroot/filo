#include "ProviderFactory.hpp"
#include "HttpLLMProvider.hpp"
#include "ProviderDefinition.hpp"
#include "ProviderClientIdentity.hpp"
#include "providers/QwenModelCatalogSelector.hpp"
#ifdef FILO_ENABLE_LLAMACPP
#include "providers/LlamaCppProvider.hpp"
#endif
#include "protocols/OpenAIProtocol.hpp"
#include "protocols/OpenAIResponsesProtocol.hpp"
#include "protocols/MistralProtocol.hpp"
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
#include "../utils/UriUtils.hpp"
#include <algorithm>
#include <cstdlib>

namespace core::llm {

using core::config::ApiType;
using core::config::ProviderConfig;

namespace {

enum class OpenAIWireApi { ChatCompletions, Responses };

std::string resolve_key(
    std::string_view config_key,
    std::string_view env_var) {
    if (!config_key.empty()) return std::string(config_key);
    if (!env_var.empty()) {
        if (const char* e = std::getenv(std::string(env_var).c_str());
            e && *e) {
            return e;
        }
        if (env_var == "KIMI_API_KEY") {
            if (const char* e = std::getenv("MOONSHOT_API_KEY");
                e && *e) {
                return e;
            }
        }
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

[[nodiscard]] bool model_prefers_kimi_code_endpoint(std::string_view model) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(model);
    return lowered == "k3"
        || lowered == "kimi-for-coding"
        || lowered == "kimi-for-coding-highspeed";
}

[[nodiscard]] bool is_public_kimi_api_endpoint(
    std::string_view base_url) noexcept {
    return base_url == "https://api.moonshot.ai/v1"
        || base_url == "https://api.moonshot.cn/v1";
}

[[nodiscard]] bool is_qwen_token_plan_endpoint(std::string_view base_url) noexcept {
    const auto host = core::utils::uri::extract_http_host(base_url);
    return host.has_value()
        && core::utils::ascii::iequals(
            *host, "token-plan.ap-southeast-1.maas.aliyuncs.com");
}

} // namespace

std::shared_ptr<LLMProvider> ProviderFactory::create_provider(
    std::string_view name, const ProviderConfig& config)
{
    const BuiltinProviderDefinition* builtin =
        find_builtin_provider_definition(name);

    // Resolve api_type and base_url: explicit config overrides built-in defaults.
    ApiType     api_type  = config.api_type;
    std::string base_url  = config.base_url;
    ProviderAuthStyle auth_style = ProviderAuthStyle::Bearer;
    std::string_view env_var;
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
        if (api_type == ApiType::Anthropic) {
            auth_style = ProviderAuthStyle::XApiKey;
        } else if (api_type == ApiType::Gemini) {
            auth_style = ProviderAuthStyle::QueryParam;
        } else if (api_type == ApiType::Ollama) {
            auth_style = ProviderAuthStyle::None;
        }
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
    const std::string config_dir =
        core::config::ConfigManager::get_instance().get_config_dir();
    auto auth_manager = core::auth::AuthenticationManager::create_with_defaults(config_dir);
    std::shared_ptr<core::auth::ICredentialSource> cred =
        auth_manager.create_credential_source(canonical_type, config);
    std::shared_ptr<IProviderClientIdentitySource> client_identity_source;
    std::shared_ptr<const IModelCatalogSelector> model_catalog_selector;
    const std::string normalized_auth_type =
        core::utils::str::to_lower_ascii_copy(config.auth_type);

    // Kimi OAuth uses the managed coding endpoint. Both the current global
    // Moonshot host and the legacy China host remain valid API-key endpoints.
    if (cred
        && canonical_type == "kimi"
        && is_public_kimi_api_endpoint(base_url)) {
        base_url = "https://api.kimi.com/coding/v1";
        core::logging::debug("Using Kimi OAuth endpoint: {}", base_url);
    }

    // The official Kimi Code model is served by the Kimi Code endpoint.
    if (canonical_type == "kimi"
        && is_public_kimi_api_endpoint(base_url)
        && model_prefers_kimi_code_endpoint(config.model)) {
        base_url = "https://api.kimi.com/coding/v1";
        core::logging::debug("Using Kimi Code endpoint for model '{}': {}", config.model, base_url);
    }

    // For OpenAI ChatGPT PKCE auth, route to the ChatGPT Codex backend by default.
    if (cred && canonical_type == "openai"
        && normalized_auth_type == "oauth_openai_pkce"
        && base_url == "https://api.openai.com/v1") {
        base_url = "https://chatgpt.com/backend-api/codex";
        core::logging::debug("Using OpenAI PKCE endpoint: {}", base_url);
    }

    // xAI account sessions use the Grok Build chat proxy. API keys continue
    // to use the public xAI API endpoint and its independent billing model.
    if (cred && canonical_type == "grok"
        && (normalized_auth_type == "oauth_xai"
            || normalized_auth_type == "oauth_grok")
        && base_url == "https://api.x.ai/v1") {
        base_url = "https://cli-chat-proxy.grok.com/v1";
        if (wire_api.empty()) wire_api = "responses";
        core::logging::debug("Using Grok OAuth session endpoint: {}", base_url);
    }

    if (cred && canonical_type == "gemini"
        && normalized_auth_type == "oauth_google") {
        base_url = core::auth::google_code_assist::code_assist_endpoint();
        core::logging::debug("Using Gemini Code Assist endpoint: {}", base_url);
    }

    const bool qwen_token_plan = canonical_type == "qwen-token-plan"
        || is_qwen_token_plan_endpoint(base_url);
    if (qwen_token_plan && env_var.empty()) {
        env_var = "QWEN_TOKEN_PLAN_API_KEY";
    }
    if (qwen_token_plan && config.model.empty()) {
        model_catalog_selector = providers::make_qwen_model_catalog_selector();
    }

    if (!cred) {
        const std::string key = resolve_key(config.api_key, env_var);
        switch (auth_style) {
        case ProviderAuthStyle::Bearer:
            if (api_type == ApiType::OpenAI
                && openai_endpoint::is_azure_openai_base_url(base_url)) {
                cred = core::auth::ApiKeyCredentialSource::as_custom_header(key, "api-key");
            } else {
                cred = core::auth::ApiKeyCredentialSource::as_bearer(
                    key,
                    canonical_type == "zai-coding" || qwen_token_plan);
            }
            break;
        case ProviderAuthStyle::QueryParam:
            cred = core::auth::ApiKeyCredentialSource::as_query_param(key);
            break;
        case ProviderAuthStyle::XApiKey:
            cred = core::auth::ApiKeyCredentialSource::as_custom_header(key, "x-api-key");
            break;
        case ProviderAuthStyle::None:
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
            if (canonical_type.starts_with("grok")) {
                protocol = std::make_unique<protocols::GrokResponsesProtocol>(
                    config.service_tier);
            } else if (base_url == "https://chatgpt.com/backend-api/codex") {
                client_identity_source = make_codex_client_identity_source(config_dir);
                protocol = std::make_unique<protocols::CodexResponsesProtocol>(
                    /*include_reasoning_encrypted=*/false,
                    config.service_tier,
                    client_identity_source);
            } else {
                protocol = std::make_unique<protocols::OpenAIResponsesProtocol>(
                    /*include_reasoning_encrypted=*/
                        canonical_type == "openai"
                        && openai_endpoint::is_native_openai_responses_base_url(base_url),
                    config.service_tier);
            }
        } else {
            // Grok uses a thin OpenAI-compatible extension for xAI headers,
            // errors, rate limits, and optional reasoning effort.
            if (canonical_type == "zai-coding") {
                protocol = std::make_unique<protocols::ZaiCodingProtocol>(
                    config.stream_usage);
            } else if (canonical_type == "zai") {
                protocol = std::make_unique<protocols::ZaiProtocol>(
                    config.stream_usage);
            } else if (canonical_type.starts_with("grok")) {
                protocols::GrokReasoningEffort effort = protocols::GrokReasoningEffort::None;
                if (config.reasoning_effort == "low")    effort = protocols::GrokReasoningEffort::Low;
                if (config.reasoning_effort == "medium") effort = protocols::GrokReasoningEffort::Medium;
                if (config.reasoning_effort == "high")   effort = protocols::GrokReasoningEffort::High;
                protocol = std::make_unique<protocols::GrokProtocol>(
                    effort, config.stream_usage);
            } else if (canonical_type == "mistral") {
                protocol = std::make_unique<protocols::MistralProtocol>(
                    config.stream_usage);
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
        // Token Plan defaults to the Responses API for native effort levels,
        // hosted tools, and server-side session caching. Ordinary DashScope
        // remains on Chat Completions unless explicitly configured otherwise.
        if (parse_openai_wire_api(
                wire_api,
                builtin ? builtin->default_wire_api : "chat_completions")
            == OpenAIWireApi::Responses) {
            protocol = std::make_unique<protocols::DashScopeResponsesProtocol>(
                protocols::DashScopeResponsesProtocol::Options{
                    .default_effort = config.reasoning_effort.empty()
                        ? "high"
                        : config.reasoning_effort,
                    .enable_hosted_tools = qwen_token_plan,
                    .deployment = qwen_token_plan
                        ? protocols::DashScopeDeployment::TokenPlan
                        : protocols::DashScopeDeployment::Standard,
                });
        } else {
            protocol = std::make_unique<protocols::DashScopeProtocol>(
                config.thinking_budget,
                config.reasoning_effort,
                qwen_token_plan
                    ? protocols::DashScopeDeployment::TokenPlan
                    : protocols::DashScopeDeployment::Standard);
        }
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
        base_url,
        std::move(cred),
        config.model,
        std::move(protocol),
        api_type,
        std::string(name),
        std::move(client_identity_source),
        std::move(model_catalog_selector));
}

} // namespace core::llm
