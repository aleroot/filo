#include "AuthenticationManager.hpp"
#include "ApiKeyPromptStrategy.hpp"
#include "ClaudeOAuthFlow.hpp"
#include "FileTokenStore.hpp"
#include "GoogleAntigravityOAuthFlow.hpp"
#include "GoogleOAuthCredentialSource.hpp"
#include "KimiOAuthFlow.hpp"
#include "MimoAuthenticationStrategy.hpp"
#include "QwenOAuthFlow.hpp"
#include "XaiOAuthFlow.hpp"
#include "XaiOAuthCredentialSource.hpp"
#include "OpenAIOAuthFlow.hpp"
#include "OAuthCredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include "ui/ConsoleAuthUI.hpp"
#include "core/logging/Logger.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace core::auth {

namespace {

// Command identifiers are matched case-insensitively against each
// strategy's login_provider() and login_aliases().
std::string normalize(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

bool matches_login_provider(const IAuthStrategy& strategy,
                            std::string_view requested) {
    if (normalize(strategy.login_provider()) == requested) {
        return true;
    }
    for (const std::string_view alias : strategy.login_aliases()) {
        if (normalize(alias) == requested) {
            return true;
        }
    }
    return false;
}

std::string join(const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ", ";
        out += values[i];
    }
    return out;
}

/**
 * @brief Google (Gemini) OAuth strategy — subscription (Google AI Pro/Ultra)
 *        login for the `gemini` provider.
 *
 * Google deprecated the gemini-cli OAuth flow for individual / Google AI Pro
 * and Ultra accounts (June 2026) and replaced it with a closed-source
 * "Antigravity CLI" that does not publish an integrable client_id. This
 * strategy signs in using OAuth client credentials supplied via environment
 * variables (see GoogleAntigravityOAuthFlow.hpp) — it is the only way left
 * to authenticate
 * a personal Google AI Pro/Ultra subscription without an API key.
 *
 * This is NOT an officially supported integration: it violates Google's
 * Antigravity Terms of Service, and Google has suspended/banned real
 * accounts for this exact pattern. `GoogleAntigravityOAuthFlow::login()`
 * always shows a risk disclosure before proceeding — using it is entirely
 * at the user's own risk and responsibility.
 */
class GoogleOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "google"; }
    std::string_view display_name() const noexcept override {
        return "Google / Gemini (unofficial — ToS risk)";
    }
    std::vector<std::string_view> login_aliases() const override {
        return {"gemini"};
    }
    std::string_view token_store_key() const noexcept override { return "google"; }

    std::shared_ptr<IOAuthTokenRevoker> logout_revocation_flow() const override {
        return std::make_shared<GoogleAntigravityOAuthFlow>();
    }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "gemini" && auth_type == "oauth_google";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow = std::make_shared<GoogleAntigravityOAuthFlow>();
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "google", std::move(flow), std::move(store),
            /*allow_interactive_login=*/false);
        return std::make_shared<GoogleOAuthCredentialSource>(
            std::move(manager), /*ide_type=*/"ANTIGRAVITY");
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        auto flow = std::make_shared<GoogleAntigravityOAuthFlow>(
            std::make_shared<ui::ConsoleAuthUI>());
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "google", std::move(flow), std::move(store));
        manager->login();
        return {
            "\xE2\x9A\xA0  Unofficial: this uses Google's Antigravity IDE OAuth client, "
            "not a supported integration. Google's Antigravity ToS prohibit this and "
            "accounts have been banned for it — use entirely at your own risk.",
            "Set \"auth_type\": \"oauth_google\" on a Gemini provider in "
            "~/.config/filo/config.json to use it.",
        };
    }
};

class ClaudeOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "claude"; }
    std::string_view display_name() const noexcept override { return "Claude"; }
    std::string_view token_store_key() const noexcept override { return "claude"; }
    // Anthropic exposes no public OAuth revocation endpoint; logout clears
    // local credentials only (same behaviour as the official Claude Code CLI).

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "claude" && auth_type == "oauth_claude";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        // For automated runs (non-interactive), we don't provide a UI.
        // If the token is missing, the flow will just check env var or fail.
        auto flow = std::make_shared<ClaudeOAuthFlow>(nullptr);
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "claude", std::move(flow), std::move(store),
            /*allow_interactive_login=*/false);
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        // For explicit login command, we provide the Console UI.
        auto flow = std::make_shared<ClaudeOAuthFlow>(std::make_shared<ui::ConsoleAuthUI>());
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "claude", std::move(flow), std::move(store));
        manager->login();
        return {
            "Set \"auth_type\": \"oauth_claude\" on a Claude provider in "
            "~/.config/filo/config.json to use the stored Claude OAuth token.",
            "OAuth mode supports Claude subscription billing (Pro/Max/Team/Enterprise)."
        };
    }
};

class OpenAIPkceStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "openai"; }
    std::string_view display_name() const noexcept override { return "OpenAI (ChatGPT login)"; }
    std::vector<std::string_view> login_aliases() const override {
        return {"openai-pkce", "openai_pkce", "openaipkce"};
    }
    std::string_view token_store_key() const noexcept override { return "openai-pkce"; }

    std::shared_ptr<IOAuthTokenRevoker> logout_revocation_flow() const override {
        return std::make_shared<OpenAIOAuthFlow>();
    }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "openai" && auth_type == "oauth_openai_pkce";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow    = std::make_shared<OpenAIOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "openai-pkce", std::move(flow), std::move(store),
            /*allow_interactive_login=*/false);
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        auto flow    = std::make_shared<OpenAIOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "openai-pkce", std::move(flow), std::move(store));
        manager->login();
        return {
            "Set \"auth_type\": \"oauth_openai_pkce\" on an OpenAI provider in "
            "~/.config/filo/config.json to use your ChatGPT Plus/Pro plan.",
            "PKCE login uses a built-in Codex-compatible client id by default "
            "(you can override it with OPENAI_OAUTH_CLIENT_ID if needed).",
        };
    }
};

class KimiOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "kimi"; }
    std::string_view token_store_key() const noexcept override { return "kimi"; }
    // Moonshot documents no public OAuth revocation endpoint; local-only logout.
    std::string_view display_name() const noexcept override { return "Kimi Code"; }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "kimi" && auth_type == "oauth_kimi";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow    = std::make_shared<KimiOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "kimi", std::move(flow), std::move(store),
            /*allow_interactive_login=*/false);
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        auto flow    = std::make_shared<KimiOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "kimi", std::move(flow), std::move(store));
        manager->login();
        return {
            "Set \"auth_type\": \"oauth_kimi\" on a Kimi provider in "
            "~/.config/filo/config.json to use the stored OAuth token.",
            "Kimi Code defaults to the international host (api.kimi.ai). "
            "Set KIMI_CODE_REGION=mainland-cn before login for the China host.",
            "You can also export KIMI_API_KEY for one-time / CI use without OAuth.",
        };
    }
};

class QwenOAuthStrategy final : public IAuthStrategy {
public:
    // Hidden guard that turns stale oauth_qwen configurations into an
    // actionable error instead of silently sending an unauthenticated request.
    std::string_view login_provider() const noexcept override { return {}; }
    std::string_view display_name() const noexcept override {
        return "Qwen OAuth (unsupported for Token Plan)";
    }
    std::string_view token_store_key() const noexcept override { return "qwen"; }
    // Qwen documents no public OAuth revocation endpoint; local-only logout.

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type.starts_with("qwen") && auth_type == "oauth_qwen";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view /*config_dir*/) const override {
        throw std::runtime_error(
            "Qwen OAuth was discontinued on 2026-04-15 and does not support OAuth "
            "for current Qwen Cloud APIs. Run `filo --auth qwen-coding` for Coding Plan, "
            "`filo --auth dashscope` for a DashScope API key, or `filo --auth qwen` "
            "for Token Plan.");
    }

    std::vector<std::string> login(std::string_view /*config_dir*/) const override {
        throw std::runtime_error(
            "Qwen OAuth was discontinued on 2026-04-15 and does not support "
            "current Qwen Cloud APIs. Use an API key instead.");
    }
};

class XaiOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "grok"; }
    std::string_view display_name() const noexcept override { return "Grok"; }
    std::vector<std::string_view> login_aliases() const override {
        return {"xai", "x.ai", "x-ai"};
    }
    std::string_view token_store_key() const noexcept override { return "grok"; }

    std::shared_ptr<IOAuthTokenRevoker> logout_revocation_flow() const override {
        return std::make_shared<XaiOAuthFlow>();
    }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "grok"
            && (auth_type == "oauth_xai" || auth_type == "oauth_grok");
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow = std::make_shared<XaiOAuthFlow>();
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "grok", std::move(flow), std::move(store),
            /*allow_interactive_login=*/false);
        auto oauth = std::make_shared<OAuthCredentialSource>(std::move(manager));
        return std::make_shared<XaiOAuthCredentialSource>(std::move(oauth));
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        auto flow = std::make_shared<XaiOAuthFlow>();
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "grok", std::move(flow), std::move(store));
        manager->login();
        return {
            "Grok OAuth is active through the Grok Build session endpoint.",
            "Export XAI_API_KEY or set auth_type to api_key to use public API billing instead.",
        };
    }
};

} // namespace

AuthenticationManager::AuthenticationManager(std::string config_dir)
    : config_dir_(std::move(config_dir)) {}

AuthenticationManager AuthenticationManager::create_with_defaults(std::string config_dir) {
    AuthenticationManager manager(std::move(config_dir));
    manager.register_strategy(std::make_shared<GoogleOAuthStrategy>());
    manager.register_strategy(std::make_shared<ClaudeOAuthStrategy>());
    manager.register_strategy(std::make_shared<OpenAIPkceStrategy>());
    manager.register_strategy(std::make_shared<KimiOAuthStrategy>());
    manager.register_strategy(std::make_shared<QwenOAuthStrategy>());
    manager.register_strategy(std::make_shared<XaiOAuthStrategy>());
    manager.register_strategy(make_mimo_authentication_strategy());
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        ApiKeyPromptStrategySpec{
            .login_provider = "qwen",
            .login_aliases = {
                "qwen-token-plan", "qwen_token_plan", "qwencloud", "qwen-cloud",
            },
            .display_name = "Qwen Cloud Token Plan",
            .profiles = {ApiKeyLoginProfile{
                .provider_name = "qwen-token-plan",
                .default_model = "",
            }},
            .env_var = "QWEN_TOKEN_PLAN_API_KEY",
            .docs_hint =
                "Uses the Token Plan Chat Completions API with Qwen reasoning and "
                "subscription billing. Manage usage at "
                "https://home.qwencloud.com/token-plan.",
        }));
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        ApiKeyPromptStrategySpec{
            .login_provider = "qwen-coding",
            .login_aliases = {
                "qwen_coding", "qwencoding", "qwen-coding-plan",
                "coding-plan", "codingplan", "bailian",
            },
            .display_name = "Qwen Coding Plan",
            .profiles = {ApiKeyLoginProfile{
                .provider_name = "qwen-coding",
                .default_model = "qwen3-coder-plus",
            }},
            .env_var = "QWEN_CODING_PLAN_API_KEY",
            .docs_hint =
                "Uses the Qwen Coding Plan endpoint at "
                "https://coding.dashscope.aliyuncs.com/v1. BAILIAN_CODING_PLAN_API_KEY "
                "is also accepted. This is not the Token Plan or public DashScope key.",
        }));
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        ApiKeyPromptStrategySpec{
            .login_provider = "dashscope",
            .login_aliases = {"qwen-api", "qwen_api", "qwen-dashscope"},
            .display_name = "Qwen",
            .profiles = {ApiKeyLoginProfile{
                .provider_name = "qwen",
                // The coder line belongs to the Coding Plan endpoint; defaulting
                // the pay-as-you-go login to it would hand out a model this
                // provider is not the catalog owner of.
                .default_model = "qwen3-max",
            }},
            .env_var = "QWEN_API_KEY",
            .docs_hint =
                "Uses the public DashScope compatible-mode API "
                "(dashscope-intl.aliyuncs.com). DASHSCOPE_API_KEY is also accepted. "
                "For Coding Plan use `filo --auth qwen-coding` instead.",
        }));
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        ApiKeyPromptStrategySpec{
            .login_provider = "zai-coding",
            .login_aliases = {
                "z.ai-coding", "z-ai-coding", "zai_coding", "zai-coding-plan",
                "z.aicodingplan", "zaicoding",
            },
            .display_name = "Z.AI Coding Plan",
            .profiles = {ApiKeyLoginProfile{
                .provider_name = "zai-coding",
                .default_model = "glm-5.3",
            }},
            .env_var = "ZAI_CODING_API_KEY",
            .docs_hint =
                "Uses the GLM Coding Plan Anthropic endpoint at "
                "https://api.z.ai/api/anthropic. ZAI_API_KEY is also accepted. "
                "This is not a General API pay-as-you-go key.",
        }));
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        ApiKeyPromptStrategySpec{
            .login_provider = "zai",
            .login_aliases = {"z.ai", "z-ai", "zai-api"},
            .display_name = "Z.AI",
            .profiles = {ApiKeyLoginProfile{
                .provider_name = "zai",
                .default_model = "glm-5.1",
            }},
            .env_var = "ZAI_API_KEY",
            .docs_hint =
                "Uses the General API at https://api.z.ai/api/paas/v4. "
                "For a GLM Coding Plan subscription use `filo --auth zai-coding` instead.",
        }));
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        ApiKeyPromptStrategySpec{
            .login_provider = "mistral",
            .display_name = "Mistral",
            .profiles = {ApiKeyLoginProfile{
                .provider_name = "mistral",
                .default_model = "mistral-vibe-cli-latest",
            }},
            .env_var = "MISTRAL_API_KEY",
            .docs_hint =
                "Uses api.mistral.ai. A Vibe or Studio key from console.mistral.ai "
                "works with your subscription plan's included monthly usage. "
                "mistral-vibe browser sign-in stores the same key as MISTRAL_API_KEY "
                "in ~/.vibe/.env.",
        }));
    return manager;
}

void AuthenticationManager::register_strategy(std::shared_ptr<IAuthStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("Cannot register a null auth strategy.");
    }
    strategies_.push_back(std::move(strategy));
}

LoginResult AuthenticationManager::login(std::string_view provider) const {
    const std::string requested = normalize(provider);

    for (const auto& strategy : strategies_) {
        if (matches_login_provider(*strategy, requested)) {
            return {
                .provider = std::string(strategy->display_name()),
                .login_provider = std::string(strategy->login_provider()),
                .hints = strategy->login(config_dir_),
            };
        }
    }

    const auto available = available_login_providers();
    throw std::runtime_error(
        "Unknown provider '" + std::string(provider)
        + "'. Available: " + join(available));
}

std::optional<AuthenticationProviderDescriptor>
AuthenticationManager::describe_provider(std::string_view provider) const {
    const std::string requested = normalize(provider);
    for (const auto& strategy : strategies_) {
        const std::string credential_id =
            normalize(strategy->token_store_key());
        if (!matches_login_provider(*strategy, requested)
            && requested != credential_id) {
            continue;
        }
        if (strategy->login_provider().empty()
            || strategy->token_store_key().empty()) {
            continue;
        }
        return AuthenticationProviderDescriptor{
            .credential_id = credential_id,
            .login_provider = std::string(strategy->login_provider()),
            .display_name = std::string(strategy->display_name()),
        };
    }
    return std::nullopt;
}

std::string AuthenticationManager::logout(std::string_view provider,
                                          bool revoke_remote) const {
    const std::string requested = normalize(provider);
    for (const auto& strategy : strategies_) {
        if (!matches_login_provider(*strategy, requested)) continue;
        const std::string_view store_key = strategy->token_store_key();
        if (store_key.empty()) {
            throw std::runtime_error(
                "Provider '" + std::string(provider)
                + "' does not use a cached OAuth session; nothing to sign out of.");
        }

        FileTokenStore store(config_dir_);
        auto store_lock = store.acquire_refresh_lock(store_key);

        // Best-effort server-side revocation before clearing local state,
        // when the provider advertises support. Local credentials are removed
        // even when the revocation request fails.
        if (revoke_remote) {
            if (const auto flow = strategy->logout_revocation_flow()) {
                if (const auto token = store.load(store_key)) {
                    try {
                        flow->revoke(*token);
                    } catch (const std::exception& error) {
                        core::logging::warn(
                            "Could not revoke {} session server-side: {}. "
                            "Clearing local credentials anyway.",
                            strategy->display_name(), error.what());
                    }
                }
            }
        }

        store.clear(store_key);
        return std::string(strategy->display_name());
    }
    throw std::runtime_error("Unknown authentication provider '" + std::string(provider) + "'.");
}

std::shared_ptr<ICredentialSource> AuthenticationManager::create_credential_source(
    std::string_view canonical_type,
    const core::config::ProviderConfig& provider_config) const {
    const std::string auth_type = normalize(provider_config.auth_type);
    if (auth_type.empty() || auth_type == "api_key") {
        return nullptr;
    }

    const std::string ptype = normalize(canonical_type);
    for (const auto& strategy : strategies_) {
        if (strategy->supports(ptype, auth_type)) {
            return strategy->create_credential_source(provider_config, config_dir_);
        }
    }

    return nullptr;
}

std::vector<std::string> AuthenticationManager::available_login_providers() const {
    std::vector<std::string> providers;
    providers.reserve(strategies_.size());
    for (const auto& strategy : strategies_) {
        const std::string provider = std::string(strategy->login_provider());
        if (!provider.empty()) {
            providers.push_back(provider);
        }
    }
    std::sort(providers.begin(), providers.end());
    providers.erase(std::unique(providers.begin(), providers.end()), providers.end());
    return providers;
}

} // namespace core::auth
