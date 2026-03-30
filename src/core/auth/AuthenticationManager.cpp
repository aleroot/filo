#include "AuthenticationManager.hpp"
#include "ClaudeOAuthFlow.hpp"
#include "FileTokenStore.hpp"
#include "GoogleOAuthFlow.hpp"
#include "KimiOAuthFlow.hpp"
#include "QwenOAuthFlow.hpp"
#include "OpenAIAuthFlow.hpp"
#include "OpenAIOAuthFlow.hpp"
#include "OAuthCredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include "ui/ConsoleAuthUI.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace core::auth {

namespace {

std::string normalize(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

std::string join(const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ", ";
        out += values[i];
    }
    return out;
}

class GoogleOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "google"; }
    std::string_view display_name() const noexcept override { return "Google"; }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "gemini" && auth_type == "oauth_google";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow = std::make_shared<GoogleOAuthFlow>();
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "google", std::move(flow), std::move(store));
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        auto flow = std::make_shared<GoogleOAuthFlow>();
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "google", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
        return {
            "Set \"auth_type\": \"oauth_google\" on a Gemini provider in "
            "~/.config/filo/config.json to use it."
        };
    }
};

class ClaudeOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "claude"; }
    std::string_view display_name() const noexcept override { return "Claude"; }

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
            "claude", std::move(flow), std::move(store));
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        // For explicit login command, we provide the Console UI.
        auto flow = std::make_shared<ClaudeOAuthFlow>(std::make_shared<ui::ConsoleAuthUI>());
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "claude", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
        return {
            "Set \"auth_type\": \"oauth_claude\" on a Claude provider in "
            "~/.config/filo/config.json to use the stored bearer token.",
            "If you need Sonnet/Opus for production SDK usage, configure an "
            "ANTHROPIC_API_KEY-based provider."
        };
    }
};

class OpenAIKeyStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "openai"; }
    std::string_view display_name() const noexcept override { return "OpenAI"; }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "openai" && auth_type == "oauth_openai";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow    = std::make_shared<OpenAIAuthFlow>(nullptr);
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "openai", std::move(flow), std::move(store));
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        auto flow    = std::make_shared<OpenAIAuthFlow>(std::make_shared<ui::ConsoleAuthUI>());
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "openai", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
        return {
            "Set \"auth_type\": \"oauth_openai\" on an OpenAI provider in "
            "~/.config/filo/config.json to use the stored API key.",
            "You can also export OPENAI_API_KEY for one-time / CI use without storing.",
        };
    }
};

class OpenAIPkceStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "openai-pkce"; }
    std::string_view display_name() const noexcept override { return "OpenAI (ChatGPT login)"; }

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
            "openai-pkce", std::move(flow), std::move(store));
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        auto flow    = std::make_shared<OpenAIOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "openai-pkce", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
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
            "kimi", std::move(flow), std::move(store));
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        auto flow    = std::make_shared<KimiOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "kimi", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
        return {
            "Set \"auth_type\": \"oauth_kimi\" on a Kimi provider in "
            "~/.config/filo/config.json to use the stored OAuth token.",
            "You can also export KIMI_API_KEY for one-time / CI use without OAuth.",
        };
    }
};

class QwenOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "qwen"; }
    std::string_view display_name() const noexcept override { return "Qwen (chat.qwen.ai)"; }

    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override {
        return provider_type == "qwen" && auth_type == "oauth_qwen";
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        auto flow    = std::make_shared<QwenOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "qwen", std::move(flow), std::move(store));
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        auto flow    = std::make_shared<QwenOAuthFlow>();
        auto store   = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "qwen", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
        return {
            "Set \"auth_type\": \"oauth_qwen\" on a Qwen provider in "
            "~/.config/filo/config.json to use the stored OAuth token.",
            "The free tier provides 1000 requests/day via the \"coder-model\" alias.",
            "You can also export DASHSCOPE_API_KEY for API key authentication.",
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
    manager.register_strategy(std::make_shared<OpenAIKeyStrategy>());
    manager.register_strategy(std::make_shared<OpenAIPkceStrategy>());
    manager.register_strategy(std::make_shared<KimiOAuthStrategy>());
    manager.register_strategy(std::make_shared<QwenOAuthStrategy>());
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
        if (normalize(strategy->login_provider()) == requested) {
            strategy->login(config_dir_);
            return {
                .provider = std::string(strategy->display_name()),
                .hints = strategy->post_login_hints(),
            };
        }
    }

    const auto available = available_login_providers();
    throw std::runtime_error(
        "Unknown provider '" + std::string(provider)
        + "'. Available: " + join(available));
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
        providers.push_back(std::string(strategy->login_provider()));
    }
    std::sort(providers.begin(), providers.end());
    providers.erase(std::unique(providers.begin(), providers.end()), providers.end());
    return providers;
}

} // namespace core::auth
