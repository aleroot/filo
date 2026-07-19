#include "AuthenticationManager.hpp"
#include "ClaudeOAuthFlow.hpp"
#include "FileTokenStore.hpp"
#include "GoogleOAuthCredentialSource.hpp"
#include "GoogleOAuthFlow.hpp"
#include "KimiOAuthFlow.hpp"
#include "QwenOAuthFlow.hpp"
#include "XaiOAuthFlow.hpp"
#include "XaiOAuthCredentialSource.hpp"
#include "OpenAIOAuthFlow.hpp"
#include "OAuthCredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include "ui/ConsoleAuthUI.hpp"
#include "core/logging/Logger.hpp"
#include "core/utils/JsonWriter.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

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

std::string normalize_login_provider(std::string_view provider) {
    std::string requested = normalize(provider);
    if (requested == "openai-pkce" || requested == "openai_pkce"
        || requested == "openaipkce") {
        return "openai";
    }
    if (requested == "xai" || requested == "x.ai" || requested == "x-ai") {
        return "grok";
    }
    if (requested == "z.ai" || requested == "z-ai" || requested == "zai-api") {
        return "zai";
    }
    if (requested == "z.ai-coding" || requested == "z-ai-coding"
        || requested == "zai_coding" || requested == "zai-coding-plan"
        || requested == "z.aicodingplan") {
        return "zai";
    }
    if (requested == "qwen-token-plan" || requested == "qwen_token_plan"
        || requested == "qwencloud" || requested == "qwen-cloud") {
        return "qwen";
    }
    return requested;
}

std::string join(const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += ", ";
        out += values[i];
    }
    return out;
}

struct AuthOverlayProvider {
    std::string model;
    std::string auth_type;
    std::string api_key;
};

struct ApiKeyProviderSeed {
    std::string provider_name;
    std::string model;
};

void write_api_key_overlay(std::string_view config_dir,
                           std::string_view default_provider,
                           const std::vector<ApiKeyProviderSeed>& provider_seeds,
                           std::string_view api_key) {
    if (api_key.empty()) {
        throw std::runtime_error("API key cannot be empty.");
    }
    if (default_provider.empty() || provider_seeds.empty()) {
        throw std::runtime_error("Provider login is not configured correctly.");
    }

    const std::filesystem::path dir(config_dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / "auth_defaults.json";

    std::unordered_map<std::string, AuthOverlayProvider> providers;
    if (std::filesystem::exists(path)) {
        try {
            simdjson::padded_string json =
                simdjson::padded_string::load(path.string());
            simdjson::dom::parser parser;
            simdjson::dom::element doc = parser.parse(json);
            simdjson::dom::object providers_obj;
            if (doc["providers"].get(providers_obj) == simdjson::SUCCESS) {
                for (auto field : providers_obj) {
                    simdjson::dom::object provider_obj;
                    if (field.value.get(provider_obj) != simdjson::SUCCESS) {
                        continue;
                    }
                    AuthOverlayProvider saved;
                    std::string_view value;
                    if (provider_obj["model"].get(value) == simdjson::SUCCESS) {
                        saved.model = std::string(value);
                    }
                    if (provider_obj["auth_type"].get(value) == simdjson::SUCCESS) {
                        saved.auth_type = std::string(value);
                    }
                    if (provider_obj["api_key"].get(value) == simdjson::SUCCESS) {
                        saved.api_key = std::string(value);
                    }
                    providers[std::string(field.key)] = std::move(saved);
                }
            }
        } catch (...) {
            providers.clear();
        }
    }

    for (const auto& seed : provider_seeds) {
        if (seed.provider_name.empty()) continue;
        AuthOverlayProvider& selected = providers[seed.provider_name];
        selected.model = seed.model;
        selected.auth_type.clear();
        selected.api_key = std::string(api_key);
    }

    std::vector<std::string> names;
    names.reserve(providers.size());
    for (const auto& [name, provider] : providers) {
        if (!provider.model.empty()
            || !provider.auth_type.empty()
            || !provider.api_key.empty()) {
            names.push_back(name);
        }
    }
    std::sort(names.begin(), names.end());

    core::utils::JsonWriter writer(512);
    {
        auto root = writer.object();
        writer.kv_str("default_provider", default_provider).comma();
        writer.kv_str("default_model_selection", "manual").comma();
        writer.key("providers");
        auto providers_object = writer.object();
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i > 0) writer.comma();
            writer.key(names[i]);
            auto provider_object = writer.object();
            const auto& provider = providers.at(names[i]);
            bool has_field = false;
            if (!provider.model.empty()) {
                writer.kv_str("model", provider.model);
                has_field = true;
            }
            if (!provider.auth_type.empty()) {
                if (has_field) writer.comma();
                writer.kv_str("auth_type", provider.auth_type);
                has_field = true;
            }
            if (!provider.api_key.empty()) {
                if (has_field) writer.comma();
                writer.kv_str("api_key", provider.api_key);
            }
        }
    }

    std::string payload = std::move(writer).take();
    payload.push_back('\n');

    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("Failed to open auth overlay for writing.");
        }
        out << payload;
        if (!out) {
            throw std::runtime_error("Failed to write auth overlay.");
        }
    }
    std::filesystem::rename(tmp, path);
}

class ApiKeyPromptStrategy final : public IAuthStrategy {
public:
    ApiKeyPromptStrategy(std::string login_provider,
                         std::string display_name,
                         std::string provider_name,
                         std::string default_model,
                         std::vector<ApiKeyProviderSeed> additional_provider_seeds,
                         std::string env_var,
                         std::string docs_hint)
        : login_provider_(std::move(login_provider))
        , display_name_(std::move(display_name))
        , provider_name_(std::move(provider_name))
        , default_model_(std::move(default_model))
        , additional_provider_seeds_(std::move(additional_provider_seeds))
        , env_var_(std::move(env_var))
        , docs_hint_(std::move(docs_hint)) {}

    std::string_view login_provider() const noexcept override { return login_provider_; }
    std::string_view display_name() const noexcept override { return display_name_; }

    bool supports(std::string_view /*provider_type*/,
                  std::string_view /*auth_type*/) const noexcept override {
        return false;
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view /*config_dir*/) const override {
        return nullptr;
    }

    void login(std::string_view config_dir) const override {
        ui::ConsoleAuthUI ui;
        ui.show_header(display_name_ + " API Key Login");
        ui.show_instructions(
            "Paste an API key for this provider. It will be saved in Filo's "
            "auth_defaults.json overlay and used as the default provider.");
        const std::string key = ui.prompt_secret("API key:");
        auto seeds = additional_provider_seeds_;
        seeds.insert(seeds.begin(), ApiKeyProviderSeed{
            .provider_name = provider_name_,
            .model = default_model_,
        });
        write_api_key_overlay(config_dir, provider_name_, seeds, key);
        ui.show_success(display_name_ + " credential saved.");
    }

    std::vector<std::string> post_login_hints() const override {
        std::vector<std::string> hints;
        if (default_model_.empty()) {
            hints.push_back("Default provider is set to '" + provider_name_
                            + "'; Filo will select the newest model from its live catalog.");
        } else {
            hints.push_back("Default provider is set to '" + provider_name_
                            + "' with model '" + default_model_ + "'.");
        }
        if (!env_var_.empty()) {
            hints.push_back("For CI or one-off use, you can also export "
                            + env_var_ + ".");
        }
        if (!docs_hint_.empty()) {
            hints.push_back(docs_hint_);
        }
        return hints;
    }

private:
    std::string login_provider_;
    std::string display_name_;
    std::string provider_name_;
    std::string default_model_;
    std::vector<ApiKeyProviderSeed> additional_provider_seeds_;
    std::string env_var_;
    std::string docs_hint_;
};

class GoogleOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "google"; }
    std::string_view display_name() const noexcept override { return "Google"; }
    std::string_view token_store_key() const noexcept override { return "google"; }

    std::shared_ptr<IOAuthTokenRevoker> logout_revocation_flow() const override {
        return std::make_shared<GoogleOAuthFlow>();
    }

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
        return std::make_shared<GoogleOAuthCredentialSource>(std::move(manager));
    }

    void login(std::string_view config_dir) const override {
        auto flow = std::make_shared<GoogleOAuthFlow>(std::make_shared<ui::ConsoleAuthUI>());
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
            "~/.config/filo/config.json to use the stored Claude OAuth token.",
            "OAuth mode supports Claude subscription billing (Pro/Max/Team/Enterprise)."
        };
    }
};

class OpenAIPkceStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "openai"; }
    std::string_view display_name() const noexcept override { return "OpenAI (ChatGPT login)"; }
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
            "Qwen Cloud Token Plan does not support OAuth. Run `filo --auth qwen` "
            "and paste its dedicated API key instead.");
    }

    void login(std::string_view /*config_dir*/) const override {
        throw std::runtime_error(
            "Qwen Cloud Token Plan does not support OAuth. Use its dedicated API key.");
    }

    std::vector<std::string> post_login_hints() const override {
        return {
            "Run `filo --auth qwen` to configure a Token Plan API key.",
        };
    }
};

class XaiOAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override { return "grok"; }
    std::string_view display_name() const noexcept override { return "Grok"; }
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

    void login(std::string_view config_dir) const override {
        auto flow = std::make_shared<XaiOAuthFlow>();
        auto store = std::make_shared<FileTokenStore>(std::string(config_dir));
        auto manager = std::make_shared<OAuthTokenManager>(
            "grok", std::move(flow), std::move(store));
        manager->login();
    }

    std::vector<std::string> post_login_hints() const override {
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
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        "qwen",
        "Qwen Cloud Token Plan",
        "qwen-token-plan",
        "",
        std::vector<ApiKeyProviderSeed>{},
        "QWEN_TOKEN_PLAN_API_KEY",
        "Uses the Token Plan Responses API with Qwen reasoning, hosted tools, "
        "session cache, and subscription billing. Manage usage at "
        "https://home.qwencloud.com/token-plan."));
    manager.register_strategy(std::make_shared<ApiKeyPromptStrategy>(
        "zai",
        "Z.AI",
        "zai",
        "glm-5.1",
        std::vector<ApiKeyProviderSeed>{{
            .provider_name = "zai-coding",
            .model = "glm-5.2",
        }},
        "ZAI_API_KEY",
        "The same key is also saved for the Coding Plan endpoint "
        "for glm-5.2, glm-5-turbo, glm-4.7, and glm-4.5-air."));
    return manager;
}

void AuthenticationManager::register_strategy(std::shared_ptr<IAuthStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("Cannot register a null auth strategy.");
    }
    strategies_.push_back(std::move(strategy));
}

LoginResult AuthenticationManager::login(std::string_view provider) const {
    const std::string requested = normalize_login_provider(provider);

    for (const auto& strategy : strategies_) {
        if (normalize(strategy->login_provider()) == requested) {
            strategy->login(config_dir_);
            return {
                .provider = std::string(strategy->display_name()),
                .login_provider = std::string(strategy->login_provider()),
                .hints = strategy->post_login_hints(),
            };
        }
    }

    const auto available = available_login_providers();
    throw std::runtime_error(
        "Unknown provider '" + std::string(provider)
        + "'. Available: " + join(available));
}

std::string AuthenticationManager::logout(std::string_view provider,
                                          bool revoke_remote) const {
    const std::string requested = normalize_login_provider(provider);
    for (const auto& strategy : strategies_) {
        if (normalize(strategy->login_provider()) != requested) continue;
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
