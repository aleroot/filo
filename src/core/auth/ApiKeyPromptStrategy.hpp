#pragma once

#include "AuthenticationManager.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

/** A user-visible choice in an interactive authentication flow. */
struct AuthMenuOption {
    std::string label;
    std::string description;
};

/** Provider configuration written when an API-key profile is selected. */
struct ApiKeyProviderSeed {
    std::string provider_name;
    std::string model;
};

/**
 * Declarative API-key profile for providers with multiple compatible
 * endpoints, plans, or regions.
 */
struct ApiKeyLoginProfile {
    AuthMenuOption selection;
    std::string provider_name;
    std::string default_model;
    std::vector<ApiKeyProviderSeed> additional_provider_seeds;
    std::string key_prompt = "API key:";
    std::vector<std::string> accepted_key_prefixes;
    std::string invalid_key_message;
};

/** Persist an API-key profile as the active default provider. */
void save_api_key_profile(std::string_view config_dir,
                          const ApiKeyLoginProfile& profile,
                          std::string_view api_key);

/** Declarative configuration for ApiKeyPromptStrategy. */
struct ApiKeyPromptStrategySpec {
    /// Primary command identifier (`filo --auth <id>`).
    std::string login_provider;
    /// Alternate command identifiers accepted for the same login.
    std::vector<std::string> login_aliases;
    std::string display_name;
    /// Shown before the prompts; defaults to the standard paste-a-key text.
    std::string instructions;
    /// One entry per compatible endpoint, plan, or region. More than one entry
    /// presents a selector menu before the secret prompt.
    std::vector<ApiKeyLoginProfile> profiles;
    /// Environment variable (or short phrase) named in the post-login hint.
    std::string env_var;
    /// Extra post-login hint.
    std::string docs_hint;
};

/**
 * Generic API-key authentication strategy.
 *
 * A single profile produces the conventional key prompt. Multiple profiles
 * present a reusable endpoint/plan selector before the secret prompt.
 */
class ApiKeyPromptStrategy final : public IAuthStrategy {
public:
    explicit ApiKeyPromptStrategy(ApiKeyPromptStrategySpec spec);

    std::string_view login_provider() const noexcept override;
    std::string_view display_name() const noexcept override;
    std::vector<std::string_view> login_aliases() const override;
    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override;
    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& provider_config,
        std::string_view config_dir) const override;
    std::vector<std::string> login(std::string_view config_dir) const override;

private:
    ApiKeyPromptStrategySpec spec_;
};

/** A child strategy exposed through a generic interactive auth menu. */
struct AuthStrategyChoice {
    AuthMenuOption selection;
    std::shared_ptr<IAuthStrategy> strategy;
};

/** Declarative configuration for CompositeAuthStrategy. */
struct CompositeAuthStrategySpec {
    /// Primary command identifier (`filo --auth <id>`).
    std::string login_provider;
    /// Alternate command identifiers accepted for the same login.
    std::vector<std::string> login_aliases;
    std::string display_name;
    std::string instructions;
    std::vector<AuthStrategyChoice> choices;
};

/**
 * Composite strategy that selects and delegates to one of several concrete
 * authentication strategies. It is reusable for providers that support more
 * than one credential acquisition method. Post-login hints come from the
 * child strategy that actually ran, so they describe what was saved.
 */
class CompositeAuthStrategy final : public IAuthStrategy {
public:
    explicit CompositeAuthStrategy(CompositeAuthStrategySpec spec);

    std::string_view login_provider() const noexcept override;
    std::string_view display_name() const noexcept override;
    std::vector<std::string_view> login_aliases() const override;
    bool supports(std::string_view provider_type,
                  std::string_view auth_type) const noexcept override;
    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& provider_config,
        std::string_view config_dir) const override;
    std::vector<std::string> login(std::string_view config_dir) const override;

private:
    CompositeAuthStrategySpec spec_;
};

} // namespace core::auth
