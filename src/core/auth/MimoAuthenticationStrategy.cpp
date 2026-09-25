#include "MimoAuthenticationStrategy.hpp"

#include "ApiKeyPromptStrategy.hpp"
#include "MimoOAuthFlow.hpp"
#include "ui/ConsoleAuthUI.hpp"
#include "core/llm/MimoModelTraits.hpp"

namespace core::auth {
namespace {

/// Env-var hint shared by both MiMo credential acquisition methods.
constexpr std::string_view kMimoEnvVar =
    "MIMO_TOKEN_PLAN_API_KEY (Token Plan) or XIAOMI_API_KEY (pay-as-you-go)";

[[nodiscard]] std::string mimo_provider_for_grant(std::string_view base_url) {
    if (!base_url.empty()) {
        if (const auto region = core::llm::mimo_region_for_endpoint(base_url)) {
            return std::string(core::llm::mimo_region_provider_name(*region));
        }
        if (core::llm::is_mimo_endpoint(base_url)) {
            // A MiMo host that is not a plan gateway is pay-as-you-go.
            return "mimo";
        }
    }
    return std::string(core::llm::mimo_region_provider_name(
        core::llm::resolve_mimo_region()));
}

class MimoBrowserAuthStrategy final : public IAuthStrategy {
public:
    std::string_view login_provider() const noexcept override {
        return "xiaomi-browser";
    }

    std::string_view display_name() const noexcept override {
        return "Xiaomi MiMo";
    }

    bool supports(std::string_view /*provider_type*/,
                  std::string_view /*auth_type*/) const noexcept override {
        return false;
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view /*config_dir*/) const override {
        return nullptr;
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        ui::ConsoleAuthUI ui;
        ui.show_header("Xiaomi MiMo Browser Login");
        ui.show_instructions(
            "Sign in at platform.xiaomimimo.com to provision an API key for "
            "Filo. The key is returned encrypted to this session and saved in "
            "Filo's auth_defaults.json overlay.");

        MimoOAuthFlow flow;
        flow.set_key_name(MimoOAuthFlow::key_name(config_dir));
        const MimoAuthorizationGrant grant = flow.login(ui);
        const std::string provider_name = mimo_provider_for_grant(grant.base_url);
        const std::string default_model = "mimo-v2.6-pro";
        save_api_key_profile(
            config_dir,
            ApiKeyLoginProfile{
                .provider_name = provider_name,
                .default_model = default_model,
            },
            grant.api_key);
        ui.show_success("Xiaomi MiMo credential saved for '" + provider_name
                        + "'.");
        return {
            "Default provider is set to '" + provider_name
                + "' with model '" + default_model + "'.",
            "For CI or one-off use, you can also export "
                + std::string(kMimoEnvVar) + ".",
        };
    }
};

[[nodiscard]] std::shared_ptr<IAuthStrategy> make_mimo_api_key_strategy() {
    return std::make_shared<ApiKeyPromptStrategy>(ApiKeyPromptStrategySpec{
        .login_provider = "xiaomi-api-key",
        .display_name = "Xiaomi MiMo",
        .instructions =
            "Select the exact Base URL shown beside this key on the Token Plan "
            "page. Token Plan keys are regional; using a different region "
            "returns 401.",
        .profiles = {
            ApiKeyLoginProfile{
                .selection = {
                    .label = "Token Plan — Europe (Amsterdam)",
                    .description = "Uses the Amsterdam Token Plan gateway.",
                },
                .provider_name = "mimo-token-plan",
                .default_model = "mimo-v2.6-pro",
                .key_prompt = "Token Plan API key:",
                .accepted_key_prefixes = {"tp-", "ttp-"},
                .invalid_key_message =
                    "Token Plan API keys begin with 'tp-' or 'ttp-'. "
                    "Choose Pay-as-you-go for an 'sk-' key.",
            },
            ApiKeyLoginProfile{
                .selection = {
                    .label = "Token Plan — Singapore",
                    .description = "Uses the Singapore Token Plan gateway.",
                },
                .provider_name = "mimo-token-plan-sgp",
                .default_model = "mimo-v2.6-pro",
                .key_prompt = "Token Plan API key:",
                .accepted_key_prefixes = {"tp-", "ttp-"},
                .invalid_key_message =
                    "Token Plan API keys begin with 'tp-' or 'ttp-'. "
                    "Choose Pay-as-you-go for an 'sk-' key.",
            },
            ApiKeyLoginProfile{
                .selection = {
                    .label = "Token Plan — China",
                    .description = "Uses the China Token Plan gateway.",
                },
                .provider_name = "mimo-token-plan-cn",
                .default_model = "mimo-v2.6-pro",
                .key_prompt = "Token Plan API key:",
                .accepted_key_prefixes = {"tp-", "ttp-"},
                .invalid_key_message =
                    "Token Plan API keys begin with 'tp-' or 'ttp-'. "
                    "Choose Pay-as-you-go for an 'sk-' key.",
            },
            ApiKeyLoginProfile{
                .selection = {
                    .label = "Pay-as-you-go",
                    .description = "Uses api.xiaomimimo.com instead of Token Plan.",
                },
                .provider_name = "mimo",
                .default_model = "mimo-v2.6-pro",
                .key_prompt = "Pay-as-you-go API key:",
                .accepted_key_prefixes = {"sk-"},
                .invalid_key_message =
                    "Pay-as-you-go API keys begin with 'sk-'. "
                    "Choose a Token Plan region for a 'tp-' or 'ttp-' key.",
            },
        },
        .env_var = std::string(kMimoEnvVar),
    });
}

} // namespace

std::shared_ptr<IAuthStrategy> make_mimo_authentication_strategy() {
    return std::make_shared<CompositeAuthStrategy>(CompositeAuthStrategySpec{
        .login_provider = "xiaomi",
        .login_aliases = {
            "mimo",
            "mimocode",
            "mimo-code",
            "mimo_code",
            "xiaomi-mimo",
            "mimo-token-plan",
            "mimo-token-plan-sgp",
            "mimo-token-plan-cn",
            "mimo-token-plan-ams",
        },
        .display_name = "Xiaomi MiMo",
        .instructions = "Choose how you want to connect MiMo.",
        .choices = {
            AuthStrategyChoice{
                .selection = {
                    .label = "Sign in with Xiaomi in a browser",
                    .description =
                        "Provisions a new key and selects its endpoint automatically.",
                },
                .strategy = std::make_shared<MimoBrowserAuthStrategy>(),
            },
            AuthStrategyChoice{
                .selection = {
                    .label = "Paste an existing API key",
                    .description =
                        "Choose its Token Plan region or pay-as-you-go endpoint.",
                },
                .strategy = make_mimo_api_key_strategy(),
            },
        },
    });
}

} // namespace core::auth
