#include "QwenSubscriptionAuthenticationStrategy.hpp"

#include "AuthBrowserLauncher.hpp"
#include "FileTokenStore.hpp"
#include "OAuthCredentialSource.hpp"
#include "OAuthTokenManager.hpp"
#include "QwenSubscriptionOAuthFlow.hpp"
#include "ui/ConsoleAuthUI.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/QwenSubscription.hpp"
#include "core/utils/StringUtils.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace core::auth {
namespace {

namespace subscription = core::llm::qwen_subscription;

class QwenSubscriptionStrategy final : public IAuthStrategy {
public:
    explicit QwenSubscriptionStrategy(QwenSubscriptionStrategyOptions options)
        : options_(std::move(options)) {}

    std::string_view login_provider() const noexcept override { return subscription::kProviderName; }
    std::string_view display_name() const noexcept override { return subscription::kDisplayName; }
    std::vector<std::string_view> login_aliases() const override {
        return {"qwen_subscription", "qwensubscription", "qwen-account"};
    }
    std::string_view token_store_key() const noexcept override { return subscription::kProviderName; }

    std::shared_ptr<IOAuthTokenRevoker> logout_revocation_flow() const override {
        return make_flow();
    }

    bool supports(std::string_view provider_type, std::string_view auth_type) const noexcept override {
        return provider_type == subscription::kProviderName && auth_type == subscription::kAuthType;
    }

    std::shared_ptr<ICredentialSource> create_credential_source(
        const core::config::ProviderConfig& /*provider_config*/,
        std::string_view config_dir) const override {
        // Requests never start a browser sign-in: an unusable session surfaces
        // as "run /auth qwen-subscription" instead.
        auto manager = std::make_shared<OAuthTokenManager>(
            std::string(subscription::kProviderName), make_flow(),
            std::make_shared<FileTokenStore>(std::string(config_dir)),
            /*allow_interactive_login=*/false);
        return std::make_shared<OAuthCredentialSource>(std::move(manager));
    }

    std::vector<std::string> login(std::string_view config_dir) const override {
        const auto ui = options_.ui ? options_.ui : std::make_shared<ui::ConsoleAuthUI>();
        ui->show_header(std::string(subscription::kDisplayName));
        ui->show_instructions(
            "Sign in with the Qwen account that holds your Qwen Code Plus or Pro "
            "subscription. Filo uses it to call the Qwen Subscription gateway.");

        const auto flow = make_flow();
        const auto device = flow->request_device_authorization();
        const std::string url = subscription::authorization_url(device.user_code);
        ui->show_url(url, "Approve this sign-in in your browser:");
        ui->show_instructions("Code: " + device.user_code + "\nWaiting for authorization...");
        open(url);
        OAuthToken token = flow->poll_for_token(device);

        auto info = flow->user_info(token.access_token);
        token.user_id = info.subject;
        token.email = info.email;
        // Keep the session even without a qualifying plan: subscribing in the
        // browser and checking again must not require another authorization.
        OAuthTokenManager(std::string(subscription::kProviderName), flow,
                          std::make_shared<FileTokenStore>(std::string(config_dir)))
            .save_token(token);

        while (!subscription::membership_qualifies(info.member_type)) {
            ui->show_error("No active Plus or Pro subscription was found for "
                           + account_label(info) + " (current plan: "
                           + subscription::display_plan(info.member_type) + ").");
            ui->show_url(subscription::plans_url(), "Qwen Code plans:");
            const auto answer = ui->prompt_text(
                "Press Enter to check again, 'o' to open the plans page, or 'q' to stop:");
            const std::string choice = answer
                ? core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_view(*answer))
                : std::string("q");
            if (choice == "q") {
                throw std::runtime_error(
                    "Qwen Subscription needs a Plus or Pro plan. You are still signed in; run "
                    "/auth qwen-subscription again after subscribing, or /logout qwen-subscription.");
            }
            if (choice == "o") {
                open(subscription::plans_url());
                continue;
            }
            info = flow->user_info(token.access_token);
        }

        std::vector<std::string> hints{
            "Signed in as " + account_label(info) + " with Qwen Code "
                + subscription::display_plan(info.member_type) + '.',
        };
        // Like an API-key sign-in, this explicit action becomes the manual
        // selection. The model stays empty so the provider's catalog selector
        // picks the gateway's first entry, as Qwen Code does at sign-in.
        std::string error;
        if (!core::config::persist_model_defaults_overlay(
                std::filesystem::path(config_dir), subscription::kProviderName,
                "manual", "", &error)) {
            throw std::runtime_error(
                "Signed in, but Filo could not activate Qwen Subscription: " + error);
        }
        hints.push_back("Filo uses the first model your subscription lists; "
                        "use /model to choose another one.");
        hints.push_back("Run /upgrade for more credits and /logout qwen-subscription to sign out.");
        return hints;
    }

private:
    [[nodiscard]] std::shared_ptr<QwenSubscriptionOAuthFlow> make_flow() const {
        return std::make_shared<QwenSubscriptionOAuthFlow>(QwenSubscriptionOAuthFlow::Options{
            .auth_origin = options_.auth_origin,
            .open_browser = options_.open_browser,
            .sleep = options_.sleep,
        });
    }

    void open(std::string_view url) const {
        if (options_.open_browser) options_.open_browser(url);
        else open_browser(url);
    }

    [[nodiscard]] static std::string account_label(
        const QwenSubscriptionOAuthFlow::UserInfo& info) {
        if (!info.email.empty()) return info.email;
        if (!info.name.empty()) return info.name;
        return "this account";
    }

    QwenSubscriptionStrategyOptions options_;
};

} // namespace

std::shared_ptr<IAuthStrategy> make_qwen_subscription_authentication_strategy(
    QwenSubscriptionStrategyOptions options) {
    return std::make_shared<QwenSubscriptionStrategy>(std::move(options));
}

} // namespace core::auth
