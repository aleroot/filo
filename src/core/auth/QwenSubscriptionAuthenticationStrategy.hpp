#pragma once

#include "AuthenticationManager.hpp"
#include "ui/AuthUI.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace core::auth {

/// Endpoint and side effects of the Qwen Subscription sign-in. Production uses
/// the defaults; tests point the auth origin at a local server.
struct QwenSubscriptionStrategyOptions {
    std::string auth_origin = {};
    std::shared_ptr<ui::AuthUI> ui = {};
    std::function<void(std::string_view)> open_browser = {};
    std::function<void(std::chrono::milliseconds)> sleep = {};
};

/**
 * `filo --auth qwen-subscription`: sign in to a Qwen account, confirm a Plus
 * or Pro membership, and make the subscription the manual selection. The
 * model is left to the provider's catalog selector, which takes the first
 * model the gateway lists.
 *
 * The session is stored like other OAuth sessions (token store key
 * `qwen-subscription`), refreshed on expiry and once on HTTP 401, and revoked
 * on `/logout qwen-subscription`.
 */
[[nodiscard]] std::shared_ptr<IAuthStrategy> make_qwen_subscription_authentication_strategy(
    QwenSubscriptionStrategyOptions options = {});

} // namespace core::auth
