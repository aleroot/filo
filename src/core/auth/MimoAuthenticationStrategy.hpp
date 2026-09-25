#pragma once

#include "AuthenticationManager.hpp"

#include <memory>

namespace core::auth {

/**
 * Xiaomi MiMo authentication — browser provisioning or pasted API key.
 *
 * Composes MiMo browser sign-in (MimoOAuthFlow) with the generic API-key
 * profile prompt as a CompositeAuthStrategy. One Xiaomi login serves every
 * regional gateway and the pay-as-you-go endpoint; the grant or the chosen
 * profile names the preset to seed. A plan key must never be copied onto the
 * pay-as-you-go provider, which rejects it.
 */
[[nodiscard]] std::shared_ptr<IAuthStrategy> make_mimo_authentication_strategy();

} // namespace core::auth
