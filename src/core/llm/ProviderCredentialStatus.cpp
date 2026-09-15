#include "ProviderCredentialStatus.hpp"

#include "LLMProvider.hpp"
#include "../auth/ICredentialSource.hpp"

#include <exception>

namespace core::llm {

ProviderCredentialState provider_credential_state(
    const std::shared_ptr<LLMProvider>& provider) {
    if (!provider) {
        return ProviderCredentialState::Unknown;
    }

    try {
        // A local backend authenticates nothing, so it is always usable.
        if (provider->capabilities().is_local) {
            return ProviderCredentialState::Ready;
        }

        const auto metadata = provider->metadata();
        if (!metadata || !metadata->credential_source) {
            return ProviderCredentialState::Unknown;
        }

        switch (metadata->credential_source->availability()) {
            case core::auth::CredentialAvailability::Ready:
                return ProviderCredentialState::Ready;
            case core::auth::CredentialAvailability::Missing:
                return ProviderCredentialState::Missing;
            case core::auth::CredentialAvailability::Unknown:
                break;
        }
    } catch (const std::exception&) {
        // A probe must never be the reason a provider disappears from the UI.
        return ProviderCredentialState::Unknown;
    }

    return ProviderCredentialState::Unknown;
}

std::string_view credential_state_note(
    ProviderCredentialState state) noexcept {
    return credentials_missing(state) ? "No credentials — run /login." : "";
}

std::vector<std::size_t> order_by_credential_state(
    std::span<const ProviderCredentialState> states) {
    std::vector<std::size_t> order;
    order.reserve(states.size());

    for (std::size_t i = 0; i < states.size(); ++i) {
        if (!credentials_missing(states[i])) {
            order.push_back(i);
        }
    }
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (credentials_missing(states[i])) {
            order.push_back(i);
        }
    }

    return order;
}

} // namespace core::llm
