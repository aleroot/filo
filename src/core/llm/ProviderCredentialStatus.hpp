#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace core::llm {

class LLMProvider;

/**
 * Credential readiness of one selectable provider service.
 *
 * This is a non-blocking probe intended for presentation code: it never
 * refreshes a token, opens a browser, or issues a network request. `Unknown`
 * is therefore a legitimate answer (e.g. a custom credential source that
 * cannot report without doing work) and callers must treat it as usable.
 */
enum class ProviderCredentialState {
    Unknown,
    Ready,
    Missing,
};

/**
 * True when the service would send an unauthenticated request as configured.
 *
 * Only an explicit `Missing` answer counts: `Unknown` never demotes a service.
 */
[[nodiscard]] constexpr bool credentials_missing(
    ProviderCredentialState state) noexcept {
    return state == ProviderCredentialState::Missing;
}

/**
 * Probe a registered provider for credential readiness.
 *
 * Local backends need no credential and always report `Ready`. Providers that
 * expose no metadata, or no credential source, report `Unknown`.
 */
[[nodiscard]] ProviderCredentialState provider_credential_state(
    const std::shared_ptr<LLMProvider>& provider);

/**
 * Human-readable suffix appended to a picker row for this state.
 *
 * Empty for every state that does not need a warning, so callers can append
 * unconditionally.
 */
[[nodiscard]] std::string_view credential_state_note(
    ProviderCredentialState state) noexcept;

/**
 * Stable ranking that keeps usable services ahead of unauthenticated ones.
 *
 * Returns a permutation of [0, states.size()): every entry whose state is not
 * `Missing` first, in input order, then the `Missing` entries in input order.
 * When nothing is authenticated the input order is preserved unchanged, so a
 * fresh install still shows its full catalogue.
 */
[[nodiscard]] std::vector<std::size_t> order_by_credential_state(
    std::span<const ProviderCredentialState> states);

} // namespace core::llm
