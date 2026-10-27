#pragma once

/**
 * @file SystemKeyring.hpp
 * @brief Read-only access to the operating system's secret store.
 *
 * Installed apps commonly keep their sign-in in the OS keyring rather than in a
 * file (go-keyring, Rust's `keyring` crate, ...). Both libraries address an
 * item by a (service, account) pair, which every backend below understands:
 *   - macOS: the login keychain, through `/usr/bin/security`;
 *   - Linux and the BSDs: the Secret Service (GNOME Keyring, KWallet), through
 *     libsecret's `secret-tool`, matching the `service` and `username`
 *     attributes both libraries write;
 *   - elsewhere: no backend, reported through IKeyring::description().
 *
 * The helpers are spawned with an argument vector, never through a shell, so
 * service and account names are passed verbatim.
 */

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace core::auth {

/// The address of one keyring item.
struct KeyringEntry {
    std::string service;
    std::string account;

    friend bool operator==(const KeyringEntry&, const KeyringEntry&) = default;
};

class IKeyring {
public:
    virtual ~IKeyring() = default;

    /// Human-readable name of the store, e.g. "the macOS login keychain";
    /// empty when this platform has no keyring Filo can read.
    [[nodiscard]] virtual std::string_view description() const noexcept = 0;

    /**
     * @brief The secret stored for @p entry.
     *
     * @return the secret; nullopt when the store holds no such item; an error
     *         naming why the store could not be consulted at all (no backend,
     *         helper not installed, no session bus, ...).
     */
    [[nodiscard]] virtual std::expected<std::optional<std::string>, std::string>
    read(const KeyringEntry& entry) const = 0;
};

/// The keyring of the platform Filo runs on.
[[nodiscard]] std::shared_ptr<const IKeyring> system_keyring();

/// "service=…, account=…", for messages.
[[nodiscard]] std::string describe_keyring_entry(const KeyringEntry& entry);

} // namespace core::auth
