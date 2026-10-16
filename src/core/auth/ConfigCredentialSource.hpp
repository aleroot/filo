#pragma once

#include "ICredentialSource.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>

namespace core::config { class ConfigSource; }

namespace core::auth {

/// Decorates an existing credential strategy with configuration-source
/// selection. FileTokenStore and provider-specific OAuth flows remain unaware
/// of layering; each uses one real directory and its normal refresh lock.
class ConfigCredentialSource final : public ICredentialSource {
public:
    using Factory = std::function<std::shared_ptr<ICredentialSource>(const std::filesystem::path&)>;
    ConfigCredentialSource(std::shared_ptr<const core::config::ConfigSource> source,
                           std::string credential_id, Factory factory);

    AuthInfo get_auth() override;
    CredentialAvailability availability() override;
    bool refresh_on_auth_failure() override;
    BillingKind billing_kind() const noexcept override { return billing_; }

private:
    ICredentialSource& current_source(); // caller holds mutex_
    std::shared_ptr<const core::config::ConfigSource> source_;
    std::string credential_id_;
    Factory factory_;
    std::filesystem::path current_directory_;
    std::shared_ptr<ICredentialSource> current_;
    BillingKind billing_;
    std::mutex mutex_;
};

} // namespace core::auth
