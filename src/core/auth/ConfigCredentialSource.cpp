#include "ConfigCredentialSource.hpp"
#include "core/config/ConfigSource.hpp"

#include <stdexcept>

namespace core::auth {

ConfigCredentialSource::ConfigCredentialSource(
    std::shared_ptr<const core::config::ConfigSource> source, std::string credential_id, Factory factory)
    : source_(std::move(source)), credential_id_(std::move(credential_id)), factory_(std::move(factory)) {
    billing_ = current_source().billing_kind();
}

ICredentialSource& ConfigCredentialSource::current_source() {
    const auto directory = source_->credential_directory(credential_id_);
    if (!current_ || directory != current_directory_) {
        auto next = factory_(directory);
        if (!next) throw std::runtime_error("credential strategy did not create a source");
        current_ = std::move(next);
        current_directory_ = directory;
    }
    return *current_;
}

AuthInfo ConfigCredentialSource::get_auth() {
    std::lock_guard lock(mutex_);
    return current_source().get_auth();
}

CredentialAvailability ConfigCredentialSource::availability() {
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return CredentialAvailability::Unknown;
    return current_->availability();
}

bool ConfigCredentialSource::refresh_on_auth_failure() {
    std::lock_guard lock(mutex_);
    return current_source().refresh_on_auth_failure();
}

} // namespace core::auth
