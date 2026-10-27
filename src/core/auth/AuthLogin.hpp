#pragma once

#include "AuthenticationManager.hpp"

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

struct AuthLoginOutcome {
    LoginResult result;
    bool profile_persisted = false;
    std::string selected_provider;
    std::string profile_error;
};

[[nodiscard]] std::vector<std::string> available_login_providers(
    std::string_view config_dir);

[[nodiscard]] std::optional<std::string> choose_login_provider_console(
    const std::vector<std::string>& providers,
    std::string_view title,
    std::istream& input,
    std::ostream& output);

/**
 * @brief Authenticate @p provider and make it the default login profile.
 *
 * @param import_source  When set, adopt the credential the provider's installed
 *                       app holds (`--from`, see AppCredentialImport.hpp)
 *                       instead of running the interactive login.
 */
[[nodiscard]] AuthLoginOutcome login_and_persist(
    std::string_view provider,
    std::string_view config_dir,
    std::optional<std::string_view> import_source = std::nullopt);

/// Describe an app import for the console: origin, remaining lifetime, warnings.
void print_app_import_report(const AppImportReport& report,
                             std::int64_t now_unix_seconds,
                             std::ostream& output);

} // namespace core::auth
