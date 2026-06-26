#pragma once

#include "AuthenticationManager.hpp"

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

[[nodiscard]] AuthLoginOutcome login_and_persist(
    std::string_view provider,
    std::string_view config_dir);

} // namespace core::auth
