#include "AuthLogin.hpp"

#include "core/config/ConfigManager.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <utility>

namespace core::auth {

std::vector<std::string> available_login_providers(std::string_view config_dir) {
    auto auth_manager = AuthenticationManager::create_with_defaults(std::string(config_dir));
    return auth_manager.available_login_providers();
}

std::optional<std::string> choose_login_provider_console(
    const std::vector<std::string>& providers,
    std::string_view title,
    std::istream& input,
    std::ostream& output) {
    if (providers.empty()) return std::nullopt;

    while (true) {
        output << "\n" << title << "\n";
        for (std::size_t i = 0; i < providers.size(); ++i) {
            output << "  " << (i + 1) << ". " << providers[i] << "\n";
        }
        output << "Select a provider [1-" << providers.size()
               << "] or press Enter to cancel: ";
        output.flush();

        std::string line;
        if (!std::getline(input, line)) return std::nullopt;
        const std::string_view choice = core::utils::str::trim_ascii_view(line);
        if (choice.empty()) return std::nullopt;

        const bool digits_only = std::ranges::all_of(choice, [](unsigned char ch) {
            return std::isdigit(ch) != 0;
        });
        if (!digits_only) {
            output << "Invalid selection. Please enter a number.\n";
            continue;
        }

        try {
            const auto selected = static_cast<std::size_t>(
                std::stoul(std::string(choice)));
            if (selected >= 1 && selected <= providers.size()) {
                return providers[selected - 1];
            }
        } catch (...) {
        }

        output << "Invalid selection. Choose a number between 1 and "
               << providers.size() << ".\n";
    }
}

AuthLoginOutcome login_and_persist(std::string_view provider,
                                   std::string_view config_dir) {
    auto auth_manager = AuthenticationManager::create_with_defaults(std::string(config_dir));

    AuthLoginOutcome outcome;
    outcome.result = auth_manager.login(provider);

    const std::string persist_provider = outcome.result.login_provider.empty()
        ? std::string(provider)
        : outcome.result.login_provider;

    std::string persist_error;
    if (core::config::ConfigManager::get_instance().persist_login_profile(
            persist_provider, &persist_error)) {
        outcome.profile_persisted = true;
        outcome.selected_provider =
            core::config::ConfigManager::get_instance().get_config().default_provider;
    } else {
        outcome.profile_error = std::move(persist_error);
    }

    return outcome;
}

} // namespace core::auth
