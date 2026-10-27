#include "AppCredentialImport.hpp"

#include "OAuthTokenManager.hpp"

#include <stdexcept>
#include <utility>

namespace core::auth {

AppImportReport import_app_credential(const AppCredentialImport& import,
                                      std::string_view source,
                                      std::string store_key,
                                      std::shared_ptr<ITokenStore> store) {
    if (!import.reader || !import.flow) {
        throw std::logic_error("import_app_credential needs both a reader and a flow");
    }
    const std::string app_name(import.reader->app_name());
    const AppCredential credential = import.reader->read(source);

    OAuthTokenManager manager(std::move(store_key), import.flow, std::move(store),
                              /*allow_interactive_login=*/false);
    OAuthToken stored;
    try {
        stored = manager.adopt(credential.token);
    } catch (const std::exception& error) {
        std::string reason = error.what();
        while (reason.ends_with('.')) reason.pop_back();
        throw std::runtime_error(
            "The " + app_name + " credential from " + credential.origin +
            " cannot be imported: " + reason + ". Open the " + app_name +
            " app once so it renews its sign-in, or drop --from and sign in through "
            "a browser instead.");
    }

    return AppImportReport{
        .app_name = app_name,
        .origin = credential.origin,
        .email = stored.email,
        .renewed = stored.access_token != credential.token.access_token,
        .expires_at = stored.expires_at,
        .warnings = import.warnings,
    };
}

} // namespace core::auth
