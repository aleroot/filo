#pragma once

#include "ICredentialSource.hpp"
#include <memory>

namespace core::auth {

/**
 * @brief Stateless credential source for static API keys.
 *
 * The AuthInfo is pre-built at construction so get_auth() is O(1) and
 * never blocks. Use the named factory functions to create instances:
 *
 *   as_query_param("my-key")            → appends ?key=my-key to the URL
 *   as_bearer("my-key")                 → Authorization: Bearer my-key
 *   as_custom_header("my-key", "x-key") → x-key: my-key
 */
class ApiKeyCredentialSource : public ICredentialSource {
public:
    AuthInfo get_auth() override { return auth_info_; }

    static std::shared_ptr<ApiKeyCredentialSource>
    as_query_param(std::string key, std::string param_name = "key") {
        AuthInfo ai;
        ai.query_params[std::move(param_name)] = std::move(key);
        return std::shared_ptr<ApiKeyCredentialSource>(new ApiKeyCredentialSource(std::move(ai)));
    }

    static std::shared_ptr<ApiKeyCredentialSource>
    as_bearer(std::string key) {
        AuthInfo ai;
        ai.headers["Authorization"] = "Bearer " + std::move(key);
        return std::shared_ptr<ApiKeyCredentialSource>(new ApiKeyCredentialSource(std::move(ai)));
    }

    static std::shared_ptr<ApiKeyCredentialSource>
    as_custom_header(std::string key, std::string header_name) {
        AuthInfo ai;
        ai.headers[std::move(header_name)] = std::move(key);
        return std::shared_ptr<ApiKeyCredentialSource>(new ApiKeyCredentialSource(std::move(ai)));
    }

private:
    explicit ApiKeyCredentialSource(AuthInfo ai) : auth_info_(std::move(ai)) {}
    AuthInfo auth_info_;
};

} // namespace core::auth
