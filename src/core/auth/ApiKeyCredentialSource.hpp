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
 *
 * Credentials for subscription plans (Coding Plans, token plans) declare
 * BillingKind::Subscription; plain pay-as-you-go keys stay Metered (the
 * default).
 */
class ApiKeyCredentialSource : public ICredentialSource {
public:
    AuthInfo get_auth() override { return auth_info_; }

    /**
     * An empty AuthInfo means no key was configured: the factory functions
     * drop the header or query parameter entirely for an empty key, so the
     * request would go out unauthenticated.
     */
    [[nodiscard]] CredentialAvailability availability() override {
        return auth_info_.headers.empty() && auth_info_.query_params.empty()
            ? CredentialAvailability::Missing
            : CredentialAvailability::Ready;
    }

    [[nodiscard]] BillingKind billing_kind() const noexcept override {
        return billing_kind_;
    }

    static std::shared_ptr<ApiKeyCredentialSource>
    as_query_param(std::string key, std::string param_name = "key") {
        AuthInfo ai;
        if (!key.empty()) {
            ai.query_params[std::move(param_name)] = std::move(key);
        }
        return std::shared_ptr<ApiKeyCredentialSource>(new ApiKeyCredentialSource(std::move(ai)));
    }

    static std::shared_ptr<ApiKeyCredentialSource>
    as_bearer(std::string key, BillingKind billing_kind = BillingKind::Metered) {
        AuthInfo ai;
        if (!key.empty()) {
            ai.headers["Authorization"] = "Bearer " + std::move(key);
        }
        return std::shared_ptr<ApiKeyCredentialSource>(
            new ApiKeyCredentialSource(std::move(ai), billing_kind));
    }

    static std::shared_ptr<ApiKeyCredentialSource>
    as_custom_header(std::string key,
                     std::string header_name,
                     BillingKind billing_kind = BillingKind::Metered) {
        AuthInfo ai;
        if (!key.empty()) {
            ai.headers[std::move(header_name)] = std::move(key);
        }
        return std::shared_ptr<ApiKeyCredentialSource>(
            new ApiKeyCredentialSource(std::move(ai), billing_kind));
    }

    static std::shared_ptr<ApiKeyCredentialSource>
    none() {
        return std::shared_ptr<ApiKeyCredentialSource>(new ApiKeyCredentialSource(AuthInfo{}));
    }

private:
    explicit ApiKeyCredentialSource(AuthInfo ai, BillingKind billing_kind = BillingKind::Metered)
        : auth_info_(std::move(ai))
        , billing_kind_(billing_kind) {}
    AuthInfo auth_info_;
    BillingKind billing_kind_ = BillingKind::Metered;
};

} // namespace core::auth
