#pragma once

#include <string>

namespace core::auth {

struct KimiSessionIdentity {
    std::string device_id;
    std::string oauth_host;
};

/// Session identity needed by the OAuth flow, independent of storage layout.
class IKimiSessionState {
public:
    virtual ~IKimiSessionState() = default;
    [[nodiscard]] virtual KimiSessionIdentity load_identity() = 0;
};

} // namespace core::auth
