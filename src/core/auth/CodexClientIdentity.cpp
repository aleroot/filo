#include "CodexClientIdentity.hpp"

namespace core::auth::codex_cli {

const std::string& originator() {
    static const std::string value(kOriginator);
    return value;
}

const std::string& client_version() {
    static const std::string value(kClientVersion);
    return value;
}

const std::string& user_agent() {
    static const std::string value = originator() + "/" + client_version();
    return value;
}

} // namespace core::auth::codex_cli
