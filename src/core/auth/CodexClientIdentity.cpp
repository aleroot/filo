#include "CodexClientIdentity.hpp"

#include "../utils/ClientPlatform.hpp"

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
    // codex-rs login/src/auth/default_client.rs get_codex_user_agent:
    // "{originator}/{version} ({os} {os_version}; {architecture}) {terminal}".
    // The backend reads it as telemetry about the client that sent the
    // request, so it names the platform the CLI runs on.
    static const std::string value = [] {
        const auto& platform = core::utils::client_platform();
        std::string agent = originator() + "/" + client_version();
        agent += " (" + platform.os + " " + platform.version + "; "
            + platform.architecture + ")";
        if (const auto terminal = core::utils::terminal_token(); !terminal.empty()) {
            agent += ' ';
            agent += terminal;
        }
        return agent;
    }();
    return value;
}

} // namespace core::auth::codex_cli
