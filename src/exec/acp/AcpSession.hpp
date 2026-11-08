#pragma once

#include "AcpProtocol.hpp"
#include "core/context/SteeringLoader.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace exec::acp {

/// What the editor lets the agent call back into while a turn is running.
/// `request` blocks until the editor answers. Implementations must be safe to
/// call from the agent's tool threads.
class ClientChannel {
public:
    virtual ~ClientChannel() = default;
    virtual void notify(std::string_view method, const core::utils::JsonValue& params) = 0;
    virtual core::utils::JsonValue request(std::string_view method,
                                           const core::utils::JsonValue& params) = 0;
    /// What the client advertised in `initialize`.
    [[nodiscard]] virtual ClientCapabilities capabilities() const { return {}; }
};

/// Thrown by Session::prompt when the provider rejected the stored sign-in and
/// the editor can run the terminal sign-in. The server answers the prompt
/// with kAuthRequiredCode so the editor offers it.
class AuthRequired : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// One ACP conversation. The server owns it and never runs two prompts at once.
class Session {
public:
    virtual ~Session() = default;

    [[nodiscard]] virtual std::string id() const = 0;
    [[nodiscard]] virtual core::utils::JsonValue opened_result() const = 0;
    virtual void announce(ClientChannel& client) = 0;
    virtual void replay(ClientChannel& client) = 0;
    /// Blocks until the turn finishes. Returns an ACP stop reason.
    [[nodiscard]] virtual std::string prompt(const core::utils::JsonValue& blocks,
                                             ClientChannel& client) = 0;
    virtual void cancel() = 0;
    /// The conversation was deleted: never write it to disk again, even from
    /// a turn still finishing.
    virtual void discard() {}
    [[nodiscard]] virtual std::optional<std::string> set_mode(std::string_view mode_id,
                                                              ClientChannel& client) = 0;
    [[nodiscard]] virtual std::optional<std::string> set_config(
        std::string_view config_id,
        const core::utils::JsonValue& value,
        ClientChannel& client) = 0;
    [[nodiscard]] virtual core::utils::JsonValue config_state() const = 0;
};

/// The saved conversations an editor can list and delete. Loading one goes
/// through the session factory.
class SessionCatalog {
public:
    virtual ~SessionCatalog() = default;
    /// `cursor` is the opaque token from a previous page; empty for the first.
    /// Fails with a message when the cursor is not one this catalog issued.
    [[nodiscard]] virtual std::expected<SessionPage, std::string> list(
        const std::optional<std::filesystem::path>& cwd,
        std::string_view cursor) const = 0;
    /// Deleting a session that does not exist succeeds.
    virtual void remove(std::string_view session_id) = 0;
};

/// Filo's own session store. Lists only conversations that have a turn.
[[nodiscard]] std::unique_ptr<SessionCatalog> make_store_catalog(
    std::filesystem::path sessions_dir);

struct OpenRequest {
    std::filesystem::path cwd;
    std::vector<std::filesystem::path> additional_directories;
    std::vector<core::config::McpServerConfig> mcp_servers;
    std::optional<std::string> resume_id;
    ClientCapabilities client;
};

/// Why `session/new` or `session/load` was refused. The server maps each kind
/// onto the JSON-RPC code the protocol assigns it.
struct OpenError {
    enum class Kind { invalid_params, not_found, internal };

    Kind kind = Kind::invalid_params;
    std::string message;

    [[nodiscard]] static OpenError invalid_params(std::string message) {
        return {Kind::invalid_params, std::move(message)};
    }
    [[nodiscard]] static OpenError not_found(std::string message) {
        return {Kind::not_found, std::move(message)};
    }
    [[nodiscard]] static OpenError internal(std::string message) {
        return {Kind::internal, std::move(message)};
    }
};

struct HostOptions {
    bool yolo = false;
    std::vector<std::string> trusted_tools;
    std::optional<std::string> startup_model;
    /// Agent effort value: "" is auto, "none" is off, otherwise a level.
    std::optional<std::string> startup_effort;
    core::context::SteeringPolicy steering{};
};

[[nodiscard]] std::expected<std::unique_ptr<Session>, OpenError>
open_session(const OpenRequest& request, const HostOptions& host);

} // namespace exec::acp
