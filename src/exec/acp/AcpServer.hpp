#pragma once

#include "AcpSession.hpp"

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace exec::acp {

/// One JSON-RPC message per line. `read_line` returns nullopt at end of stream.
class AcpTransport {
public:
    virtual ~AcpTransport() = default;
    [[nodiscard]] virtual std::optional<std::string> read_line() = 0;
    virtual void write_line(std::string_view line) = 0;
};

using SessionFactory = std::function<std::expected<std::unique_ptr<Session>, OpenError>(
    const OpenRequest&)>;

struct ServeOptions {
    /// Answers session/list and session/delete; without one the list is empty.
    SessionCatalog* catalog = nullptr;
    /// Provider the terminal sign-in method logs in to. Empty offers none.
    std::string login_provider;
};

/// Speaks ACP until the transport closes. `open` builds each session.
int serve(AcpTransport& transport, const SessionFactory& open, ServeOptions options = {});

/// `filo --acp`. Stdout is reserved for the protocol; logs go to stderr.
int run(const HostOptions& options);

} // namespace exec::acp
