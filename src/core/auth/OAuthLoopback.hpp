#pragma once

#include "ui/AuthUI.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace core::auth {

class OAuthLoopbackBindError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct OAuthLoopbackResult {
    std::string code;
    std::string state;
    std::string error;
    bool timed_out = false;
};

struct OAuthManualAuthInput {
    std::string code;
    std::optional<std::string> state;
};

struct OAuthLoopbackOptions {
    std::string bind_host = "127.0.0.1";
    /// Used in the redirect_uri shown to the authorization server.
    /// Empty = same as bind_host (with 0.0.0.0/:: normalized to 127.0.0.1).
    std::string redirect_host;
    int port_start = 17890;
    int port_end = 17920;
    /// A value of zero requests an ephemeral OS-assigned port.
    std::optional<int> fixed_port;
    /// Exact registered callback ports to try, in order. When non-empty this
    /// takes precedence over the inclusive port range.
    std::vector<int> candidate_ports;
    std::string callback_path = "/callback";
    std::vector<std::string> extra_paths;
    /// When set, reject mismatched callbacks before displaying success.
    std::optional<std::string> expected_state;
    std::chrono::seconds timeout{std::chrono::minutes(5)};
    std::string success_html =
        "<html><body style=\"font-family:system-ui;padding:2rem\">"
        "<h2>Login successful</h2>"
        "<p>You can close this tab and return to Filo.</p>"
        "</body></html>";
};

/**
 * Local HTTP callback server for OAuth authorization-code redirects.
 *
 * Lifecycle:
 *   1. construct  -> registers handlers and binds a free (or fixed) port
 *   2. redirect_uri() for authorize URL / DCR
 *   3. start()     -> background listen
 *   4. wait()      -> blocks until code/error/timeout
 *   5. destructor  -> stop + join
 */
class OAuthLoopbackServer {
public:
    explicit OAuthLoopbackServer(OAuthLoopbackOptions options = {});
    ~OAuthLoopbackServer();

    OAuthLoopbackServer(const OAuthLoopbackServer&) = delete;
    OAuthLoopbackServer& operator=(const OAuthLoopbackServer&) = delete;

    [[nodiscard]] int port() const noexcept;
    [[nodiscard]] const std::string& redirect_uri() const noexcept;

    void start();
    [[nodiscard]] OAuthLoopbackResult wait();
    void stop() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Parse a pasted callback URL or a bare authorization code.
 * Throws std::runtime_error when no code can be extracted.
 */
[[nodiscard]] OAuthManualAuthInput
parse_oauth_manual_auth_input(std::string_view raw_input);

/**
 * If @p result timed out or has no code, optionally prompt via @p ui for a
 * manual callback URL/code and merge into @p result.
 * Returns false when still incomplete.
 */
bool complete_oauth_loopback_with_manual_fallback(
    OAuthLoopbackResult& result,
    std::string_view expected_state,
    ui::AuthUI* ui,
    std::string_view prompt_label = "Callback URL or authorization code: ");

} // namespace core::auth
