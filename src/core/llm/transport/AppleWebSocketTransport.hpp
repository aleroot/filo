#pragma once

#include "CurlWebSocketTransport.hpp"

namespace core::llm::transport {

// Uses the system TLS trust store and WebSocket implementation on macOS,
// where Apple's libcurl does not advertise ws/wss support.
class AppleWebSocketTransport {
public:
    AppleWebSocketTransport();
    ~AppleWebSocketTransport();
    void reset();
    [[nodiscard]] bool matches_connection(std::string_view key) const;
    [[nodiscard]] WebSocketStreamResult stream_text(
        std::string_view url, std::string_view key, const cpr::Header& headers,
        std::string_view payload, const CurlWebSocketTransport::MessageCallback& callback,
        const std::atomic_bool* cancelled, std::chrono::milliseconds idle);
private:
    struct Impl;
    // Both require Impl::mutex to be held; they exist so stream_text() can set
    // a connection up and tear one down without re-entering a non-recursive
    // lock from a path that already holds it.
    void reset_locked();
    [[nodiscard]] bool matches_connection_locked(std::string_view key) const;
    std::unique_ptr<Impl> impl_;
};

} // namespace core::llm::transport
