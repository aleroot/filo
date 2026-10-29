#pragma once

#include <cpr/cpr.h>

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

// macOS streams through NSURLSession because Apple's libcurl has no ws/wss
// support. Defining FILO_WEBSOCKET_FORCE_CURL builds the libcurl path there
// too, against a WebSocket-capable libcurl, so it can be exercised on a Mac.
#if defined(__APPLE__) && !defined(FILO_WEBSOCKET_FORCE_CURL)
#define FILO_WEBSOCKET_NATIVE_APPLE 1
#endif

namespace core::llm::transport {

inline constexpr std::size_t kMaxWebSocketMessageBytes = 16 * 1024 * 1024;

#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
class AppleWebSocketTransport;
#endif

struct WebSocketStreamResult {
    enum class Status {
        Completed,
        ConnectFailed,
        UpgradeRequired,
        StreamFailed,
        // The caller is already streaming on this transport, further up its
        // own stack (it re-entered from a message callback). Nothing was sent
        // and the outer stream is untouched; serve this request another way.
        Busy,
        // reset() abandoned the connection while this stream was using it. The
        // owner has discarded the conversation state the response belongs to,
        // so it must not be committed, and it must not be retried.
        Aborted,
    };

    Status status = Status::Completed;
    long http_status = 0;
    std::string message = {};
    cpr::Header response_headers = {};
    bool request_sent = false;
    bool connection_reused = false;

    [[nodiscard]] bool completed() const noexcept {
        return status == Status::Completed;
    }

};

// One WebSocket connection, used by one stream at a time.
//
// Locking contract, which callers rely on to stay deadlock-free:
//  - No transport lock is held while `on_message` runs, or during the network
//    handshake, so reset() and matches_connection() never wait on the network.
//  - A stream_text() call from another thread waits (cancellably) for the
//    active stream to finish, exactly like a mutex would.
//  - A stream_text() call from inside the active stream's own callback cannot
//    wait for itself, so it returns Status::Busy immediately.
class CurlWebSocketTransport {
public:
    using MessageCallback = std::function<bool(std::string_view)>;

    explicit CurlWebSocketTransport(std::chrono::milliseconds idle_timeout = std::chrono::seconds(120));
    ~CurlWebSocketTransport();

    [[nodiscard]] static bool runtime_supports_url(std::string_view url) noexcept;

    /// Sends one request and delivers every server message to `on_message`,
    /// which returns true once the response is complete.
    [[nodiscard]] WebSocketStreamResult stream_text(
        std::string_view url,
        std::string_view connection_key,
        const cpr::Header& headers,
        std::string_view request_payload,
        const MessageCallback& on_message,
        const std::atomic_bool* cancel_requested = nullptr,
        std::optional<std::chrono::milliseconds> idle_timeout = std::nullopt);

    /// Abandons the connection. A stream in flight is not waited for: it stops
    /// at its next step and reports Status::Aborted. Safe from any thread,
    /// including from inside a message callback.
    void reset();
    [[nodiscard]] bool matches_connection(std::string_view connection_key);

private:
    struct Connection;
    class StreamClaim;

    // Claims the single stream slot, waiting for another thread's stream if
    // necessary. Returns the refusal to report if the slot cannot be claimed.
    [[nodiscard]] std::optional<WebSocketStreamResult> claim_stream(
        const std::atomic_bool* cancel_requested, std::uint64_t& generation);
    // Drops `connection` if it is still the current one. Used by a stream that
    // failed, so it never touches a connection someone else has since opened.
    void discard(const std::shared_ptr<Connection>& connection);

    [[nodiscard]] static std::expected<std::shared_ptr<Connection>, WebSocketStreamResult> connect(
        std::string_view url,
        std::string_view connection_key,
        const cpr::Header& headers,
        const std::atomic_bool* cancel_requested,
        const std::atomic<std::uint64_t>& generation,
        std::uint64_t expected_generation);

    std::chrono::milliseconds idle_timeout_;
    // Plain, and only ever held for bookkeeping: never across on_message, the
    // handshake, or a receive. A recursive mutex here once let a re-entrant
    // reset() destroy the connection underneath the loop still using it.
    std::mutex mutex_;
    std::condition_variable stream_released_;
    // Shared so the stream keeps its handle alive after reset() drops ours.
    std::shared_ptr<Connection> connection_;
    // Thread running the active stream; it tells a re-entrant call (Busy) from
    // a concurrent one (waits).
    std::optional<std::thread::id> streaming_thread_;
    // Bumped by reset(). A stream compares it with the value it started under.
    std::atomic<std::uint64_t> generation_{0};
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
    std::unique_ptr<AppleWebSocketTransport> native_;
#endif
};

} // namespace core::llm::transport
