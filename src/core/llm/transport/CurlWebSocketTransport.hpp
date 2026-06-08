#pragma once

#include <cpr/cpr.h>

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace core::llm::transport {

struct WebSocketStreamResult {
    enum class Status {
        Completed,
        ConnectFailed,
        UpgradeRequired,
        StreamFailed,
    };

    Status status = Status::Completed;
    long http_status = 0;
    std::string message;
    cpr::Header response_headers;
    bool request_sent = false;
    bool connection_reused = false;

    [[nodiscard]] bool completed() const noexcept {
        return status == Status::Completed;
    }

};

class CurlWebSocketTransport {
public:
    using MessageCallback = std::function<bool(std::string_view)>;

    explicit CurlWebSocketTransport(std::chrono::milliseconds idle_timeout = std::chrono::seconds(120));
    ~CurlWebSocketTransport();

    [[nodiscard]] static bool runtime_supports_url(std::string_view url) noexcept;

    [[nodiscard]] WebSocketStreamResult stream_text(
        std::string_view url,
        std::string_view connection_key,
        const cpr::Header& headers,
        std::string_view request_payload,
        const MessageCallback& on_message);

    void reset();

private:
    struct Connection;

    [[nodiscard]] WebSocketStreamResult ensure_connected(
        std::string_view url,
        std::string_view connection_key,
        const cpr::Header& headers);

    std::chrono::milliseconds idle_timeout_;
    std::mutex mutex_;
    std::unique_ptr<Connection> connection_;
};

} // namespace core::llm::transport
