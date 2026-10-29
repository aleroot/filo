#include "CurlWebSocketTransport.hpp"
#include "HttpHeaderUtils.hpp"
#include "core/net/EgressScope.hpp"
#include "core/net/NetworkTraffic.hpp"
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
#include "AppleWebSocketTransport.hpp"
#endif

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

namespace core::llm::transport {

namespace {

struct CurlGlobal {
    CurlGlobal() { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobal() { curl_global_cleanup(); }
};

void ensure_curl_global() {
    static CurlGlobal global;
}

[[nodiscard]] std::string_view websocket_scheme(std::string_view url) noexcept {
    if (url.starts_with("wss://")) return "wss";
    if (url.starts_with("ws://")) return "ws";
    return {};
}

[[nodiscard]] bool runtime_supports_websocket_protocol(std::string_view url) {
    const std::string_view required_scheme = websocket_scheme(url);
    if (required_scheme.empty()) return false;

    const auto* version = curl_version_info(CURLVERSION_NOW);
    if (!version || !version->protocols) return false;

    for (const char* const* protocol = version->protocols; *protocol; ++protocol) {
        if (std::string_view(*protocol) == required_scheme) {
            return true;
        }
    }
    return false;
}

struct CurlDeleter {
    void operator()(CURL* curl) const noexcept {
        if (curl) curl_easy_cleanup(curl);
    }
};

using CurlPtr = std::unique_ptr<CURL, CurlDeleter>;

struct SlistDeleter {
    void operator()(curl_slist* list) const noexcept {
        if (list) curl_slist_free_all(list);
    }
};

using SlistPtr = std::unique_ptr<curl_slist, SlistDeleter>;

[[nodiscard]] SlistPtr build_header_list(const cpr::Header& headers) {
    curl_slist* raw = nullptr;
    for (const auto& [name, value] : headers) {
        std::string line = name + ": " + value;
        raw = curl_slist_append(raw, line.c_str());
    }
    return SlistPtr(raw);
}

[[nodiscard]] WebSocketStreamResult busy() {
    return WebSocketStreamResult{
        .status = WebSocketStreamResult::Status::Busy,
        .message = "websocket transport is already streaming on this thread",
    };
}

[[nodiscard]] WebSocketStreamResult cancelled_before_send() {
    return WebSocketStreamResult{
        .status = WebSocketStreamResult::Status::StreamFailed,
        .message = "request cancelled by user",
    };
}

[[nodiscard]] WebSocketStreamResult failed(
    WebSocketStreamResult::Status status,
    CURL* curl,
    CURLcode code,
    std::string message,
    cpr::Header response_headers = {},
    bool request_sent = false,
    bool connection_reused = false) {
    long http_status = 0;
    if (curl) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_status);
    }
    if (message.empty() && code != CURLE_OK) {
        message = curl_easy_strerror(code);
    }
    return WebSocketStreamResult{
        .status = status,
        .http_status = http_status,
        .message = std::move(message),
        .response_headers = std::move(response_headers),
        .request_sent = request_sent,
        .connection_reused = connection_reused,
    };
}

[[nodiscard]] WebSocketStreamResult::Status connect_failure_status(long http_status) {
    if (http_status == 426) {
        return WebSocketStreamResult::Status::UpgradeRequired;
    }
    return WebSocketStreamResult::Status::ConnectFailed;
}

struct HeaderCapture {
    cpr::Header& headers;
    uint64_t bytes_received = 0;
};

std::size_t header_callback(char* buffer,
                            std::size_t size,
                            std::size_t nitems,
                            void* userdata) {
    const std::size_t bytes = size * nitems;
    auto* capture = static_cast<HeaderCapture*>(userdata);
    capture->bytes_received += static_cast<uint64_t>(bytes);
    std::string_view line(buffer, bytes);
    if (auto parsed = parse_header_line(line); parsed.has_value()) {
        capture->headers[std::move(parsed->first)] = std::move(parsed->second);
    }
    return bytes;
}

// Bounds the upgrade handshake, which runs without the transport lock and so
// has to notice cancellation and reset() on its own. libcurl calls this
// roughly once a second, or sooner, for the whole of curl_easy_perform.
struct HandshakeWatch {
    const std::atomic_bool* cancel_requested;
    const std::atomic<std::uint64_t>* generation;
    std::uint64_t expected_generation;
    std::chrono::steady_clock::time_point deadline;

    [[nodiscard]] bool cancelled() const noexcept {
        return cancel_requested != nullptr && cancel_requested->load(std::memory_order_acquire);
    }
    [[nodiscard]] bool abandoned() const noexcept {
        return generation->load(std::memory_order_acquire) != expected_generation;
    }
    [[nodiscard]] bool expired() const noexcept {
        return std::chrono::steady_clock::now() >= deadline;
    }
};

int handshake_progress(void* userdata, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    const auto& watch = *static_cast<const HandshakeWatch*>(userdata);
    return watch.cancelled() || watch.abandoned() || watch.expired() ? 1 : 0;
}

constexpr auto kHandshakeTimeout = std::chrono::seconds(15);

#if !defined(FILO_WEBSOCKET_NATIVE_APPLE)
struct SendResult {
    CURLcode code = CURLE_OK;
    bool sent_any = false;
    uint64_t bytes_sent = 0;
};

[[nodiscard]] SendResult send_all(CURL* curl,
                                  std::string_view payload,
                                  const std::atomic_bool* cancel_requested,
                                  std::chrono::milliseconds idle) {
    bool sent_any = false;
    uint64_t bytes_sent = 0;
    auto last_activity = std::chrono::steady_clock::now();
    while (!payload.empty()) {
        if (cancel_requested != nullptr
            && cancel_requested->load(std::memory_order_acquire)) {
            return SendResult{CURLE_ABORTED_BY_CALLBACK, sent_any, bytes_sent};
        }
        std::size_t sent = 0;
        const CURLcode code = curl_ws_send(
            curl,
            payload.data(),
            payload.size(),
            &sent,
            0,
            CURLWS_TEXT);
        const bool sent_bytes = sent > 0;
        sent_any = sent_any || sent_bytes;
        if (sent_bytes) {
            bytes_sent += static_cast<uint64_t>(std::min(sent, payload.size()));
            last_activity = std::chrono::steady_clock::now();
        }
        if (code == CURLE_AGAIN) {
            if (std::chrono::steady_clock::now() - last_activity >= idle) {
                return SendResult{CURLE_OPERATION_TIMEDOUT, sent_any, bytes_sent};
            }
            if (sent_bytes) {
                payload.remove_prefix(std::min(sent, payload.size()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (code != CURLE_OK) {
            return SendResult{code, sent_any, bytes_sent};
        }
        if (sent_bytes) {
            payload.remove_prefix(std::min(sent, payload.size()));
        }
        if (!sent_bytes && !payload.empty()) {
            return SendResult{CURLE_SEND_ERROR, sent_any, bytes_sent};
        }
    }
    return SendResult{CURLE_OK, sent_any, bytes_sent};
}

#endif

} // namespace

struct CurlWebSocketTransport::Connection {
    CurlPtr curl;
    SlistPtr header_list;
    cpr::Header response_headers;
    std::string url;
    std::string key;
    long http_status = 0;
    std::chrono::steady_clock::time_point opened_at = std::chrono::steady_clock::now();

    [[nodiscard]] bool reusable_for(std::string_view connection_key) const {
        return key == connection_key
            && std::chrono::steady_clock::now() - opened_at < std::chrono::minutes(55);
    }
};

// Holds the single stream slot for the lifetime of one stream_text() call and
// releases it on every exit path, including a consumer callback that throws.
class CurlWebSocketTransport::StreamClaim {
public:
    explicit StreamClaim(CurlWebSocketTransport& transport) noexcept : transport_(transport) {}
    StreamClaim(const StreamClaim&) = delete;
    StreamClaim& operator=(const StreamClaim&) = delete;
    ~StreamClaim() {
        {
            std::lock_guard lock(transport_.mutex_);
            transport_.streaming_thread_.reset();
        }
        transport_.stream_released_.notify_all();
    }

private:
    CurlWebSocketTransport& transport_;
};

CurlWebSocketTransport::CurlWebSocketTransport(std::chrono::milliseconds idle_timeout)
    : idle_timeout_(idle_timeout)
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
    , native_(std::make_unique<AppleWebSocketTransport>())
#endif
{}

CurlWebSocketTransport::~CurlWebSocketTransport() = default;

bool CurlWebSocketTransport::runtime_supports_url(std::string_view url) noexcept {
    // A WebSocket connection cannot be bound to a network policy decision, so
    // while one has to be, the caller uses HTTPS, which can.
    if (core::net::binding_required()) return false;
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
    return !websocket_scheme(url).empty();
#else
    return runtime_supports_websocket_protocol(url);
#endif
}

void CurlWebSocketTransport::reset() {
    std::shared_ptr<Connection> dropped;
    {
        std::lock_guard lock(mutex_);
        // Bumped first: a stream in flight watches the generation, not
        // connection_, and keeps its own handle alive until it has unwound.
        generation_.fetch_add(1, std::memory_order_acq_rel);
        dropped = std::move(connection_);
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
        // Lock order is always this mutex, then the native transport's.
        native_->reset();
#endif
    }
    // The handle is closed here, outside the lock, unless a stream still holds
    // it, in which case that stream closes it once it notices the reset.
}

bool CurlWebSocketTransport::matches_connection(std::string_view key) {
    std::lock_guard lock(mutex_);
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
    return native_->matches_connection(key);
#else
    return connection_ && connection_->reusable_for(key);
#endif
}

std::optional<WebSocketStreamResult> CurlWebSocketTransport::claim_stream(
    const std::atomic_bool* cancel_requested, std::uint64_t& generation) {
    const auto self = std::this_thread::get_id();
    std::unique_lock lock(mutex_);
    while (streaming_thread_.has_value()) {
        // Waiting for ourselves would never end.
        if (*streaming_thread_ == self) return busy();
        if (cancel_requested != nullptr && cancel_requested->load(std::memory_order_acquire)) {
            return cancelled_before_send();
        }
        // Polled, because cancellation is an atomic nobody notifies us about.
        stream_released_.wait_for(lock, std::chrono::milliseconds(50));
    }
    streaming_thread_ = self;
    generation = generation_.load(std::memory_order_acquire);
    return std::nullopt;
}

void CurlWebSocketTransport::discard(const std::shared_ptr<Connection>& connection) {
    std::shared_ptr<Connection> dropped;
    std::lock_guard lock(mutex_);
    if (connection_ == connection) dropped = std::move(connection_);
}

WebSocketStreamResult CurlWebSocketTransport::stream_text(
    std::string_view url,
    std::string_view connection_key,
    const cpr::Header& headers,
    std::string_view request_payload,
    const MessageCallback& on_message,
    const std::atomic_bool* cancel_requested,
    std::optional<std::chrono::milliseconds> idle_timeout) {
    const std::chrono::milliseconds idle = idle_timeout.value_or(idle_timeout_);
    const auto is_cancelled = [&] {
        return cancel_requested != nullptr && cancel_requested->load(std::memory_order_acquire);
    };
    if (is_cancelled()) return cancelled_before_send();

    std::uint64_t generation = 0;
    if (auto refused = claim_stream(cancel_requested, generation)) return std::move(*refused);
    const StreamClaim claim(*this);
    const auto abandoned = [&] {
        return generation_.load(std::memory_order_acquire) != generation;
    };
    // No transport lock is held while the consumer runs, so it may reset the
    // transport or start another request; the latter is refused as Busy.
    const auto deliver = [&](std::string_view message) {
        try {
            return on_message(message);
        } catch (...) {
            // A consumer may fail before draining the generation. Its unread
            // events must never be delivered as part of a later request.
            reset();
            throw;
        }
    };

#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
    // The native transport synchronizes itself and holds its task for the
    // whole call; a reset() cancels that task, which ends the stream.
    auto result = native_->stream_text(url, connection_key, headers, request_payload,
                                       deliver, cancel_requested, idle);
    if (abandoned()) {
        result.status = WebSocketStreamResult::Status::Aborted;
        result.message = "websocket connection was reset";
    }
    return result;
#else
    std::shared_ptr<Connection> conn;
    bool connection_reused = false;
    {
        std::shared_ptr<Connection> stale;
        std::lock_guard lock(mutex_);
        if (connection_ && connection_->reusable_for(connection_key)) {
            conn = connection_;
            connection_reused = true;
        } else {
            stale = std::move(connection_);
        }
    }
    if (!conn) {
        // The handshake runs unlocked: reset() and matches_connection() must
        // never wait on a server that is slow, or silent, to upgrade.
        auto opened = connect(url, connection_key, headers, cancel_requested, generation_, generation);
        if (!opened) return std::move(opened.error());
        std::lock_guard lock(mutex_);
        if (abandoned()) {
            return WebSocketStreamResult{
                .status = WebSocketStreamResult::Status::Aborted,
                .message = "websocket connection was reset",
            };
        }
        connection_ = conn = std::move(*opened);
    }

    const auto fail = [&](WebSocketStreamResult::Status status,
                          CURLcode code,
                          std::string message,
                          bool request_sent) {
        auto result = failed(status, conn->curl.get(), code, std::move(message),
                             conn->response_headers, request_sent, connection_reused);
        if (abandoned()) {
            result.status = WebSocketStreamResult::Status::Aborted;
        } else {
            // A broken connection is never reused. Only ours is dropped: a
            // reset() may already have let another stream open a new one.
            discard(conn);
        }
        return result;
    };

    SendResult send_result = send_all(conn->curl.get(), request_payload, cancel_requested, idle);
    core::net::NetworkTrafficStats::get_instance().record_sent(send_result.bytes_sent);
    if (send_result.code != CURLE_OK) {
        return fail(WebSocketStreamResult::Status::StreamFailed, send_result.code, {},
                    send_result.sent_any);
    }

    std::string message;
    auto last_activity = std::chrono::steady_clock::now();
    while (true) {
        if (abandoned()) {
            // reset() ran, from a consumer callback or from another thread. The
            // owner has moved on: report it rather than wait for a response
            // nobody will use.
            return fail(WebSocketStreamResult::Status::Aborted, CURLE_ABORTED_BY_CALLBACK,
                        "websocket connection was reset", true);
        }
        if (is_cancelled()) {
            return fail(WebSocketStreamResult::Status::StreamFailed, CURLE_ABORTED_BY_CALLBACK,
                        "request cancelled by user", true);
        }

        char buffer[64 * 1024];
        std::size_t received = 0;
        const curl_ws_frame* meta = nullptr;
        CURLcode code = curl_ws_recv(
            conn->curl.get(), buffer, sizeof(buffer), &received, &meta);
        if (received > 0) {
            core::net::NetworkTrafficStats::get_instance().record_received(received);
        }

        if (code == CURLE_AGAIN) {
            if (std::chrono::steady_clock::now() - last_activity > idle) {
                return fail(WebSocketStreamResult::Status::StreamFailed, CURLE_OPERATION_TIMEDOUT,
                            "idle timeout waiting for websocket event", true);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (code != CURLE_OK) {
            return fail(WebSocketStreamResult::Status::StreamFailed, code, {}, true);
        }
        if (!meta) {
            continue;
        }

        last_activity = std::chrono::steady_clock::now();

        if (meta->flags & CURLWS_CLOSE) {
            return fail(WebSocketStreamResult::Status::StreamFailed, CURLE_OK,
                        "websocket closed before response.completed", true);
        }
        if (!(meta->flags & (CURLWS_TEXT | CURLWS_CONT))) {
            continue;
        }

        if (received > kMaxWebSocketMessageBytes - message.size()) {
            return fail(WebSocketStreamResult::Status::StreamFailed, CURLE_RECV_ERROR,
                        "websocket message exceeds size limit", true);
        }
        message.append(buffer, received);
        if (meta->bytesleft != 0 || (meta->flags & CURLWS_CONT)) {
            continue;
        }

        const bool done = deliver(message);
        message.clear();
        if (done) {
            return WebSocketStreamResult{
                // A reset() that raced the final message still wins: the
                // response belongs to state the owner has thrown away.
                .status = abandoned() ? WebSocketStreamResult::Status::Aborted
                                      : WebSocketStreamResult::Status::Completed,
                .http_status = conn->http_status,
                .response_headers = conn->response_headers,
                .request_sent = true,
                .connection_reused = connection_reused,
            };
        }
    }
#endif
}

std::expected<std::shared_ptr<CurlWebSocketTransport::Connection>, WebSocketStreamResult>
CurlWebSocketTransport::connect(
    std::string_view url,
    std::string_view connection_key,
    const cpr::Header& headers,
    const std::atomic_bool* cancel_requested,
    const std::atomic<std::uint64_t>& generation,
    std::uint64_t expected_generation) {
    ensure_curl_global();

    if (!runtime_supports_websocket_protocol(url)) {
        return std::unexpected(WebSocketStreamResult{
            .status = WebSocketStreamResult::Status::ConnectFailed,
            .message = "libcurl was built without websocket protocol support for this URL",
        });
    }

    CurlPtr curl(curl_easy_init());
    if (!curl) {
        return std::unexpected(WebSocketStreamResult{
            .status = WebSocketStreamResult::Status::ConnectFailed,
            .message = "failed to initialize libcurl",
        });
    }

    cpr::Header response_headers;
    HeaderCapture header_capture{response_headers};
    const HandshakeWatch watch{
        .cancel_requested = cancel_requested,
        .generation = &generation,
        .expected_generation = expected_generation,
        .deadline = std::chrono::steady_clock::now() + kHandshakeTimeout,
    };
    const uint64_t request_header_bytes =
        core::net::estimated_http_header_bytes(headers);
    SlistPtr header_list = build_header_list(headers);
    std::string url_string(url);

    curl_easy_setopt(curl.get(), CURLOPT_URL, url_string.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, header_list.get());
    curl_easy_setopt(curl.get(), CURLOPT_CONNECT_ONLY, 2L);
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(std::chrono::milliseconds(kHandshakeTimeout).count()));
    curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &header_capture);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, handshake_progress);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &watch);
    curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);

    CURLcode code = curl_easy_perform(curl.get());
    // Both callbacks point into this frame; the handle outlives it.
    curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, nullptr);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, nullptr);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, nullptr);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, nullptr);

    long http_status = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_status);
    core::net::NetworkTrafficStats::get_instance().record({
        .bytes_sent = (code == CURLE_OK || http_status != 0 || header_capture.bytes_received > 0)
            ? request_header_bytes
            : 0,
        .bytes_received = header_capture.bytes_received,
    });
    if (code == CURLE_ABORTED_BY_CALLBACK) {
        if (watch.abandoned()) {
            return std::unexpected(WebSocketStreamResult{
                .status = WebSocketStreamResult::Status::Aborted,
                .message = "websocket connection was reset",
            });
        }
        return std::unexpected(failed(
            WebSocketStreamResult::Status::ConnectFailed,
            curl.get(),
            code,
            watch.cancelled() ? "request cancelled by user" : "websocket handshake timed out",
            std::move(response_headers)));
    }
    if (code != CURLE_OK) {
        return std::unexpected(failed(
            connect_failure_status(http_status),
            curl.get(),
            code,
            {},
            std::move(response_headers)));
    }

    if (http_status == 426) {
        return std::unexpected(failed(
            WebSocketStreamResult::Status::UpgradeRequired,
            curl.get(),
            CURLE_OK,
            "websocket upgrade required",
            std::move(response_headers)));
    }
    if (http_status != 101) {
        return std::unexpected(failed(
            WebSocketStreamResult::Status::ConnectFailed,
            curl.get(),
            CURLE_OK,
            "websocket upgrade failed",
            std::move(response_headers)));
    }

    return std::make_shared<Connection>(
        std::move(curl),
        std::move(header_list),
        std::move(response_headers),
        std::move(url_string),
        std::string(connection_key),
        http_status);
}

} // namespace core::llm::transport
