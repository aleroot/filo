#include "CurlWebSocketTransport.hpp"
#include "HttpHeaderUtils.hpp"

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

std::size_t header_callback(char* buffer,
                            std::size_t size,
                            std::size_t nitems,
                            void* userdata) {
    const std::size_t bytes = size * nitems;
    auto* headers = static_cast<cpr::Header*>(userdata);
    std::string_view line(buffer, bytes);
    if (auto parsed = parse_header_line(line); parsed.has_value()) {
        (*headers)[std::move(parsed->first)] = std::move(parsed->second);
    }
    return bytes;
}

struct SendResult {
    CURLcode code = CURLE_OK;
    bool sent_any = false;
};

[[nodiscard]] SendResult send_all(CURL* curl, std::string_view payload) {
    bool sent_any = false;
    while (!payload.empty()) {
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
        if (code == CURLE_AGAIN) {
            if (sent_bytes) {
                payload.remove_prefix(std::min(sent, payload.size()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (code != CURLE_OK) {
            return SendResult{code, sent_any};
        }
        if (sent_bytes) {
            payload.remove_prefix(std::min(sent, payload.size()));
        }
        if (!sent_bytes && !payload.empty()) {
            return SendResult{CURLE_SEND_ERROR, sent_any};
        }
    }
    return SendResult{};
}

} // namespace

struct CurlWebSocketTransport::Connection {
    CurlPtr curl;
    SlistPtr header_list;
    cpr::Header response_headers;
    std::string url;
    std::string key;
    long http_status = 0;
};

CurlWebSocketTransport::CurlWebSocketTransport(std::chrono::milliseconds idle_timeout)
    : idle_timeout_(idle_timeout) {}

CurlWebSocketTransport::~CurlWebSocketTransport() = default;

bool CurlWebSocketTransport::runtime_supports_url(std::string_view url) noexcept {
    return runtime_supports_websocket_protocol(url);
}

void CurlWebSocketTransport::reset() {
    std::lock_guard lock(mutex_);
    connection_.reset();
}

WebSocketStreamResult CurlWebSocketTransport::stream_text(
    std::string_view url,
    std::string_view connection_key,
    const cpr::Header& headers,
    std::string_view request_payload,
    const MessageCallback& on_message) {
    std::lock_guard lock(mutex_);

    auto connect_result = ensure_connected(url, connection_key, headers);
    if (!connect_result.completed()) {
        return connect_result;
    }

    const bool connection_reused = connect_result.connection_reused;
    SendResult send_result = send_all(connection_->curl.get(), request_payload);
    if (send_result.code != CURLE_OK) {
        auto response_headers = connection_->response_headers;
        auto* curl = connection_->curl.get();
        auto result = failed(
            WebSocketStreamResult::Status::StreamFailed,
            curl,
            send_result.code,
            {},
            std::move(response_headers),
            send_result.sent_any,
            connection_reused);
        connection_.reset();
        return result;
    }

    std::string message;
    auto last_activity = std::chrono::steady_clock::now();
    while (true) {
        char buffer[64 * 1024];
        std::size_t received = 0;
        const curl_ws_frame* meta = nullptr;
        CURLcode code = curl_ws_recv(
            connection_->curl.get(), buffer, sizeof(buffer), &received, &meta);

        if (code == CURLE_AGAIN) {
            if (std::chrono::steady_clock::now() - last_activity > idle_timeout_) {
                auto response_headers = connection_->response_headers;
                auto* curl = connection_->curl.get();
                auto result = failed(
                    WebSocketStreamResult::Status::StreamFailed,
                    curl,
                    CURLE_OPERATION_TIMEDOUT,
                    "idle timeout waiting for websocket event",
                    std::move(response_headers),
                    /*request_sent=*/true,
                    connection_reused);
                connection_.reset();
                return result;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (code != CURLE_OK) {
            auto response_headers = connection_->response_headers;
            auto* curl = connection_->curl.get();
            auto result = failed(
                WebSocketStreamResult::Status::StreamFailed,
                curl,
                code,
                {},
                std::move(response_headers),
                /*request_sent=*/true,
                connection_reused);
            connection_.reset();
            return result;
        }
        if (!meta) {
            continue;
        }

        last_activity = std::chrono::steady_clock::now();

        if (meta->flags & CURLWS_CLOSE) {
            auto response_headers = connection_->response_headers;
            auto* curl = connection_->curl.get();
            auto result = failed(
                WebSocketStreamResult::Status::StreamFailed,
                curl,
                CURLE_OK,
                "websocket closed before response.completed",
                std::move(response_headers),
                /*request_sent=*/true,
                connection_reused);
            connection_.reset();
            return result;
        }
        if (!(meta->flags & (CURLWS_TEXT | CURLWS_CONT))) {
            continue;
        }

        message.append(buffer, received);
        if (meta->bytesleft != 0) {
            continue;
        }

        const bool done = on_message(message);
        message.clear();
        if (done) {
            return WebSocketStreamResult{
                .status = WebSocketStreamResult::Status::Completed,
                .http_status = connection_->http_status,
                .response_headers = connection_->response_headers,
                .request_sent = true,
                .connection_reused = connection_reused,
            };
        }
    }
}

WebSocketStreamResult CurlWebSocketTransport::ensure_connected(
    std::string_view url,
    std::string_view connection_key,
    const cpr::Header& headers) {
    if (connection_ && connection_->key == connection_key) {
        return WebSocketStreamResult{
            .status = WebSocketStreamResult::Status::Completed,
            .http_status = connection_->http_status,
            .response_headers = connection_->response_headers,
            .connection_reused = true,
        };
    }

    connection_.reset();
    ensure_curl_global();

    if (!runtime_supports_websocket_protocol(url)) {
        return WebSocketStreamResult{
            .status = WebSocketStreamResult::Status::ConnectFailed,
            .message = "libcurl was built without websocket protocol support for this URL",
        };
    }

    CurlPtr curl(curl_easy_init());
    if (!curl) {
        return WebSocketStreamResult{
            .status = WebSocketStreamResult::Status::ConnectFailed,
            .message = "failed to initialize libcurl",
        };
    }

    cpr::Header response_headers;
    SlistPtr header_list = build_header_list(headers);
    std::string url_string(url);

    curl_easy_setopt(curl.get(), CURLOPT_URL, url_string.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, header_list.get());
    curl_easy_setopt(curl.get(), CURLOPT_CONNECT_ONLY, 2L);
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, 15000L);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERFUNCTION, header_callback);
    curl_easy_setopt(curl.get(), CURLOPT_HEADERDATA, &response_headers);

    CURLcode code = curl_easy_perform(curl.get());
    if (code != CURLE_OK) {
        long http_status = 0;
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_status);
        return failed(
            connect_failure_status(http_status),
            curl.get(),
            code,
            {},
            std::move(response_headers));
    }

    long http_status = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_status);
    if (http_status == 426) {
        return failed(
            WebSocketStreamResult::Status::UpgradeRequired,
            curl.get(),
            CURLE_OK,
            "websocket upgrade required",
            std::move(response_headers));
    }
    if (http_status != 101) {
        return failed(
            WebSocketStreamResult::Status::ConnectFailed,
            curl.get(),
            CURLE_OK,
            "websocket upgrade failed",
            std::move(response_headers));
    }

    connection_ = std::make_unique<Connection>(
        std::move(curl),
        std::move(header_list),
        std::move(response_headers),
        std::move(url_string),
        std::string(connection_key),
        http_status);

    return WebSocketStreamResult{
        .status = WebSocketStreamResult::Status::Completed,
        .http_status = http_status,
        .response_headers = connection_->response_headers,
        .connection_reused = false,
    };
}

} // namespace core::llm::transport
