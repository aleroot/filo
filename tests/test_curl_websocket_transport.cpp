#include <catch2/catch_test_macros.hpp>

#include "core/agent/HistoryCompactor.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/protocols/ApiProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/transport/CurlWebSocketTransport.hpp"
#include "core/utils/Base64.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using core::llm::transport::CurlWebSocketTransport;
using core::llm::transport::WebSocketStreamResult;

namespace {

#if !defined(_WIN32)

void close_socket(int fd) noexcept {
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
        ::close(fd);
    }
}

[[nodiscard]] std::array<std::uint8_t, 20> sha1(std::string_view input) {
    std::uint32_t h0 = 0x67452301u;
    std::uint32_t h1 = 0xEFCDAB89u;
    std::uint32_t h2 = 0x98BADCFEu;
    std::uint32_t h3 = 0x10325476u;
    std::uint32_t h4 = 0xC3D2E1F0u;

    std::vector<std::uint8_t> data(input.begin(), input.end());
    const std::uint64_t bit_len = static_cast<std::uint64_t>(data.size()) * 8u;
    data.push_back(0x80u);
    while (data.size() % 64u != 56u) {
        data.push_back(0u);
    }
    for (int i = 7; i >= 0; --i) {
        data.push_back(static_cast<std::uint8_t>((bit_len >> (i * 8)) & 0xFFu));
    }

    for (std::size_t offset = 0; offset < data.size(); offset += 64u) {
        std::uint32_t w[80]{};
        for (int i = 0; i < 16; ++i) {
            const std::size_t at = offset + static_cast<std::size_t>(i) * 4u;
            w[i] = (std::uint32_t(data[at]) << 24u)
                 | (std::uint32_t(data[at + 1]) << 16u)
                 | (std::uint32_t(data[at + 2]) << 8u)
                 | std::uint32_t(data[at + 3]);
        }
        for (int i = 16; i < 80; ++i) {
            w[i] = std::rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }

        std::uint32_t a = h0;
        std::uint32_t b = h1;
        std::uint32_t c = h2;
        std::uint32_t d = h3;
        std::uint32_t e = h4;

        for (int i = 0; i < 80; ++i) {
            std::uint32_t f = 0;
            std::uint32_t k = 0;
            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999u;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1u;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDCu;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6u;
            }

            const std::uint32_t temp = std::rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = std::rotl(b, 30);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<std::uint8_t, 20> digest{};
    const std::uint32_t words[] = {h0, h1, h2, h3, h4};
    for (std::size_t i = 0; i < 5; ++i) {
        digest[i * 4] = static_cast<std::uint8_t>((words[i] >> 24u) & 0xFFu);
        digest[i * 4 + 1] = static_cast<std::uint8_t>((words[i] >> 16u) & 0xFFu);
        digest[i * 4 + 2] = static_cast<std::uint8_t>((words[i] >> 8u) & 0xFFu);
        digest[i * 4 + 3] = static_cast<std::uint8_t>(words[i] & 0xFFu);
    }
    return digest;
}

[[nodiscard]] std::string websocket_accept(std::string_view key) {
    std::string challenge(key);
    challenge += "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    const auto digest = sha1(challenge);
    return core::utils::Base64::encode(
        std::span<const std::uint8_t>(digest.data(), digest.size()));
}

[[nodiscard]] bool write_all(int fd, std::string_view data) {
    while (!data.empty()) {
        const auto sent = ::send(fd, data.data(), data.size(), 0);
        if (sent <= 0) return false;
        data.remove_prefix(static_cast<std::size_t>(sent));
    }
    return true;
}

[[nodiscard]] bool read_exact(int fd, std::span<std::uint8_t> out) {
    while (!out.empty()) {
        const auto received = ::recv(fd, out.data(), out.size(), 0);
        if (received <= 0) return false;
        out = out.subspan(static_cast<std::size_t>(received));
    }
    return true;
}

[[nodiscard]] std::string read_headers(int fd) {
    std::string headers;
    char ch = '\0';
    while (headers.find("\r\n\r\n") == std::string::npos) {
        const auto received = ::recv(fd, &ch, 1, 0);
        if (received <= 0) return {};
        headers.push_back(ch);
    }
    return headers;
}

[[nodiscard]] std::optional<std::string> header_value(
    std::string_view headers,
    std::string_view name) {
    std::string lower_headers(headers);
    std::string lower_name(name);
    std::ranges::transform(lower_headers, lower_headers.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::ranges::transform(lower_name, lower_name.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    std::size_t line_start = 0;
    while (line_start < lower_headers.size()) {
        const std::size_t line_end = lower_headers.find("\r\n", line_start);
        const std::size_t count = line_end == std::string::npos
            ? lower_headers.size() - line_start
            : line_end - line_start;
        const std::string_view lower_line(lower_headers.data() + line_start, count);
        if (lower_line.starts_with(lower_name) && lower_line.size() > lower_name.size()
            && lower_line[lower_name.size()] == ':') {
            const std::size_t value_start = line_start + lower_name.size() + 1;
            const std::size_t value_end = line_start + count;
            std::string value(headers.substr(value_start, value_end - value_start));
            const auto first = value.find_first_not_of(" \t");
            const auto last = value.find_last_not_of(" \t");
            if (first == std::string::npos) return std::string{};
            return value.substr(first, last - first + 1);
        }
        if (line_end == std::string::npos) break;
        line_start = line_end + 2;
    }
    return std::nullopt;
}

[[nodiscard]] std::string text_frame(std::string_view payload, std::uint8_t flags = 0x81) {
    std::string frame;
    frame.push_back(static_cast<char>(flags));
    if (payload.size() <= 125) {
        frame.push_back(static_cast<char>(payload.size()));
    } else if (payload.size() <= 0xFFFFu) {
        frame.push_back(static_cast<char>(126));
        frame.push_back(static_cast<char>((payload.size() >> 8u) & 0xFFu));
        frame.push_back(static_cast<char>(payload.size() & 0xFFu));
    } else {
        frame.push_back(static_cast<char>(127));
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<char>((payload.size() >> shift) & 0xFFu));
        }
    }
    frame.append(payload);
    return frame;
}

[[nodiscard]] std::optional<std::string> read_client_text_frame(int fd) {
    std::array<std::uint8_t, 2> header{};
    if (!read_exact(fd, header)) return std::nullopt;

    const std::uint8_t opcode = header[0] & 0x0Fu;
    if (opcode == 0x8u) return std::nullopt;
    if (opcode != 0x1u && opcode != 0x0u) return std::nullopt;

    const bool masked = (header[1] & 0x80u) != 0;
    std::uint64_t len = header[1] & 0x7Fu;
    if (len == 126u) {
        std::array<std::uint8_t, 2> extended{};
        if (!read_exact(fd, extended)) return std::nullopt;
        len = (std::uint64_t(extended[0]) << 8u) | std::uint64_t(extended[1]);
    } else if (len == 127u) {
        std::array<std::uint8_t, 8> extended{};
        if (!read_exact(fd, extended)) return std::nullopt;
        len = 0;
        for (const auto byte : extended) {
            len = (len << 8u) | std::uint64_t(byte);
        }
    }

    std::array<std::uint8_t, 4> mask{};
    if (masked && !read_exact(fd, mask)) return std::nullopt;

    std::string payload(static_cast<std::size_t>(len), '\0');
    if (!payload.empty()) {
        auto* bytes = reinterpret_cast<std::uint8_t*>(payload.data());
        if (!read_exact(fd, std::span<std::uint8_t>(bytes, payload.size()))) {
            return std::nullopt;
        }
        if (masked) {
            for (std::size_t i = 0; i < payload.size(); ++i) {
                bytes[i] ^= mask[i % 4u];
            }
        }
    }

    return payload;
}

class LoopbackWebSocketServer {
public:
    explicit LoopbackWebSocketServer(std::vector<std::vector<std::string>> responses,
                                     std::vector<std::size_t> reconnect_after = {},
                                     bool fragment_messages = false,
                                     std::chrono::milliseconds response_delay = {})
        : responses_(std::move(responses)), reconnect_after_(std::move(reconnect_after)),
          fragment_messages_(fragment_messages), response_delay_(response_delay) {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) throw std::runtime_error("socket failed");

        int reuse = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close_socket(listen_fd_);
            throw std::runtime_error("bind failed");
        }
        if (::listen(listen_fd_, 1) != 0) {
            close_socket(listen_fd_);
            throw std::runtime_error("listen failed");
        }

        socklen_t len = sizeof(addr);
        if (::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
            close_socket(listen_fd_);
            throw std::runtime_error("getsockname failed");
        }
        port_ = ntohs(addr.sin_port);
        thread_ = std::thread([this] { run(); });
    }

    ~LoopbackWebSocketServer() {
        close_socket(client_fd_.exchange(-1));
        close_socket(listen_fd_);
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    [[nodiscard]] int port() const noexcept { return port_; }

    [[nodiscard]] std::vector<std::string> received_payloads() const {
        std::lock_guard lock(mutex_);
        return received_payloads_;
    }

private:
    int accept_client() {
        const int client = ::accept(listen_fd_, nullptr, nullptr);
        if (client < 0) return -1;
        client_fd_.store(client);

        const std::string headers = read_headers(client);
        const auto key = header_value(headers, "Sec-WebSocket-Key");
        if (!key.has_value()) return -1;

        const std::string response =
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: " + websocket_accept(*key) + "\r\n"
            "x-test-websocket: loopback\r\n"
            "\r\n";
        if (!write_all(client, response)) return -1;
        return client;
    }

    void run() {
        int client = -1;
        std::size_t group_index = 0;
        for (const auto& response_group : responses_) {
            if (client < 0) client = accept_client();
            if (client < 0) return;
            auto payload = read_client_text_frame(client);
            if (!payload.has_value()) return;
            {
                std::lock_guard lock(mutex_);
                received_payloads_.push_back(std::move(*payload));
            }
            std::this_thread::sleep_for(response_delay_);
            for (const auto& message : response_group) {
                if (fragment_messages_ && message.size() > 1) {
                    const auto split = message.size() / 2;
                    if (!write_all(client, text_frame(std::string_view(message).substr(0, split), 0x01))
                        || !write_all(client, text_frame(std::string_view(message).substr(split), 0x80))) return;
                } else if (!write_all(client, text_frame(message))) return;
            }
            if (std::ranges::find(reconnect_after_, group_index++) != reconnect_after_.end()) {
                close_socket(client_fd_.exchange(-1));
                client = -1;
            }
        }
    }

    int listen_fd_ = -1;
    std::atomic<int> client_fd_{-1};
    int port_ = 0;
    std::vector<std::vector<std::string>> responses_;
    std::vector<std::size_t> reconnect_after_;
    bool fragment_messages_ = false;
    std::chrono::milliseconds response_delay_;
    mutable std::mutex mutex_;
    std::vector<std::string> received_payloads_;
    std::thread thread_;
};

#endif

class UnsupportedRuntimeWebSocketProtocol final
    : public core::llm::protocols::ApiProtocolBase {
public:
    explicit UnsupportedRuntimeWebSocketProtocol(std::shared_ptr<std::atomic<int>> ws_serializes)
        : ws_serializes_(std::move(ws_serializes)) {}

    [[nodiscard]] std::string serialize(
        const core::llm::ChatRequest&) const override {
        return "{}";
    }

    [[nodiscard]] cpr::Header build_headers(
        const core::auth::AuthInfo&) const override {
        return {{"Content-Type", "application/json"}};
    }

    [[nodiscard]] std::string build_url(
        std::string_view base_url,
        std::string_view) const override {
        return std::string(base_url) + "/responses";
    }

    [[nodiscard]] std::string_view event_delimiter() const noexcept override {
        return "\n\n";
    }

    [[nodiscard]] core::llm::protocols::ParseResult parse_event(
        std::string_view) override {
        return {};
    }

    [[nodiscard]] std::string_view name() const noexcept override {
        return "unsupported_runtime_websocket";
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<UnsupportedRuntimeWebSocketProtocol>(*this);
    }

    [[nodiscard]] bool supports_websocket_transport() const noexcept override {
        return true;
    }

    [[nodiscard]] std::string build_websocket_url(
        std::string_view,
        std::string_view) const override {
        return "ftp://127.0.0.1/responses";
    }

    [[nodiscard]] std::string serialize_websocket_request(
        const core::llm::ChatRequest&) const override {
        ++(*ws_serializes_);
        return R"({"type":"response.create"})";
    }

private:
    std::shared_ptr<std::atomic<int>> ws_serializes_;
};

} // namespace

TEST_CASE("HttpLLMProvider does not serialize websocket payloads for unsupported runtime URLs",
          "[transport][websocket][fallback]") {
    auto ws_serializes = std::make_shared<std::atomic<int>>(0);
    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        "http://127.0.0.1:1",
        nullptr,
        "gpt-5",
        std::make_unique<UnsupportedRuntimeWebSocketProtocol>(ws_serializes),
        core::config::ApiType::OpenAI,
        "test");

    core::llm::ChatRequest request;
    request.model = "gpt-5";
    request.messages.push_back(core::llm::Message{
        .role = "user",
        .content = "hello",
    });

    std::promise<void> done;
    auto future = done.get_future();
    std::atomic<bool> completed = false;
    provider->stream_response(request, [&](const core::llm::StreamChunk& chunk) {
        if ((chunk.is_final || chunk.is_error) && !completed.exchange(true)) {
            done.set_value();
        }
    });

    REQUIRE(future.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    REQUIRE(ws_serializes->load() == 0);
}

TEST_CASE("CurlWebSocketTransport streams websocket frames and reuses the connection",
          "[transport][websocket]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    LoopbackWebSocketServer server({
        {
            R"({"type":"response.output_text.delta","delta":"hello"})",
            R"({"type":"response.completed","response":{"id":"resp_1"}})",
        },
        {
            R"({"type":"response.output_text.delta","delta":"again"})",
            R"({"type":"response.completed","response":{"id":"resp_2"}})",
        },
    });

    CurlWebSocketTransport transport(std::chrono::seconds(2));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    std::vector<std::string> first_events;
    const auto first = transport.stream_text(
        url,
        "same-connection",
        {},
        R"({"type":"response.create","sequence":1})",
        [&](std::string_view message) {
            first_events.emplace_back(message);
            return message.find("response.completed") != std::string_view::npos;
        });

    REQUIRE(first.completed());
    REQUIRE(first.request_sent);
    REQUIRE_FALSE(first.connection_reused);
    REQUIRE(first.response_headers.at("x-test-websocket") == "loopback");
    REQUIRE(first_events.size() == 2);

    std::vector<std::string> second_events;
    const auto second = transport.stream_text(
        url,
        "same-connection",
        {},
        R"({"type":"response.create","sequence":2})",
        [&](std::string_view message) {
            second_events.emplace_back(message);
            return message.find("response.completed") != std::string_view::npos;
        });

    REQUIRE(second.completed());
    REQUIRE(second.request_sent);
    REQUIRE(second.connection_reused);
    REQUIRE(second_events.size() == 2);

    const auto received = server.received_payloads();
    REQUIRE(received.size() == 2);
    REQUIRE(received[0].find(R"("sequence":1)") != std::string::npos);
    REQUIRE(received[1].find(R"("sequence":2)") != std::string::npos);
#endif
}

TEST_CASE("CurlWebSocketTransport marks sent requests on websocket stream failure",
          "[transport][websocket][fallback]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    LoopbackWebSocketServer server({{}});

    CurlWebSocketTransport transport(std::chrono::milliseconds(200));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    const auto result = transport.stream_text(
        url,
        "failure-connection",
        {},
        R"({"type":"response.create","sequence":1})",
        [](std::string_view) {
            return false;
        });

    REQUIRE_FALSE(result.completed());
    REQUIRE(result.status == WebSocketStreamResult::Status::StreamFailed);
    REQUIRE(result.request_sent);
#endif
}

TEST_CASE("Codex WebSocket retries discard provisional output and rewarm a new connection",
          "[transport][websocket][codex][regression][retry]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("WebSocket transport unavailable");
    }
    bool disconnected = false;
    SECTION("retryable server error") {}
    SECTION("connection closes after provisional output") { disconnected = true; }
    std::vector<std::string> failed{
        R"({"type":"response.output_text.delta","delta":"discard this"})",
        R"({"type":"response.output_item.done","item":{"type":"reasoning","encrypted_content":"discard opaque"}})",
        R"({"type":"response.output_item.done","item":{"type":"function_call","call_id":"discard_call","name":"exec","arguments":"{}"}})",
    };
    if (!disconnected) failed.push_back(
        R"({"type":"response.failed","response":{"error":{"code":"server_error","message":"temporary"}}})");
    LoopbackWebSocketServer server({
        {R"({"type":"response.completed","response":{"id":"warm_old"}})"},
        failed,
        {R"({"type":"response.completed","response":{"id":"warm_new"}})"},
        {R"({"type":"response.output_text.delta","delta":"complete"})",
         R"({"type":"response.completed","response":{"id":"resp_ok","usage":{"input_tokens":2,"output_tokens":1}}})"},
    }, {1});
    core::llm::HttpStreamTransportOptions options;
    options.retries.max_retries = 1;
    options.retries.retry_only_before_output = false;
    options.retries.initial_backoff = std::chrono::milliseconds(1);
    options.retries.minimum_delay = std::chrono::milliseconds(0);
    options.retries.jitter_ratio = 0;
    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(server.port()), nullptr, "gpt-5.6-sol",
        std::make_unique<core::llm::protocols::CodexResponsesProtocol>(),
        core::config::ApiType::Unknown, "", nullptr, nullptr, "", options);
    core::llm::ChatRequest request;
    request.model = "gpt-5.6-sol";
    request.session_id = "ws-retry";
    request.transport_turn_id = "ws-turn";
    request.messages = {{.role = "user", .content = "inspect"}};
    std::string text;
    int terminals = 0, resets = 0;
    bool error = false;
    std::vector<core::llm::ToolCall> tools;
    std::vector<core::llm::ContinuationItem> continuation;
    provider->stream_response(request, [&](const auto& chunk) {
        if (chunk.reset_attempt) { ++resets; text.clear(); tools.clear(); continuation.clear(); return; }
        text += chunk.content;
        tools.insert(tools.end(), chunk.tools.begin(), chunk.tools.end());
        continuation.insert(continuation.end(), chunk.continuation_items.begin(), chunk.continuation_items.end());
        terminals += chunk.is_final;
        error |= chunk.is_error;
    });
    CHECK_FALSE(error);
    CHECK(text == "complete");
    CHECK(terminals == 1);
    CHECK(resets == 1);
    CHECK(tools.empty());
    CHECK(continuation.empty());
    CHECK(provider->get_last_usage().completion_tokens == 1);
    const auto payloads = server.received_payloads();
    REQUIRE(payloads.size() == 4);
    CHECK(payloads[0].find("generate\":false") != std::string::npos);
    CHECK(payloads[2].find("previous_response_id") == std::string::npos);
    CHECK(payloads[2].find("inspect") != std::string::npos);
    CHECK(payloads[3].find("warm_new") != std::string::npos);
    CHECK(payloads[3].find("warm_old") == std::string::npos);
#endif
}

TEST_CASE("WebSocket cancellation interrupts a waiting receive and invalidates the socket",
          "[transport][websocket][regression][cancel]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) SKIP("WebSocket transport unavailable");
    LoopbackWebSocketServer server({{}});
    CurlWebSocketTransport transport;
    std::atomic_bool cancelled{false};
    std::jthread cancel([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (server.received_payloads().empty() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        cancelled.store(true);
    });
    const auto started = std::chrono::steady_clock::now();
    const auto result = transport.stream_text(
        "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses",
        "cancel", {}, R"({"type":"response.create"})", [](auto) { return false; }, &cancelled);
    CHECK_FALSE(result.completed());
    CHECK(result.request_sent);
    CHECK_FALSE(transport.matches_connection("cancel"));
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(3));
#endif
}

TEST_CASE("WebSocket transport assembles fragmented messages before parsing JSON",
          "[transport][websocket][regression][fragmentation]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) SKIP("WebSocket transport unavailable");
    const std::string event = R"({"type":"response.completed","response":{"id":"fragmented"}})";
    LoopbackWebSocketServer server({{event}}, {}, true);
    CurlWebSocketTransport transport(std::chrono::seconds(2));
    std::vector<std::string> messages;
    const auto result = transport.stream_text(
        "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses", "fragmented", {},
        R"({"type":"response.create"})", [&](std::string_view message) {
            messages.emplace_back(message);
            return true;
        });
    REQUIRE(result.completed());
    REQUIRE(messages.size() == 1);
    CHECK(messages[0] == event);
#endif
}

TEST_CASE("WebSocket transport discards a connection when its consumer throws",
          "[transport][websocket][regression][exception]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) SKIP("WebSocket transport unavailable");
    LoopbackWebSocketServer server({{R"({"type":"response.output_text.delta","delta":"partial"})"}});
    CurlWebSocketTransport transport(std::chrono::seconds(2));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    CHECK_THROWS_AS(transport.stream_text(url, "throwing-consumer", {}, "{}",
        [](std::string_view) -> bool { throw std::runtime_error("consumer failed"); }), std::runtime_error);
    CHECK_FALSE(transport.matches_connection("throwing-consumer"));
#endif
}

// These pin NSURLSession behaviour, so they follow the transport actually built
// rather than the platform: FILO_WEBSOCKET_FORCE_CURL selects libcurl on a Mac.
#if defined(FILO_WEBSOCKET_NATIVE_APPLE)
TEST_CASE("Native WebSocket handshake timeout does not shorten the generation idle budget",
          "[transport][websocket][regression][timeout]") {
    // Codex reasoning can remain silent longer than the 15-second handshake
    // timeout. Foundation must leave established receives under our idle budget.
    LoopbackWebSocketServer server({{R"({"type":"response.completed"})"}}, {}, false,
                                  std::chrono::seconds(16));
    CurlWebSocketTransport transport(std::chrono::seconds(25));
    const auto result = transport.stream_text(
        "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses", "long-reasoning", {},
        "{}", [](std::string_view) { return true; });
    CHECK(result.completed());
}

TEST_CASE("Native WebSocket transport rejects invalid UTF-8 without an Objective-C exception",
          "[transport][websocket][regression][validation]") {
    LoopbackWebSocketServer server({{}});
    CurlWebSocketTransport transport(std::chrono::seconds(2));
    const auto result = transport.stream_text(
        "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses", "invalid-utf8", {},
        std::string(1, static_cast<char>(0xff)), [](auto) { return false; });
    CHECK_FALSE(result.completed());
    CHECK_FALSE(result.request_sent);
    CHECK_FALSE(transport.matches_connection("invalid-utf8"));
}
#endif

TEST_CASE("WebSocket terminal chunk callback can reset conversation state and chain requests without deadlocking",
          "[transport][websocket][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("WebSocket transport unavailable");
    }

    LoopbackWebSocketServer server({
        {R"({"type":"response.completed","response":{"id":"warm_1"}})"},
        {R"({"type":"response.output_text.delta","delta":"summary checkpoint"})",
         R"({"type":"response.completed","response":{"id":"resp_1"}})"},
        {R"({"type":"response.completed","response":{"id":"warm_2"}})"},
        {R"({"type":"response.output_text.delta","delta":"follow-up turn"})",
         R"({"type":"response.completed","response":{"id":"resp_2"}})"},
    }, {1});

    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(server.port()),
        nullptr,
        "gpt-5.6-sol",
        std::make_unique<core::llm::protocols::CodexResponsesProtocol>(),
        core::config::ApiType::Unknown,
        "codex");

    core::llm::ChatRequest first_request;
    first_request.model = "gpt-5.6-sol";
    first_request.session_id = "ws-deadlock-regression";
    first_request.transport_turn_id = "turn-1";
    first_request.messages = {{.role = "user", .content = "compact conversation"}};

    std::string first_text;
    std::string second_text;
    std::promise<void> completed;
    auto completed_future = completed.get_future();

    std::jthread worker([&] {
        provider->stream_response(first_request, [&](const core::llm::StreamChunk& chunk) {
            first_text += chunk.content;
            if (!chunk.is_final) return;

            // Reproduces compaction / turn continuation callbacks that reset
            // conversation state and start a follow-up request from inside the
            // terminal chunk callback.
            provider->reset_conversation_state();

            core::llm::ChatRequest second_request;
            second_request.model = "gpt-5.6-sol";
            second_request.session_id = "ws-deadlock-regression";
            second_request.transport_turn_id = "turn-2";
            second_request.messages = {{.role = "user", .content = "continue"}};

            provider->stream_response(second_request, [&](const core::llm::StreamChunk& inner) {
                second_text += inner.content;
                if (inner.is_final) {
                    completed.set_value();
                }
            });
        });
    });

    REQUIRE(completed_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(first_text == "summary checkpoint");
    CHECK(second_text == "follow-up turn");
#endif
}

TEST_CASE("HistoryCompactor over WebSocket forks the provider and resets conversation state without deadlocking",
          "[transport][websocket][compaction][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("WebSocket transport unavailable");
    }

    LoopbackWebSocketServer server({
        {R"({"type":"response.output_text.delta","delta":"Compacted summary checkpoint."})",
         R"({"type":"response.completed","response":{"id":"resp_compact"}})"},
    });

    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(server.port()),
        nullptr,
        "gpt-5.6-sol",
        std::make_unique<core::llm::protocols::CodexResponsesProtocol>(),
        core::config::ApiType::Unknown,
        "codex");

    std::mutex history_mutex;
    std::string applied_summary;
    std::promise<void> finished;
    auto finished_future = finished.get_future();

    core::agent::HistoryCompactor compactor;
    compactor.compact_async(
        core::agent::HistoryCompactionRequest{
            .history = {{.role = "user", .content = "Summarize this session."}},
            .provider = provider,
            .model = "gpt-5.6-sol",
            .reason = core::agent::HistoryCompactionReason::Auto,
        },
        core::agent::HistoryCompactionCallbacks{
            .on_summary = [&](std::string summary) {
                {
                    std::lock_guard lock(history_mutex);
                    applied_summary = std::move(summary);
                }
                provider->reset_conversation_state();
                return core::agent::HistoryCompactionApplyStatus::Applied;
            },
            .on_finished = [&]() {
                finished.set_value();
            },
        });

    REQUIRE(finished_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    // A compaction request carries no transport_turn_id, so
    // CodexResponsesProtocol sends no prewarm frame ahead of it: exactly one
    // request goes out, and it must be the one the response below answers.
    CHECK(server.received_payloads().size() == 1);
    std::lock_guard lock(history_mutex);
    CHECK(applied_summary == "Compacted summary checkpoint.");
#endif
}

namespace {

#if !defined(_WIN32)
// Runs one stream_text() call off-thread and hands the result back, so a
// regression that re-introduces a deadlock fails the assertion instead of
// hanging the whole suite. Everything the worker touches is shared, so it stays
// valid even if the worker has to be abandoned.
class StreamProbe {
public:
    void start(CurlWebSocketTransport& transport,
               const std::string& url,
               std::string_view connection_key,
               CurlWebSocketTransport::MessageCallback on_message,
               std::string payload = R"({"type":"response.create","sequence":1})") {
        worker_ = std::thread([state = state_, &transport, url, key = std::string(connection_key),
                               on_message = std::move(on_message), payload = std::move(payload)] {
            auto streamed = transport.stream_text(url, key, {}, payload, on_message);
            {
                std::lock_guard lock(state->mutex);
                state->result = std::move(streamed);
            }
            state->ready.notify_all();
        });
    }

    // Returns false (and abandons the thread) if the stream never finished.
    [[nodiscard]] bool wait(std::chrono::seconds budget) {
        bool ready = false;
        {
            std::unique_lock lock(state_->mutex);
            ready = state_->ready.wait_for(lock, budget, [&] { return state_->result.has_value(); });
        }
        if (ready) {
            worker_.join();
        } else {
            worker_.detach();
        }
        return ready;
    }

    [[nodiscard]] WebSocketStreamResult result() const {
        std::lock_guard lock(state_->mutex);
        return *state_->result;
    }

    ~StreamProbe() {
        if (worker_.joinable()) worker_.detach();
    }

private:
    struct State {
        std::mutex mutex;
        std::condition_variable ready;
        std::optional<WebSocketStreamResult> result;
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
    std::thread worker_;
};

// Accepts connections and never answers them: a server that is slow, or
// silent, to upgrade.
class SilentServer {
public:
    SilentServer() {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) throw std::runtime_error("socket failed");
        int reuse = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t len = sizeof(addr);
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0
            || ::listen(listen_fd_, 4) != 0
            || ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
            close_socket(listen_fd_);
            throw std::runtime_error("silent server setup failed");
        }
        port_ = ntohs(addr.sin_port);
        thread_ = std::thread([this] {
            while (true) {
                const int client = ::accept(listen_fd_, nullptr, nullptr);
                if (client < 0) return;
                std::lock_guard lock(mutex_);
                clients_.push_back(client);
            }
        });
    }
    ~SilentServer() {
        close_socket(listen_fd_);
        if (thread_.joinable()) thread_.join();
        std::lock_guard lock(mutex_);
        for (const int client : clients_) close_socket(client);
    }
    [[nodiscard]] int port() const noexcept { return port_; }

private:
    int listen_fd_ = -1;
    int port_ = 0;
    std::mutex mutex_;
    std::vector<int> clients_;
    std::thread thread_;
};

[[nodiscard]] bool wait_until(const std::function<bool()>& condition, std::chrono::seconds budget) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!condition()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}
#endif

} // namespace

TEST_CASE("WebSocket consumer may reset the transport from inside a non-final message",
          "[transport][websocket][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    LoopbackWebSocketServer server({{
        R"({"type":"response.output_text.delta","delta":"one"})",
        R"({"type":"response.output_text.delta","delta":"two"})",
        R"({"type":"response.completed","response":{"id":"resp_1"}})",
    }});

    // A generous idle budget: the stream has to end because it was reset, not
    // because it ran out of time.
    CurlWebSocketTransport transport(std::chrono::seconds(30));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    auto delivered = std::make_shared<std::atomic<int>>(0);

    StreamProbe probe;
    const auto started = std::chrono::steady_clock::now();
    probe.start(transport, url, "reset-from-callback",
                [delivered, &transport](std::string_view) {
                    delivered->fetch_add(1, std::memory_order_acq_rel);
                    // Abandon the stream from inside the consumer callback and
                    // report "not done". The transport used to hold its own
                    // mutex across this call, so reset() either self-deadlocked
                    // on it or — once the mutex was made recursive — destroyed
                    // the connection underneath the receive loop still using it.
                    transport.reset();
                    return false;
                });

    REQUIRE(probe.wait(std::chrono::seconds(20)));
    const auto elapsed = std::chrono::steady_clock::now() - started;

    CHECK(delivered->load(std::memory_order_acquire) == 1);
    CHECK(probe.result().status == WebSocketStreamResult::Status::Aborted);
    // It must have noticed the reset, not sat out the idle budget waiting for a
    // connection that reset() had already cancelled.
    CHECK(elapsed < std::chrono::seconds(10));
    CHECK(probe.result().message.find("idle timeout") == std::string::npos);
#endif
}

TEST_CASE("Resetting a WebSocket stream from another thread aborts it instead of blocking",
          "[transport][websocket][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    // One message, then the server parks reading a second client frame, so the
    // connection stays open and silent for the rest of the test.
    LoopbackWebSocketServer server({
        {R"({"type":"response.output_text.delta","delta":"only"})"},
        {},
    });

    CurlWebSocketTransport transport(std::chrono::seconds(30));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    auto signalled = std::make_shared<std::atomic<bool>>(false);
    auto first_message = std::make_shared<std::promise<void>>();

    StreamProbe probe;
    probe.start(transport, url, "concurrent-reset",
                [signalled, first_message](std::string_view) {
                    if (!signalled->exchange(true, std::memory_order_acq_rel)) {
                        first_message->set_value();
                    }
                    return false;
                });

    REQUIRE(first_message->get_future().wait_for(std::chrono::seconds(20)) == std::future_status::ready);

    // While the receive loop held the transport mutex, this blocked until the
    // stream above finished — for a quiet connection, the whole idle budget.
    // That is how a background reset froze the thread that asked for it.
    const auto reset_started = std::chrono::steady_clock::now();
    transport.reset();
    const auto reset_elapsed = std::chrono::steady_clock::now() - reset_started;

    REQUIRE(probe.wait(std::chrono::seconds(20)));
    CHECK(reset_elapsed < std::chrono::seconds(5));
    CHECK(probe.result().status == WebSocketStreamResult::Status::Aborted);
#endif
}

TEST_CASE("A WebSocket stream from another thread waits for the active one instead of breaking it",
          "[transport][websocket][regression][concurrency]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    // Each response is held back, so the second caller arrives while the first
    // is still streaming.
    LoopbackWebSocketServer server({
        {R"({"type":"response.output_text.delta","delta":"first"})",
         R"({"type":"response.completed","response":{"id":"resp_1"}})"},
        {R"({"type":"response.output_text.delta","delta":"second"})",
         R"({"type":"response.completed","response":{"id":"resp_2"}})"},
    }, {}, false, std::chrono::milliseconds(300));

    CurlWebSocketTransport transport(std::chrono::seconds(10));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    const auto until_completed = [](std::string_view message) {
        return message.find("response.completed") != std::string_view::npos;
    };

    StreamProbe first;
    first.start(transport, url, "shared", until_completed,
                R"({"type":"response.create","sequence":1})");
    REQUIRE(wait_until([&] { return server.received_payloads().size() == 1; }, std::chrono::seconds(10)));

    // A shared provider (the API gateway, MCP sampling) does this. The second
    // caller used to be refused, and the refusal reset the transport, which
    // aborted the first caller's stream with it.
    StreamProbe second;
    second.start(transport, url, "shared", until_completed,
                 R"({"type":"response.create","sequence":2})");

    REQUIRE(first.wait(std::chrono::seconds(20)));
    REQUIRE(second.wait(std::chrono::seconds(20)));
    CHECK(first.result().completed());
    CHECK(second.result().completed());
    CHECK(second.result().connection_reused);
    const auto received = server.received_payloads();
    REQUIRE(received.size() == 2);
    CHECK(received[1].find(R"("sequence":2)") != std::string::npos);
#endif
}

TEST_CASE("A WebSocket stream re-entered from its own callback is refused as Busy",
          "[transport][websocket][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    LoopbackWebSocketServer server({{
        R"({"type":"response.output_text.delta","delta":"outer"})",
        R"({"type":"response.completed","response":{"id":"resp_1"}})",
    }});

    CurlWebSocketTransport transport(std::chrono::seconds(10));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";
    auto inner = std::make_shared<std::optional<WebSocketStreamResult>>();

    StreamProbe outer;
    outer.start(transport, url, "reentrant", [&transport, url, inner](std::string_view message) {
        if (!inner->has_value()) {
            // Waiting here for the outer stream would wait for ourselves.
            *inner = transport.stream_text(url, "reentrant", {},
                                           R"({"type":"response.create","sequence":2})",
                                           [](std::string_view) { return true; });
        }
        return message.find("response.completed") != std::string_view::npos;
    });

    REQUIRE(outer.wait(std::chrono::seconds(20)));
    REQUIRE(inner->has_value());
    CHECK((*inner)->status == WebSocketStreamResult::Status::Busy);
    CHECK_FALSE((*inner)->request_sent);
    // The refusal left the outer stream alone.
    CHECK(outer.result().completed());
    CHECK(server.received_payloads().size() == 1);
#endif
}

TEST_CASE("Resetting a WebSocket transport during the handshake neither blocks nor waits for the server",
          "[transport][websocket][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("linked libcurl does not support ws://");
    }

    SilentServer server;
    CurlWebSocketTransport transport(std::chrono::seconds(30));
    const std::string url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/responses";

    StreamProbe probe;
    const auto started = std::chrono::steady_clock::now();
    probe.start(transport, url, "silent-upgrade", [](std::string_view) { return true; });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // The handshake used to run under the transport mutex, so this waited for
    // the server: forever, without a handshake deadline. It is what a UI thread
    // clearing the conversation would have been stuck behind.
    const auto reset_started = std::chrono::steady_clock::now();
    transport.reset();
    CHECK(std::chrono::steady_clock::now() - reset_started < std::chrono::seconds(1));
    CHECK_FALSE(transport.matches_connection("silent-upgrade"));

    REQUIRE(probe.wait(std::chrono::seconds(20)));
    CHECK(probe.result().status == WebSocketStreamResult::Status::Aborted);
    CHECK_FALSE(probe.result().request_sent);
    // Well inside the 15 s handshake deadline: the reset ended it.
    CHECK(std::chrono::steady_clock::now() - started < std::chrono::seconds(5));
#endif
}

TEST_CASE("Concurrent requests on one Codex provider share its WebSocket without aborting each other",
          "[transport][websocket][codex][regression][concurrency]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("WebSocket transport unavailable");
    }

    // No transport_turn_id, so no prewarm: one frame and one response per
    // request. The responses are identical because either caller may go first.
    LoopbackWebSocketServer server({
        {R"({"type":"response.output_text.delta","delta":"shared answer"})",
         R"({"type":"response.completed","response":{"id":"resp_1"}})"},
        {R"({"type":"response.output_text.delta","delta":"shared answer"})",
         R"({"type":"response.completed","response":{"id":"resp_2"}})"},
    }, {}, false, std::chrono::milliseconds(300));

    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(server.port()),
        nullptr,
        "gpt-5.6-sol",
        std::make_unique<core::llm::protocols::CodexResponsesProtocol>(),
        core::config::ApiType::Unknown,
        "codex");

    struct Outcome {
        std::string text;
        bool error = false;
        bool final = false;
    };
    const auto run = [&](Outcome& outcome, std::string session) {
        core::llm::ChatRequest request;
        request.model = "gpt-5.6-sol";
        request.session_id = std::move(session);
        request.messages = {{.role = "user", .content = "hello"}};
        provider->stream_response(request, [&](const core::llm::StreamChunk& chunk) {
            if (chunk.reset_attempt) {
                outcome.text.clear();
                return;
            }
            outcome.text += chunk.content;
            outcome.error |= chunk.is_error;
            outcome.final |= chunk.is_final;
        });
    };

    Outcome a;
    Outcome b;
    // One session, like two gateway requests on the same conversation key, so
    // both map onto the one connection the loopback server accepts.
    std::thread first([&] { run(a, "concurrent"); });
    REQUIRE(wait_until([&] { return server.received_payloads().size() == 1; }, std::chrono::seconds(10)));
    std::thread second([&] { run(b, "concurrent"); });
    first.join();
    second.join();

    CHECK(a.final);
    CHECK(b.final);
    CHECK_FALSE(a.error);
    CHECK_FALSE(b.error);
    CHECK(a.text == "shared answer");
    CHECK(b.text == "shared answer");
    // Both went over the WebSocket; neither was pushed onto a retry or HTTPS.
    CHECK(server.received_payloads().size() == 2);
#endif
}

TEST_CASE("Resetting conversation state mid-stream aborts the WebSocket response instead of retrying it",
          "[transport][websocket][codex][regression][deadlock]") {
#if defined(_WIN32)
    SKIP("loopback websocket transport test is POSIX-only");
#else
    if (!CurlWebSocketTransport::runtime_supports_url("ws://127.0.0.1/")) {
        SKIP("WebSocket transport unavailable");
    }

    // Half a response, then silence: the stream is in flight when it is reset.
    LoopbackWebSocketServer server({
        {R"({"type":"response.output_text.delta","delta":"partial"})"},
        {},
    });

    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        "http://127.0.0.1:" + std::to_string(server.port()),
        nullptr,
        "gpt-5.6-sol",
        std::make_unique<core::llm::protocols::CodexResponsesProtocol>(),
        core::config::ApiType::Unknown,
        "codex");

    core::llm::ChatRequest request;
    request.model = "gpt-5.6-sol";
    request.session_id = "reset-mid-stream";
    request.messages = {{.role = "user", .content = "hello"}};

    auto partial_seen = std::make_shared<std::promise<void>>();
    auto finished = std::make_shared<std::promise<core::llm::StreamChunk>>();
    auto partial_future = partial_seen->get_future();
    auto finished_future = finished->get_future();
    std::thread worker([provider, request, partial_seen, finished] {
        bool partial_signalled = false;
        provider->stream_response(request, [&](const core::llm::StreamChunk& chunk) {
            if (!chunk.content.empty() && !chunk.is_final && !partial_signalled) {
                partial_signalled = true;
                partial_seen->set_value();
            }
            if (chunk.is_final) finished->set_value(chunk);
        });
    });

    REQUIRE(partial_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    // What /clear or an applied compaction does while a turn is streaming.
    provider->reset_conversation_state();

    REQUIRE(finished_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    const auto terminal = finished_future.get();
    worker.join();
    CHECK(terminal.is_error);
    CHECK(terminal.content.find("reset before the response completed") != std::string::npos);
    // Resending would replay history the owner has just replaced.
    CHECK(server.received_payloads().size() == 1);
#endif
}
