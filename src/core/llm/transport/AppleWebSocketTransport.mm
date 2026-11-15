#include "AppleWebSocketTransport.hpp"
#include "core/net/NetworkTraffic.hpp"

#import <Foundation/Foundation.h>

#include <condition_variable>
#include <deque>
#include <mutex>

namespace {

struct SocketState {
    std::mutex mutex;
    std::condition_variable changed;
    bool opened = false;
    bool closed = false;
    long status = 0;
    cpr::Header headers;
    std::string error;
    // Frames queued for the live connection while a stream runs. The stream
    // loop sends them, so a caller never writes to the task itself.
    std::deque<std::string> pending_sends;
};

std::string utf8(NSString* value) {
    if (!value) return {};
    const char* bytes = value.UTF8String;
    return bytes ? std::string(bytes) : std::string{};
}

NSString* ns_string(std::string_view value) {
    return [[NSString alloc] initWithBytes:value.data() length:value.size()
                                 encoding:NSUTF8StringEncoding];
}

void capture_response(SocketState& state, NSURLSessionTask* task) {
    if (![task.response isKindOfClass:[NSHTTPURLResponse class]]) return;
    NSHTTPURLResponse* response = (NSHTTPURLResponse*)task.response;
    state.status = response.statusCode;
    for (NSString* key in response.allHeaderFields) {
        state.headers[utf8(key)] = utf8([response.allHeaderFields[key] description]);
    }
}

// All completion blocks own their result state. Cancellation may return before
// Foundation delivers a callback, so no block may capture stack references.
struct Operation {
    std::mutex mutex;
    std::condition_variable changed;
    bool done = false;
    std::string error;
    std::string text;
};

template<class State, class Predicate>
bool wait_for(const std::shared_ptr<State>& state, Predicate ready,
              const std::atomic_bool* cancelled, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::unique_lock lock(state->mutex);
    while (!ready(*state)) {
        if (cancelled && cancelled->load(std::memory_order_acquire)) return false;
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        state->changed.wait_until(lock, std::min(deadline, now + std::chrono::milliseconds(50)));
    }
    return true;
}

/// Waits for the message, the cancellation, or a frame queued for the
/// connection. The wait is sliced, so a queued frame is noticed promptly even
/// while the endpoint stays silent.
bool wait_for_message(const std::shared_ptr<Operation>& received,
                      const std::shared_ptr<SocketState>& state,
                      const std::atomic_bool* cancelled,
                      std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (true) {
        {
            std::unique_lock lock(received->mutex);
            if (received->done) return true;
            received->changed.wait_until(
                lock, std::chrono::steady_clock::now() + std::chrono::milliseconds(20));
        }
        if (cancelled && cancelled->load(std::memory_order_acquire)) return false;
        {
            std::lock_guard lock(state->mutex);
            if (!state->pending_sends.empty()) return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
    }
}

} // namespace

@interface FiloWebSocketDelegate : NSObject <NSURLSessionWebSocketDelegate>
- (instancetype)initWithState:(std::shared_ptr<SocketState>)state;
@end

@implementation FiloWebSocketDelegate {
    std::shared_ptr<SocketState> _state;
}
- (instancetype)initWithState:(std::shared_ptr<SocketState>)state {
    self = [super init];
    if (self) _state = std::move(state);
    return self;
}
- (void)URLSession:(NSURLSession*)session webSocketTask:(NSURLSessionWebSocketTask*)task
 didOpenWithProtocol:(NSString*)protocol {
    (void)session; (void)protocol;
    std::lock_guard lock(_state->mutex);
    capture_response(*_state, task);
    _state->opened = true;
    _state->changed.notify_all();
}
- (void)URLSession:(NSURLSession*)session webSocketTask:(NSURLSessionWebSocketTask*)task
 didCloseWithCode:(NSURLSessionWebSocketCloseCode)code reason:(NSData*)reason {
    (void)session; (void)code; (void)reason;
    std::lock_guard lock(_state->mutex);
    capture_response(*_state, task);
    _state->closed = true;
    _state->changed.notify_all();
}
- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task
 didCompleteWithError:(NSError*)error {
    (void)session;
    std::lock_guard lock(_state->mutex);
    capture_response(*_state, task);
    _state->closed = true;
    _state->error = utf8(error.localizedDescription);
    _state->changed.notify_all();
}
- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task
 willPerformHTTPRedirection:(NSHTTPURLResponse*)response newRequest:(NSURLRequest*)request
 completionHandler:(void (^)(NSURLRequest*))completionHandler {
    (void)session; (void)task; (void)response; (void)request;
    // Never forward an authenticated handshake to a redirect target.
    completionHandler(nil);
}
@end

namespace core::llm::transport {

struct AppleWebSocketTransport::Impl {
    NSURLSession* session = nil;
    NSURLSessionWebSocketTask* task = nil;
    std::shared_ptr<SocketState> state;
    std::string key;
    std::chrono::steady_clock::time_point opened_at;
    // Guards the fields above. It is never held across the consumer callback:
    // stream_text() copies what it needs into locals and then runs the whole
    // send/receive loop on those, so reset() — from the callback itself or from
    // another thread — cannot pull the task out from under an in-flight stream.
    std::mutex mutex;
};

AppleWebSocketTransport::AppleWebSocketTransport() : impl_(std::make_unique<Impl>()) {}
AppleWebSocketTransport::~AppleWebSocketTransport() { reset(); }

void AppleWebSocketTransport::reset() {
    std::lock_guard lock(impl_->mutex);
    reset_locked();
}

void AppleWebSocketTransport::reset_locked() {
    @autoreleasepool {
        [impl_->task cancelWithCloseCode:NSURLSessionWebSocketCloseCodeGoingAway reason:nil];
        [impl_->session invalidateAndCancel];
        impl_->task = nil;
        impl_->session = nil;
        impl_->state.reset();
        impl_->key.clear();
    }
}

bool AppleWebSocketTransport::matches_connection(std::string_view key) const {
    std::lock_guard lock(impl_->mutex);
    return matches_connection_locked(key);
}

void AppleWebSocketTransport::send_text(std::string_view payload) {
    if (payload.empty()) return;
    std::shared_ptr<SocketState> state;
    {
        std::lock_guard lock(impl_->mutex);
        state = impl_->state;
    }
    // No connection, no endpoint to interrupt.
    if (!state) return;
    std::lock_guard lock(state->mutex);
    state->pending_sends.emplace_back(payload);
    state->changed.notify_all();
}

bool AppleWebSocketTransport::matches_connection_locked(std::string_view key) const {
    if (!impl_->task || !impl_->state || impl_->key != key
        || std::chrono::steady_clock::now() - impl_->opened_at >= std::chrono::minutes(55)) return false;
    std::lock_guard lock(impl_->state->mutex);
    return impl_->state->opened && !impl_->state->closed;
}

WebSocketStreamResult AppleWebSocketTransport::stream_text(
    std::string_view url, std::string_view key, const cpr::Header& headers,
    std::string_view payload, const CurlWebSocketTransport::MessageCallback& callback,
    const std::atomic_bool* cancelled, std::chrono::milliseconds idle) {
    @autoreleasepool {
        const auto is_cancelled = [&] { return cancelled && cancelled->load(std::memory_order_acquire); };
        bool reused = false;
        bool request_sent = false;

        // The connection this stream owns, captured once under mutex_. A
        // concurrent reset() drops the transport's own references but ARC keeps
        // these alive, and the cancel it issues fails the pending receive — so
        // an abandoned stream ends promptly instead of idling out or messaging
        // a task that no longer exists.
        NSURLSessionWebSocketTask* task = nil;
        std::shared_ptr<SocketState> state;

        const auto fail = [&](std::string message) {
            WebSocketStreamResult result{
                .status = request_sent ? WebSocketStreamResult::Status::StreamFailed
                                       : WebSocketStreamResult::Status::ConnectFailed,
                .message = is_cancelled() ? "request cancelled by user" : std::move(message),
                .request_sent = request_sent,
                .connection_reused = reused,
            };
            if (state) {
                std::lock_guard lock(state->mutex);
                result.http_status = state->status;
                result.response_headers = state->headers;
                if (result.http_status == 426) result.status = WebSocketStreamResult::Status::UpgradeRequired;
            }
            reset();
            return result;
        };

        if (is_cancelled()) return fail("request cancelled by user");

        std::string connect_error;
        {
            std::lock_guard lock(impl_->mutex);
            reused = matches_connection_locked(key);
            if (!reused) {
                reset_locked();
                NSString* endpoint_string = ns_string(url);
                NSURL* endpoint = endpoint_string ? [NSURL URLWithString:endpoint_string] : nil;
                if (!endpoint || (![[endpoint scheme] isEqualToString:@"ws"]
                    && ![[endpoint scheme] isEqualToString:@"wss"])) {
                    connect_error = "invalid websocket URL";
                } else {
                    NSMutableURLRequest* request = [NSMutableURLRequest requestWithURL:endpoint];
                    request.timeoutInterval = 15;
                    for (const auto& [name, value] : headers) {
                        [request setValue:ns_string(value) forHTTPHeaderField:ns_string(name)];
                    }
                    NSURLSessionConfiguration* config = [NSURLSessionConfiguration ephemeralSessionConfiguration];
                    config.URLCache = nil;
                    config.HTTPCookieStorage = nil;
                    config.URLCredentialStorage = nil;
                    config.HTTPShouldSetCookies = NO;
                    impl_->state = std::make_shared<SocketState>();
                    auto delegate = [[FiloWebSocketDelegate alloc] initWithState:impl_->state];
                    impl_->session = [NSURLSession sessionWithConfiguration:config delegate:delegate delegateQueue:nil];
                    impl_->task = [impl_->session webSocketTaskWithRequest:request];
                    impl_->task.maximumMessageSize = kMaxWebSocketMessageBytes;
                    impl_->key = key;
                    impl_->opened_at = std::chrono::steady_clock::now();
                }
            }
            task = impl_->task;
            state = impl_->state;
        }
        // fail() takes mutex_, so connection errors are reported after the
        // setup scope has released it.
        if (!connect_error.empty()) return fail(std::move(connect_error));
        if (!task || !state) return fail("websocket connection is not available");

        if (!reused) {
            [task resume];
            if (!wait_for(state, [](const auto& s) { return s.opened || s.closed; },
                          cancelled, std::chrono::seconds(15))) return fail("websocket handshake timed out");
            bool opened;
            std::string error;
            {
                std::lock_guard lock(state->mutex);
                opened = state->opened;
                error = state->error;
            }
            if (!opened) return fail(error.empty() ? "websocket upgrade failed" : error);
            core::net::NetworkTrafficStats::get_instance().record_sent(
                core::net::estimated_http_header_bytes(headers));
        }

        auto sent = std::make_shared<Operation>();
        NSString* payload_string = ns_string(payload);
        if (!payload_string) return fail("websocket request is not valid UTF-8");
        auto message = [[NSURLSessionWebSocketMessage alloc] initWithString:payload_string];
        request_sent = true;
        [task sendMessage:message completionHandler:^(NSError* error) {
            std::lock_guard lock(sent->mutex);
            sent->error = utf8(error.localizedDescription);
            sent->done = true;
            sent->changed.notify_all();
        }];
        if (!wait_for(sent, [](const auto& s) { return s.done; }, cancelled, idle)) return fail("websocket send timed out");
        if (!sent->error.empty()) return fail(sent->error);
        message = nil;
        payload_string = nil;
        core::net::NetworkTrafficStats::get_instance().record_sent(payload.size());

        // Sends the frames queued while the stream runs: an interrupt for the
        // response the endpoint is serving. The loop owns the task, so it
        // writes to it itself, and it waits for the frame to be handed over
        // before the connection goes away.
        const auto send_pending_frames = [&]() {
            std::deque<std::string> frames;
            {
                std::lock_guard lock(state->mutex);
                frames.swap(state->pending_sends);
            }
            for (const auto& frame : frames) {
                NSString* text = ns_string(frame);
                if (!text) continue;
                auto frame_sent = std::make_shared<Operation>();
                auto frame_message = [[NSURLSessionWebSocketMessage alloc] initWithString:text];
                [task sendMessage:frame_message completionHandler:^(NSError* error) {
                    std::lock_guard lock(frame_sent->mutex);
                    frame_sent->error = utf8(error.localizedDescription);
                    frame_sent->done = true;
                    frame_sent->changed.notify_all();
                }];
                (void)wait_for(frame_sent, [](const auto& s) { return s.done; }, nullptr,
                               std::chrono::seconds(2));
            }
        };

        // Set while a cancelled stream waits for the endpoint's answer to the
        // interrupt it was sent.
        std::optional<std::chrono::steady_clock::time_point> drain_deadline;
        while (true) {
            if (is_cancelled() && !drain_deadline) {
                // The endpoint must hear that the response may stop before
                // the connection goes away, and its answer tells the caller
                // whether it stopped.
                drain_deadline = std::chrono::steady_clock::now()
                    + CurlWebSocketTransport::kInterruptAckWindow;
                send_pending_frames();
            }
            auto received = std::make_shared<Operation>();
            [task receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage* incoming, NSError* error) {
                std::lock_guard lock(received->mutex);
                received->error = utf8(error.localizedDescription);
                if (incoming.type == NSURLSessionWebSocketMessageTypeString && incoming.string) {
                    received->text = utf8(incoming.string);
                } else if (!error) {
                    received->error = "expected a websocket text message";
                }
                received->done = true;
                received->changed.notify_all();
            }];
            if (drain_deadline) {
                // The endpoint answers whether it stopped the response: the
                // caller wants to know, so the loop reads until the window
                // closes.
                const auto remaining = *drain_deadline - std::chrono::steady_clock::now();
                const auto window = remaining > std::chrono::milliseconds::zero()
                    ? std::chrono::duration_cast<std::chrono::milliseconds>(remaining)
                    : std::chrono::milliseconds::zero();
                if (!wait_for_message(received, state, nullptr, window)) {
                    return fail("request cancelled by user");
                }
            } else if (!wait_for_message(received, state, cancelled, idle)) {
                if (!is_cancelled()) {
                    return fail("idle timeout waiting for websocket event");
                }
                // The user asked the turn to stop while the endpoint was
                // silent: the endpoint must hear that the response may stop,
                // and the pending receive is still outstanding, so its answer
                // can still arrive.
                drain_deadline = std::chrono::steady_clock::now()
                    + CurlWebSocketTransport::kInterruptAckWindow;
                send_pending_frames();
                const auto remaining = *drain_deadline - std::chrono::steady_clock::now();
                if (remaining <= std::chrono::milliseconds::zero()
                    || !wait_for_message(received, state, nullptr,
                                         std::chrono::duration_cast<std::chrono::milliseconds>(remaining))) {
                    return fail("request cancelled by user");
                }
            }
            if (!received->done) {
                // A frame was queued while the endpoint stayed silent: it
                // goes out before the next message is read.
                send_pending_frames();
                continue;
            }
            if (!received->error.empty()) return fail(received->error);
            core::net::NetworkTrafficStats::get_instance().record_received(received->text.size());
            // No transport lock is held here, so the consumer may reset this
            // transport — directly or by starting the next request — and the
            // next wait_for() then sees the cancelled task fail.
            if (callback(received->text)) {
                WebSocketStreamResult result{
                    .status = WebSocketStreamResult::Status::Completed,
                    .request_sent = true,
                    .connection_reused = reused,
                };
                std::lock_guard lock(state->mutex);
                result.http_status = state->status;
                result.response_headers = state->headers;
                return result;
            }
        }
    }
}

} // namespace core::llm::transport
