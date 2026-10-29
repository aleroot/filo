#include "AppleWebSocketTransport.hpp"
#include "core/net/NetworkTraffic.hpp"

#import <Foundation/Foundation.h>

#include <condition_variable>
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
};

AppleWebSocketTransport::AppleWebSocketTransport() : impl_(std::make_unique<Impl>()) {}
AppleWebSocketTransport::~AppleWebSocketTransport() { reset(); }

void AppleWebSocketTransport::reset() {
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
        bool reused = matches_connection(key);
        bool request_sent = false;
        const auto fail = [&](std::string message) {
            WebSocketStreamResult result{
                .status = request_sent ? WebSocketStreamResult::Status::StreamFailed
                                       : WebSocketStreamResult::Status::ConnectFailed,
                .message = is_cancelled() ? "request cancelled by user" : std::move(message),
                .request_sent = request_sent,
                .connection_reused = reused,
            };
            if (impl_->state) {
                std::lock_guard lock(impl_->state->mutex);
                result.http_status = impl_->state->status;
                result.response_headers = impl_->state->headers;
                if (result.http_status == 426) result.status = WebSocketStreamResult::Status::UpgradeRequired;
            }
            reset();
            return result;
        };
        if (is_cancelled()) return fail("request cancelled by user");
        if (!reused) {
            reset();
            NSString* endpoint_string = ns_string(url);
            NSURL* endpoint = endpoint_string ? [NSURL URLWithString:endpoint_string] : nil;
            if (!endpoint || (![[endpoint scheme] isEqualToString:@"ws"]
                && ![[endpoint scheme] isEqualToString:@"wss"])) return fail("invalid websocket URL");
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
            [impl_->task resume];
            if (!wait_for(impl_->state, [](const auto& s) { return s.opened || s.closed; },
                          cancelled, std::chrono::seconds(15))) return fail("websocket handshake timed out");
            bool opened;
            std::string error;
            {
                std::lock_guard lock(impl_->state->mutex);
                opened = impl_->state->opened;
                error = impl_->state->error;
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
        [impl_->task sendMessage:message completionHandler:^(NSError* error) {
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
        while (!is_cancelled()) {
            auto received = std::make_shared<Operation>();
            [impl_->task receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage* incoming, NSError* error) {
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
            if (!wait_for(received, [](const auto& s) { return s.done; }, cancelled, idle)) return fail("idle timeout waiting for websocket event");
            if (!received->error.empty()) return fail(received->error);
            core::net::NetworkTrafficStats::get_instance().record_received(received->text.size());
            if (callback(received->text)) {
                WebSocketStreamResult result{
                    .status = WebSocketStreamResult::Status::Completed,
                    .request_sent = true,
                    .connection_reused = reused,
                };
                if (impl_->state) {
                    std::lock_guard lock(impl_->state->mutex);
                    result.http_status = impl_->state->status;
                    result.response_headers = impl_->state->headers;
                }
                return result;
            }
        }
        return fail("request cancelled by user");
    }
}

} // namespace core::llm::transport
