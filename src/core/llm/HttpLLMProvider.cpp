#include "HttpLLMProvider.hpp"
#include "ModelRegistry.hpp"
#include "protocols/ApiProtocol.hpp"
#include "../logging/Logger.hpp"
#include <cpr/cpr.h>
#include <thread>
#include <chrono>
#include <format>
#include <random>

namespace core::llm {

namespace {
    // Exponential backoff with jitter for rate limit retries.
    // Uses retry_after from the API when available (429/529 responses).
    void backoff_sleep_with_retry_after(int attempt, int retry_after_seconds) {
        constexpr int base_ms = 500;
        constexpr int cap_ms = 30000;  // Max 30 seconds
        
        int delay_ms;
        if (retry_after_seconds > 0) {
            // Use server-provided retry_after with small buffer
            delay_ms = retry_after_seconds * 1000 + 100;
        } else {
            // Exponential backoff: 500ms, 1000ms, 2000ms, 4000ms...
            delay_ms = std::min(base_ms * (1 << (attempt - 1)), cap_ms);
        }
        
        // Add jitter: ±25% randomization to prevent thundering herd
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> jitter(-delay_ms / 4, delay_ms / 4);
        delay_ms = std::max(100, delay_ms + jitter(rng));
        
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }

}

HttpLLMProvider::HttpLLMProvider(std::string                                    base_url,
                                 std::shared_ptr<core::auth::ICredentialSource> cred_source,
                                 std::string                                    default_model,
                                 std::unique_ptr<protocols::ApiProtocolBase>    protocol)
    : base_url_(std::move(base_url))
    , cred_source_(std::move(cred_source))
    , default_model_(std::move(default_model))
    , protocol_(std::move(protocol))
{}

int HttpLLMProvider::max_context_size() const noexcept {
    return core::llm::get_max_context_size(default_model_);
}

std::optional<ModelInfo> HttpLLMProvider::get_model_info() const {
    return ModelRegistry::instance().get_info(default_model_);
}

std::vector<std::string> HttpLLMProvider::validate_request(const ChatRequest& request) const {
    std::vector<std::string> errors;
    
    const std::string& model = request.model.empty() ? default_model_ : request.model;
    const auto* info = ModelRegistry::instance().lookup(model);
    
    if (!info) {
        // Unknown model - can't validate, but not necessarily an error
        // (could be a new model not yet in registry)
        return errors;
    }
    
    // Validate max_tokens
    if (request.max_tokens.has_value()) {
        if (!ModelRegistry::instance().validate_max_tokens(model, *request.max_tokens)) {
            int effective_max = info->effective_max_tokens();
            errors.push_back(std::format(
                "max_tokens {} exceeds model limit of {}", 
                *request.max_tokens, effective_max));
        }
    }
    
    // Validate temperature
    if (request.temperature.has_value()) {
        if (!info->validate_parameter("temperature", *request.temperature)) {
            if (info->constraints.temperature) {
                errors.push_back(std::format(
                    "temperature {} is outside valid range [{}, {}]",
                    *request.temperature,
                    info->constraints.temperature->min,
                    info->constraints.temperature->max));
            }
        }
    }
    
    // Validate tool support
    if (!request.tools.empty() && !info->supports(ModelCapability::FunctionCalling)) {
        errors.push_back("model does not support function calling");
    }
    
    // Validate JSON mode support
    if (request.response_format.is_structured() && !info->supports(ModelCapability::JsonMode)) {
        errors.push_back("model does not support structured outputs (JSON mode)");
    }
    
    return errors;
}

bool HttpLLMProvider::supports(ModelCapability cap) const {
    return ModelRegistry::instance().supports(default_model_, cap);
}

double HttpLLMProvider::estimate_cost(int input_tokens, int output_tokens) const {
    return ModelRegistry::instance().estimate_cost(default_model_, input_tokens, output_tokens);
}

bool HttpLLMProvider::should_estimate_cost() const {
    return !cred_source_ || !cred_source_->uses_subscription_billing();
}

void HttpLLMProvider::stream_response(const ChatRequest&                      request,
                                      std::function<void(const StreamChunk&)> callback) {

    std::shared_ptr<HttpLLMProvider> keepalive;
    try {
        keepalive = shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        callback(StreamChunk::make_error(
            "\n[Internal error: HttpLLMProvider::stream_response requires shared_ptr ownership.]"));
        callback(StreamChunk::make_final());
        return;
    }

    // Resolve per-call state upfront (may block for token refresh).
    auto protocol = protocol_->clone();
    auto auth     = cred_source_->get_auth();

    ChatRequest req = request;
    if (req.model.empty()) {
        req.model = default_model_;
    }
    protocol->prepare_request(req);

    std::string payload   = protocol->serialize(req);
    std::string delimiter = std::string(protocol->event_delimiter());

    // Build URL; append AuthInfo::query_params (e.g. Gemini ?key=…).
    std::string url = protocol->build_url(base_url_, req.model);
    {
        bool first = (url.find('?') == std::string::npos);
        for (const auto& [k, v] : auth.query_params) {
            url += (first ? '?' : '&');
            url += k + '=' + v;
            first = false;
        }
    }

    cpr::Header headers = protocol->build_headers(auth);
    headers["Accept"]   = "text/event-stream";

    std::thread([self      = std::move(keepalive),
                 url       = std::move(url),
                 headers   = std::move(headers),
                 payload   = std::move(payload),
                 delimiter = std::move(delimiter),
                 protocol  = std::move(protocol),
                 callback] () mutable {

        // Retry configuration for 429/529 errors
        constexpr int max_retries = 3;
        int retry_attempt = 0;
        int retry_after_seconds = 0;
        
        while (true) {
            std::string buffer;
            bool        done_signalled = false;

            // Prevent stale usage from previous requests if this request does not
            // emit a usage chunk.
            self->set_last_usage(0, 0);

            cpr::Session session;
            session.SetUrl(cpr::Url{url});
            session.SetHeader(headers);
            session.SetBody(cpr::Body{payload});

            session.SetWriteCallback(cpr::WriteCallback([self, &buffer, &done_signalled, &delimiter, &protocol, callback] (std::string_view data, intptr_t /*userdata*/) -> bool {
                buffer.append(data);

                std::size_t pos = 0;
                while ((pos = buffer.find(delimiter)) != std::string::npos) {
                    std::string event = buffer.substr(0, pos);
                    buffer.erase(0, pos + delimiter.size());

                    if (event.empty()) continue;
                    protocols::ParseResult result = protocol->parse_event(event);

                    // Forward all content/tool chunks produced by this event.
                    bool has_terminal_chunk = false;
                    for (auto& chunk : result.chunks) {
                        callback(chunk);
                        if (chunk.is_final) has_terminal_chunk = true;
                    }

                    if (has_terminal_chunk) {
                        done_signalled = true;
                        return true;
                    }

                    // Report token usage before emitting the final chunk so that
                    // get_last_usage() is populated before callers observe is_final.
                    if (result.prompt_tokens > 0 || result.completion_tokens > 0) {
                        self->set_last_usage(result.prompt_tokens, result.completion_tokens);
                    }

                    if (result.done) {
                        done_signalled = true;
                        callback(StreamChunk::make_final());
                        return true;
                    }
                }
                return true;
            }));

            cpr::Response r = session.Post();
            if (r.status_code != 200) {
                core::logging::debug("[HTTP] Response status={}, body={}", r.status_code, r.text.substr(0, 500));
            }

            // Fire the response lifecycle hook.  Protocols use this to extract
            // rate-limit headers, update metrics, or prepare any per-response state.
            // The hook is a no-op for protocols that do not override it.
            const protocols::HttpResponse http_resp{
                static_cast<int>(r.status_code), r.text, r.header};
            protocol->on_response(http_resp);

            // Check if we should retry (429 rate limit or 529 overloaded)
            if (r.status_code == 429 || r.status_code == 529) {
                if (retry_attempt < max_retries) {
                    retry_attempt++;
                    
                    // Extract retry_after from the protocol's rate limit info
                    // For Anthropic, this comes from the retry-after header
                    auto rate_limit_info = protocol->last_rate_limit();
                    retry_after_seconds = rate_limit_info.retry_after;
                    
                    // Notify user of retry
                    if (retry_after_seconds > 0) {
                        callback(StreamChunk::make_error(
                            std::format("\n[Rate limited ({}). Retrying in {}s (attempt {}/{})...]",
                                       r.status_code, retry_after_seconds, retry_attempt, max_retries)));
                    } else {
                        callback(StreamChunk::make_error(
                            std::format("\n[Server overloaded ({}). Retrying with backoff (attempt {}/{})...]",
                                       r.status_code, retry_attempt, max_retries)));
                    }
                    
                    backoff_sleep_with_retry_after(retry_attempt, retry_after_seconds);
                    continue;  // Retry the request
                }
            }

            // Update cached rate limit info for status bar display
            // This works for any protocol that implements on_response() to populate rate limits
            auto rate_limit_info = protocol->last_rate_limit();
            self->set_last_rate_limit_info(rate_limit_info);

            if (r.error.code != cpr::ErrorCode::OK) {
                core::logging::error("[HTTP] Connection error: {}", r.error.message);
                callback(StreamChunk::make_error(
                    "\n[Error connecting to " + url + ": " + r.error.message + "]"));
            } else if (r.status_code != 200) {
                core::logging::error("[HTTP] Error status={}", r.status_code);
                callback(StreamChunk::make_error(
                    "\n" + protocol->format_error_message(http_resp)));
            } else if (!done_signalled) {
                callback(StreamChunk::make_final());
            }
            break;  // Exit retry loop on non-retryable response
        }
    }).detach();
}

} // namespace core::llm
