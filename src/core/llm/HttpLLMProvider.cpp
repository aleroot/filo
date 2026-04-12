#include "HttpLLMProvider.hpp"
#include "ModelRegistry.hpp"
#include "protocols/ApiProtocol.hpp"
#include "protocols/GeminiProtocol.hpp"
#include "../logging/Logger.hpp"
#include "../utils/UriUtils.hpp"
#include <cpr/cpr.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <random>
#include <thread>

namespace core::llm {

namespace {
    struct PreparedHttpStreamRequest {
        std::unique_ptr<protocols::ApiProtocolBase> protocol;
        std::string                                  payload;
        std::string                                  delimiter;
        std::string                                  model;
        std::string                                  url;
        cpr::Header                                  headers;
    };

    [[nodiscard]] std::string normalize_metadata_model(
        std::string_view model,
        const protocols::ApiProtocolBase* protocol) {
        if (!protocol) return std::string(model);

        const std::string_view protocol_name = protocol->name();
        if (protocol_name == "gemini" || protocol_name == "gemini_code_assist") {
            return protocols::normalize_requested_gemini_model(model);
        }

        return std::string(model);
    }

    [[nodiscard]] PreparedHttpStreamRequest prepare_stream_request(
        const ChatRequest& request,
        std::string_view default_model,
        std::string_view base_url,
        const std::shared_ptr<core::auth::ICredentialSource>& cred_source,
        const protocols::ApiProtocolBase& protocol_template) {
        PreparedHttpStreamRequest prepared;
        prepared.protocol = protocol_template.clone();

        core::auth::AuthInfo auth;
        if (cred_source) {
            auth = cred_source->get_auth();
        }

        ChatRequest req = request;
        if (req.model.empty()) {
            req.model = std::string(default_model);
        }
        req.auth_properties = auth.properties;
        prepared.protocol->prepare_request(req);
        prepared.model = req.model;
        prepared.payload = prepared.protocol->serialize(req);
        prepared.delimiter = std::string(prepared.protocol->event_delimiter());
        prepared.url = prepared.protocol->build_url(base_url, req.model);
        bool first = (prepared.url.find('?') == std::string::npos);
        for (const auto& [k, v] : auth.query_params) {
            prepared.url += (first ? '?' : '&');
            prepared.url += k + '=' + v;
            first = false;
        }

        prepared.headers = prepared.protocol->build_headers(auth);
        prepared.headers["Accept"] = "text/event-stream";
        return prepared;
    }

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

    [[nodiscard]] bool looks_like_loopback_base_url(std::string_view base_url) {
        return core::utils::uri::is_loopback_http_url(base_url);
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
    return core::llm::get_max_context_size(
        normalize_metadata_model(default_model_, protocol_.get()));
}

std::optional<ModelInfo> HttpLLMProvider::get_model_info() const {
    return ModelRegistry::instance().get_info(
        normalize_metadata_model(default_model_, protocol_.get()));
}

std::string HttpLLMProvider::get_last_model() const {
    std::lock_guard lock(state_mutex_);
    return last_model_;
}

std::vector<std::string> HttpLLMProvider::validate_request(const ChatRequest& request) const {
    std::vector<std::string> errors;
    
    const std::string model = normalize_metadata_model(
        request.model.empty() ? std::string_view(default_model_) : std::string_view(request.model),
        protocol_.get());
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

    if (request_has_image_input(request) && !info->supports(ModelCapability::Vision)) {
        errors.push_back("model does not support image input");
    }
    
    return errors;
}

bool HttpLLMProvider::supports(ModelCapability cap) const {
    return ModelRegistry::instance().supports(
        normalize_metadata_model(default_model_, protocol_.get()),
        cap);
}

double HttpLLMProvider::estimate_cost(int input_tokens, int output_tokens) const {
    return ModelRegistry::instance().estimate_cost(
        normalize_metadata_model(default_model_, protocol_.get()),
        input_tokens,
        output_tokens);
}

bool HttpLLMProvider::should_estimate_cost() const {
    return !cred_source_ || !cred_source_->uses_subscription_billing();
}

ProviderCapabilities HttpLLMProvider::capabilities() const {
    const bool is_ollama = protocol_ && protocol_->name() == "ollama";
    return ProviderCapabilities{
        .supports_tool_calls = true,
        .is_local = is_ollama && looks_like_loopback_base_url(base_url_),
    };
}

void HttpLLMProvider::reset_conversation_state() {
    if (protocol_) {
        protocol_->reset_state();
    }
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

    PreparedHttpStreamRequest prepared;
    ChatRequest effective_request = request;

    try {
        const std::string metadata_model = normalize_metadata_model(
            effective_request.model.empty()
                ? std::string_view(default_model_)
                : std::string_view(effective_request.model),
            protocol_.get());
        if (!ModelRegistry::instance().supports(metadata_model, ModelCapability::Vision)) {
            degrade_historical_image_inputs(effective_request);
        }

        if (const auto errors = validate_request(effective_request); !errors.empty()) {
            std::string message = "\n[Request validation error: ";
            for (std::size_t i = 0; i < errors.size(); ++i) {
                if (i > 0) message += "; ";
                message += errors[i];
            }
            message += "]";
            callback(StreamChunk::make_error(std::move(message)));
            return;
        }

        prepared = prepare_stream_request(
            effective_request, default_model_, base_url_, cred_source_, *protocol_);
        {
            std::lock_guard lock(state_mutex_);
            last_model_ = prepared.model;
        }
    } catch (const std::exception& e) {
        core::logging::error("[HTTP] Failed to start request: {}", e.what());
        callback(StreamChunk::make_error(std::string("\n[Failed to start request: ") + e.what() + "]"));
        return;
    } catch (...) {
        core::logging::error("[HTTP] Failed to start request: unknown exception");
        callback(StreamChunk::make_error("\n[Failed to start request: unknown exception]"));
        return;
    }

    std::thread([self      = std::move(keepalive),
                 url       = std::move(prepared.url),
                 headers   = std::move(prepared.headers),
                 payload   = std::move(prepared.payload),
                 delimiter = std::move(prepared.delimiter),
                 protocol  = std::move(prepared.protocol),
                 callback] () mutable {

        try {
            // Retry configuration for 429/529 errors
            constexpr int max_retries = 3;
            int retry_attempt = 0;
            int retry_after_seconds = 0;
            bool attempted_auth_recovery = false;

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

                auto forward_parsed_event = [&] (std::string_view event_payload) {
                    if (event_payload.empty()) return;
                    protocols::ParseResult result = protocol->parse_event(event_payload);

                    // Forward all content/tool chunks produced by this event.
                    bool has_terminal_chunk = false;
                    for (auto& chunk : result.chunks) {
                        callback(chunk);
                        if (chunk.is_final) has_terminal_chunk = true;
                    }

                    if (has_terminal_chunk) {
                        done_signalled = true;
                        return;
                    }

                    // Report token usage before emitting the final chunk so that
                    // get_last_usage() is populated before callers observe is_final.
                    if (result.prompt_tokens > 0 || result.completion_tokens > 0) {
                        self->set_last_usage(result.prompt_tokens, result.completion_tokens);
                    }

                    if (result.done) {
                        done_signalled = true;
                        callback(StreamChunk::make_final());
                    }
                };

                auto drain_complete_events = [&] {
                    while (!done_signalled) {
                        std::size_t pos = buffer.find(delimiter);
                        std::size_t delim_len = delimiter.size();

                        // Most SSE providers use "\n\n", but some use CRLF framing.
                        if (delimiter == "\n\n") {
                            const std::size_t crlf_pos = buffer.find("\r\n\r\n");
                            if (crlf_pos != std::string::npos
                                && (pos == std::string::npos || crlf_pos < pos)) {
                                pos = crlf_pos;
                                delim_len = 4;
                            }
                        }

                        if (pos == std::string::npos) break;

                        std::string event = buffer.substr(0, pos);
                        buffer.erase(0, pos + delim_len);
                        forward_parsed_event(event);
                    }
                };

                session.SetWriteCallback(cpr::WriteCallback([&buffer, &drain_complete_events]
                                                            (std::string_view data,
                                                             intptr_t /*userdata*/) -> bool {
                    buffer.append(data);
                    drain_complete_events();
                    return true;
                }));

                cpr::Response r = session.Post();
                if (!done_signalled) {
                    // Some providers close the stream without a trailing delimiter.
                    // Parse any non-whitespace remainder as one final event.
                    if (buffer.find_first_not_of(" \t\r\n") != std::string::npos) {
                        forward_parsed_event(buffer);
                    }
                    buffer.clear();
                }
                if (r.status_code != 200) {
                    core::logging::debug("[HTTP] Response status={}, body={}", r.status_code, r.text.substr(0, 500));
                }

                // Fire the response lifecycle hook.  Protocols use this to extract
                // rate-limit headers, update metrics, or prepare any per-response state.
                // The hook is a no-op for protocols that do not override it.
                const protocols::HttpResponse http_resp{
                    static_cast<int>(r.status_code), r.text, r.header};
                protocol->on_response(http_resp);

                // Attempt one forced credential refresh on auth failures (OAuth providers).
                const bool oauth_revoked_403 =
                    (r.status_code == 403
                     && r.text.find("OAuth token has been revoked") != std::string::npos);
                if ((r.status_code == 401 || oauth_revoked_403)
                    && !attempted_auth_recovery
                    && self->cred_source_) {
                    attempted_auth_recovery = true;
                    if (self->cred_source_->refresh_on_auth_failure()) {
                        try {
                            auto refreshed_auth = self->cred_source_->get_auth();
                            headers = protocol->build_headers(refreshed_auth);
                            headers["Accept"] = "text/event-stream";
                            callback(StreamChunk::make_error(
                                "\n[Authentication expired. Retrying with refreshed credentials...]"));
                            continue;
                        } catch (const std::exception& e) {
                            core::logging::warn("Credential refresh retry failed: {}", e.what());
                        }
                    }
                }

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

                try {
                    protocol->enrich_rate_limit(self->base_url_, headers, http_resp);
                } catch (const std::exception& e) {
                    core::logging::warn(
                        "Protocol '{}' rate-limit enrichment failed: {}",
                        protocol->name(),
                        e.what());
                } catch (...) {
                    core::logging::warn(
                        "Protocol '{}' rate-limit enrichment failed: unknown exception",
                        protocol->name());
                }

                auto rate_limit_info = protocol->last_rate_limit();

                // Update cached rate limit info for status bar display.
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
        } catch (const std::exception& e) {
            core::logging::error("[HTTP] Unhandled streaming exception: {}", e.what());
            callback(StreamChunk::make_error(std::string("\n[Internal streaming error: ") + e.what() + "]"));
        } catch (...) {
            core::logging::error("[HTTP] Unhandled streaming exception: unknown exception");
            callback(StreamChunk::make_error("\n[Internal streaming error: unknown exception]"));
        }
    }).detach();
}

} // namespace core::llm
