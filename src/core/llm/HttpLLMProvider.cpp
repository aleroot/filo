#include "HttpLLMProvider.hpp"
#include "ModelRegistry.hpp"
#include "protocols/ApiProtocol.hpp"
#include "protocols/GeminiProtocol.hpp"
#include "transport/CurlWebSocketTransport.hpp"
#include "transport/HttpHeaderUtils.hpp"
#include "../logging/Logger.hpp"
#include "../utils/UriUtils.hpp"
#include <cpr/cpr.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
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
        ChatRequest                                  request;
        core::auth::AuthInfo                         auth;
    };

    struct PreparedWebSocketStreamRequest {
        std::string websocket_url;
        protocols::WebSocketRequestFrame websocket_frame;
        cpr::Header websocket_headers;
        std::string websocket_connection_key;
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
        prepared.protocol->prepare_media_uploads(req, base_url, auth);
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
        prepared.protocol->prepare_headers(prepared.headers, req, base_url);
        prepared.request = req;
        prepared.auth = std::move(auth);
        return prepared;
    }

    [[nodiscard]] std::optional<PreparedWebSocketStreamRequest> prepare_websocket_stream_request(
        protocols::ApiProtocolBase& protocol,
        const ChatRequest& request,
        std::string_view base_url,
        const core::auth::AuthInfo& auth) {
        PreparedWebSocketStreamRequest prepared;
        prepared.websocket_url = protocol.build_websocket_url(base_url, request.model);
        if (prepared.websocket_url.empty()
            || !transport::CurlWebSocketTransport::runtime_supports_url(
                prepared.websocket_url)) {
            return std::nullopt;
        }

        prepared.websocket_headers = protocol.build_headers(auth);
        protocol.prepare_websocket_headers(prepared.websocket_headers, request, base_url);
        prepared.websocket_frame = protocol.initial_websocket_request_frame(request);
        if (prepared.websocket_frame.payload.empty()) {
            protocol.abandon_websocket_request(request);
            return std::nullopt;
        }

        prepared.websocket_connection_key = protocol.websocket_connection_key(
            prepared.websocket_url, prepared.websocket_headers, request);
        return prepared;
    }

    [[nodiscard]] std::optional<std::uintmax_t> local_file_size(
        const std::string& path) noexcept {
        std::error_code ec;
        const auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            return std::nullopt;
        }
        return size;
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
                                 std::unique_ptr<protocols::ApiProtocolBase>    protocol,
                                 core::config::ApiType                          api_type,
                                 std::string                                    provider_name,
                                 std::shared_ptr<IProviderClientIdentitySource>  client_identity_source)
    : base_url_(std::move(base_url))
    , cred_source_(std::move(cred_source))
    , default_model_(std::move(default_model))
    , protocol_(std::move(protocol))
    , api_type_(api_type)
    , provider_name_(std::move(provider_name))
    , client_identity_source_(std::move(client_identity_source))
{}

HttpLLMProvider::~HttpLLMProvider() = default;

HttpLLMProvider::WebSocketTransportState::WebSocketTransportState()
    : client(std::make_unique<transport::CurlWebSocketTransport>()) {}

HttpLLMProvider::WebSocketTransportState::~WebSocketTransportState() = default;

bool HttpLLMProvider::WebSocketTransportState::available() const noexcept {
    return enabled.load(std::memory_order_relaxed) && client != nullptr;
}

void HttpLLMProvider::WebSocketTransportState::disable() noexcept {
    enabled.store(false, std::memory_order_relaxed);
}

void HttpLLMProvider::WebSocketTransportState::reset() {
    if (client) {
        client->reset();
    }
    enabled.store(true, std::memory_order_relaxed);
}

int HttpLLMProvider::max_context_size() const noexcept {
    return core::llm::get_max_context_size(
        normalize_metadata_model(default_model_, protocol_.get()));
}

std::optional<ModelInfo> HttpLLMProvider::get_model_info() const {
    return ModelRegistry::instance().get_info(
        normalize_metadata_model(default_model_, protocol_.get()));
}

void HttpLLMProvider::discover_models(
    const ModelCatalogDiscoveryOptions& options) const {
    if (!protocol_ || provider_name_.empty()) {
        return;
    }

    discover_and_register_models_in_background(
        provider_name_,
        api_type_,
        base_url_,
        cred_source_,
        protocol_->clone(),
        options);
}

std::string HttpLLMProvider::get_last_model() const {
    std::lock_guard lock(state_mutex_);
    return last_model_;
}

std::optional<ProviderMetadata> HttpLLMProvider::metadata() const {
    return ProviderMetadata{
        .api_type = api_type_,
        .provider_name = provider_name_,
        .base_url = base_url_,
        .default_model = default_model_,
        .credential_source = cred_source_,
        .client_identity_source = client_identity_source_,
    };
}

std::vector<std::string> HttpLLMProvider::validate_request(const ChatRequest& request) const {
    std::vector<std::string> errors;
    
    const std::string model = normalize_metadata_model(
        request.model.empty() ? std::string_view(default_model_) : std::string_view(request.model),
        protocol_.get());
    const auto info = ModelRegistry::instance().lookup(model);
    
    if (!info) {
        // Unknown model - can't validate, but not necessarily an error
        // (could be a new model not yet in registry)
        return errors;
    }

    const bool has_declared_capabilities = info->capabilities != 0;
    
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
    if (has_declared_capabilities
        && !request.tools.empty()
        && !info->supports(ModelCapability::FunctionCalling)) {
        errors.push_back("model does not support function calling");
    }
    
    // Validate JSON mode support
    if (has_declared_capabilities
        && request.response_format.is_structured()
        && !info->supports(ModelCapability::JsonMode)) {
        errors.push_back("model does not support structured outputs (JSON mode)");
    }

    if (has_declared_capabilities
        && request_has_image_input(request)
        && !info->supports(ModelCapability::Vision)) {
        errors.push_back("model does not support image input");
    }

    if (has_declared_capabilities
        && request_has_video_input(request)
        && !info->supports(ModelCapability::VideoInput)) {
        errors.push_back("model does not support video input");
    }

    const bool can_upload_video = protocol_ && protocol_->supports_video_upload();
    for (const auto& msg : request.messages) {
        for (const auto& part : msg.content_parts) {
            if (part.type != ContentPartType::Video || part.path.empty()) {
                continue;
            }
            const auto size = local_file_size(part.path);
            if (!size.has_value()) {
                errors.push_back("video input file is not readable: " + part.path);
                continue;
            }
            if (*size == 0) {
                errors.push_back("video input file is empty: " + part.path);
            } else if (*size > kMaxLocalVideoBytes) {
                errors.push_back(std::format(
                    "video input file '{}' is {} bytes, which exceeds the {} MB limit",
                    part.path,
                    *size,
                    kMaxLocalVideoBytes / (1024ULL * 1024ULL)));
            } else if (!can_upload_video && *size > kMaxInlineVideoBytes) {
                errors.push_back(std::format(
                    "video input file '{}' is {} bytes; this provider can only inline videos up to {} MB",
                    part.path,
                    *size,
                    kMaxInlineVideoBytes / (1024ULL * 1024ULL)));
            }
        }
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
    websocket_.reset();
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
        const auto metadata_info = ModelRegistry::instance().get_info(metadata_model);
        if (metadata_info.has_value()
            && metadata_info->capabilities != 0) {
            degrade_historical_media_inputs(
                effective_request,
                {
                    .images = !metadata_info->supports(ModelCapability::Vision),
                    .videos = !metadata_info->supports(ModelCapability::VideoInput),
                });
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
                 request_metadata = std::move(prepared.request),
                 auth = std::move(prepared.auth),
                 callback] () mutable {

        try {
            if (self->websocket_.available() && protocol->supports_websocket_transport()) {
                std::optional<PreparedWebSocketStreamRequest> websocket_request;
                try {
                    websocket_request = prepare_websocket_stream_request(
                        *protocol, request_metadata, self->base_url_, auth);
                } catch (const std::exception& e) {
                    core::logging::debug(
                        "[WebSocket] Failed to prepare request; falling back to HTTP: {}",
                        e.what());
                    protocol->abandon_websocket_request(request_metadata);
                } catch (...) {
                    core::logging::debug(
                        "[WebSocket] Failed to prepare request; falling back to HTTP: unknown exception");
                    protocol->abandon_websocket_request(request_metadata);
                }

                if (websocket_request.has_value()
                    && !websocket_request->websocket_frame.payload.empty()) {
                    protocols::WebSocketRequestFrame websocket_frame =
                        std::move(websocket_request->websocket_frame);

                    while (!websocket_frame.payload.empty()) {
                        bool websocket_done = false;
                        bool suppressed_error = false;
                        self->set_last_usage(0, 0);

                        auto websocket_result = self->websocket_.client->stream_text(
                            websocket_request->websocket_url,
                            websocket_request->websocket_connection_key,
                            websocket_request->websocket_headers,
                            websocket_frame.payload,
                            [&](std::string_view event_payload) {
                                protocols::ParseResult result = protocol->parse_event(event_payload);

                                for (auto& chunk : result.chunks) {
                                    if (websocket_frame.suppress_output) {
                                        suppressed_error = suppressed_error || chunk.is_error;
                                        if (chunk.is_final) websocket_done = true;
                                        continue;
                                    }

                                    callback(chunk);
                                    if (chunk.is_final) websocket_done = true;
                                }

                                if (websocket_done) {
                                    return true;
                                }

                                if (!websocket_frame.suppress_output
                                    && (result.prompt_tokens > 0 || result.completion_tokens > 0)) {
                                    self->set_last_usage(
                                        result.prompt_tokens, result.completion_tokens);
                                }

                                if (result.done) {
                                    websocket_done = true;
                                    if (!websocket_frame.suppress_output) {
                                        callback(StreamChunk::make_final());
                                    }
                                }
                                return websocket_done;
                            });

                        protocol->observe_response_headers(
                            websocket_result.response_headers, request_metadata);

                        if (websocket_result.completed() && !suppressed_error) {
                            if (!websocket_frame.suppress_output) {
                                const protocols::HttpResponse websocket_response{
                                    200, "", websocket_result.response_headers};
                                protocol->on_response(websocket_response);
                                self->set_last_rate_limit_info(protocol->last_rate_limit());
                                return;
                            }

                            websocket_frame = protocol->next_websocket_request_frame(
                                request_metadata, websocket_frame);
                            if (!websocket_frame.payload.empty()) {
                                continue;
                            }
                        }

                        self->websocket_.client->reset();
                        self->websocket_.disable();
                        protocol->abandon_websocket_request(request_metadata);

                        if (websocket_result.request_sent
                            && !websocket_frame.suppress_output) {
                            core::logging::debug(
                                "[WebSocket] Stream failed after request send status={} reason={}",
                                websocket_result.http_status,
                                websocket_result.message);
                            callback(StreamChunk::make_error(
                                "\n[WebSocket stream failed after the request was sent: "
                                + websocket_result.message + "]"));
                            callback(StreamChunk::make_final());
                            return;
                        }

                        core::logging::debug(
                            "[WebSocket] Falling back to HTTP status={} reason={}",
                            websocket_result.http_status,
                            websocket_result.message);
                        protocol = protocol->clone();
                        break;
                    }
                }
            }

            // Retry configuration for 429/529 errors
            constexpr int max_retries = 3;
            int retry_attempt = 0;
            int retry_after_seconds = 0;
            bool attempted_auth_recovery = false;

            while (true) {
                std::string buffer;
                std::size_t buffer_start = 0;
                bool        done_signalled = false;

                // Prevent stale usage from previous requests if this request does not
                // emit a usage chunk.
                self->set_last_usage(0, 0);

                cpr::Session session;
                session.SetUrl(cpr::Url{url});
                session.SetHeader(headers);
                session.SetBody(cpr::Body{payload});

                cpr::Header response_headers_seen;
                bool observed_transport_headers = false;
                session.SetHeaderCallback(cpr::HeaderCallback(
                    [&protocol, &response_headers_seen,
                     &observed_transport_headers, &request_metadata]
                    (std::string_view line, intptr_t /*userdata*/) -> bool {
                        if (core::utils::str::trim_ascii_view(line).empty()) {
                            if (!observed_transport_headers) {
                                observed_transport_headers = true;
                                protocol->observe_response_headers(
                                    response_headers_seen, request_metadata);
                            }
                            return true;
                        }

                        if (auto header = transport::parse_header_line(line); header.has_value()) {
                            response_headers_seen[std::move(header->first)] =
                                std::move(header->second);
                        }
                        return true;
                    }));

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
                        std::size_t pos = buffer.find(delimiter, buffer_start);
                        std::size_t delim_len = delimiter.size();

                        // Most SSE providers use "\n\n", but some use CRLF framing.
                        if (delimiter == "\n\n") {
                            const std::size_t crlf_pos = buffer.find("\r\n\r\n", buffer_start);
                            if (crlf_pos != std::string::npos
                                && (pos == std::string::npos || crlf_pos < pos)) {
                                pos = crlf_pos;
                                delim_len = 4;
                            }
                        }

                        if (pos == std::string::npos) break;

                        const std::string_view event_payload{
                            buffer.data() + buffer_start, pos - buffer_start};
                        forward_parsed_event(event_payload);
                        buffer_start = pos + delim_len;
                    }

                    // Compaction: avoid O(n) erase per event; erase in larger chunks.
                    if (buffer_start == 0) return;
                    if (buffer_start >= buffer.size()) {
                        buffer.clear();
                        buffer_start = 0;
                        return;
                    }
                    if (buffer_start >= 8192 || buffer_start * 2 >= buffer.size()) {
                        buffer.erase(0, buffer_start);
                        buffer_start = 0;
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
                    const std::string_view remainder{
                        buffer.data() + buffer_start, buffer.size() - buffer_start};
                    if (remainder.find_first_not_of(" \t\r\n") != std::string::npos) {
                        forward_parsed_event(remainder);
                    }
                    buffer.clear();
                    buffer_start = 0;
                }
                if (r.status_code != 200) {
                    core::logging::debug("[HTTP] Response status={}, body={}", r.status_code, r.text.substr(0, 500));
                }

                // Fire the response lifecycle hook.  Protocols use this to extract
                // rate-limit headers, update metrics, or prepare any per-response state.
                // The hook is a no-op for protocols that do not override it.
                const protocols::HttpResponse http_resp{
                    static_cast<int>(r.status_code), r.text, r.header};
                protocol->observe_response_headers(r.header, request_metadata);
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
                            protocol->prepare_headers(headers, request_metadata, self->base_url_);
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
