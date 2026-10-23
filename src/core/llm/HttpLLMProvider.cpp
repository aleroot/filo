#include "HttpLLMProvider.hpp"
#include "ModelMetadata.hpp"
#include "ModelRegistry.hpp"
#include "protocols/ApiProtocol.hpp"
#include "transport/CurlWebSocketTransport.hpp"
#include "transport/HttpHeaderUtils.hpp"
#include "../auth/OAuthErrors.hpp"
#include "../logging/Logger.hpp"
#include "../net/NetworkTraffic.hpp"
#include "../utils/UriUtils.hpp"
#include <cpr/cpr.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>

namespace core::llm {

namespace {
    // CPR does not populate Response::text when a streaming write callback is
    // installed. Preserve a bounded prefix so non-2xx JSON error bodies remain
    // available to protocol-specific error formatters without buffering a
    // successful streaming response in full.
    constexpr std::size_t kMaxResponseBodyCaptureBytes = 64 * 1024;

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
        return protocol ? protocol->model_id(model) : std::string(model);
    }

    [[nodiscard]] TokenUsage token_usage_from(const protocols::ParseResult& result) noexcept {
        return TokenUsage{
            .prompt_tokens = result.prompt_tokens,
            .completion_tokens = result.completion_tokens,
            .total_tokens = result.prompt_tokens + result.completion_tokens,
            .cached_prompt_tokens = result.cached_prompt_tokens,
            .cache_creation_prompt_tokens = result.cache_creation_prompt_tokens,
            .reasoning_tokens = result.reasoning_tokens,
        };
    }

    [[nodiscard]] PreparedHttpStreamRequest prepare_stream_request(
        ChatRequest req,
        std::string_view default_model,
        std::string_view base_url,
        const std::shared_ptr<core::auth::ICredentialSource>& cred_source,
        const protocols::ApiProtocolBase& protocol_template,
        const std::atomic_bool& cancelled) {
        PreparedHttpStreamRequest prepared;
        prepared.protocol = protocol_template.clone();

        core::auth::AuthInfo auth;
        if (cred_source) {
            auth = cred_source->get_auth();
        }

        if (req.model.empty()) {
            req.model = std::string(default_model);
        }
        req.auth_properties = auth.properties;
        prepared.protocol->prepare_request(req);
        prepared.protocol->prepare_media_uploads(req, base_url, auth);
        prepared.protocol->prepare_context(req, base_url, auth,
            [&] { return cancelled.load(std::memory_order_acquire); });
        prepared.model = req.model;
        // The WebSocket path serializes its own frame. Avoid holding a second
        // complete encoded history for an HTTP fallback that may never run.
        if (!prepared.protocol->supports_websocket_transport()) {
            prepared.payload = prepared.protocol->serialize(req);
        }
        prepared.delimiter = std::string(prepared.protocol->event_delimiter());
        prepared.url = core::utils::uri::append_query_params(
            prepared.protocol->build_url(base_url, req.model), auth.query_params);

        prepared.headers = prepared.protocol->build_headers(auth);
        prepared.headers["Accept"] = "text/event-stream";
        prepared.protocol->prepare_headers(prepared.headers, req, base_url);
        prepared.request = std::move(req);
        prepared.auth = std::move(auth);
        return prepared;
    }

    [[nodiscard]] std::optional<PreparedWebSocketStreamRequest> prepare_websocket_stream_request(
        protocols::ApiProtocolBase& protocol,
        const ChatRequest& request,
        std::string_view base_url,
        const core::auth::AuthInfo& auth,
        transport::CurlWebSocketTransport& client) {
        PreparedWebSocketStreamRequest prepared;
        prepared.websocket_url = protocol.build_websocket_url(base_url, request.model);
        if (prepared.websocket_url.empty()
            || !transport::CurlWebSocketTransport::runtime_supports_url(
                prepared.websocket_url)) {
            return std::nullopt;
        }

        prepared.websocket_headers = protocol.build_headers(auth);
        protocol.prepare_websocket_headers(prepared.websocket_headers, request, base_url);
        prepared.websocket_connection_key = protocol.websocket_connection_key(
            prepared.websocket_url, prepared.websocket_headers, request);
        if (!client.matches_connection(prepared.websocket_connection_key)) {
            protocol.abandon_websocket_request(request);
        }
        prepared.websocket_frame = protocol.initial_websocket_request_frame(request);
        if (prepared.websocket_frame.payload.empty()) {
            protocol.abandon_websocket_request(request);
            return std::nullopt;
        }

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

    [[nodiscard]] bool looks_like_loopback_base_url(std::string_view base_url) {
        return core::utils::uri::is_loopback_http_url(base_url);
    }

    [[nodiscard]] StreamChunk make_reauthentication_chunk(
        const core::auth::ReauthenticationRequired& error,
        bool retry_safe) {
        const std::string provider_id(error.provider_id());
        return StreamChunk::make_authentication_error(
            "\n[Authentication required.]\n",
            AuthenticationRecoveryRequest{
                .provider_id = provider_id,
                .reason = error.what(),
                .retry_safe = retry_safe,
            });
    }

}

HttpLLMProvider::HttpLLMProvider(std::string                                    base_url,
                                 std::shared_ptr<core::auth::ICredentialSource> cred_source,
                                 std::string                                    default_model,
                                 std::unique_ptr<protocols::ApiProtocolBase>    protocol,
                                 core::config::ApiType                          api_type,
                                 std::string                                    provider_name,
                                 std::shared_ptr<IProviderClientIdentitySource>  client_identity_source,
                                 std::shared_ptr<const IModelCatalogSelector>    model_catalog_selector,
                                 std::string                                    service_id,
                                 HttpStreamTransportOptions                     transport_options)
    : base_url_(std::move(base_url))
    , cred_source_(std::move(cred_source))
    , default_model_(std::move(default_model))
    , protocol_(std::move(protocol))
    , api_type_(api_type)
    , provider_name_(std::move(provider_name))
    , service_id_(std::move(service_id))
    , client_identity_source_(std::move(client_identity_source))
    , model_catalog_selector_(std::move(model_catalog_selector))
    , transport_options_(transport_options)
{}

HttpLLMProvider::~HttpLLMProvider() = default;

std::string_view HttpLLMProvider::catalog_id() const noexcept {
    return service_id_.empty() ? std::string_view{provider_name_}
                               : std::string_view{service_id_};
}

void HttpLLMProvider::cancel() {
    cancel_requested_.store(true, std::memory_order_release);
}

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
    const std::string model =
        normalize_metadata_model(default_model_, protocol_.get());
    if (const auto info = resolved_model_info(model);
        info && info->context_window > 0) {
        return info->context_window;
    }
    return core::llm::get_max_context_size(model);
}

std::optional<ModelInfo> HttpLLMProvider::get_model_info() const {
    const std::string model =
        normalize_metadata_model(default_model_, protocol_.get());
    ensure_model_metadata(model);
    return resolved_model_info(model);
}

void HttpLLMProvider::discover_models(
    const ModelCatalogDiscoveryOptions& options) const {
    if (!protocol_ || provider_name_.empty()) {
        return;
    }

    discover_and_register_models_in_background(
        std::string(catalog_id()),
        api_type_,
        base_url_,
        cred_source_,
        protocol_->clone(),
        options);
}

std::string HttpLLMProvider::resolve_default_model() const {
    if (!default_model_.empty() || !model_catalog_selector_) {
        return default_model_;
    }

    auto snapshot = ModelCatalogAvailability::instance().snapshot(catalog_id());
    if (snapshot.models.empty() && snapshot.refresh_due()) {
        discover_models({.timeout_ms = 2500});
    }
    if (snapshot.models.empty()) {
        snapshot = ModelCatalogAvailability::instance().wait_for_snapshot(
            catalog_id(), std::chrono::milliseconds{3000});
    }

    const std::string registry_provider = model_registry_provider_key(
        provider_name_, api_type_);
    const std::vector<ModelInfo> registry_models =
        ModelRegistry::instance().get_by_provider(registry_provider);
    const ResolvedModelCatalog catalog = resolve_model_catalog(
        snapshot.models, registry_models);
    const ModelCatalogSelection selection =
        model_catalog_selector_->select(catalog.models);
    if (!selection.ok()) {
        throw std::runtime_error(
            selection.error.empty()
                ? "Neither the provider catalog nor the internal registry "
                  "provided a usable default model."
                : selection.error);
    }
    return selection.model;
}

void HttpLLMProvider::ensure_model_metadata(std::string_view model) const {
    if (model.empty()
        || provider_name_.empty()
        || api_type_ == config::ApiType::Unknown
        || !protocol_) {
        return;
    }

    auto snapshot =
        ModelCatalogAvailability::instance().snapshot(catalog_id());
    const bool provider_knows_model = static_cast<bool>(
        resolve_model_metadata(model, snapshot.models, nullptr));
    const bool registry_knows_model =
        ModelRegistry::instance().has_model(model);
    const bool targeted_refresh =
        snapshot.state == ModelCatalogDiscoveryState::Succeeded
        && !provider_knows_model
        && !registry_knows_model;

    // A retained provider catalog remains the primary tier while a refresh is
    // due or has failed transiently. Only block when no API data has ever been
    // obtained; this keeps requests reliable without adding recurring latency.
    if (snapshot.state != ModelCatalogDiscoveryState::PermanentSkip
        && (targeted_refresh || snapshot.refresh_due())) {
        discover_models({
            .timeout_ms = 2500,
            .force_refresh =
                targeted_refresh
                || snapshot.state == ModelCatalogDiscoveryState::Succeeded,
        });
        snapshot =
            ModelCatalogAvailability::instance().snapshot(catalog_id());
    }
    if (snapshot.state != ModelCatalogDiscoveryState::PermanentSkip
        && (targeted_refresh
            || (snapshot.models.empty()
                && (snapshot.refresh_in_progress || !snapshot.checked)))) {
        snapshot = ModelCatalogAvailability::instance().wait_for_snapshot(
            catalog_id(),
            std::chrono::milliseconds{3000});
    }
}

std::optional<ModelInfo> HttpLLMProvider::resolved_model_info(
    std::string_view model) const {
    const auto snapshot =
        ModelCatalogAvailability::instance().snapshot(catalog_id());
    const ResolvedModelMetadata resolved = resolve_model_metadata(
        model,
        snapshot.models,
        ModelRegistry::instance().lookup(model));
    return resolved.model;
}

std::optional<ModelInfo> HttpLLMProvider::provider_model_info(std::string_view model) const {
    const auto snapshot = ModelCatalogAvailability::instance().snapshot(catalog_id());
    return resolve_model_metadata(model, snapshot.models, nullptr).model;
}

std::optional<ModelReasoningProfile> HttpLLMProvider::resolved_reasoning_profile(
    std::string_view model) const {
    if (protocol_ && protocol_->uses_provider_reasoning_catalog()) {
        const auto snapshot = ModelCatalogAvailability::instance().snapshot(catalog_id());
        const auto resolved = resolve_model_metadata(model, snapshot.models, nullptr);
        if (resolved.model) return resolved.model->reasoning;
        return std::nullopt;
    }
    if (const auto info = resolved_model_info(model)) return info->reasoning;
    return std::nullopt;
}

std::string HttpLLMProvider::get_last_model() const {
    std::lock_guard lock(state_mutex_);
    return last_model_;
}

std::optional<ProviderMetadata> HttpLLMProvider::metadata() const {
    return ProviderMetadata{
        .api_type = api_type_,
        .provider_name = provider_name_,
        .service_id = service_id_.empty() ? provider_name_ : service_id_,
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
    const auto info = resolved_model_info(model);
    
    if (!info) {
        // Unknown model - can't validate, but not necessarily an error
        // (could be a new model not yet in registry)
        return errors;
    }

    // Absence is evidence of non-support only for an exhaustive capability
    // catalog. Sparse provider APIs (OpenAI, xAI, Ollama, and compatible
    // gateways) commonly advertise a useful subset and omit the rest.
    const bool can_reliably_deny_capabilities =
        info->capabilities_complete;
    
    // Validate max_tokens
    if (request.max_tokens.has_value()) {
        const int effective_max = info->effective_max_tokens();
        if ((effective_max > 0 && *request.max_tokens > effective_max)
            || *request.max_tokens < info->constraints.max_tokens_min) {
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
    if (can_reliably_deny_capabilities
        && !request.tools.empty()
        && !info->supports(ModelCapability::FunctionCalling)) {
        errors.push_back("model does not support function calling");
    }
    
    // Validate JSON mode support
    if (can_reliably_deny_capabilities
        && request.response_format.is_structured()
        && !info->supports(ModelCapability::JsonMode)) {
        errors.push_back("model does not support structured outputs (JSON mode)");
    }

    if (can_reliably_deny_capabilities
        && request_has_image_input(request)
        && !info->supports(ModelCapability::Vision)) {
        errors.push_back("model does not support image input");
    }

    if (can_reliably_deny_capabilities
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
    const auto info = resolved_model_info(
        normalize_metadata_model(default_model_, protocol_.get()));
    return info && info->supports(cap);
}

double HttpLLMProvider::estimate_cost(int input_tokens, int output_tokens) const {
    const auto info = resolved_model_info(
        normalize_metadata_model(default_model_, protocol_.get()));
    if (!info) return -1.0;
    return info->estimate_cost(input_tokens, output_tokens);
}

bool HttpLLMProvider::should_estimate_cost() const {
    return !cred_source_
        || cred_source_->billing_kind() != core::auth::BillingKind::Subscription;
}

ReasoningCapabilities HttpLLMProvider::reasoning_capabilities(
    std::string_view model) const noexcept {
    const auto protocol_capabilities = protocol_
        ? protocol_->reasoning_capabilities(model)
        : ReasoningCapabilities{};
    if (!protocol_capabilities.empty() && !protocol_->uses_provider_reasoning_catalog()) {
        return protocol_capabilities;
    }
    // A complete endpoint catalog is authoritative; family-name heuristics
    // cannot know which reasoning controls a particular deployment accepts.
    try {
        if (const auto profile = resolved_reasoning_profile(model); profile && profile->complete) {
            return profile->effort;
        }
    } catch (...) {
        // A catalog lookup must not turn a capability query into a failure.
    }
    return protocol_capabilities;
}

PromptCacheProfile HttpLLMProvider::prompt_cache_profile(
    std::string_view model) const noexcept {
    if (!protocol_ || looks_like_loopback_base_url(base_url_)) {
        return PromptCacheProfile::none();
    }
    return protocol_->prompt_cache_profile(model);
}

ProviderCapabilities HttpLLMProvider::capabilities() const {
    const bool is_ollama = protocol_ && protocol_->name() == "ollama";
    return ProviderCapabilities{
        .supports_tool_calls = true,
        .is_local = is_ollama && looks_like_loopback_base_url(base_url_),
        .supports_parallel_requests = true,
    };
}

std::shared_ptr<LLMProvider> HttpLLMProvider::fork_for_parallel_request() const {
    std::lock_guard lock(state_mutex_);
    if (!protocol_) return {};
    return std::make_shared<HttpLLMProvider>(
        base_url_,
        cred_source_,
        default_model_,
        protocol_->clone(),
        api_type_,
        provider_name_,
        client_identity_source_,
        model_catalog_selector_,
        service_id_,
        transport_options_);
}

void HttpLLMProvider::reset_conversation_state() {
    if (protocol_) {
        protocol_->reset_state();
    }
    websocket_.reset();
}

void HttpLLMProvider::stream_response(const ChatRequest&                      request,
                                      std::function<void(const StreamChunk&)> callback) {

    cancel_requested_.store(false, std::memory_order_release);

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
        const std::string effective_default_model = effective_request.model.empty()
            ? resolve_default_model()
            : default_model_;
        if (effective_request.model.empty() && !effective_default_model.empty()) {
            effective_request.model = effective_default_model;
        }

        const std::string metadata_model = normalize_metadata_model(
            effective_request.model.empty()
                ? std::string_view(effective_default_model)
                : std::string_view(effective_request.model),
            protocol_.get());
        const std::string continuation_service = provider_name_ + "|" + base_url_;
        for (auto& message : effective_request.messages) {
            std::erase_if(message.continuation_items, [&](const auto& item) {
                if (item.origin_model.empty() || item.origin_service.empty()) return !protocol_->accepts_legacy_continuation();
                return item.origin_model != metadata_model || item.origin_service != continuation_service;
            });
        }
        ensure_model_metadata(metadata_model);
        const auto metadata_info = resolved_model_info(metadata_model);
        if (const auto live = provider_model_info(metadata_model); live && live->context_window > 0) {
            effective_request.catalog_context_window = live->context_window;
        }
        // Model discovery is intentionally provider-scoped: a model ID exposed
        // by one configured endpoint must not overwrite the global registry
        // card for another endpoint. Copy its concrete output limit onto this
        // request instead, so serializers that require or honor max_tokens do
        // not fall back to their generic, often much smaller, default.
        if (!effective_request.max_tokens.has_value() && metadata_info.has_value()) {
            const int32_t max_output_tokens = metadata_info->max_output_tokens;
            if (max_output_tokens > 0) {
                effective_request.max_tokens = max_output_tokens;
            }
        }
        // Same reasoning as max_tokens: a model this endpoint serves but the
        // global registry has never seen still has to reach the serializer.
        const auto reasoning_profile = resolved_reasoning_profile(metadata_model);
        if (!effective_request.catalog_reasoning.has_value()
            && reasoning_profile.has_value()
            && (reasoning_profile->complete
                || reasoning_profile->catalog_default_effort
                       .has_value())) {
            effective_request.catalog_reasoning = reasoning_profile;
        }
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
            std::move(effective_request), effective_default_model, base_url_, cred_source_, *protocol_, cancel_requested_);
        if (cancel_requested_.load(std::memory_order_acquire)) {
            callback(StreamChunk::make_final());
            return;
        }
        {
            std::lock_guard lock(state_mutex_);
            last_model_ = prepared.model;
        }
    } catch (const core::auth::ReauthenticationRequired& error) {
        core::logging::warn(
            "[HTTP] OAuth session for '{}' requires sign-in: {}",
            error.provider_id(),
            error.what());
        callback(make_reauthentication_chunk(error, /*retry_safe=*/true));
        return;
    } catch (const std::exception& e) {
        core::logging::error("[HTTP] Failed to start request: {}", e.what());
        callback(StreamChunk::make_error(std::string("\n[Failed to start request: ") + e.what() + "]"));
        return;
    } catch (...) {
        core::logging::error("[HTTP] Failed to start request: unknown exception");
        callback(StreamChunk::make_error("\n[Failed to start request: unknown exception]"));
        return;
    }

    const std::string continuation_service = provider_name_ + "|" + base_url_;
    const std::string continuation_model = prepared.model;
    auto downstream = std::move(callback);
    callback = [downstream = std::move(downstream), continuation_service, continuation_model](const StreamChunk& chunk) {
        if (chunk.continuation_items.empty()) { downstream(chunk); return; }
        auto scoped = chunk;
        for (auto& item : scoped.continuation_items) {
            item.origin_model = continuation_model;
            item.origin_service = continuation_service;
        }
        downstream(scoped);
    };

    [self      = std::move(keepalive),
     url       = std::move(prepared.url),
     headers   = std::move(prepared.headers),
     payload   = std::move(prepared.payload),
     delimiter = std::move(prepared.delimiter),
     protocol  = std::move(prepared.protocol),
     request_metadata = std::move(prepared.request),
     auth = std::move(prepared.auth),
     callback] () mutable {

        const auto abandon_websocket = [&] {
            if (protocol->supports_websocket_transport()) {
                self->websocket_.client->reset();
                protocol->abandon_websocket_request(request_metadata);
            }
        };
        const auto read_auth = [&] {
            if (self->cred_source_) {
                auth = self->cred_source_->get_auth();
                request_metadata.auth_properties = auth.properties;
            }
        };
        const auto prepare_http_headers = [&] {
            headers = protocol->build_headers(auth);
            headers["Accept"] = "text/event-stream";
            protocol->prepare_headers(headers, request_metadata, self->base_url_);
            url = core::utils::uri::append_query_params(
                protocol->build_url(self->base_url_, request_metadata.model), auth.query_params);
        };
        try {
            if (self->cancel_requested_.load(std::memory_order_acquire)) {
                callback(StreamChunk::make_final());
                return;
            }

            // Both transports use the provider's bounded retry policy. Retrying
            // retracts provisional text, tools, and encrypted continuation state.
            const auto timeouts = self->transport_options_.timeouts == transport::StreamTimeoutPolicy{}
                ? protocol->stream_timeouts() : self->transport_options_.timeouts;
            const auto retries = self->transport_options_.retries == transport::RetryPolicy{}
                ? protocol->stream_retry_policy() : self->transport_options_.retries;
            bool attempted_auth_recovery = false;
            const auto emit_retry_reset = [&](const transport::RetrySchedule& retry,
                                              std::string_view reason) {
                callback(StreamChunk::make_attempt_reset(std::format(
                    "[{} retrying after {} ({}/{})]",
                    protocol->name(), reason, retry.attempt, retry.max_retries)));
            };

            if (self->websocket_.available() && protocol->supports_websocket_transport()) {
                transport::RetryController websocket_retries(retries);
                while (true) {
                    std::optional<PreparedWebSocketStreamRequest> ws;
                    try {
                        ws = prepare_websocket_stream_request(
                            *protocol, request_metadata, self->base_url_, auth,
                            *self->websocket_.client);
                    } catch (const std::exception& e) {
                        core::logging::debug("[WebSocket] Failed to prepare request: {}", e.what());
                        abandon_websocket();
                    }
                    if (!ws) break;
                    auto frame = std::move(ws->websocket_frame);
                    bool retry_connection = false;
                    while (!frame.payload.empty()) {
                        bool terminal_forwarded = false;
                        bool output_emitted = false;
                        bool stream_error = false;
                        bool retryable = false;
                        int32_t retry_after = 0;
                        std::string error_message;
                        self->set_last_usage(0, 0);
                        const auto result = self->websocket_.client->stream_text(
                            ws->websocket_url, ws->websocket_connection_key,
                            ws->websocket_headers, frame.payload,
                            [&](std::string_view event) {
                                auto parsed = protocol->parse_event(event);
                                if (parsed.stream_error) {
                                    stream_error = true;
                                    retryable = parsed.retryable_stream_error;
                                    retry_after = parsed.rate_limit.retry_after;
                                    error_message = std::move(parsed.stream_error_message);
                                    return true;
                                }
                                if (!frame.suppress_output) {
                                    if (parsed.prompt_tokens > 0 || parsed.completion_tokens > 0) {
                                        self->set_last_usage(token_usage_from(parsed));
                                    }
                                    for (const auto& chunk : parsed.chunks) {
                                        output_emitted |= !chunk.content.empty() || !chunk.reasoning_content.empty()
                                            || !chunk.tools.empty() || !chunk.continuation_items.empty();
                                        callback(chunk);
                                        terminal_forwarded |= chunk.is_final;
                                    }
                                    if (parsed.done && !terminal_forwarded) {
                                        callback(StreamChunk::make_final(parsed.stop_reason, parsed.incomplete_tool_call));
                                        terminal_forwarded = true;
                                    }
                                } else {
                                    for (const auto& chunk : parsed.chunks) {
                                        if (chunk.is_error) {
                                            stream_error = true;
                                            error_message = chunk.content;
                                        }
                                    }
                                }
                                return parsed.done || terminal_forwarded || stream_error;
                            }, &self->cancel_requested_, timeouts.inactivity);
                        protocol->observe_response_headers(result.response_headers, request_metadata);
                        if (self->cancel_requested_.load(std::memory_order_acquire)) {
                            abandon_websocket();
                            if (!terminal_forwarded) callback(StreamChunk::make_final());
                            return;
                        }
                        if (result.completed() && !stream_error) {
                            if (frame.suppress_output) {
                                frame = protocol->next_websocket_request_frame(request_metadata, frame);
                                continue;
                            }
                            const protocols::HttpResponse response{200, "", result.response_headers};
                            protocol->on_response(response);
                            self->set_last_rate_limit_info(protocol->last_rate_limit());
                            return;
                        }
                        abandon_websocket();
                        if (result.http_status == 401 && !attempted_auth_recovery && self->cred_source_) {
                            attempted_auth_recovery = true;
                            if (self->cred_source_->refresh_on_auth_failure()) {
                                read_auth();
                                protocol = protocol->clone();
                                retry_connection = true;
                                break;
                            }
                        }
                        if (error_message.empty()) error_message = result.message;
                        retry_after = std::max(retry_after, transport::retry_after_seconds(result.response_headers));
                        const bool veto = transport::retry_vetoed(result.response_headers);
                        if (veto || (stream_error && !retryable)) {
                            callback(StreamChunk::make_error("\n[WebSocket stream failed: " + error_message + "]"));
                            return;
                        }
                        // A failed warmup or unsupported handshake can go directly
                        // to HTTP. No generated model output has been committed.
                        if (!frame.suppress_output && result.request_sent) {
                            const auto retry = websocket_retries.schedule(
                                !veto && (!stream_error || retryable), output_emitted,
                                std::chrono::seconds(retry_after));
                            if (retry) {
                                emit_retry_reset(*retry, "a WebSocket failure");
                                if (!transport::wait_for_retry(retry->delay, self->cancel_requested_)) {
                                    callback(StreamChunk::make_final());
                                    return;
                                }
                                protocol = protocol->clone();
                                // Re-read credentials after backoff; a refreshed token
                                // must establish a new connection with full input.
                                read_auth();
                                retry_connection = true;
                                break;
                            }
                            if (output_emitted && retries.retry_only_before_output) {
                                callback(StreamChunk::make_error("\n[WebSocket stream failed: " + error_message + "]"));
                                return;
                            }
                            callback(StreamChunk::make_attempt_reset(
                                "[WebSocket retries exhausted; continuing over HTTPS]"));
                        }
                        // Handshake errors and warmups can carry server advice
                        // too. A transport switch must not bypass its deadline.
                        if (!transport::wait_for_retry(std::chrono::seconds(retry_after), self->cancel_requested_)) {
                            callback(StreamChunk::make_final());
                            return;
                        }
                        self->websocket_.disable();
                        protocol = protocol->clone();
                        break;
                    }
                    if (!retry_connection) break;
                }
                // HTTP always starts from complete history and current credentials.
                payload = protocol->serialize(request_metadata);
                prepare_http_headers();
            }

            if (payload.empty()) payload = protocol->serialize(request_metadata);
            transport::RetryController retry_controller(retries);
            // Cleared once the server rejects an encoded body, so it is resent plain.
            bool encode_body = true;

            const auto prepare_retry = [&](const transport::RetrySchedule& retry) {
                if (!transport::wait_for_retry(
                        retry.delay, self->cancel_requested_)) {
                    callback(StreamChunk::make_final());
                    return false;
                }
                protocol->reset_attempt_state();
                prepare_http_headers();
                return true;
            };

            while (true) {
                std::string buffer;
                std::size_t buffer_start = 0;
                bool        done_signalled = false;
                bool        stream_started = false;
                bool        stream_error_seen = false;
                bool        stream_error_retryable = false;
                int32_t     stream_retry_after = 0;
                std::string stream_error_type;
                std::string stream_error_message;
                bool        output_emitted = false;
                const bool requires_terminal_event =
                    protocol->requires_terminal_event();
                transport::StreamWatchdog watchdog(timeouts);

                // Only the wire bytes change; `payload` stays the retry source.
                const auto encoded_body = encode_body
                    ? protocol->encode_request_body(payload, url)
                    : std::nullopt;
                if (encoded_body) {
                    headers["Content-Encoding"] = encoded_body->content_encoding;
                } else {
                    headers.erase("Content-Encoding");
                }
                const std::string_view wire_body = encoded_body
                    ? std::string_view(encoded_body->bytes)
                    : std::string_view(payload);

                uint64_t response_bytes_received = 0;
                const uint64_t request_bytes_sent =
                    static_cast<uint64_t>(wire_body.size())
                    + core::net::estimated_http_header_bytes(headers);
                cpr::Header response_headers_seen;
                bool observed_transport_headers = false;
                int response_status = 0;

                // Prevent stale usage from previous requests if this request does not
                // emit a usage chunk.
                self->set_last_usage(0, 0);

                cpr::Session session;
                session.SetUrl(cpr::Url{url});
                session.SetHeader(headers);
                // A view, not a copy: payload outlives the session, and both
                // cpr::Body and curl's COPYPOSTFIELDS would duplicate a body
                // that grows with the whole conversation.
                session.SetBodyView(cpr::BodyView{wire_body});
                session.SetProgressCallback(cpr::ProgressCallback(
                    [cancel_requested = &self->cancel_requested_,
                     &watchdog](
                        cpr::cpr_pf_arg_t,
                        cpr::cpr_pf_arg_t,
                        cpr::cpr_pf_arg_t,
                        cpr::cpr_pf_arg_t,
                        intptr_t) -> bool {
                        if (cancel_requested->load(std::memory_order_acquire)) {
                            return false;
                        }
                        return watchdog.poll();
                    }));

                session.SetHeaderCallback(cpr::HeaderCallback(
                    [&protocol, &response_headers_seen,
                     &observed_transport_headers, &request_metadata,
                     &response_bytes_received, &watchdog, &response_status]
                    (std::string_view line, intptr_t /*userdata*/) -> bool {
                        if (!watchdog.observe_activity()) return false;
                        response_bytes_received += static_cast<uint64_t>(line.size());
                        if (const auto status = transport::parse_status_line(line)) {
                            response_status = *status;
                            response_headers_seen.clear();
                            observed_transport_headers = false;
                            return true;
                        }
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
                    if (response_status >= 400) return;
                    protocols::ParseResult result = protocol->parse_event(event_payload);

                    stream_started = stream_started || result.stream_started;
                    if (result.stream_error) {
                        stream_error_seen = true;
                        stream_error_retryable = result.retryable_stream_error;
                        stream_error_type = std::move(result.stream_error_type);
                        stream_error_message = std::move(result.stream_error_message);
                        stream_retry_after = result.rate_limit.retry_after;
                        done_signalled = true;
                        return;
                    }

                    // Forward all content/tool chunks produced by this event.
                    bool has_terminal_chunk = false;
                    for (auto& chunk : result.chunks) {
                        if (!chunk.content.empty()
                            || !chunk.reasoning_content.empty()
                            || !chunk.tools.empty()) {
                            output_emitted = true;
                        }
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
                        self->set_last_usage(token_usage_from(result));
                    }

                    if (result.done) {
                        done_signalled = true;
                        callback(StreamChunk::make_final(result.stop_reason, result.incomplete_tool_call));
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

                std::string response_body_capture;
                response_body_capture.reserve(1024);
                session.SetWriteCallback(cpr::WriteCallback([&buffer, &drain_complete_events,
                                                             &response_body_capture,
                                                             &response_bytes_received,
                                                             &stream_error_seen,
                                                             &watchdog,
                                                             cancel_requested = &self->cancel_requested_]
                                                            (std::string_view data,
                                                             intptr_t /*userdata*/) -> bool {
                    if (cancel_requested->load(std::memory_order_acquire)) {
                        return false;
                    }
                    if (!watchdog.observe_activity()) return false;
                    response_bytes_received += static_cast<uint64_t>(data.size());
                    if (response_body_capture.size() < kMaxResponseBodyCaptureBytes) {
                        const std::size_t remaining =
                            kMaxResponseBodyCaptureBytes - response_body_capture.size();
                        response_body_capture.append(data.substr(0, remaining));
                    }
                    buffer.append(data);
                    drain_complete_events();
                    return !stream_error_seen
                        && !cancel_requested->load(std::memory_order_acquire);
                }));

                cpr::Response r = session.Post();
                // CPR's custom header callback can leave Response::header empty.
                // Give protocol lifecycle/retry hooks the headers we collected.
                for (const auto& [key, value] : response_headers_seen) {
                    r.header.try_emplace(key, value);
                }
                core::net::NetworkTraffic traffic{
                    .bytes_sent = (r.status_code != 0
                                   || observed_transport_headers
                                   || response_bytes_received > 0)
                        ? request_bytes_sent
                        : 0,
                    .bytes_received = response_bytes_received,
                };
                core::net::NetworkTrafficStats::get_instance().record(traffic);
                if (self->cancel_requested_.load(std::memory_order_acquire)) {
                    if (!done_signalled) {
                        callback(StreamChunk::make_final());
                    }
                    break;
                }
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
                const std::string_view response_body = r.text.empty()
                    ? std::string_view(response_body_capture)
                    : std::string_view(r.text);
                if (r.status_code != 200) {
                    core::logging::debug(
                        "[HTTP] Response status={}, body={}",
                        r.status_code,
                        response_body.substr(0, 500));
                }

                // Fire the response lifecycle hook.  Protocols use this to extract
                // rate-limit headers, update metrics, or prepare any per-response state.
                // The hook is a no-op for protocols that do not override it.
                // Preserve the historical empty body for successful streaming
                // responses; the bounded callback capture exists only to recover
                // diagnostic bodies that CPR otherwise drops on HTTP failures.
                const std::string_view lifecycle_body = r.status_code == 200
                    ? std::string_view(r.text)
                    : response_body;
                const protocols::HttpResponse http_resp{
                    static_cast<int>(r.status_code), lifecycle_body, r.header};
                protocol->observe_response_headers(r.header, request_metadata);
                protocol->on_response(http_resp);

                if (encoded_body && !output_emitted && protocol->rejects_encoded_body(http_resp)) {
                    core::logging::warn(
                        "[HTTP] {} rejected the {} request body (HTTP {}); resending it unencoded",
                        protocol->name(), encoded_body->content_encoding, r.status_code);
                    encode_body = false;
                    continue;
                }

                const bool incomplete_required_stream =
                    requires_terminal_event
                    && r.status_code == 200
                    && !done_signalled
                    && !stream_error_seen;

                if (stream_error_seen || incomplete_required_stream) {
                    const bool retryable_stream_failure =
                        !transport::retry_vetoed(r.header)
                        && (incomplete_required_stream || stream_error_retryable);
                    const auto rate_limit_info = protocol->last_rate_limit();
                    const auto retry = retry_controller.schedule(
                        retryable_stream_failure,
                        output_emitted,
                        std::chrono::seconds(std::max(rate_limit_info.retry_after, stream_retry_after)));
                    if (retry.has_value()) {
                        core::logging::warn(
                            "[HTTP] Retrying {} stream failure type='{}' message='{}' attempt={}/{}",
                            protocol->name(),
                            incomplete_required_stream ? "incomplete_stream" : stream_error_type,
                            incomplete_required_stream
                                ? (stream_started
                                       ? "stream ended before terminal event"
                                       : "stream ended before start event")
                                : stream_error_message,
                            retry->attempt,
                            retry->max_retries);

                        emit_retry_reset(
                            *retry,
                            incomplete_required_stream
                                ? "an incomplete stream"
                                : "a generation failure");
                        if (!prepare_retry(*retry)) break;
                        continue;
                    }

                    std::string message;
                    if (incomplete_required_stream) {
                        message = std::format(
                            "\n[{} stream ended before the terminal event; response may be incomplete]",
                            protocol->name());
                    } else {
                        message = "\n[" + std::string(protocol->name()) + " stream error";
                        if (!stream_error_type.empty()) {
                            message += ": ";
                            message += stream_error_type;
                        }
                        if (!stream_error_message.empty()) {
                            message += " - ";
                            message += stream_error_message;
                        }
                        message += "]";
                    }
                    self->set_last_rate_limit_info(rate_limit_info);
                    callback(StreamChunk::make_error(std::move(message)));
                    break;
                }

                const bool transport_failed =
                    r.error.code != cpr::ErrorCode::OK && !done_signalled;
                const auto transport_retry = retry_controller.schedule(
                    transport_failed && !transport::retry_vetoed(r.header), output_emitted);
                if (transport_retry.has_value()) {
                    const std::string_view timeout_label =
                        transport::timeout_log_label(watchdog.timeout_kind());
                    const std::string_view failure = timeout_label.empty()
                        ? std::string_view(r.error.message)
                        : timeout_label;
                    core::logging::warn(
                        "[HTTP] Retrying transport failure '{}' attempt={}/{}",
                        failure,
                        transport_retry->attempt,
                        transport_retry->max_retries);
                    emit_retry_reset(*transport_retry, "a transport failure");
                    if (!prepare_retry(*transport_retry)) break;
                    continue;
                }

                // Attempt one forced credential refresh on auth failures (OAuth providers).
                const bool oauth_revoked_403 =
                    (r.status_code == 403
                     && response_body.find("OAuth token has been revoked")
                         != std::string_view::npos);
                if ((r.status_code == 401 || oauth_revoked_403)
                    && !attempted_auth_recovery
                    && self->cred_source_) {
                    attempted_auth_recovery = true;
                    try {
                        if (self->cred_source_->refresh_on_auth_failure()) {
                            read_auth();
                            protocol->reset_attempt_state();
                            payload = protocol->serialize(request_metadata);
                            prepare_http_headers();
                            if (self->cancel_requested_.load(std::memory_order_acquire)) {
                                callback(StreamChunk::make_final());
                                break;
                            }
                            core::logging::info(
                                "OAuth credentials refreshed after an authentication failure; "
                                "retrying the request");
                            continue;
                        }
                    } catch (const core::auth::ReauthenticationRequired& error) {
                        core::logging::warn(
                            "OAuth session for '{}' requires sign-in: {}",
                            error.provider_id(),
                            error.what());
                        callback(make_reauthentication_chunk(
                            error,
                            /*retry_safe=*/!stream_started));
                        break;
                    } catch (const std::exception& error) {
                        core::logging::warn(
                            "Credential refresh retry failed: {}",
                            error.what());
                    }
                }

                // Protocols own status-code semantics (including permanent
                // quota failures that deliberately bypass retries).
                const auto retry_rate_limit = protocol->last_rate_limit();
                const auto http_retry = retry_controller.schedule(
                    protocol->is_retryable(http_resp),
                    output_emitted,
                    std::chrono::seconds(retry_rate_limit.retry_after));
                if (http_retry.has_value()) {
                    emit_retry_reset(
                        *http_retry,
                        std::format("HTTP {}", r.status_code));
                    if (!prepare_retry(*http_retry)) break;
                    continue;
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

                if (transport_failed) {
                    const std::string_view timeout_message =
                        transport::timeout_user_message(watchdog.timeout_kind());
                    const std::string transport_message = timeout_message.empty()
                        ? r.error.message
                        : std::string(timeout_message);
                    core::logging::error(
                        "[HTTP] Connection error: {}", transport_message);
                    callback(StreamChunk::make_error(
                        "\n[Error connecting to " + url + ": "
                        + transport_message + "]"));
                } else if (r.status_code != 200) {
                    core::logging::error("[HTTP] Error status={}", r.status_code);
                    callback(StreamChunk::make_error(
                        "\n" + protocol->format_error_message(http_resp)));
                } else if (!done_signalled) {
                    callback(StreamChunk::make_final());
                }
                break;  // Exit retry loop on non-retryable response
            }
        } catch (const core::auth::ReauthenticationRequired& error) {
            abandon_websocket();
            callback(make_reauthentication_chunk(error, /*retry_safe=*/true));
        } catch (const std::exception& e) {
            abandon_websocket();
            core::logging::error("[HTTP] Unhandled streaming exception: {}", e.what());
            callback(StreamChunk::make_error(std::string("\n[Internal streaming error: ") + e.what() + "]"));
        } catch (...) {
            abandon_websocket();
            core::logging::error("[HTTP] Unhandled streaming exception: unknown exception");
            callback(StreamChunk::make_error("\n[Internal streaming error: unknown exception]"));
        }
    }();
}

} // namespace core::llm
