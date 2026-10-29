#include "MiniMaxProtocol.hpp"
#include "MiniMaxImage.hpp"
#include "SseUtils.hpp"
#include "../MiniMaxModelTraits.hpp"
#include "../catalog/MiniMaxModelCatalogProvider.hpp"
#include "../ReasoningEffortResolution.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../providers/MiniMaxRequestBudget.hpp"
#include "core/auth/MiniMaxClientIdentity.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <stdexcept>
#include <charconv>
#include <limits>

namespace core::llm::protocols {
namespace {
struct MiniMaxError {
    int code = 0;
    std::string message;
};

int positive_code(std::string_view text) {
    int code = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), code);
    return ec == std::errc{} && end == text.data() + text.size() && code > 0 ? code : 0;
}

MiniMaxError parse_error(std::string_view body) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    MiniMaxError error;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return error;
    simdjson::dom::element node;
    if (doc["error"].get(node) != simdjson::SUCCESS) node = doc;
    const auto code_from = [](simdjson::dom::element node, std::string_view key) {
        int64_t numeric = 0;
        if (node[key].get(numeric) == simdjson::SUCCESS) {
            return numeric > 0 && numeric <= std::numeric_limits<int>::max() ? static_cast<int>(numeric) : 0;
        }
        std::string_view text;
        if (node[key].get(text) == simdjson::SUCCESS) {
            return positive_code(text);
        }
        return 0;
    };
    error.code = code_from(node, "code");
    std::string_view message;
    if (node["message"].get(message) == simdjson::SUCCESS) error.message = message;
    simdjson::dom::element base;
    if (doc["base_resp"].get(base) == simdjson::SUCCESS) {
        if (const int code = code_from(base, "status_code"); code > 0) {
            error.code = code;
            if (base["status_msg"].get(message) == simdjson::SUCCESS) error.message = message;
        }
    }
    // Messages errors sometimes embed the business code in the message.
    if (!error.code && error.message.ends_with(')')) {
        const auto open = error.message.rfind('(');
        if (open != std::string::npos) {
            error.code = positive_code(std::string_view(error.message).substr(
                open + 1, error.message.size() - open - 2));
        }
    }
    return error;
}
bool transient(int code) {
    return code == 1000 || code == 1001 || code == 1002 || code == 1024
        || code == 1033 || code == 1041 || code == 1045 || code == 2045
        || code == 2046 || code == 2047 || code == 2062 || code == 2064 || code == 2066;
}
std::string_view guidance(int code, int http_status) {
    switch (code) {
    case 1004: case 2049: return "Check your MiniMax key and account region.";
    case 2056: case 2067: return "M Plan usage exhausted; wait for reset or manage your plan/credits.";
    case 2057: return "Use your M Plan Subscription Key, not a pay-as-you-go API key.";
    case 2061: case 2068: return "This model is not included in your plan.";
    case 2062: return "M Plan request rate limited; reduce request frequency or concurrency.";
    case 2069: return "This model was retired; select an available MiniMax model.";
    case 2022: return "This model does not exist or is not available to this key.";
    case 1008: case 2053: return "MiniMax balance or credits are insufficient.";
    case 1026: return "MiniMax's content filter rejected the input; rephrase the request.";
    case 1027: return "MiniMax's content filter rejected the output; rephrase the request.";
    case 1039: return "MiniMax token limit reached; retry shortly or compact the conversation.";
    case 1042: return "The input contains too many invisible or illegal characters.";
    case 1046: return "MiniMax rejected an identity claim in the prompt.";
    case 2013: case 2014: return "MiniMax rejected the request parameters.";
    default: break;
    }
    if (http_status == 401 || http_status == 403) {
        return "MiniMax rejected the key; run `filo --auth minimax` to replace it.";
    }
    return {};
}

/// Only an effort the model accepts reaches the wire, and a short output
/// budget lowers it so thinking cannot consume the whole answer (see
/// minimax_effort_ceiling).
std::string budgeted_effort(const std::vector<std::string>& supported, std::string effort,
                            std::optional<int> max_tokens) {
    if (std::ranges::find(supported, effort) == supported.end()) effort = "max";
    if (!max_tokens) return effort;
    const auto ceiling = minimax_effort_ceiling(*max_tokens);
    const auto rank = [&](std::string_view value) {
        return std::ranges::find(supported, value) - supported.begin();
    };
    return !ceiling.empty() && rank(ceiling) < rank(effort) ? std::string(ceiling) : effort;
}

} // namespace

int minimax_error_code(std::string_view body) {
    try { return parse_error(body).code; } catch (...) { return 0; }
}

std::string describe_minimax_error(int http_status, std::string_view body) {
    MiniMaxError error;
    try { error = parse_error(body); } catch (...) {}
    std::string message = "MiniMax error " + std::to_string(error.code ? error.code : http_status);
    if (const auto hint = guidance(error.code, http_status); !hint.empty()) message += ": " + std::string(hint);
    if (!error.message.empty()) message += " " + error.message;
    return message;
}

MiniMaxProtocol::MiniMaxProtocol(bool subscription, std::string default_effort,
                               std::shared_ptr<IMiniMaxUsageSource> usage, int context_window)
    : AnthropicProtocol(kMiniMaxOutputLimit), context_window_(context_window), subscription_(subscription),
      default_effort_(std::move(default_effort)),
      usage_(usage ? std::move(usage) : make_minimax_usage_source()) {
    if (context_window_ != kMiniMaxContextWindow && context_window_ != kMiniMaxConservativeContextWindow) {
        throw std::invalid_argument("MiniMax context_window must be 512000 or 1000000.");
    }
}

std::string MiniMaxProtocol::model_id(std::string_view model) const { return canonical_minimax_model(model); }

void MiniMaxProtocol::prepare_request(ChatRequest& req) {
    req.model = model_id(req.model);
    const auto info = minimax_model_info(req.model);
    if (!info || (subscription_ && !is_minimax_plan_model(req.model))) {
        throw std::invalid_argument("Select a supported MiniMax model for this provider.");
    }
    if (req.effort.empty()) req.effort = default_effort_;
    // JSON-object callers (reader, review) instruct and validate JSON
    // themselves, as with every other Messages endpoint. A schema is a
    // guarantee MiniMax does not document, so it is refused rather than faked.
    if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
        throw std::invalid_argument("MiniMax does not support JSON-schema output; use JSON-object output and validate it.");
    }
    prepare_minimax_images(req);
    req.max_tokens = std::clamp(req.max_tokens.value_or(kMiniMaxOutputLimit), 1, kMiniMaxOutputLimit);
}

void MiniMaxProtocol::prepare_context(ChatRequest& req, std::string_view base_url,
                                     const core::auth::AuthInfo& auth,
                                     const std::function<bool()>& cancelled,
                                     const ProtocolRequester& requester) {
    // Exact counting pays for its extra round trip only for large histories.
    // The request goes to the same configured endpoint, with no redirect or
    // cross-origin fallback. Failure retains the conservative local budget.
    if (cancelled && cancelled()) return;
    const int context_window = req.catalog_context_window && *req.catalog_context_window > 0
        ? std::min(context_window_, *req.catalog_context_window) : context_window_;
    std::size_t counted = 0;
    if (providers::MiniMaxRequestBudget::estimate_request(req) >= static_cast<std::size_t>(context_window / 3)) {
        try {
            const auto payload = serialize(req);
            simdjson::dom::parser parser;
            simdjson::padded_string padded(payload);
            simdjson::dom::object document;
            if (parser.parse(padded).get_object().get(document) != simdjson::SUCCESS) throw std::runtime_error("Invalid token-count request");
            std::string body = "{";
            for (const auto field : document) {
                if (field.key != "model" && field.key != "messages" && field.key != "system" && field.key != "tools") continue;
                if (body.size() > 1) body += ',';
                body += '"' + std::string(field.key) + "\":" + simdjson::to_string(field.value);
            }
            body += '}';
            auto headers = build_headers(auth);
            prepare_headers(headers, req, base_url);
            // The provider performs the request, so the count is decided like
            // every other request this call makes. Without one the local
            // estimate below is used.
            if (requester) {
                const auto response = requester(protocols::ProtocolRequest{
                    .url = build_url(base_url, req.model) + "/count_tokens",
                    .headers = headers,
                    .body = std::move(body),
                    .timeout = std::chrono::milliseconds{1500},
                    .connect_timeout = std::chrono::milliseconds{500},
                    .max_response_bytes = 16 * 1024,
                });
                if (response.status_code == 200 && response.error.empty()) {
                    simdjson::padded_string result(response.body);
                    simdjson::dom::element value;
                    int64_t tokens = 0;
                    if (parser.parse(result).get(value) == simdjson::SUCCESS
                        && value["input_tokens"].get(tokens) == simdjson::SUCCESS && tokens > 0) counted = static_cast<std::size_t>(tokens);
                }
            }
        } catch (...) { /* Optional estimation must not make the model unavailable. */ }
    }
    if (cancelled && cancelled()) return;
    // Apply the budget once: exact counts must be able to admit histories that
    // the fallback estimate overstates and restore the requested output room.
    providers::MiniMaxRequestBudget::prepare(req, context_window, counted);
}

std::string MiniMaxProtocol::serialize(const ChatRequest& req) const {
    AnthropicSerializationOptions options;
    options.image_urls = true;
    options.extra_fields = [&](std::string& payload, const ChatRequest& request) {
        const auto info = minimax_model_info(request.model);
        if (!info) throw std::invalid_argument("Unknown MiniMax model policy.");
        if (info->wire.thinking_always_on) {
            const auto configured = core::utils::str::to_lower_ascii_copy(
                core::utils::str::trim_ascii_copy(request.effort.empty() ? default_effort_ : request.effort));
            const auto effort = resolve_reasoning_effort(configured, info->reasoning.effort);
            const auto effective = budgeted_effort(
                info->reasoning.supported_efforts, effort.effective, request.max_tokens);
            payload += R"(,"thinking":{"type":"adaptive"},"output_config":{"effort":")";
            payload += core::utils::escape_json_string(effective);
            payload += R"("})";
        } else {
            const auto effort = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_copy(request.effort));
            const bool requested = !effort.empty() && effort != "auto" && effort != "default"
                && effort != "unset" && effort != "off" && effort != "none" && effort != "disabled";
            // Optional thinking is dropped where even the shallowest effort
            // could exhaust the budget before any answer text.
            const bool fits = !request.max_tokens
                || minimax_effort_ceiling(*request.max_tokens) != "low";
            if (requested && fits) payload += R"(,"thinking":{"type":"adaptive"})";
        }
    };
    auto payload = AnthropicSerializer::serialize(req, kMiniMaxOutputLimit, options);
    if (payload.size() > 64 * 1024 * 1024) throw std::invalid_argument("MiniMax request exceeds the 64 MiB body limit.");
    return payload;
}

cpr::Header MiniMaxProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    auto headers = AnthropicProtocol::build_headers(auth);
    const auto key = transport::find_header(headers, "x-api-key");
    if (key && ((subscription_ && key->starts_with("sk-api-")) || (!subscription_ && key->starts_with("sk-cp-")))) {
        throw std::invalid_argument("MiniMax key type does not match this provider; select M Plan or API access explicitly.");
    }
    core::auth::minimax_code::apply_identity_headers(headers, {});
    return headers;
}

void MiniMaxProtocol::prepare_headers(cpr::Header& headers, const ChatRequest& request,
                                      std::string_view) {
    if (!request.session_id.empty()) {
        headers[std::string(core::auth::minimax_code::kSessionIdHeader)] = request.session_id;
    }
}

transport::RetryPolicy MiniMaxProtocol::stream_retry_policy() const noexcept {
    // Plan rate limits (2062) and cluster load (2064) clear in seconds, not on
    // the generic 0.5 s schedule.
    transport::RetryPolicy policy;
    policy.max_retries = 5;
    policy.initial_backoff = std::chrono::seconds(1);
    policy.maximum_backoff = std::chrono::seconds(30);
    return policy;
}
ReasoningCapabilities MiniMaxProtocol::reasoning_capabilities(std::string_view model) const noexcept {
    try { if (auto info = minimax_model_info(model)) return info->reasoning.effort; } catch (...) {}
    return {};
}
std::unique_ptr<ModelCatalogProvider> MiniMaxProtocol::model_catalog(std::string_view provider) const {
    return std::make_unique<MiniMaxModelCatalogProvider>(std::string(provider), subscription_, context_window_);
}
ParseResult MiniMaxProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView event;
    if (sse::parse_event_payload(raw_event, event)
        && (event.event == "error" || event.data.find("\"error\"") != std::string_view::npos
            || event.data.find("\"base_resp\"") != std::string_view::npos)) {
        const auto error = parse_error(event.data);
        if (error.code) {
            ParseResult failed;
            failed.stream_error = true;
            failed.retryable_stream_error = transient(error.code);
            failed.stream_error_message = format_error_message({400, event.data, {}});
            return failed;
        }
    }
    return AnthropicProtocol::parse_event(raw_event);
}
std::unique_ptr<ApiProtocolBase> MiniMaxProtocol::clone() const {
    return std::make_unique<MiniMaxProtocol>(subscription_, default_effort_, usage_, context_window_);
}
void MiniMaxProtocol::on_response(const HttpResponse& response) {
    rate_limit_ = {};
    rate_limit_.is_rate_limited = response.status_code == 429;
    rate_limit_.retry_after = transport::retry_after_seconds(response.headers);
}
void MiniMaxProtocol::enrich_rate_limit(std::string_view base_url, const cpr::Header& headers,
                                      const HttpResponse&) {
    if (subscription_ && usage_) {
        try { if (auto snapshot = usage_->read(base_url, headers)) {
            snapshot->is_rate_limited |= rate_limit_.is_rate_limited;
            snapshot->retry_after = rate_limit_.retry_after;
            rate_limit_ = std::move(*snapshot);
        } } catch (...) {}
    }
}
std::string MiniMaxProtocol::format_error_message(const HttpResponse& response) const {
    return describe_minimax_error(response.status_code, response.body);
}
bool MiniMaxProtocol::is_retryable(const HttpResponse& response) const noexcept {
    if (transport::retry_vetoed(response.headers)) return false;
    try { const auto error = parse_error(response.body); if (error.code) return transient(error.code); } catch (...) { return false; }
    return response.status_code == 429 || response.status_code == 500 || response.status_code == 502
        || response.status_code == 503 || response.status_code == 504 || response.status_code == 529;
}
void MiniMaxProtocol::reset_state() { AnthropicProtocol::reset_state(); rate_limit_ = {}; }
} // namespace core::llm::protocols
