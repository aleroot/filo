#include "MimoProtocol.hpp"

#include "../MimoModelTraits.hpp"
#include "../StrictToolPolicy.hpp"
#include "core/version/Version.hpp"
#include "../../utils/JsonUtils.hpp"

#include <simdjson.h>

namespace core::llm::protocols {
namespace {

/// Gateway error payload: `{"error":{"code":…,"message":…,"param":…}}`.
struct MimoGatewayError {
    std::string code;
    std::string message;
    std::string param;

    [[nodiscard]] bool empty() const noexcept {
        return code.empty() && message.empty() && param.empty();
    }
};

[[nodiscard]] std::string field_to_string(simdjson::dom::element element) {
    std::string_view text;
    if (element.get(text) == simdjson::SUCCESS) return std::string(text);

    std::int64_t number = 0;
    if (element.get(number) == simdjson::SUCCESS) {
        return std::to_string(number);
    }
    return {};
}

[[nodiscard]] MimoGatewayError parse_gateway_error(std::string_view body) {
    MimoGatewayError parsed;
    if (body.empty()) return parsed;

    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string padded(body);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return parsed;

    simdjson::dom::element error;
    if (document["error"].get(error) != simdjson::SUCCESS) {
        // Some gateway responses put the message at the top level.
        if (simdjson::dom::element message;
            document["message"].get(message) == simdjson::SUCCESS) {
            parsed.message = field_to_string(message);
        }
        return parsed;
    }

    if (simdjson::dom::element code; error["code"].get(code) == simdjson::SUCCESS) {
        parsed.code = field_to_string(code);
    }
    if (simdjson::dom::element message;
        error["message"].get(message) == simdjson::SUCCESS) {
        parsed.message = field_to_string(message);
    }
    if (simdjson::dom::element param;
        error["param"].get(param) == simdjson::SUCCESS) {
        parsed.param = field_to_string(param);
    }
    return parsed;
}

/**
 * Compose the human-readable half of a gateway error.
 *
 * The gateway leaves `error.message` generic and puts the actionable reason in
 * `error.param`, so both are surfaced when they differ.
 */
[[nodiscard]] std::string describe(const MimoGatewayError& error) {
    std::string base(mimo_gateway_code_label(error.code));
    if (base.empty()) base = error.message;
    if (base.empty()) return error.param;
    if (!error.param.empty() && error.param != base) {
        return base + ": " + error.param;
    }
    return base;
}

[[nodiscard]] ChatRequest apply_mimo_defaults(ChatRequest req) {
    // MiMo's reference client pins every MiMo model to temperature 1; the
    // family is tuned for it. An explicitly configured value still wins so a
    // deliberate provider override is not silently discarded.
    if (is_mimo_model(req.model) && !req.temperature.has_value()) {
        req.temperature = 1.0F;
    }
    return req;
}

} // namespace

std::string MimoProtocol::serialize(const ChatRequest& req) const {
    const ChatRequest adjusted = apply_mimo_defaults(req);

    Serializer::Options options;
    options.strict_tools = configured_strict_tool_dialect(
        ToolSchemaWire::OpenAI, adjusted.model);
    // MiMo models are interleaved-reasoning models: the reference client
    // replays `reasoning_content` on every assistant message, present even
    // when empty. Omitting it on a follow-up turn is what a tool loop does,
    // so this is load-bearing for multi-step agent work.
    options.reasoning_content_policy =
        Serializer::ReasoningContentPolicy::EveryAssistant;
    options.reasoning_protocol = std::string(name());

    std::string payload = Serializer::serialize(adjusted, options);
    if (payload.ends_with('}')) {
        payload.pop_back();
        append_extra_fields(payload, adjusted);
        if ((stream_usage_ || adjusted.stream_include_usage) && adjusted.stream) {
            payload += R"(,"stream_options":{"include_usage":true})";
        }
        payload += '}';
    }
    return payload;
}

ReasoningCapabilities MimoProtocol::reasoning_capabilities(
    std::string_view /*model*/) const noexcept {
    // Thinking is always on and has no request-side control.
    return {};
}

cpr::Header MimoProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    cpr::Header headers = OpenAIProtocol::build_headers(auth);
    // The MiMo gateway attributes Token Plan traffic by client source, the
    // same value the reference MiMo Code client sends.
    headers["X-Mimo-Source"] = "mimocode-cli";
    headers["User-Agent"] = std::string(core::version::user_agent);
    return headers;
}

std::string MimoProtocol::format_error_message(
    const HttpResponse& response) const {
    const MimoGatewayError error = parse_gateway_error(response.body);
    const std::string detail = error.empty() ? std::string() : describe(error);

    switch (response.status_code) {
        case 400:
            // Moderation and risk-control blocks are reported as plain 400s;
            // the code is the only thing that tells them apart.
            if (!mimo_gateway_code_label(error.code).empty()) {
                return "[MiMo API Error 400: " + detail + "]";
            }
            break;
        case 401:
            return "[MiMo API Error 401: Authentication failed. "
                   "Check MIMO_TOKEN_PLAN_API_KEY or XIAOMI_API_KEY, or run "
                   "`filo --auth xiaomi`.]";
        case 402:
            return "[MiMo API Error 402: Payment required. "
                   + (detail.empty()
                          ? std::string("Your Token Plan quota may be exhausted; "
                                        "check https://platform.xiaomimimo.com.")
                          : detail)
                   + "]";
        case 403:
            return "[MiMo API Error 403: Permission denied. "
                   + (detail.empty()
                          ? std::string("This key may not have access to the "
                                        "requested model or regional gateway.")
                          : detail)
                   + "]";
        case 429:
            return "[MiMo API Error 429: Rate limited or Token Plan quota "
                   "exhausted. "
                   + (detail.empty()
                          ? std::string("Wait and retry, or review plan usage at "
                                        "https://platform.xiaomimimo.com.")
                          : detail)
                   + "]";
        default:
            break;
    }

    if (!detail.empty()) {
        return "[MiMo API Error " + std::to_string(response.status_code)
               + ": " + detail + "]";
    }
    return OpenAIProtocol::format_error_message(response);
}

} // namespace core::llm::protocols
