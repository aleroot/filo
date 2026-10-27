#include "GeminiCodeAssistProtocol.hpp"
#include "GeminiProtocol.hpp"
#include "SseUtils.hpp"
#include "../../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <stdexcept>

namespace core::llm::protocols {

std::string serialize_gemini_code_assist_request(const ChatRequest& req,
                                                 const std::string& default_model,
                                                 const GeminiRequestExtras* extras) {
    const std::string model = normalize_requested_gemini_model(
        req.model.empty() ? std::string_view(default_model) : std::string_view(req.model));
    auto project_it = req.auth_properties.find("project_id");

    std::string payload = "{";
    payload += R"("model":")";
    payload += core::utils::escape_json_string(model);
    payload += '"';
    if (project_it != req.auth_properties.end() && !project_it->second.empty()) {
        payload += R"(,"project":")";
        payload += core::utils::escape_json_string(project_it->second);
        payload += '"';
    }
    payload += R"(,"request":)";
    payload += serialize_gemini_request(req, default_model, extras);
    payload += '}';
    return payload;
}

void GeminiCodeAssistProtocol::prepare_request(ChatRequest& request) {
    if (request.model.empty()) return;
    request.model = model_id(request.model);
}

std::string GeminiCodeAssistProtocol::model_id(std::string_view model) const {
    return normalize_requested_gemini_model(model);
}

std::string GeminiCodeAssistProtocol::serialize(const ChatRequest& req) const {
    return serialize_gemini_code_assist_request(req, req.model);
}

cpr::Header GeminiCodeAssistProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{{"Content-Type", "application/json"}};
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string GeminiCodeAssistProtocol::build_url(std::string_view base_url,
                                                std::string_view /*model*/) const {
    return std::string(base_url) + "/v1internal:streamGenerateContent?alt=sse";
}

ParseResult GeminiCodeAssistProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};

    if (parsed.is_done) {
        ParseResult result;
        result.done = true;
        return result;
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(parsed.data.data(), parsed.data.size());
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return {};

    simdjson::dom::element response_el;
    if (doc["response"].get(response_el) != simdjson::SUCCESS) return {};

    ParseResult result;
    result.rate_limit = {};

    std::string_view response_id;
    if (response_el["responseId"].get(response_id) == simdjson::SUCCESS) {
        result.response_id = std::string(response_id);
    }

    // The wrapped object is the same Gemini response shape, so it shares the
    // plain-Gemini parser (thought summaries, complete tool calls, usage).
    const std::string response_json = simdjson::minify(response_el);
    auto parts = parse_gemini_sse_chunk(response_json);
    const GeminiUsageMetadata usage = extract_gemini_usage_metadata(response_json);
    result.prompt_tokens = usage.prompt_tokens;
    result.completion_tokens = usage.completion_tokens;
    result.cached_prompt_tokens = usage.cached_prompt_tokens;

    if (!parts.content.empty() || !parts.reasoning.empty() || !parts.tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(parts.content);
        if (!parts.reasoning.empty()) {
            chunk.reasoning_content = std::move(parts.reasoning);
            chunk.reasoning_protocol = "gemini";
        }
        chunk.tools = std::move(parts.tools);
        result.chunks.push_back(std::move(chunk));
    }
    return result;
}

} // namespace core::llm::protocols
