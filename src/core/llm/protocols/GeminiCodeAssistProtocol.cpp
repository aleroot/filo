#include "GeminiCodeAssistProtocol.hpp"
#include "GeminiProtocol.hpp"
#include "SseUtils.hpp"
#include "../../utils/JsonUtils.hpp"
#include <atomic>
#include <simdjson.h>
#include <stdexcept>

namespace core::llm::protocols {

namespace {

[[nodiscard]] std::pair<std::string, std::vector<ToolCall>>
parse_code_assist_response(simdjson::dom::element response_el) {
    std::string           content_str;
    std::vector<ToolCall> tools;

    simdjson::dom::array candidates;
    if (response_el["candidates"].get(candidates) != simdjson::SUCCESS) return {};

    for (auto candidate : candidates) {
        simdjson::dom::object content_obj;
        if (candidate["content"].get(content_obj) != simdjson::SUCCESS) continue;

        simdjson::dom::array parts;
        if (content_obj["parts"].get(parts) != simdjson::SUCCESS) continue;

        for (auto part : parts) {
            std::string_view text_v;
            if (part["text"].get(text_v) == simdjson::SUCCESS) {
                content_str += std::string(text_v);
                continue;
            }

            simdjson::dom::object function_call;
            if (part["functionCall"].get(function_call) == simdjson::SUCCESS) {
                ToolCall call;
                static std::atomic_uint64_t call_counter{0};
                call.id   = "call_gemini_" + std::to_string(++call_counter);
                call.type = "function";

                std::string_view name_v;
                if (function_call["name"].get(name_v) == simdjson::SUCCESS) {
                    call.function.name = std::string(name_v);
                }

                simdjson::dom::element args_el;
                if (function_call["args"].get(args_el) == simdjson::SUCCESS) {
                    call.function.arguments = simdjson::minify(args_el);
                }
                tools.push_back(std::move(call));
            }
        }
    }

    return {std::move(content_str), std::move(tools)};
}

} // namespace

std::string serialize_gemini_code_assist_request(const ChatRequest& req,
                                                 const std::string& default_model) {
    const std::string model = normalize_requested_gemini_model(
        req.model.empty() ? std::string_view(default_model) : std::string_view(req.model));
    auto project_it = req.auth_properties.find("project_id");

    std::string payload = "{";
    payload += R"("model":")";
    payload += core::utils::escape_json_string(model);
    if (project_it != req.auth_properties.end() && !project_it->second.empty()) {
        payload += R"(","project":")";
        payload += core::utils::escape_json_string(project_it->second);
        payload += '"';
    }
    payload += R"(,"request":)";
    payload += serialize_gemini_request(req, default_model);
    payload += '}';
    return payload;
}

void GeminiCodeAssistProtocol::prepare_request(ChatRequest& request) {
    if (request.model.empty()) return;
    request.model = normalize_requested_gemini_model(request.model);
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
    auto [content, tools] = parse_code_assist_response(response_el);
    if (!content.empty() || !tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(content);
        chunk.tools   = std::move(tools);
        result.chunks.push_back(std::move(chunk));
    }

    result.rate_limit = {};
    result.prompt_tokens = 0;
    result.completion_tokens = 0;

    const GeminiUsageMetadata usage =
        extract_gemini_usage_metadata(simdjson::minify(response_el));
    result.prompt_tokens = usage.prompt_tokens;
    result.completion_tokens = usage.completion_tokens;
    return result;
}

} // namespace core::llm::protocols
