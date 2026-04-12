#include "GeminiProtocol.hpp"
#include "SseUtils.hpp"
#include "../Models.hpp"
#include "../../utils/JsonUtils.hpp"
#include <simdjson.h>

namespace core::llm::protocols {

// ─────────────────────────────────────────────────────────────────────────────
// serialize_gemini_request
// ─────────────────────────────────────────────────────────────────────────────

std::string serialize_gemini_request(const ChatRequest& req,
                                      const std::string& /*default_model*/) {
    std::string payload;
    payload.reserve(8192);
    payload += "{";

    // System instruction (top-level, not a content turn).
    std::string system_prompt;
    std::vector<Message> filtered_messages;
    for (const auto& msg : req.messages) {
        if (msg.role == "system") {
            system_prompt += msg.content + "\n";
        } else {
            filtered_messages.push_back(msg);
        }
    }

    if (!system_prompt.empty()) {
        payload += R"("systemInstruction":{"parts":[{"text":")" +
                   core::utils::escape_json_string(system_prompt) +
                   R"("}]},)";
    }

    // Tools (Gemini uses functionDeclarations with uppercase type names).
    if (!req.tools.empty()) {
        payload += R"("tools":[{"functionDeclarations":[)";
        for (size_t i = 0; i < req.tools.size(); ++i) {
            const auto& def = req.tools[i].function;
            payload += R"({"name":")";
            payload += core::utils::escape_json_string(def.name);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(def.description);
            payload += R"(","parameters":{"type":"OBJECT","properties":{)";

            for (size_t j = 0; j < def.parameters.size(); ++j) {
                const auto& p = def.parameters[j];
                payload += '"';
                payload += core::utils::escape_json_string(p.name);
                payload += R"(":{"type":")";
                payload += core::utils::escape_json_string(p.type);
                payload += R"(","description":")";
                payload += core::utils::escape_json_string(p.description);
                payload += "\"}";
                if (j < def.parameters.size() - 1) payload += ',';
            }

            payload += R"(},"required":[)";
            bool first_req = true;
            for (const auto& p : def.parameters) {
                if (p.required) {
                    if (!first_req) payload += ',';
                    payload += '"';
                    payload += core::utils::escape_json_string(p.name);
                    payload += '"';
                    first_req = false;
                }
            }
            payload += "]}}";
            if (i < req.tools.size() - 1) payload += ',';
        }
        payload += "]}],";
    }

    // generationConfig (temperature, maxOutputTokens).
    payload += R"("generationConfig":{)";
    bool has_config = false;
    if (req.temperature.has_value()) {
        payload += R"("temperature":)" + std::to_string(req.temperature.value());
        has_config = true;
    }
    if (req.max_tokens.has_value()) {
        if (has_config) payload += ',';
        payload += R"("maxOutputTokens":)" + std::to_string(req.max_tokens.value());
    }
    payload += "},";

    // Contents (the conversation).
    payload += R"("contents":[)";
    for (size_t i = 0; i < filtered_messages.size(); ++i) {
        const auto& msg = filtered_messages[i];
        std::string role = msg.role;
        if (role == "assistant") role = "model";
        if (role == "tool")      role = "function";

        payload += R"({"role":")";
        payload += core::utils::escape_json_string(role);
        payload += R"(","parts":[)";

        if (role == "function") {
            payload += R"({"functionResponse":{"name":")";
            payload += core::utils::escape_json_string(msg.name);
            payload += R"(","response":{"result":)";
            payload += msg.content;
            payload += "}}}";
        } else if (!msg.tool_calls.empty()) {
            for (size_t j = 0; j < msg.tool_calls.size(); ++j) {
                const auto& tc = msg.tool_calls[j];
                std::string args = tc.function.arguments.empty() ? "{}" : tc.function.arguments;
                payload += R"({"functionCall":{"name":")";
                payload += core::utils::escape_json_string(tc.function.name);
                payload += R"(","args":)";
                payload += args;
                payload += "}}";
                if (j < msg.tool_calls.size() - 1) payload += ',';
            }
        } else if (!msg.content_parts.empty()) {
            bool first_part = true;
            for (const auto& part : msg.content_parts) {
                if (part.type == ContentPartType::Text) {
                    if (part.text.empty()) continue;
                    if (!first_part) payload += ',';
                    payload += R"({"text":")";
                    payload += core::utils::escape_json_string(part.text);
                    payload += R"("})";
                    first_part = false;
                    continue;
                }

                if (!first_part) payload += ',';
                if (const auto encoded = encode_image_part(part); encoded.has_value()) {
                    payload += R"({"inlineData":{"mimeType":")";
                    payload += core::utils::escape_json_string(encoded->mime_type);
                    payload += R"(","data":")";
                    payload += core::utils::escape_json_string(encoded->base64_data);
                    payload += R"("}})";
                } else {
                    payload += R"({"text":")";
                    payload += core::utils::escape_json_string(
                        unavailable_image_attachment_text(part.path));
                    payload += R"("})";
                }
                first_part = false;
            }
        } else {
            payload += R"({"text":")";
            payload += core::utils::escape_json_string(msg.content);
            payload += "\"}";
        }

        payload += "]}";
        if (i < filtered_messages.size() - 1) payload += ',';
    }
    payload += "]}";
    return payload;
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_gemini_sse_chunk
// ─────────────────────────────────────────────────────────────────────────────

std::pair<std::string, std::vector<ToolCall>>
parse_gemini_sse_chunk(std::string_view json_sv) {
    if (json_sv.empty()) return {};

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_sv);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return {};

    std::string           content_str;
    std::vector<ToolCall> tools;

    simdjson::ondemand::array candidates;
    if (doc["candidates"].get_array().get(candidates) != simdjson::SUCCESS) return {};

    for (auto candidate : candidates) {
        simdjson::ondemand::object content_obj;
        if (candidate["content"].get_object().get(content_obj) != simdjson::SUCCESS) continue;
        simdjson::ondemand::array parts;
        if (content_obj["parts"].get_array().get(parts) != simdjson::SUCCESS) continue;

        for (auto part : parts) {
            std::string_view text_v;
            if (part["text"].get_string().get(text_v) == simdjson::SUCCESS) {
                content_str = std::string(text_v);
                continue;
            }

            simdjson::ondemand::object function_call;
            if (part["functionCall"].get_object().get(function_call) == simdjson::SUCCESS) {
                ToolCall call;
                static int call_counter = 0;
                call.id   = "call_gemini_" + std::to_string(++call_counter);
                call.type = "function";

                std::string_view name_v;
                if (function_call["name"].get_string().get(name_v) == simdjson::SUCCESS)
                    call.function.name = std::string(name_v);

                simdjson::ondemand::object args_obj;
                if (function_call["args"].get_object().get(args_obj) == simdjson::SUCCESS) {
                    std::string_view raw_v;
                    if (args_obj.raw_json().get(raw_v) == simdjson::SUCCESS)
                        call.function.arguments = std::string(raw_v);
                }
                tools.push_back(std::move(call));
            }
        }
    }

    return {std::move(content_str), std::move(tools)};
}

// ─────────────────────────────────────────────────────────────────────────────
// GeminiProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string GeminiProtocol::serialize(const ChatRequest& req) const {
    return serialize_gemini_request(req, req.model);
}

cpr::Header GeminiProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{{"Content-Type", "application/json"}};
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string GeminiProtocol::build_url(std::string_view base_url,
                                       std::string_view model) const {
    return std::string(base_url)
         + "/v1beta/models/"
         + std::string(model)
         + ":streamGenerateContent?alt=sse";
}

ParseResult GeminiProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    const std::string_view json_sv = parsed.data;

    if (parsed.is_done) {
        ParseResult r;
        r.done = true;
        return r;
    }

    auto [content, tools] = parse_gemini_sse_chunk(json_sv);

    ParseResult result;
    if (!content.empty() || !tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(content);
        chunk.tools   = std::move(tools);
        result.chunks.push_back(std::move(chunk));
    }
    return result;
}

} // namespace core::llm::protocols
