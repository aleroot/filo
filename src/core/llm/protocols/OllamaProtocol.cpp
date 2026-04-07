#include "OllamaProtocol.hpp"
#include "../Models.hpp"
#include <charconv>
#include <cstdio>
#include <simdjson.h>

namespace core::llm::protocols {

// ─────────────────────────────────────────────────────────────────────────────
// parse_ollama_ndjson_line
// ─────────────────────────────────────────────────────────────────────────────

OllamaNdJsonResult parse_ollama_ndjson_line(std::string_view json_str) {
    if (json_str.empty()) return {};

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(simdjson::padded_string(json_str)).get(doc) != simdjson::SUCCESS) return {};

    OllamaNdJsonResult result;

    bool done_flag = false;
    [[maybe_unused]] auto e = doc["done"].get(done_flag);
    result.done = done_flag;

    simdjson::dom::object message;
    if (doc["message"].get(message) != simdjson::SUCCESS) return result;

    std::string_view content;
    if (message["content"].get(content) == simdjson::SUCCESS)
        result.content = std::string(content);

    simdjson::dom::array tool_calls_arr;
    if (message["tool_calls"].get(tool_calls_arr) == simdjson::SUCCESS) {
        for (simdjson::dom::element tc : tool_calls_arr) {
            ToolCall call;
            simdjson::dom::object func;
            if (tc["function"].get(func) == simdjson::SUCCESS) {
                std::string_view name_v;
                if (func["name"].get(name_v) == simdjson::SUCCESS)
                    call.function.name = std::string(name_v);

                // Ollama delivers arguments as a JSON object, not a JSON string.
                simdjson::dom::object args_obj;
                if (func["arguments"].get(args_obj) == simdjson::SUCCESS)
                    call.function.arguments = simdjson::minify(args_obj);
            }
            result.tools.push_back(std::move(call));
        }
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// OllamaProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string OllamaProtocol::serialize(const ChatRequest& req) const {
    if (!request_has_image_input(req)) {
        return Serializer::serialize(req);
    }

    std::string payload;
    payload.reserve(4096);
    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += R"(","stream":)";
    payload += req.stream ? "true" : "false";

    if (req.temperature.has_value()) {
        payload += R"(,"temperature":)";
        char tmp[32];
        int n = std::snprintf(tmp, sizeof(tmp), "%.6g", req.temperature.value());
        if (n > 0 && n < static_cast<int>(sizeof(tmp))) {
            payload.append(tmp, n);
        }
    }
    if (req.max_tokens.has_value()) {
        payload += R"(,"max_tokens":)";
        char tmp[32];
        auto [ptr, _] = std::to_chars(tmp, tmp + sizeof(tmp), req.max_tokens.value());
        payload.append(tmp, ptr);
    }

    if (req.response_format.is_structured()) {
        payload += R"(,"response_format":{"type":")";
        payload += req.response_format.to_string();
        payload += '"';
        if (req.response_format.type == ResponseFormat::Type::JsonSchema
            && !req.response_format.schema.empty()) {
            payload += R"(,"schema":)";
            payload += req.response_format.schema;
        }
        payload += "}";
    }

    if (!req.tools.empty()) {
        payload += R"(,"tools":[)";
        for (std::size_t i = 0; i < req.tools.size(); ++i) {
            const auto& def = req.tools[i].function;
            payload += R"({"type":")";
            payload += req.tools[i].type;
            payload += R"(","function":{"name":")";
            payload += core::utils::escape_json_string(def.name);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(def.description);
            payload += R"(","parameters":{"type":"object","properties":{)";

            for (std::size_t j = 0; j < def.parameters.size(); ++j) {
                const auto& p = def.parameters[j];
                payload += "\"";
                payload += core::utils::escape_json_string(p.name);
                payload += R"(":{"type":")";
                payload += core::utils::escape_json_string(p.type);
                payload += R"(","description":")";
                payload += core::utils::escape_json_string(p.description);
                payload += "\"";
                if (!p.items_schema.empty()) {
                    payload += R"(,"items":)";
                    payload += p.items_schema;
                }
                payload += "}";
                if (j + 1 < def.parameters.size()) payload += ",";
            }

            payload += R"(},"required":[)";
            bool first_req = true;
            for (const auto& p : def.parameters) {
                if (!p.required) continue;
                if (!first_req) payload += ",";
                payload += "\"";
                payload += core::utils::escape_json_string(p.name);
                payload += "\"";
                first_req = false;
            }
            payload += "]}}}";
            if (i + 1 < req.tools.size()) payload += ",";
        }
        payload += "]";
    }

    payload += R"(,"messages":[)";
    for (std::size_t i = 0; i < req.messages.size(); ++i) {
        const auto& msg = req.messages[i];
        payload += R"({"role":")";
        payload += core::utils::escape_json_string(msg.role);
        payload += "\"";

        if (!msg.name.empty()) {
            payload += R"(,"name":")";
            payload += core::utils::escape_json_string(msg.name);
            payload += "\"";
        }
        if (!msg.tool_call_id.empty()) {
            payload += R"(,"tool_call_id":")";
            payload += core::utils::escape_json_string(msg.tool_call_id);
            payload += "\"";
        }

        std::string content = msg.content_parts.empty()
            ? msg.content
            : collapse_text_parts(msg.content_parts);
        std::vector<std::string> images;
        for (const auto& part : msg.content_parts) {
            if (part.type != ContentPartType::Image) continue;
            if (const auto encoded = encode_image_part(part); encoded.has_value()) {
                images.push_back(encoded->base64_data);
            } else {
                if (!content.empty() && !content.ends_with('\n')) {
                    content.push_back('\n');
                }
                content += unavailable_image_attachment_text(part.path);
            }
        }

        if (!content.empty()) {
            payload += R"(,"content":")";
            payload += core::utils::escape_json_string(content);
            payload += "\"";
        } else if (msg.tool_calls.empty()) {
            payload += R"(,"content":null)";
        }

        if (!images.empty()) {
            payload += R"(,"images":[)";
            for (std::size_t j = 0; j < images.size(); ++j) {
                if (j > 0) payload += ",";
                payload += "\"";
                payload += core::utils::escape_json_string(images[j]);
                payload += "\"";
            }
            payload += "]";
        }

        if (!msg.tool_calls.empty()) {
            payload += R"(,"tool_calls":[)";
            for (std::size_t j = 0; j < msg.tool_calls.size(); ++j) {
                const auto& tc = msg.tool_calls[j];
                payload += R"({"id":")";
                payload += core::utils::escape_json_string(tc.id);
                payload += R"(","type":")";
                payload += core::utils::escape_json_string(tc.type);
                payload += R"(","function":{"name":")";
                payload += core::utils::escape_json_string(tc.function.name);
                payload += R"(","arguments":")";
                payload += core::utils::escape_json_string(tc.function.arguments);
                payload += R"("}})";
                if (j + 1 < msg.tool_calls.size()) payload += ",";
            }
            payload += "]";
        }

        payload += "}";
        if (i + 1 < req.messages.size()) payload += ",";
    }

    payload += "]}";
    return payload;
}

cpr::Header OllamaProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{{"Content-Type", "application/json"}};
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string OllamaProtocol::build_url(std::string_view base_url,
                                       [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/api/chat";
}

ParseResult OllamaProtocol::parse_event(std::string_view raw_event) {
    if (raw_event.empty()) return {};

    auto ndjson = parse_ollama_ndjson_line(raw_event);

    ParseResult result;
    result.done = ndjson.done;

    if (!ndjson.content.empty() || !ndjson.tools.empty()) {
        StreamChunk chunk;
        chunk.content  = std::move(ndjson.content);
        chunk.tools    = std::move(ndjson.tools);
        chunk.is_final = ndjson.done;
        result.chunks.push_back(std::move(chunk));
    }

    return result;
}

} // namespace core::llm::protocols
