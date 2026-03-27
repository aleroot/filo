#include "OllamaProtocol.hpp"
#include "../Models.hpp"
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
    return Serializer::serialize(req);
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
