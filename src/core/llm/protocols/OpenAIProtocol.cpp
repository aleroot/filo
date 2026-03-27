#include "OpenAIProtocol.hpp"
#include "../Models.hpp"
#include "../../logging/Logger.hpp"
#include <simdjson.h>

namespace core::llm::protocols {

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — shared pure parser for any OpenAI-compatible stream
// ─────────────────────────────────────────────────────────────────────────────

std::pair<std::string, std::vector<ToolCall>>
parse_openai_sse_chunk(std::string_view json_str) {
    if (json_str.empty()) {
        return {};
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_str);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        return {};
    }

    simdjson::dom::array choices;
    if (doc["choices"].get(choices) != simdjson::SUCCESS) {
        return {};
    }
    std::string              content_str;
    std::vector<ToolCall>    tools;

    for (simdjson::dom::element choice : choices) {
        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) {
            continue;
        }

        std::string_view content;
        if (delta["content"].get(content) == simdjson::SUCCESS) {
            content_str = std::string(content);
        }

        simdjson::dom::array tool_calls_arr;
        if (delta["tool_calls"].get(tool_calls_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element tc : tool_calls_arr) {
                ToolCall call;
                int64_t index_v;
                if (tc["index"].get(index_v) == simdjson::SUCCESS)
                    call.index = static_cast<int>(index_v);
                std::string_view id_v;
                if (tc["id"].get(id_v) == simdjson::SUCCESS)
                    call.id = std::string(id_v);
                std::string_view type_v;
                if (tc["type"].get(type_v) == simdjson::SUCCESS)
                    call.type = std::string(type_v);
                simdjson::dom::object func;
                if (tc["function"].get(func) == simdjson::SUCCESS) {
                    std::string_view name_v;
                    if (func["name"].get(name_v) == simdjson::SUCCESS)
                        call.function.name = std::string(name_v);
                    std::string_view args_v;
                    if (func["arguments"].get(args_v) == simdjson::SUCCESS)
                        call.function.arguments = std::string(args_v);
                }
                tools.push_back(std::move(call));
            }
        }
    }

    return {std::move(content_str), std::move(tools)};
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAIProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string OpenAIProtocol::serialize(const ChatRequest& req) const {
    std::string payload = Serializer::serialize(req);

    if (payload.ends_with('}')) {
        payload.pop_back();
        append_extra_fields(payload, req);
        if (stream_usage_ && req.stream) {
            payload += R"(,"stream_options":{"include_usage":true})";
        }
        payload += '}';
    }

    return payload;
}

cpr::Header OpenAIProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "text/event-stream"},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string OpenAIProtocol::build_url(std::string_view base_url,
                                       [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/chat/completions";
}

ParseResult OpenAIProtocol::parse_event(std::string_view raw_event) {
    if (!raw_event.starts_with("data: ")) return {};
    std::string_view json_sv = raw_event.substr(6);

    if (json_sv == "[DONE]") {
        ParseResult r;
        r.done = true;
        return r;
    }

    ParseResult result;

    // Extract usage from stream_options chunk (choices array will be empty).
    {
        thread_local simdjson::dom::parser usage_parser;
        simdjson::padded_string ps(json_sv);
        simdjson::dom::element doc;
        if (usage_parser.parse(ps).get(doc) == simdjson::SUCCESS) {
            simdjson::dom::object usage_obj;
            if (doc["usage"].get(usage_obj) == simdjson::SUCCESS) {
                int64_t pt = 0, ct = 0;
                [[maybe_unused]] auto e1 = usage_obj["prompt_tokens"].get(pt);
                [[maybe_unused]] auto e2 = usage_obj["completion_tokens"].get(ct);
                if (pt > 0 || ct > 0) {
                    result.prompt_tokens     = static_cast<int32_t>(pt);
                    result.completion_tokens = static_cast<int32_t>(ct);
                }
            }
        }
    }

    auto [content, tools] = parse_openai_sse_chunk(json_sv);
    if (!content.empty() || !tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(content);
        chunk.tools   = std::move(tools);
        result.chunks.push_back(std::move(chunk));
    }

    return result;
}

void OpenAIProtocol::on_response(const HttpResponse& response) {
    RateLimitInfo info;

    auto parse_int = [&](std::string_view key) -> int32_t {
        if (const auto it = response.headers.find(std::string(key)); it != response.headers.end()) {
            try { return static_cast<int32_t>(std::stoi(it->second)); } catch (...) {}
        }
        return 0;
    };

    info.requests_limit     = parse_int("x-ratelimit-limit-requests");
    info.requests_remaining = parse_int("x-ratelimit-remaining-requests");
    info.tokens_limit       = parse_int("x-ratelimit-limit-tokens");
    info.tokens_remaining   = parse_int("x-ratelimit-remaining-tokens");
    info.retry_after        = parse_int("retry-after");
    info.is_rate_limited    = (response.status_code == 429 || info.retry_after > 0);

    // When a limit is known but remaining was not explicitly reported, assume full
    // to avoid false "0 remaining" alarms on endpoints that omit the header.
    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited) {
        info.requests_remaining = info.requests_limit;
    }
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited) {
        info.tokens_remaining = info.tokens_limit;
    }

    last_rate_limit_ = info;
}

} // namespace core::llm::protocols
