#include "OpenAIResponsesProtocol.hpp"
#include "../Models.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <format>
#include <random>

namespace core::llm::protocols {

namespace {

struct SseEnvelopeView {
    std::string_view event;
    std::string_view data;
};

[[nodiscard]] std::string_view trim_ascii(std::string_view sv) {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }
    return sv;
}

[[nodiscard]] bool parse_sse_envelope(std::string_view raw,
                                      SseEnvelopeView& out,
                                      std::string& data_scratch) {
    out = {};
    data_scratch.clear();

    std::string_view first_data_line;
    bool saw_data = false;

    std::size_t start = 0;
    while (start <= raw.size()) {
        const std::size_t end =
            (start < raw.size()) ? raw.find('\n', start) : std::string_view::npos;

        std::string_view line = (end == std::string_view::npos)
            ? raw.substr(start)
            : raw.substr(start, end - start);

        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        if (line.starts_with("event:")) {
            out.event = trim_ascii(line.substr(6));
        } else if (line.starts_with("data:")) {
            std::string_view data_part = line.substr(5);
            if (!data_part.empty() && data_part.front() == ' ') data_part.remove_prefix(1);
            if (!saw_data) {
                first_data_line = data_part;
                out.data = first_data_line;
                saw_data = true;
            } else if (data_scratch.empty()) {
                data_scratch.reserve(first_data_line.size() + 1 + data_part.size());
                data_scratch.append(first_data_line);
                data_scratch.push_back('\n');
                data_scratch.append(data_part);
                out.data = data_scratch;
            } else {
                data_scratch.push_back('\n');
                data_scratch.append(data_part);
                out.data = data_scratch;
            }
        }

        if (end == std::string_view::npos) break;
        start = end + 1;
    }

    if (saw_data) return true;

    const std::string_view trimmed = trim_ascii(raw);
    if (trimmed == "[DONE]" || (!trimmed.empty() && (trimmed.front() == '{' || trimmed.front() == '['))) {
        out.data = trimmed;
        return true;
    }

    return false;
}

[[nodiscard]] std::string generate_prompt_cache_key() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t hi = dist(rng);
    const uint64_t lo = dist(rng);
    return std::format("filo-{0:016x}{1:016x}", hi, lo);
}

void append_tool_schema(std::string& payload, const std::vector<Tool>& tools) {
    payload += R"(,"tools":[)";
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const auto& def = tools[i].function;
        payload += R"({"type":"function","name":")";
        payload += core::utils::escape_json_string(def.name);
        payload += R"(","description":")";
        payload += core::utils::escape_json_string(def.description);
        payload += R"(","strict":false,"parameters":{"type":"object","properties":{)";

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
        payload += "]}}";
        if (i + 1 < tools.size()) payload += ",";
    }
    payload += "]";
}

[[nodiscard]] std::string collect_instructions(const ChatRequest& req) {
    std::string instructions;
    bool first = true;
    for (const auto& msg : req.messages) {
        if (msg.role != "system" || msg.content.empty()) continue;
        if (!first) instructions += "\n\n";
        instructions += msg.content;
        first = false;
    }
    return instructions;
}

void append_message_item(std::string& payload,
                         std::string_view role,
                         std::string_view content) {
    payload += R"({"type":"message","role":")";
    payload += core::utils::escape_json_string(role);
    payload += R"(","content":[{"type":")";
    payload += (role == "assistant") ? "output_text" : "input_text";
    payload += R"(","text":")";
    payload += core::utils::escape_json_string(content);
    payload += R"("}]})";
}

void append_function_call_item(std::string& payload,
                               const ToolCall& call,
                               std::size_t fallback_idx) {
    const std::string call_id = !call.id.empty()
        ? call.id
        : "call_" + std::to_string(fallback_idx);
    const std::string arguments = call.function.arguments.empty()
        ? "{}"
        : call.function.arguments;

    payload += R"({"type":"function_call","call_id":")";
    payload += core::utils::escape_json_string(call_id);
    payload += R"(","name":")";
    payload += core::utils::escape_json_string(call.function.name);
    payload += R"(","arguments":")";
    payload += core::utils::escape_json_string(arguments);
    payload += R"("})";
}

void append_function_call_output_item(std::string& payload,
                                      const Message& msg,
                                      std::size_t fallback_idx) {
    const std::string call_id = !msg.tool_call_id.empty()
        ? msg.tool_call_id
        : "call_" + std::to_string(fallback_idx);
    payload += R"({"type":"function_call_output","call_id":")";
    payload += core::utils::escape_json_string(call_id);
    payload += R"(","output":")";
    payload += core::utils::escape_json_string(msg.content);
    payload += R"("})";
}

void append_input_items(std::string& payload, const std::vector<Message>& messages) {
    payload += R"(,"input":[)";
    bool first = true;
    std::size_t fallback_idx = 0;

    const auto append_sep = [&]() {
        if (!first) payload += ",";
        first = false;
    };

    for (const auto& msg : messages) {
        if (msg.role == "system") continue;

        if (msg.role == "tool") {
            if (msg.content.empty() && msg.tool_call_id.empty()) continue;
            append_sep();
            append_function_call_output_item(payload, msg, ++fallback_idx);
            continue;
        }

        if (!msg.content.empty()) {
            append_sep();
            append_message_item(payload, msg.role, msg.content);
        }

        if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            for (const auto& tc : msg.tool_calls) {
                if (tc.function.name.empty()) continue;
                append_sep();
                append_function_call_item(payload, tc, ++fallback_idx);
            }
        }
    }

    payload += "]";
}

void extract_usage_from_completed(simdjson::dom::element doc, ParseResult& result) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return;

    simdjson::dom::object usage_obj;
    if (response_obj["usage"].get(usage_obj) != simdjson::SUCCESS) return;

    int64_t input_tokens = 0;
    int64_t output_tokens = 0;

    if (usage_obj["input_tokens"].get(input_tokens) != simdjson::SUCCESS) {
        [[maybe_unused]] auto _ = usage_obj["prompt_tokens"].get(input_tokens);
    }
    if (usage_obj["output_tokens"].get(output_tokens) != simdjson::SUCCESS) {
        [[maybe_unused]] auto _ = usage_obj["completion_tokens"].get(output_tokens);
    }

    if (input_tokens > 0) {
        result.prompt_tokens = static_cast<int32_t>(
            std::min<int64_t>(input_tokens, INT32_MAX));
    }
    if (output_tokens > 0) {
        result.completion_tokens = static_cast<int32_t>(
            std::min<int64_t>(output_tokens, INT32_MAX));
    }
}

[[nodiscard]] std::string parse_response_id(simdjson::dom::element doc) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return {};

    std::string_view id;
    if (response_obj["id"].get(id) == simdjson::SUCCESS && !id.empty()) {
        return std::string(id);
    }
    return {};
}

[[nodiscard]] std::string extract_assistant_output_text(simdjson::dom::object item) {
    simdjson::dom::array content_items;
    if (item["content"].get(content_items) != simdjson::SUCCESS) return {};

    std::string text;
    for (simdjson::dom::element content_el : content_items) {
        simdjson::dom::object content_obj;
        if (content_el.get(content_obj) != simdjson::SUCCESS) continue;

        std::string_view content_type;
        if (content_obj["type"].get(content_type) != simdjson::SUCCESS
            || content_type != "output_text") {
            continue;
        }

        std::string_view text_part;
        if (content_obj["text"].get(text_part) == simdjson::SUCCESS) {
            text.append(text_part);
        }
    }
    return text;
}

[[nodiscard]] std::string parse_failed_message(simdjson::dom::element doc) {
    auto from_error_obj = [](simdjson::dom::object error_obj) -> std::string {
        std::string_view message;
        if (error_obj["message"].get(message) == simdjson::SUCCESS && !message.empty()) {
            return std::string(message);
        }
        return {};
    };

    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) == simdjson::SUCCESS) {
        simdjson::dom::object error_obj;
        if (response_obj["error"].get(error_obj) == simdjson::SUCCESS) {
            if (std::string msg = from_error_obj(error_obj); !msg.empty()) return msg;
        }

        std::string_view top_message;
        if (response_obj["message"].get(top_message) == simdjson::SUCCESS && !top_message.empty()) {
            return std::string(top_message);
        }
    }

    simdjson::dom::object error_obj;
    if (doc["error"].get(error_obj) == simdjson::SUCCESS) {
        if (std::string msg = from_error_obj(error_obj); !msg.empty()) return msg;
    }

    std::string_view top_message;
    if (doc["message"].get(top_message) == simdjson::SUCCESS && !top_message.empty()) {
        return std::string(top_message);
    }

    return "Responses API stream failed.";
}

[[nodiscard]] std::string parse_incomplete_reason(simdjson::dom::element doc) {
    simdjson::dom::object response_obj;
    if (doc["response"].get(response_obj) != simdjson::SUCCESS) return {};

    simdjson::dom::object details_obj;
    if (response_obj["incomplete_details"].get(details_obj) != simdjson::SUCCESS) return {};

    std::string_view reason;
    if (details_obj["reason"].get(reason) == simdjson::SUCCESS && !reason.empty()) {
        return std::string(reason);
    }
    return {};
}

} // namespace

void OpenAIResponsesProtocol::prepare_request(ChatRequest& req) {
    std::scoped_lock lock(shared_state_->mutex);

    if (shared_state_->prompt_cache_key.empty()) {
        shared_state_->prompt_cache_key = generate_prompt_cache_key();
    }
    if (req.prompt_cache_key.empty()) {
        req.prompt_cache_key = shared_state_->prompt_cache_key;
    }
    if (req.previous_response_id.empty() && !shared_state_->previous_response_id.empty()) {
        req.previous_response_id = shared_state_->previous_response_id;
    }
}

std::string OpenAIResponsesProtocol::serialize(const ChatRequest& req) const {
    std::string payload;
    payload.reserve(8192);

    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += R"(","instructions":")";
    payload += core::utils::escape_json_string(collect_instructions(req));
    payload += R"(","stream":)";
    payload += req.stream ? "true" : "false";
    payload += R"(,"store":false)";

    if (!req.previous_response_id.empty()) {
        payload += R"(,"previous_response_id":")";
        payload += core::utils::escape_json_string(req.previous_response_id);
        payload += '"';
    }

    if (req.temperature.has_value()) {
        payload += R"(,"temperature":)";
        payload += std::to_string(*req.temperature);
    }
    if (req.max_tokens.has_value()) {
        payload += R"(,"max_output_tokens":)";
        payload += std::to_string(*req.max_tokens);
    }

    append_input_items(payload, req.messages);

    payload += R"(,"tool_choice":"auto","parallel_tool_calls":)";
    payload += req.tools.empty() ? "false" : "true";

    if (!req.tools.empty()) {
        append_tool_schema(payload, req.tools);
    }

    if (!req.prompt_cache_key.empty()) {
        payload += R"(,"prompt_cache_key":")";
        payload += core::utils::escape_json_string(req.prompt_cache_key);
        payload += '"';
    }

    const std::string_view service_tier = req.service_tier.empty()
        ? std::string_view{default_service_tier_}
        : std::string_view{req.service_tier};
    if (!service_tier.empty()) {
        payload += R"(,"service_tier":")";
        payload += core::utils::escape_json_string(service_tier);
        payload += '"';
    }

    if (include_reasoning_encrypted_) {
        payload += R"(,"include":["reasoning.encrypted_content"])";
    } else {
        payload += R"(,"include":[])";
    }

    if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
        payload += R"(,"text":{"format":{"type":"json_schema","name":"structured_output","strict":true,"schema":)";
        payload += req.response_format.schema.empty() ? R"({"type":"object"})" : req.response_format.schema;
        payload += "}}";
    } else if (req.response_format.type == ResponseFormat::Type::JsonObject) {
        payload += R"(,"text":{"format":{"type":"json_schema","name":"json_object","strict":false,"schema":{"type":"object","additionalProperties":true}}})";
    }

    payload += "}";
    return payload;
}

cpr::Header OpenAIResponsesProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "text/event-stream"},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string OpenAIResponsesProtocol::build_url(std::string_view base_url,
                                               [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/responses";
}

ParseResult OpenAIResponsesProtocol::parse_event(std::string_view raw_event) {
    SseEnvelopeView envelope;
    thread_local std::string data_scratch;
    if (!parse_sse_envelope(raw_event, envelope, data_scratch)) return {};

    const std::string_view payload_sv = trim_ascii(envelope.data);
    if (payload_sv == "[DONE]") {
        ParseResult result;
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(payload_sv);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        core::logging::debug("[responses] Failed to parse SSE payload JSON");
        return {};
    }

    std::string_view event_type = trim_ascii(envelope.event);
    if (event_type.empty()) {
        std::string_view type_from_payload;
        if (doc["type"].get(type_from_payload) == simdjson::SUCCESS) {
            event_type = type_from_payload;
        }
    }

    ParseResult result;

    if (event_type == "response.created") {
        in_progress_response_id_ = parse_response_id(doc);
        return result;
    }

    if (event_type == "response.output_text.delta") {
        std::string_view delta;
        if (doc["delta"].get(delta) == simdjson::SUCCESS && !delta.empty()) {
            saw_text_delta_ = true;
            result.chunks.push_back(StreamChunk::make_content(std::string(delta)));
        }
        return result;
    }

    if (event_type == "response.output_text.done") {
        std::string_view text;
        if (doc["text"].get(text) == simdjson::SUCCESS && !text.empty()) {
            saw_text_delta_ = true;
            result.chunks.push_back(StreamChunk::make_content(std::string(text)));
        }
        return result;
    }

    if (event_type == "response.reasoning_text.delta"
        || event_type == "response.reasoning_summary_text.delta") {
        std::string_view delta;
        if (doc["delta"].get(delta) == simdjson::SUCCESS && !delta.empty()) {
            result.chunks.push_back(StreamChunk::make_reasoning(std::string(delta)));
        }
        return result;
    }

    if (event_type == "response.output_item.done" || event_type == "response.output_item.added") {
        simdjson::dom::object item;
        if (doc["item"].get(item) != simdjson::SUCCESS) return result;

        std::string_view item_type;
        if (item["type"].get(item_type) != simdjson::SUCCESS) return result;

        if (item_type == "function_call") {
            ToolCall call;
            call.type = "function";

            int64_t index = -1;
            if (item["index"].get(index) == simdjson::SUCCESS && index >= 0 && index <= INT32_MAX) {
                call.index = static_cast<int>(index);
            }

            std::string_view call_id;
            if (item["call_id"].get(call_id) == simdjson::SUCCESS) {
                call.id = std::string(call_id);
            } else {
                std::string_view id;
                if (item["id"].get(id) == simdjson::SUCCESS) {
                    call.id = std::string(id);
                }
            }

            std::string_view name;
            if (item["name"].get(name) == simdjson::SUCCESS) {
                call.function.name = std::string(name);
            }

            std::string_view arguments;
            if (item["arguments"].get(arguments) == simdjson::SUCCESS) {
                call.function.arguments = std::string(arguments);
            }
            if (call.function.arguments.empty()) {
                call.function.arguments = "{}";
            }

            if (!call.function.name.empty()) {
                StreamChunk chunk;
                chunk.tools.push_back(std::move(call));
                result.chunks.push_back(std::move(chunk));
            }
        } else if (item_type == "message" && !saw_text_delta_) {
            std::string_view role;
            if (item["role"].get(role) != simdjson::SUCCESS || role != "assistant") {
                return result;
            }

            std::string text = extract_assistant_output_text(item);
            if (!text.empty()) {
                result.chunks.push_back(StreamChunk::make_content(std::move(text)));
            }
        }

        return result;
    }

    if (event_type == "response.completed") {
        extract_usage_from_completed(doc, result);
        std::string response_id = parse_response_id(doc);
        if (response_id.empty() && !in_progress_response_id_.empty()) {
            response_id = in_progress_response_id_;
        }
        if (!response_id.empty()) {
            last_response_id_ = std::move(response_id);
        }
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    if (event_type == "response.incomplete") {
        std::string reason = parse_incomplete_reason(doc);
        if (reason.empty()) reason = "unknown";
        result.chunks.push_back(StreamChunk::make_error(
            "\n[Responses API incomplete response: " + reason + "]"));
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    if (event_type == "response.failed" || event_type == "error") {
        const std::string message = parse_failed_message(doc);
        result.chunks.push_back(StreamChunk::make_error("\n[Responses API error: " + message + "]"));
        result.done = true;
        saw_text_delta_ = false;
        return result;
    }

    return result;
}

void OpenAIResponsesProtocol::on_response(const HttpResponse& response) {
    RateLimitInfo info;

    const auto parse_int = [&](std::string_view key) -> int32_t {
        if (const auto it = response.headers.find(std::string(key)); it != response.headers.end()) {
            try {
                return static_cast<int32_t>(std::stoi(it->second));
            } catch (...) {}
        }
        return 0;
    };

    info.requests_limit     = parse_int("x-ratelimit-limit-requests");
    info.requests_remaining = parse_int("x-ratelimit-remaining-requests");
    info.tokens_limit       = parse_int("x-ratelimit-limit-tokens");
    info.tokens_remaining   = parse_int("x-ratelimit-remaining-tokens");
    info.retry_after        = parse_int("retry-after");
    info.is_rate_limited    = (response.status_code == 429 || info.retry_after > 0);

    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited) {
        info.requests_remaining = info.requests_limit;
    }
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited) {
        info.tokens_remaining = info.tokens_limit;
    }

    last_rate_limit_ = info;

    if (response.status_code == 200 && !last_response_id_.empty()) {
        std::scoped_lock lock(shared_state_->mutex);
        shared_state_->previous_response_id = last_response_id_;
    }
}

} // namespace core::llm::protocols
