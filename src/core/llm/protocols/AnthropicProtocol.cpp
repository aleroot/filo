#include "AnthropicProtocol.hpp"
#include "../Models.hpp"
#include "../../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <cstdio>
#include <ctime>

namespace core::llm::protocols {

namespace {
    constexpr std::string_view ANTHROPIC_VERSION       = "2023-06-01";
    constexpr std::string_view ANTHROPIC_BETA_THINKING = "interleaved-thinking-2025-05-14";
    constexpr std::string_view ANTHROPIC_BILLING_HEADER = "cc_version=2.1.78.13b; cc_entrypoint=cli; cch=0;";
    
    // Helper to safely convert string header to int32_t
    int32_t safe_stoi32(std::string_view sv, int32_t default_val = 0) {
        if (sv.empty()) return default_val;
        try {
            return static_cast<int32_t>(std::stoi(std::string(sv)));
        } catch (...) {
            return default_val;
        }
    }
    
    // Helper to safely convert string header to int64_t
    int64_t safe_stoi64(std::string_view sv, int64_t default_val = 0) {
        if (sv.empty()) return default_val;
        try {
            return std::stoll(std::string(sv));
        } catch (...) {
            return default_val;
        }
    }
    
    // Helper to safely convert string header to float
    float safe_stof(std::string_view sv, float default_val = 0.0f) {
        if (sv.empty()) return default_val;
        try {
            return std::stof(std::string(sv));
        } catch (...) {
            return default_val;
        }
    }
    
    // Parse ISO 8601 timestamp to unix seconds (Anthropic format: "2025-01-15T12:00:30Z").
    int64_t parse_iso8601_timestamp(std::string_view sv) {
        if (sv.empty()) return 0;
        int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;
        // Use a std::string to guarantee null-termination before passing to sscanf.
        const std::string s(sv);
        if (std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%dZ",
                        &year, &month, &day, &hour, &min, &sec) == 6) {
            std::tm t{};
            t.tm_year = year - 1900;
            t.tm_mon  = month - 1;
            t.tm_mday = day;
            t.tm_hour = hour;
            t.tm_min  = min;
            t.tm_sec  = sec;
            return static_cast<int64_t>(timegm(&t));
        }
        // Fallback: some implementations send a plain unix timestamp integer.
        return safe_stoi64(sv);
    }

    // ── Response lifecycle hook implementations ──────────────────────────────
    // Named with an `impl_` prefix to avoid name-hiding issues when called
    // from member functions that share the same base name (e.g. the virtual
    // override `format_error_message` would shadow a plain `format_error_message`
    // inside the class's own method body due to C++ name-lookup rules).
    // ─────────────────────────────────────────────────────────────────────────

    RateLimitInfo impl_parse_rate_limit_headers(const cpr::Header& headers) {
        RateLimitInfo info;

        // Standard rate limit headers (requests per minute)
        if (auto it = headers.find("anthropic-ratelimit-requests-limit"); it != headers.end()) {
            info.requests_limit = safe_stoi32(it->second);
        }
        if (auto it = headers.find("anthropic-ratelimit-requests-remaining"); it != headers.end()) {
            info.requests_remaining = safe_stoi32(it->second);
        }
        if (auto it = headers.find("anthropic-ratelimit-requests-reset"); it != headers.end()) {
            info.requests_reset = parse_iso8601_timestamp(it->second);
        }

        // Token-based rate limits (tokens per minute)
        if (auto it = headers.find("anthropic-ratelimit-tokens-limit"); it != headers.end()) {
            info.tokens_limit = safe_stoi32(it->second);
        }
        if (auto it = headers.find("anthropic-ratelimit-tokens-remaining"); it != headers.end()) {
            info.tokens_remaining = safe_stoi32(it->second);
        }
        if (auto it = headers.find("anthropic-ratelimit-tokens-reset"); it != headers.end()) {
            info.tokens_reset = parse_iso8601_timestamp(it->second);
        }

        // Retry-after header (present on 429 responses)
        if (auto it = headers.find("retry-after"); it != headers.end()) {
            info.retry_after = safe_stoi32(it->second);
            if (info.retry_after > 0) {
                info.is_rate_limited = true;
            }
        }

        // Subscription usage windows (for OAuth/claude.ai users).
        // Anthropic returns two windows: a 5-hour rolling window and a 7-day window.
        // The representative-claim header indicates which is currently authoritative.
        // Add new labels here if Anthropic introduces additional windows.
        for (const std::string_view window : {"5h", "7d"}) {
            const std::string header_name =
                "anthropic-ratelimit-unified-" + std::string(window) + "-utilization";
            if (auto it = headers.find(header_name); it != headers.end()) {
                if (const float val = safe_stof(it->second); val > 0.0f) {
                    info.usage_windows.push_back({std::string(window), val});
                }
            }
        }
        if (auto it = headers.find("anthropic-ratelimit-unified-status"); it != headers.end()) {
            info.unified_status = it->second;
            if (it->second == "rate_limited") {
                info.is_rate_limited = true;
            }
        }
        if (auto it = headers.find("anthropic-ratelimit-unified-representative-claim"); it != headers.end()) {
            info.unified_representative_claim = it->second;
        }

        return info;
    }

    std::string impl_format_error_message(int status_code, std::string_view body) {
        switch (status_code) {
            case 400:
                return "[Anthropic API Error 400: Invalid request. The request body is malformed or contains invalid parameters.]";
            case 401:
                return "[Anthropic API Error 401: Authentication failed. Please check your API key or session token.]";
            case 403:
                return "[Anthropic API Error 403: Permission denied. Your account may not have access to this model or feature.]";
            case 404:
                return "[Anthropic API Error 404: Not found. The requested model or endpoint does not exist.]";
            case 429: {
                std::string msg = "[Anthropic API Error 429: Rate limit exceeded. ";
                if (!body.empty()) {
                    msg += "Please wait before retrying. Consider reducing request frequency or context size.]";
                } else {
                    msg += "Please wait before retrying.]";
                }
                return msg;
            }
            case 500:
                return "[Anthropic API Error 500: Internal server error. This is a temporary issue on Anthropic's side.]";
            case 529:
                return "[Anthropic API Error 529: Server overloaded. Anthropic is experiencing high load. "
                       "This is NOT a rate limit - retrying with backoff is recommended.]";
            default:
                if (status_code >= 500) {
                    return "[Anthropic API Error " + std::to_string(status_code) +
                           ": Server error. Please retry with exponential backoff.]";
                } else if (status_code >= 400) {
                    return "[Anthropic API Error " + std::to_string(status_code) +
                           ": Client error. Please check your request parameters.]";
                }
                return "[Anthropic API Error " + std::to_string(status_code) + "]";
        }
    }

    bool impl_is_retryable_status(int status_code) noexcept {
        return status_code == 429 ||  // Rate limit — retry after delay
               status_code == 500 ||  // Internal server error
               status_code == 502 ||  // Bad gateway
               status_code == 503 ||  // Service unavailable
               status_code == 504 ||  // Gateway timeout
               status_code == 529;    // Overloaded
    }

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSerializer
// ─────────────────────────────────────────────────────────────────────────────

std::string AnthropicSerializer::serialize(const ChatRequest& req,
                                            int default_max_tokens,
                                            const AnthropicThinkingConfig& thinking) {
    std::string payload;
    payload.reserve(8192);

    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += R"(","stream":)";
    payload += req.stream ? "true" : "false";

    // max_tokens is mandatory in the Anthropic API.
    payload += R"(,"max_tokens":)";
    payload += std::to_string(req.max_tokens.value_or(default_max_tokens));

    // Anthropic requires temperature=1 when extended thinking is enabled.
    // Emit the constraint unconditionally so the API never receives a conflicting value.
    if (thinking.enabled) {
        payload += R"(,"temperature":1)";
    } else if (req.temperature.has_value()) {
        payload += R"(,"temperature":)";
        payload += std::to_string(req.temperature.value());
    }

    // Extended thinking block (must come before messages).
    if (thinking.enabled) {
        payload += R"(,"thinking":{"type":"enabled","budget_tokens":)";
        payload += std::to_string(thinking.budget_tokens);
        payload += '}';
    }

    // Extract the first system message — Claude places it at the top level.
    for (const auto& msg : req.messages) {
        if (msg.role == "system" && !msg.content.empty()) {
            payload += R"(,"system":")";
            payload += core::utils::escape_json_string(msg.content);
            payload += '"';
            break;
        }
    }

    // Tools (Anthropic uses "input_schema" instead of "parameters").
    if (!req.tools.empty()) {
        payload += R"(,"tools":[)";
        for (size_t i = 0; i < req.tools.size(); ++i) {
            const auto& def = req.tools[i].function;
            payload += R"({"name":")";
            payload += core::utils::escape_json_string(def.name);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(def.description);
            payload += R"(","input_schema":{"type":"object","properties":{)";

            for (size_t j = 0; j < def.parameters.size(); ++j) {
                const auto& p = def.parameters[j];
                payload += '"';
                payload += core::utils::escape_json_string(p.name);
                payload += R"(":{"type":")";
                payload += core::utils::escape_json_string(p.type);
                payload += R"(","description":")";
                payload += core::utils::escape_json_string(p.description);
                payload += '"';
                if (!p.items_schema.empty()) {
                    payload += R"(,"items":)";
                    payload += p.items_schema;
                }
                payload += '}';
                if (j + 1 < def.parameters.size()) payload += ',';
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
            if (i + 1 < req.tools.size()) payload += ',';
        }
        payload += ']';
    }

    // Messages — system messages are skipped (handled above as top-level field).
    payload += R"(,"messages":[)";
    bool first_msg = true;
    for (const auto& msg : req.messages) {
        if (msg.role == "system") continue;

        if (!first_msg) payload += ',';
        first_msg = false;

        if (msg.role == "tool") {
            // OpenAI tool role → Claude user message with tool_result content block.
            payload += R"({"role":"user","content":[{"type":"tool_result","tool_use_id":")";
            payload += core::utils::escape_json_string(msg.tool_call_id);
            payload += R"(","content":")";
            payload += core::utils::escape_json_string(msg.content);
            payload += "\"}]}";

        } else if (msg.role == "assistant" && !msg.tool_calls.empty()) {
            // OpenAI assistant tool_calls → Claude content blocks.
            payload += R"({"role":"assistant","content":[)";
            bool first_block = true;

            if (!msg.content.empty()) {
                payload += R"({"type":"text","text":")";
                payload += core::utils::escape_json_string(msg.content);
                payload += "\"}";
                first_block = false;
            }

            for (const auto& tc : msg.tool_calls) {
                if (!first_block) payload += ',';
                first_block = false;
                payload += R"({"type":"tool_use","id":")";
                payload += core::utils::escape_json_string(tc.id);
                payload += R"(","name":")";
                payload += core::utils::escape_json_string(tc.function.name);
                // "input" is a raw JSON object — embed directly.
                payload += R"(","input":)";
                payload += tc.function.arguments.empty() ? "{}" : tc.function.arguments;
                payload += '}';
            }
            payload += "]}";

        } else {
            payload += R"({"role":")";
            payload += core::utils::escape_json_string(msg.role);
            payload += R"(","content":")";
            payload += core::utils::escape_json_string(msg.content);
            payload += "\"}";
        }
    }
    payload += "]}";
    return payload;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSSEParser
// ─────────────────────────────────────────────────────────────────────────────

AnthropicSSEParser::Result AnthropicSSEParser::process_event(std::string_view event_type,
                                                              std::string_view json_str) {
    Result result;
    if (json_str.empty()) return result;
    if (event_type == "ping") return result;

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string input(json_str);
    simdjson::dom::element doc;
    if (parser.parse(input).get(doc) != simdjson::SUCCESS) return result;

    if (event_type == "message_start") {
        simdjson::dom::object msg_obj;
        if (doc["message"].get(msg_obj) == simdjson::SUCCESS) {
            simdjson::dom::object usage_obj;
            if (msg_obj["usage"].get(usage_obj) == simdjson::SUCCESS) {
                int64_t it = 0;
                [[maybe_unused]] const auto err = usage_obj["input_tokens"].get(it);
                result.input_tokens = static_cast<int32_t>(it);
            }
        }
        return result;
    }

    if (event_type == "message_delta") {
        simdjson::dom::object usage_obj;
        if (doc["usage"].get(usage_obj) == simdjson::SUCCESS) {
            int64_t ot = 0;
            [[maybe_unused]] const auto err = usage_obj["output_tokens"].get(ot);
            result.output_tokens = static_cast<int32_t>(ot);
        }
        return result;
    }

    if (event_type == "content_block_start") {
        simdjson::dom::object content_block;
        if (doc["content_block"].get(content_block) != simdjson::SUCCESS) return result;

        std::string_view type_v;
        if (content_block["type"].get(type_v) != simdjson::SUCCESS) return result;

        if (type_v == "tool_use") {
            AnthropicToolBlockState state;
            int64_t index_v;
            if (doc["index"].get(index_v) == simdjson::SUCCESS)
                state.index = static_cast<int>(index_v);
            std::string_view id_v;
            if (content_block["id"].get(id_v) == simdjson::SUCCESS)
                state.id = std::string(id_v);
            std::string_view name_v;
            if (content_block["name"].get(name_v) == simdjson::SUCCESS)
                state.name = std::string(name_v);
            current_tool_ = std::move(state);
        }
        return result;
    }

    if (event_type == "content_block_delta") {
        simdjson::dom::object delta;
        if (doc["delta"].get(delta) != simdjson::SUCCESS) return result;

        std::string_view delta_type;
        if (delta["type"].get(delta_type) != simdjson::SUCCESS) return result;

        if (delta_type == "text_delta") {
            std::string_view text;
            if (delta["text"].get(text) == simdjson::SUCCESS)
                result.text = std::string(text);
        } else if (delta_type == "input_json_delta" && current_tool_.has_value()) {
            std::string_view partial_json;
            if (delta["partial_json"].get(partial_json) == simdjson::SUCCESS)
                current_tool_->accumulated_args += partial_json;
        }
        // thinking_delta: intentionally ignored.
        return result;
    }

    if (event_type == "content_block_stop") {
        if (current_tool_.has_value()) {
            ToolCall tc;
            tc.index              = current_tool_->index;
            tc.id                 = std::move(current_tool_->id);
            tc.type               = "function";
            tc.function.name      = std::move(current_tool_->name);
            tc.function.arguments = std::move(current_tool_->accumulated_args);
            result.completed_tools.push_back(std::move(tc));
            current_tool_.reset();
        }
        return result;
    }

    if (event_type == "message_stop") {
        result.done = true;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string AnthropicProtocol::serialize(const ChatRequest& req) const {
    return AnthropicSerializer::serialize(req, default_max_tokens_, thinking_);
}

cpr::Header AnthropicProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"anthropic-version",          std::string(ANTHROPIC_VERSION)},
        {"Content-Type",               "application/json"},
        {"Accept",                     "text/event-stream"},
        {"x-anthropic-billing-header", std::string(ANTHROPIC_BILLING_HEADER)},
    };
    if (thinking_.enabled) {
        headers["anthropic-beta"] = std::string(ANTHROPIC_BETA_THINKING);
    }
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

std::string AnthropicProtocol::build_url(std::string_view base_url,
                                          [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/v1/messages";
}

ParseResult AnthropicProtocol::parse_event(std::string_view raw_event) {
    std::string event_type;
    std::string event_data;

    size_t line_start = 0;
    while (line_start < raw_event.size()) {
        size_t line_end = raw_event.find('\n', line_start);
        if (line_end == std::string_view::npos) line_end = raw_event.size();

        std::string_view line(raw_event.data() + line_start, line_end - line_start);
        if (line.starts_with("event: "))     event_type = std::string(line.substr(7));
        else if (line.starts_with("data: ")) event_data = std::string(line.substr(6));
        line_start = line_end + 1;
    }

    if (event_type.empty() || event_data.empty()) return {};

    auto r = sse_parser_.process_event(event_type, event_data);

    ParseResult result;
    if (r.input_tokens  > 0) accumulated_input_  = r.input_tokens;
    if (r.output_tokens > 0) accumulated_output_ = r.output_tokens;

    if (!r.text.empty())           result.chunks.push_back(StreamChunk::make_content(r.text));
    if (!r.completed_tools.empty()) result.chunks.push_back(StreamChunk::make_tools(r.completed_tools));

    if (r.done) {
        result.done              = true;
        result.prompt_tokens     = accumulated_input_;
        result.completion_tokens = accumulated_output_;
    }

    return result;
}

std::unique_ptr<ApiProtocolBase> AnthropicProtocol::clone() const {
    return std::make_unique<AnthropicProtocol>(thinking_, default_max_tokens_);
}

// ── Response lifecycle hook overrides ────────────────────────────────────────

void AnthropicProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = impl_parse_rate_limit_headers(response.headers);
}

std::string AnthropicProtocol::format_error_message(const HttpResponse& response) const {
    return impl_format_error_message(response.status_code, response.body);
}

bool AnthropicProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return impl_is_retryable_status(response.status_code);
}

} // namespace core::llm::protocols
