#include "OpenAIProtocol.hpp"
#include "OpenAIUsage.hpp"
#include "SseUtils.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "../Models.hpp"
#include "../OpenAIEndpointUtils.hpp"
#include "../../logging/Logger.hpp"
#include "core/utils/TimeUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>

namespace core::llm::protocols {

ReasoningCapabilities openai_reasoning_capabilities(std::string_view model) noexcept {
    using core::utils::ascii::istarts_with;
    if (!istarts_with(model, "gpt-5")
        && !istarts_with(model, "o1")
        && !istarts_with(model, "o3")
        && !istarts_with(model, "o4")) {
        return {};
    }

    ReasoningCapabilities features{ReasoningCapability::Effort};
    if (istarts_with(model, "gpt-5.6")) {
        features = features | ReasoningCapability::MaxEffort;
    }
    return features;
}

namespace {

[[nodiscard]] std::string lower_ascii(std::string_view value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (const char ch : value) {
        lowered.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    return lowered;
}

[[nodiscard]] std::optional<std::string_view>
find_header_case_insensitive(const cpr::Header& headers, std::string_view key) {
    if (const auto it = headers.find(std::string(key)); it != headers.end()) {
        return it->second;
    }

    const std::string key_lower = lower_ascii(key);
    for (const auto& [k, v] : headers) {
        if (k.size() != key_lower.size()) continue;
        bool equal = true;
        for (std::size_t i = 0; i < k.size(); ++i) {
            const char lhs = static_cast<char>(
                std::tolower(static_cast<unsigned char>(k[i])));
            if (lhs != key_lower[i]) {
                equal = false;
                break;
            }
        }
        if (equal) return v;
    }
    return std::nullopt;
}

[[nodiscard]] int32_t parse_int_header_case_insensitive(const cpr::Header& headers,
                                                        std::string_view key) noexcept {
    const auto value = find_header_case_insensitive(headers, key);
    if (!value.has_value() || value->empty()) return 0;
    try {
        return static_cast<int32_t>(std::stoi(std::string(*value)));
    } catch (...) {
        return 0;
    }
}

[[nodiscard]] RateLimitInfo parse_openai_compatible_rate_limit_headers(
    const cpr::Header& headers,
    int response_status_code) noexcept {
    RateLimitInfo info;

    info.requests_limit =
        parse_int_header_case_insensitive(headers, "x-ratelimit-limit-requests");
    info.requests_remaining =
        parse_int_header_case_insensitive(headers, "x-ratelimit-remaining-requests");
    if (auto r_reset = find_header_case_insensitive(headers, "x-ratelimit-reset-requests")) {
        info.requests_reset = core::utils::time::parse_timestamp_or_duration(*r_reset);
    }
    info.tokens_limit =
        parse_int_header_case_insensitive(headers, "x-ratelimit-limit-tokens");
    info.tokens_remaining =
        parse_int_header_case_insensitive(headers, "x-ratelimit-remaining-tokens");
    if (auto t_reset = find_header_case_insensitive(headers, "x-ratelimit-reset-tokens")) {
        info.tokens_reset = core::utils::time::parse_timestamp_or_duration(*t_reset);
    }
    info.retry_after = parse_int_header_case_insensitive(headers, "retry-after");
    info.is_rate_limited = (response_status_code == 429 || info.retry_after > 0);

    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited) {
        info.requests_remaining = info.requests_limit;
    }
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited) {
        info.tokens_remaining = info.tokens_limit;
    }

    return info;
}

[[nodiscard]] std::string normalize_openai_effort(std::string_view raw_effort,
                                                  std::string_view model) {
    std::string effort = lower_ascii(raw_effort);
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch);
    });
    if (effort == "auto" || effort == "unset" || effort == "default") {
        return {};
    }
    if (effort == "low" || effort == "medium" || effort == "high") {
        return effort;
    }
    if (effort == "max") {
        return openai_reasoning_capabilities(model).supports(
            ReasoningCapability::MaxEffort) ? "max" : "high";
    }
    return {};
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — shared pure parser for any OpenAI-compatible stream
// ─────────────────────────────────────────────────────────────────────────────

OpenAIChatChunk
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
    std::string              reasoning;

    for (simdjson::dom::element choice : choices) {
        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) {
            continue;
        }

        std::string_view content;
        if (delta["content"].get(content) == simdjson::SUCCESS) {
            content_str = std::string(content);
        }

        // OpenAI-compatible thinking models (DeepSeek-R1, Qwen-thinking via
        // OpenRouter, etc.) stream chain-of-thought as delta.reasoning_content.
        // Capture it so the TUI reasoning disclosure lights up for these models.
        std::string_view reasoning_delta;
        if (delta["reasoning_content"].get(reasoning_delta) == simdjson::SUCCESS
            && !reasoning_delta.empty()) {
            reasoning.append(reasoning_delta.data(), reasoning_delta.size());
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

    return {std::move(content_str), std::move(tools), std::move(reasoning)};
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAIProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string OpenAIProtocol::serialize(const ChatRequest& req) const {
    std::string payload = Serializer::serialize(req);

    if (payload.ends_with('}')) {
        payload.pop_back();
        if (openai_reasoning_capabilities(req.model).supports_effort()) {
            const std::string effort = normalize_openai_effort(req.effort, req.model);
            if (!effort.empty()) {
                payload += R"(,"reasoning_effort":")";
                payload += core::utils::escape_json_string(effort);
                payload += '"';
            }
        }
        append_extra_fields(payload, req);
        if ((stream_usage_ || req.stream_include_usage) && req.stream) {
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

    // OpenAI Codex backend account-scoped tokens require this header.
    if (auto it = auth.properties.find("account_id");
        it != auth.properties.end() && !it->second.empty()
        && headers.count("chatgpt-account-id") == 0) {
        headers["chatgpt-account-id"] = it->second;
    }

    return headers;
}

std::string OpenAIProtocol::build_url(std::string_view base_url,
                                      std::string_view model) const {
    if (openai_endpoint::is_azure_openai_base_url(base_url)) {
        const char* api_version_env = std::getenv("AZURE_OPENAI_API_VERSION");
        const std::string api_version = (api_version_env && api_version_env[0] != '\0')
            ? std::string(api_version_env)
            : "2024-12-01-preview";
        return openai_endpoint::build_azure_chat_completions_url(
            base_url,
            model,
            api_version);
    }
    std::string normalized(base_url);
    while (!normalized.empty() && normalized.back() == '/') {
        normalized.pop_back();
    }
    return normalized + "/chat/completions";
}

ParseResult OpenAIProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    const std::string_view json_sv = parsed.data;

    if (parsed.is_done) {
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
                (void)parse_openai_usage(usage_obj, result);
            }
        }
    }

    auto [content, tools, reasoning] = parse_openai_sse_chunk(json_sv);
    if (!content.empty() || !tools.empty() || !reasoning.empty()) {
        // Combine content, tool calls, and reasoning into a single chunk. The
        // Agent routes each field independently; keeping them together also
        // lets subclasses augment this result without creating a duplicate
        // reasoning-only chunk.
        StreamChunk chunk;
        chunk.content           = std::move(content);
        chunk.tools             = std::move(tools);
        chunk.reasoning_content = std::move(reasoning);
        if (!chunk.reasoning_content.empty()) {
            chunk.reasoning_protocol = std::string(name());
        }
        result.chunks.push_back(std::move(chunk));
    }

    return result;
}

void OpenAIProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ =
        parse_openai_compatible_rate_limit_headers(response.headers, response.status_code);
}

bool OpenAIProtocol::is_retryable(
    const HttpResponse& response) const noexcept {
    return is_openai_retryable_status(response.status_code);
}

} // namespace core::llm::protocols
