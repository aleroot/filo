#include "MistralProtocol.hpp"

#include "SseUtils.hpp"
#include "../StrictToolPolicy.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "core/version/Version.hpp"

#include <algorithm>
#include <array>
#include <simdjson.h>

namespace core::llm::protocols {
namespace {

[[nodiscard]] std::string_view normalize_mistral_effort(
    std::string_view effort) noexcept {
    using core::utils::ascii::iequals;

    if (iequals(effort, "low")) return "none";
    if (iequals(effort, "medium")
        || iequals(effort, "high")
        || iequals(effort, "max")) {
        return "high";
    }
    return {};
}

[[nodiscard]] bool is_mistral_reasoning_model(std::string_view model) noexcept {
    using core::utils::ascii::iequals;
    static constexpr std::array<std::string_view, 6> kModels{
        "mistral-vibe-cli-latest",
        "mistral-medium-3.5",
        "mistral-medium-3-5",
        "mistral-medium-latest",
        "mistral-small-2603",
        "mistral-small-latest",
    };
    return std::ranges::any_of(kModels, [&](std::string_view candidate) {
        return iequals(model, candidate);
    });
}

[[nodiscard]] bool should_replay_thinking(const ChatRequest& req) noexcept {
    return is_mistral_reasoning_model(req.model)
        && !normalize_mistral_effort(req.effort).empty();
}

[[nodiscard]] ChatRequest apply_mistral_reasoning_defaults(ChatRequest req) {
    if (!is_mistral_reasoning_model(req.model)) return req;

    if (req.effort.empty()) {
        // mistral-vibe's default model card is thinking=high.
        req.effort = "high";
    }
    if (!normalize_mistral_effort(req.effort).empty()) {
        req.temperature = 1.0F;
    }
    return req;
}

[[nodiscard]] std::string extract_error_message(std::string_view body) {
    if (body.empty()) return {};

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return {};

    std::string_view message;
    if (document["error"]["message"].get(message) == simdjson::SUCCESS
        || document["message"].get(message) == simdjson::SUCCESS) {
        return std::string(message);
    }
    return {};
}

} // namespace

std::string MistralProtocol::serialize(const ChatRequest& req) const {
    const ChatRequest adjusted = apply_mistral_reasoning_defaults(req);

    Serializer::Options options;
    options.strict_tools = configured_strict_tool_dialect(
        ToolSchemaWire::OpenAI, adjusted.model);
    if (should_replay_thinking(adjusted)) {
        options.reasoning_layout = Serializer::ReasoningContentLayout::ThinkingBlocks;
        options.reasoning_protocol = std::string(name());
    }

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

void MistralProtocol::append_extra_fields(std::string& payload,
                                          const ChatRequest& req) const {
    if (!reasoning_capabilities(req.model).supports_effort()) return;

    const std::string_view effort = normalize_mistral_effort(req.effort);
    if (effort.empty()) return;

    payload += R"(,"reasoning_effort":")";
    payload += effort;
    payload += '"';
}

ReasoningCapabilities MistralProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    if (!is_mistral_reasoning_model(model)) return {};
    return ReasoningCapability::Effort | ReasoningCapability::MaxEffort;
}

cpr::Header MistralProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    cpr::Header headers = OpenAIProtocol::build_headers(auth);
    headers["User-Agent"] = std::string(core::version::user_agent);
    return headers;
}

std::string MistralProtocol::format_error_message(
    const HttpResponse& response) const {
    const std::string api_message = extract_error_message(response.body);
    const std::string detail = api_message.empty()
        ? std::string(response.body)
        : api_message;

    switch (response.status_code) {
        case 401:
            return "[Mistral API Error 401: Authentication failed. "
                   "Check MISTRAL_API_KEY, or run `filo --auth mistral`. "
                   "A Vibe/Studio key from console.mistral.ai uses your "
                   "subscription plan's included usage.]";
        case 402:
            return "[Mistral API Error 402: Payment required. "
                   "Your plan's included usage may be exhausted. "
                   "Enable pay-as-you-go in Admin or wait for the next billing period.]";
        case 403:
            return "[Mistral API Error 403: Permission denied. "
                   + (detail.empty()
                          ? std::string("This key may not have access to the requested model.")
                          : detail)
                   + "]";
        case 429:
            return "[Mistral API Error 429: Rate limited or quota exhausted. "
                   + (detail.empty()
                          ? std::string("Wait and retry, or check remaining plan usage.")
                          : detail)
                   + "]";
        case 500:
        case 502:
        case 503:
        case 504:
            return "[Mistral API Error " + std::to_string(response.status_code)
                   + ": Transient upstream failure"
                   + (detail.empty() ? std::string{"]"} : (": " + detail + "]"));
        default:
            break;
    }

    if (!detail.empty()) {
        return "[Mistral API Error " + std::to_string(response.status_code)
               + ": " + detail + "]";
    }
    return OpenAIProtocol::format_error_message(response);
}

ParseResult MistralProtocol::parse_event(std::string_view raw_event) {
    // Preserve the shared OpenAI-compatible handling for string content, tool
    // calls, usage, and terminal events.
    ParseResult result = OpenAIProtocol::parse_event(raw_event);
    if (result.done) return result;

    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed) || parsed.is_done) return result;

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(parsed.data);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return result;

    std::string content;
    std::string reasoning;
    simdjson::dom::array choices;
    if (document["choices"].get(choices) != simdjson::SUCCESS) return result;

    for (simdjson::dom::element choice : choices) {
        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) continue;

        // Note: delta.reasoning_content is already captured by the base
        // OpenAIProtocol::parse_event call above (combined into the front
        // chunk's reasoning_content), so it is not re-parsed here. This loop
        // only handles Mistral's typed content blocks.

        // Current Mistral models can stream typed content blocks.  This is the
        // wire representation consumed by mistral-vibe's ThinkChunk/TextChunk
        // mapper and is not part of the generic OpenAI protocol.
        simdjson::dom::array blocks;
        if (delta["content"].get(blocks) != simdjson::SUCCESS) continue;
        for (simdjson::dom::element block : blocks) {
            std::string_view type;
            if (block["type"].get(type) != simdjson::SUCCESS) continue;

            if (type == "text") {
                std::string_view text;
                if (block["text"].get(text) == simdjson::SUCCESS) content.append(text);
                continue;
            }
            if (type != "thinking") continue;

            simdjson::dom::array thinking_blocks;
            if (block["thinking"].get(thinking_blocks) != simdjson::SUCCESS) continue;
            for (simdjson::dom::element thinking_block : thinking_blocks) {
                std::string_view text;
                if (thinking_block["text"].get(text) == simdjson::SUCCESS) {
                    reasoning.append(text);
                } else if (thinking_block.get(text) == simdjson::SUCCESS) {
                    reasoning.append(text);
                }
            }
        }
    }

    if (content.empty() && reasoning.empty()) return result;
    if (result.chunks.empty()) result.chunks.emplace_back();
    result.chunks.front().content += content;
    result.chunks.front().reasoning_content += reasoning;
    if (!result.chunks.front().reasoning_content.empty()) {
        result.chunks.front().reasoning_protocol = std::string(name());
    }
    return result;
}

} // namespace core::llm::protocols
