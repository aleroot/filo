#include "MistralProtocol.hpp"

#include "SseUtils.hpp"
#include "core/utils/AsciiUtils.hpp"
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

} // namespace

std::string MistralProtocol::serialize(const ChatRequest& req) const {
    ChatRequest adjusted = req;
    if (reasoning_capabilities(req.model).supports_effort()
        && !normalize_mistral_effort(req.effort).empty()) {
        // Mirrors mistral-vibe: reasoning requests require temperature 1.
        adjusted.temperature = 1.0F;
    }
    return OpenAIProtocol::serialize(adjusted);
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
    return result;
}

} // namespace core::llm::protocols
