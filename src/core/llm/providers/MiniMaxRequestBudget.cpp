#include "MiniMaxRequestBudget.hpp"

#include <algorithm>
#include <stdexcept>

namespace core::llm::providers {
namespace {
constexpr std::size_t kEnvelopeMargin = 4096;
constexpr std::size_t kImageTokenEstimate = 15000;
constexpr int64_t kAdmissionMargin = 16000;
constexpr int64_t kMinimumOutputRoom = 1024;
} // namespace

std::size_t MiniMaxRequestBudget::estimate_history(std::span<const Message> messages) noexcept {
    std::size_t bytes = 0;
    std::size_t media = 0;
    for (const auto& message : messages) {
        bytes += message.content.size() + message.reasoning_content.size() + 16;
        for (const auto& item : message.continuation_items) {
            if (item.kind != "assistant_content") bytes += item.payload.size();
        }
        for (const auto& tool : message.tool_calls) {
            bytes += tool.function.name.size() + tool.function.arguments.size() + 32;
        }
        for (const auto& part : message.content_parts) {
            if (part.type == ContentPartType::Image) {
                media += kImageTokenEstimate;
            } else if (part.type == ContentPartType::Text
                       && message.content.find(part.text) == std::string::npos) {
                bytes += part.text.size();
            }
        }
    }
    return (bytes + 1) / 2 + media + kEnvelopeMargin;
}

std::size_t MiniMaxRequestBudget::estimate_request(const ChatRequest& request) {
    auto tokens = estimate_history(request.messages);
    for (const auto& tool : request.tools) {
        tokens += (core::tools::schema::canonical_input_schema(tool.function).size()
                   + tool.function.name.size() + tool.function.description.size() + 1) / 2;
    }
    for (const auto& layer : request.prompt_plan.layers()) {
        tokens += (layer.content.size() + 1) / 2;
    }
    return tokens;
}

void MiniMaxRequestBudget::prepare(ChatRequest& request, int context_window,
                                   std::size_t counted_input) {
    const auto input = counted_input ? counted_input : estimate_request(request);
    const auto room = static_cast<int64_t>(context_window)
        - static_cast<int64_t>(input) - kAdmissionMargin;
    if (room < kMinimumOutputRoom) {
        throw std::invalid_argument(
            "MiniMax context is full; compact the conversation before retrying.");
    }
    request.max_tokens = static_cast<int>(std::min<int64_t>(
        std::clamp(request.max_tokens.value_or(kMiniMaxOutputLimit), 1, kMiniMaxOutputLimit), room));
}

} // namespace core::llm::providers
