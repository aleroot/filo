#include "MiniMaxRequestBudget.hpp"

#include <algorithm>
#include <stdexcept>

namespace core::llm::providers {
namespace {
constexpr std::size_t kEnvelopeMargin = 4096;
constexpr std::size_t kImageTokenEstimate = 15000;
constexpr std::size_t kMessageOverhead = 8;
constexpr std::size_t kToolCallOverhead = 16;
constexpr int64_t kAdmissionMargin = 16000;
constexpr int64_t kMinimumOutputRoom = 1024;

/// The same CJK ranges minimax-code counts as one token per character.
[[nodiscard]] constexpr bool is_cjk(char32_t cp) noexcept {
    return (cp >= 0x3000 && cp <= 0x30FF) || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF)
        || (cp >= 0xFF00 && cp <= 0xFFEF) || (cp >= 0xAC00 && cp <= 0xD7AF)
        || (cp >= 0x20000 && cp <= 0x2FA1F);
}
} // namespace

std::size_t MiniMaxRequestBudget::estimate_text(std::string_view utf8) noexcept {
    std::size_t cjk = 0;
    std::size_t other = 0;
    const auto byte = [&](std::size_t index) { return static_cast<char32_t>(static_cast<unsigned char>(utf8[index])); };
    for (std::size_t i = 0; i < utf8.size();) {
        const auto lead = byte(i);
        const std::size_t length = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        char32_t cp = 0;
        if (length == 3 && i + 3 <= utf8.size()) {
            cp = ((lead & 0x0F) << 12) | ((byte(i + 1) & 0x3F) << 6) | (byte(i + 2) & 0x3F);
        } else if (length == 4 && i + 4 <= utf8.size()) {
            cp = ((lead & 0x07) << 18) | ((byte(i + 1) & 0x3F) << 12)
                | ((byte(i + 2) & 0x3F) << 6) | (byte(i + 3) & 0x3F);
        }
        ++(is_cjk(cp) ? cjk : other);
        i += std::min(length, utf8.size() - i);
    }
    return cjk + (other + 2) / 3;
}

std::size_t MiniMaxRequestBudget::estimate_history(std::span<const Message> messages) noexcept {
    std::size_t tokens = 0;
    std::size_t media = 0;
    for (const auto& message : messages) {
        tokens += estimate_text(message.content) + estimate_text(message.reasoning_content) + kMessageOverhead;
        for (const auto& item : message.continuation_items) {
            if (item.kind != "assistant_content") tokens += estimate_text(item.payload);
        }
        for (const auto& tool : message.tool_calls) {
            tokens += estimate_text(tool.function.name) + estimate_text(tool.function.arguments) + kToolCallOverhead;
        }
        for (const auto& part : message.content_parts) {
            if (part.type == ContentPartType::Image) {
                media += kImageTokenEstimate;
            } else if (part.type == ContentPartType::Text
                       && message.content.find(part.text) == std::string::npos) {
                tokens += estimate_text(part.text);
            }
        }
    }
    return tokens + media + kEnvelopeMargin;
}

std::size_t MiniMaxRequestBudget::estimate_request(const ChatRequest& request) {
    auto tokens = estimate_history(request.messages);
    for (const auto& tool : request.tools) {
        tokens += estimate_text(core::tools::schema::canonical_input_schema(tool.function))
            + estimate_text(tool.function.name) + estimate_text(tool.function.description);
    }
    for (const auto& layer : request.prompt_plan.layers()) {
        tokens += estimate_text(layer.content);
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
