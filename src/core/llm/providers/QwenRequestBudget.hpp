#pragma once

#include <cstddef>
#include <span>

namespace core::llm {
struct Message;
struct ChatRequest;
struct ModelInfo;
}

namespace core::llm::providers {
/// Qwen's outbound budget policy, independent of HTTP transport and wire API.
class QwenRequestBudget {
public:
    [[nodiscard]] static std::size_t estimate_history(std::span<const Message> messages) noexcept;
    [[nodiscard]] static std::size_t estimate_request(const ChatRequest& request);
    static void prepare(ChatRequest& request, const ModelInfo* model = nullptr,
                        std::size_t calibrated_prompt_tokens = 0);
};
} // namespace core::llm::providers
