#pragma once
#include "core/llm/Models.hpp"
#include "core/llm/MiniMaxModelTraits.hpp"
#include <span>

namespace core::llm::providers {
/// One conservative estimator shared by request admission and compaction.
struct MiniMaxRequestBudget {
    static std::size_t estimate_history(std::span<const Message> messages) noexcept;
    static std::size_t estimate_request(const ChatRequest& request);
    static void prepare(ChatRequest& request, int context_window,
                        std::size_t counted_input = 0);
};
} // namespace core::llm::providers
