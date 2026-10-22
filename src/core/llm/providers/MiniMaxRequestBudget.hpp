#pragma once
#include "core/llm/Models.hpp"
#include "core/llm/MiniMaxModelTraits.hpp"
#include <span>

namespace core::llm::providers {
/// One conservative estimator shared by request admission and compaction.
struct MiniMaxRequestBudget {
    /// CJK code points cost about one token each; everything else about three
    /// characters per token. minimax-code uses four as its fallback, but Filo
    /// admits requests below a third of the window on this estimate alone
    /// (larger ones are counted exactly), and code and JSON tokenize denser
    /// than prose.
    static std::size_t estimate_text(std::string_view utf8) noexcept;
    static std::size_t estimate_history(std::span<const Message> messages) noexcept;
    static std::size_t estimate_request(const ChatRequest& request);
    static void prepare(ChatRequest& request, int context_window,
                        std::size_t counted_input = 0);
};
} // namespace core::llm::providers
