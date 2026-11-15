#pragma once

#include "../llm/LLMProvider.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core::agent {

enum class HistoryCompactionReason {
    Auto,
    Manual,
    /// Written ahead, while an idle conversation's cache is still warm, and
    /// kept aside until the cache expires.
    Idle,
};

enum class HistoryCompactionApplyStatus {
    Applied,
    Stale,
};

struct HistoryCompactionRequest {
    std::vector<core::llm::Message> history;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string model;
    HistoryCompactionReason reason = HistoryCompactionReason::Auto;
    /// When set, the checkpoint is asked as a continuation of this request,
    /// already in the provider's prompt cache: its tools, system prompt and
    /// messages, then `history` (only what followed it), then the
    /// instruction. The summary then re-reads the warm cache instead of
    /// sending the conversation again under a new prefix.
    std::shared_ptr<const core::llm::ChatRequest> cached_request;
    /// The conversation exactly as `provider` reads it: its last request
    /// followed by the messages that answered it. When set and the provider
    /// supports native compaction, its own checkpoint of this conversation is
    /// requested beside the summary (see ChatRequest::native_compaction).
    std::shared_ptr<const core::llm::ChatRequest> native_request;
};

struct HistoryCompactionCallbacks {
    std::function<void(const std::string&)> on_status;
    /// The portable summary, and the provider's own checkpoint when one was
    /// requested and produced. A checkpoint never arrives without a summary.
    std::function<HistoryCompactionApplyStatus(
        std::string, std::optional<core::llm::ContinuationItem>)> on_summary;
    std::function<void()> on_finished;
};

class HistoryCompactor {
public:
    void compact_async(HistoryCompactionRequest request,
                       HistoryCompactionCallbacks callbacks) const;

private:
    [[nodiscard]] static core::llm::ChatRequest build_request(
        const HistoryCompactionRequest& request);
};

} // namespace core::agent
