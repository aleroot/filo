#pragma once

#include "../llm/LLMProvider.hpp"

#include <functional>
#include <memory>
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
};

struct HistoryCompactionCallbacks {
    std::function<void(const std::string&)> on_status;
    std::function<HistoryCompactionApplyStatus(std::string)> on_summary;
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
