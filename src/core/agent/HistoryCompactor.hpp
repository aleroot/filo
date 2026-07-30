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
