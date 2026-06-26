#pragma once

#include "../llm/LLMProvider.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace core::agent {

struct HistoryCompactionRequest {
    std::vector<core::llm::Message> history;
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string model;
};

struct HistoryCompactionCallbacks {
    std::function<void(const std::string&)> on_status;
    std::function<void(std::string)> on_summary;
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
