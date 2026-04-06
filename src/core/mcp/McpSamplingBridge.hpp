#pragma once

#include "../llm/LLMProvider.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace core::mcp {

class McpSamplingBridge {
public:
    McpSamplingBridge(std::shared_ptr<core::llm::LLMProvider> provider,
                      std::string default_model);

    void set_backend(std::shared_ptr<core::llm::LLMProvider> provider,
                     std::string default_model);

    [[nodiscard]] std::string create_message(std::string_view server_name,
                                             std::string_view params_json) const;

private:
    std::shared_ptr<core::llm::LLMProvider> provider_;
    std::string default_model_;
    mutable std::mutex mutex_;
};

} // namespace core::mcp
