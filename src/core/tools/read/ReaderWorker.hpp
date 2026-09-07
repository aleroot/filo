#pragma once
#include "ReadTypes.hpp"
#include "../../llm/LLMProvider.hpp"
#include <chrono>
#include <functional>
#include <span>

namespace core::tools::read {
struct WorkerProfile {
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string provider_name;
    std::string model;
};
struct Citation {
    std::size_t source = 0; // zero-based source index, never model-supplied URI
    int first_line = 1;
    int last_line = 1;
    std::string quote;
};
struct Answer {
    std::string text;
    std::vector<Citation> citations;
    std::string model;
    std::size_t input_bytes = 0;
    bool partial = false;
};
// The resolver creates an isolated provider per call. Dependency injection
// keeps transport/configuration separate from evidence validation.
class ReaderWorker {
public:
    using Resolver = std::function<std::expected<WorkerProfile, std::string>()>;
    explicit ReaderWorker(Resolver resolver = {}, std::chrono::milliseconds timeout = std::chrono::seconds(45));
    [[nodiscard]] std::expected<Answer, std::string> answer(
        std::span<const Resource> resources, std::string_view question,
        const ToolInvocationContext& invocation) const;
private:
    Resolver resolver_;
    std::chrono::milliseconds timeout_;
};
} // namespace core::tools::read
