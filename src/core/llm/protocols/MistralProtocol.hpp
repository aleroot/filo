#pragma once

/**
 * @file MistralProtocol.hpp
 * @brief Mistral extension of the OpenAI Chat Completions wire protocol.
 *
 * Mistral's API is OpenAI-compatible, but current reasoning models use a
 * provider-specific `reasoning_effort` contract.  Keeping that translation in
 * a dedicated protocol leaves the shared OpenAI serializer vendor-neutral.
 */

#include "OpenAIProtocol.hpp"

namespace core::llm::protocols {

class MistralProtocol final : public OpenAIProtocol {
public:
    explicit MistralProtocol(bool stream_usage = false)
        : OpenAIProtocol(stream_usage) {}

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "mistral"; }
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<MistralProtocol>(*this);
    }

protected:
    void append_extra_fields(std::string& payload,
                             const ChatRequest& req) const override;
};

} // namespace core::llm::protocols
