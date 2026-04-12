#pragma once

#include "ApiProtocol.hpp"

namespace core::llm::protocols {

[[nodiscard]] std::string
serialize_gemini_code_assist_request(const ChatRequest& req, const std::string& default_model);

class GeminiCodeAssistProtocol : public ApiProtocolBase {
public:
    GeminiCodeAssistProtocol() = default;

    void prepare_request(ChatRequest& request) override;
    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string build_url(std::string_view base_url,
                                        std::string_view model) const override;

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "gemini_code_assist"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GeminiCodeAssistProtocol>(*this);
    }
};

} // namespace core::llm::protocols
