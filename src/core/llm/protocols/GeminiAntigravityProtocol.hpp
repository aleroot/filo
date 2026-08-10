#pragma once

#include "GeminiCodeAssistProtocol.hpp"

namespace core::llm::protocols {

/**
 * @brief Cloud Code Assist protocol variant used by the unofficial
 *        "Antigravity" OAuth strategy (see core::auth::GoogleAntigravityOAuthFlow).
 *
 * Wire format on the happy path is identical to gemini_code_assist
 * (`{"model":...,"project":...,"request":{...}}` posted to
 * `/v1internal:streamGenerateContent?alt=sse`); this variant additionally:
 *  - adds `"userAgent"` / `"requestId"` fields to the request body, and
 *  - sends `User-Agent` / `X-Goog-Api-Client` / `Client-Metadata` headers
 *    that identify the request as coming from the Antigravity IDE,
 * matching what Google's backend expects from Antigravity-issued OAuth
 * tokens (community-observed; not documented by Google).
 *
 * This is an unofficial, ToS-violating integration path. See
 * GoogleAntigravityOAuthFlow.hpp for details.
 */
class GeminiAntigravityProtocol : public GeminiCodeAssistProtocol {
public:
    GeminiAntigravityProtocol() = default;

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "gemini_antigravity"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GeminiAntigravityProtocol>(*this);
    }
};

} // namespace core::llm::protocols
