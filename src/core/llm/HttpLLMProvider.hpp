#pragma once

/**
 * @file HttpLLMProvider.hpp
 * @brief Protocol-driven HTTP LLM provider.
 *
 * HttpLLMProvider is a single `LLMProvider` implementation that can drive
 * **any** HTTP-based LLM API by delegating all wire-format concerns to an
 * `ApiProtocolBase` instance.
 *
 *
 * ## Usage from C++
 *
 * @code{.cpp}
 * #include "core/llm/HttpLLMProvider.hpp"
 * #include "core/llm/protocols/OpenAIProtocol.hpp"
 * #include "core/auth/ApiKeyCredentialSource.hpp"
 *
 * auto creds    = core::auth::ApiKeyCredentialSource::as_bearer("sk-...");
 * auto protocol = std::make_unique<core::llm::protocols::OpenAIProtocol>();
 * auto provider = std::make_shared<core::llm::HttpLLMProvider>(
 *     "https://api.groq.com/openai/v1",
 *     std::move(creds),
 *     "llama-3.3-70b-versatile",
 *     std::move(protocol));
 * @endcode
 */

#include "LLMProvider.hpp"
#include "ModelRegistry.hpp"
#include "protocols/ApiProtocol.hpp"
#include "../auth/ICredentialSource.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core::llm {

/**
 * @brief Generic HTTP LLM provider driven by an ApiProtocolBase.
 *
 * `stream_response()` follows the same pattern as all built-in providers:
 *   1. Clone the protocol to get a fresh, stateful parsing session.
 *   2. Resolve auth credentials from the ICredentialSource.
 *   3. Let the protocol prepare request-level defaults/continuation.
 *   4. Serialise the request via the protocol.
 *   5. POST to the endpoint URL returned by the protocol.
 *   6. Split the response stream on the protocol's event delimiter.
 *   7. Call `parse_event()` for each event, forwarding chunks and signals.
 *   8. Call `set_last_usage()` when usage data is reported.
 *   9. Emit `make_final()` on `done` or at the end of the HTTP response.
 *
 * @note For Gemini, AuthInfo::query_params are appended to the URL as
 *       `?key=…` query parameters.
 */
class HttpLLMProvider : public LLMProvider,
                        public std::enable_shared_from_this<HttpLLMProvider> {
public:
    /**
     * @param base_url       Root URL for this endpoint, e.g.
     *                       `"https://api.groq.com/openai/v1"`.
     *                       Must not have a trailing slash.
     * @param cred_source    Credential source (API key, OAuth, or null-auth).
     *                       `get_auth()` is called once per stream invocation
     *                       so token refresh happens transparently.
     * @param default_model  Model used when `ChatRequest::model` is empty.
     * @param protocol       Wire-protocol implementation.  Cloned once per
     *                       `stream_response()` call to isolate per-session state.
     */
    HttpLLMProvider(std::string                                    base_url,
                    std::shared_ptr<core::auth::ICredentialSource> cred_source,
                    std::string                                    default_model,
                    std::unique_ptr<protocols::ApiProtocolBase>    protocol);

    void stream_response(const ChatRequest&                    request,
                         std::function<void(const StreamChunk&)> callback) override;

    /**
     * @brief Return whether synthetic token-cost estimation should run.
     *
     * OAuth-backed providers (subscription plans) return false so we don't
     * display misleading per-token USD totals in session reports.
     */
    [[nodiscard]] bool should_estimate_cost() const override;
    void reset_conversation_state() override;

    /**
     * @brief Return the maximum context size for the default model.
     *        Recognizes common models and their context window sizes.
     */
    [[nodiscard]] int max_context_size() const noexcept override;

    /**
     * @brief Get full model information from the registry.
     * @return ModelInfo if found in registry, nullopt otherwise.
     */
    [[nodiscard]] std::optional<ModelInfo> get_model_info() const;

    /**
     * @brief Validate request parameters against model constraints.
     * 
     * Checks:
     * - max_tokens is within model limits
     * - temperature is within valid range
     * - model supports requested features (tools, JSON mode, etc.)
     * 
     * @param request The chat request to validate
     * @return Empty vector if valid, list of error messages if invalid
     */
    [[nodiscard]] std::vector<std::string> validate_request(const ChatRequest& request) const;

    /**
     * @brief Check if the default model supports a specific capability.
     */
    [[nodiscard]] bool supports(ModelCapability cap) const;

    /**
     * @brief Estimate cost for a request.
     * @return Cost in USD, or -1.0 if model not in registry.
     */
    [[nodiscard]] double estimate_cost(int input_tokens, int output_tokens) const;

private:
    std::string                                    base_url_;
    std::shared_ptr<core::auth::ICredentialSource> cred_source_;
    std::string                                    default_model_;
    std::unique_ptr<protocols::ApiProtocolBase>    protocol_;
};

} // namespace core::llm
