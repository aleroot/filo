#include "QwenSubscriptionProtocol.hpp"

#include "core/llm/ModelCatalogProvider.hpp"
#include "core/llm/QwenModelTraits.hpp"
#include "core/llm/QwenSubscription.hpp"
#include "core/llm/StrictToolPolicy.hpp"
#include "core/llm/catalog/QwenSubscriptionModelCatalogProvider.hpp"
#include "core/version/Version.hpp"

#include <chrono>

namespace core::llm::protocols {

namespace subscription = core::llm::qwen_subscription;

ReasoningCapabilities QwenSubscriptionProtocol::reasoning_capabilities(
    std::string_view) const noexcept {
    // Off plus three budgets. xhigh lands on the high budget and max, which
    // is not offered, falls back to high as well.
    return ReasoningCapability::Effort | ReasoningCapability::Disable
        | ReasoningCapability::MapsMinimalToLow | ReasoningCapability::MapsXHighToHigh;
}

std::string QwenSubscriptionProtocol::serialize(const ChatRequest& req) const {
    Serializer::Options options;
    options.strict_tools = configured_strict_tool_dialect(ToolSchemaWire::OpenAI, req.model);
    // Qwen replays its own thinking across tool turns and expects `content:""`
    // rather than null on a tool-call-only assistant turn.
    options.reasoning_content_policy = Serializer::ReasoningContentPolicy::NonEmptyOwned;
    options.reasoning_protocol = std::string(name());
    options.empty_assistant_content_policy =
        qwen_model_requires_nonnull_assistant_content(req.model)
        ? Serializer::EmptyAssistantContentPolicy::EmptyString
        : Serializer::EmptyAssistantContentPolicy::EmptyStringWhenReasoning;

    std::string payload = Serializer::serialize(req, options);
    if (!payload.ends_with('}')) return payload;
    payload.pop_back();

    const std::string_view effort = req.effort.empty()
        ? std::string_view(default_effort_) : std::string_view(req.effort);
    if (const auto budget = subscription::thinking_budget(effort)) {
        payload += R"(,"enable_thinking":true,"thinking_budget":)" + std::to_string(*budget);
    } else {
        payload += R"(,"enable_thinking":false)";
    }
    if (req.stream) payload += R"(,"stream_options":{"include_usage":true})";
    payload += '}';
    return payload;
}

cpr::Header QwenSubscriptionProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers = OpenAIProtocol::build_headers(auth);
    headers["User-Agent"] = std::string(core::version::user_agent);
    return headers;
}

std::string QwenSubscriptionProtocol::format_error_message(const HttpResponse& response) const {
    if (response.status_code == 401) return std::string(subscription::kSignInRequiredMessage);
    const std::int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (auto message = subscription::gateway_error_message(
            response.status_code, response.body, now)) {
        if (response.status_code == 402) return *message;
        return "Qwen Subscription request failed (HTTP "
            + std::to_string(response.status_code) + "): " + *message;
    }
    if (response.status_code == 429) return "Qwen Subscription: too many requests. Please try again later.";
    return "Qwen Subscription request failed (HTTP " + std::to_string(response.status_code) + ").";
}

std::unique_ptr<ModelCatalogProvider> QwenSubscriptionProtocol::model_catalog(
    std::string_view provider_name) const {
    return std::make_unique<QwenSubscriptionModelCatalogProvider>(std::string(provider_name));
}

} // namespace core::llm::protocols
