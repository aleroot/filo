#include "WebBackendAdapters.hpp"

#include "McpHttpToolCall.hpp"
#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../llm/ProviderDefinition.hpp"
#include "../llm/QwenSubscription.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace core::tools::web {
namespace {

namespace subscription = core::llm::qwen_subscription;

constexpr std::string_view kBackendName = "qwen-subscription-web-search";
constexpr std::string_view kService = "Qwen Subscription web search";
constexpr std::string_view kRequestedProtocolVersion = "2025-11-25";
constexpr std::size_t kMaxQueryCodePoints = 500;

[[nodiscard]] bool is_subscription_provider(const core::llm::ProviderMetadata& metadata) {
    const auto* definition = core::llm::find_builtin_provider_definition(metadata.provider_name);
    return definition && definition->prefix == subscription::kProviderName;
}

[[nodiscard]] std::size_t code_points(std::string_view text) {
    return static_cast<std::size_t>(std::ranges::count_if(text, [](char ch) {
        return (static_cast<unsigned char>(ch) & 0xC0) != 0x80;
    }));
}

[[nodiscard]] std::int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

/// The gateway's own wording for account failures; the shared one otherwise.
[[nodiscard]] std::string describe(const detail::McpHttpFailure& failure) {
    if (failure.kind == detail::McpHttpFailure::Kind::Http) {
        if (failure.status == 401) return std::string(subscription::kSignInRequiredMessage);
        if (auto message = subscription::gateway_error_message(
                static_cast<int>(failure.status), failure.detail, now_unix_seconds())) {
            return std::move(*message);
        }
        return std::string(kService) + " failed (HTTP " + std::to_string(failure.status) + ").";
    }
    return detail::describe_mcp_failure(failure, kService);
}

/// The service answers in prose and lists its sources as bare URL lines.
[[nodiscard]] SearchResponse search_response(std::string answer, int limit) {
    SearchResponse response{.backend = std::string(kBackendName), .answer = std::move(answer)};
    const std::string_view text = response.answer;
    const auto max_hits = static_cast<std::size_t>(std::max(limit, 1));
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find('\n', start);
        const std::string_view line = core::utils::str::trim_ascii_view(text.substr(
            start, end == std::string_view::npos ? std::string_view::npos : end - start));
        const bool url = (line.starts_with("https://") || line.starts_with("http://"))
            && line.find_first_of(" \t") == std::string_view::npos;
        if (url && response.results.size() < max_hits) {
            detail::add_hit_if_present(response, SearchHit{.url = std::string(line)});
        }
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return response;
}

class QwenSubscriptionWebSearchBackend final : public IWebSearchBackend {
public:
    explicit QwenSubscriptionWebSearchBackend(QwenSubscriptionWebSearchOptions options)
        : options_(std::move(options)) {
        if (options_.search_url.empty()) options_.search_url = std::string(subscription::kWebSearchUrl);
        if (options_.timeout_ms <= 0) options_.timeout_ms = 30000;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return kBackendName; }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata && metadata->credential_source && is_subscription_provider(*metadata);
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request, const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata || !supports(context)) {
            return std::unexpected("Qwen Subscription web search requires an active Qwen Subscription session.");
        }
        if (!request.domains.allowed_domains.empty() || !request.domains.blocked_domains.empty()) {
            return std::unexpected(
                "Qwen Subscription web search does not support allowed_domains or blocked_domains filters.");
        }
        if (code_points(request.query) > kMaxQueryCodePoints) {
            return std::unexpected("Qwen Subscription search queries must not exceed 500 characters.");
        }

        core::utils::JsonWriter arguments(64 + request.query.size());
        {
            auto _root = arguments.object();
            arguments.kv_str("query", request.query);
        }
        try {
            auto answer = detail::call_mcp_http_tool({
                .url = options_.search_url,
                .tool = "web_search",
                .arguments_json = std::move(arguments).take(),
                .protocol_version = kRequestedProtocolVersion,
                .credentials = metadata->credential_source.get(),
                .timeout_ms = options_.timeout_ms,
                .cancelled = context.cancellation_requested,
            });
            if (!answer) return std::unexpected(describe(answer.error()));
            std::string text = core::utils::str::trim_ascii_copy(*answer);
            if (text.empty()) return std::unexpected(std::string(kService) + " returned no text.");
            return search_response(std::move(text), request.limit);
        } catch (const std::exception& error) {
            // Credential failures (for example an expired session) arrive as
            // exceptions; report them as a failed search.
            return std::unexpected(std::string(error.what()));
        }
    }

private:
    QwenSubscriptionWebSearchOptions options_;
};

} // namespace

std::shared_ptr<IWebSearchBackend> make_qwen_subscription_web_search_backend(
    QwenSubscriptionWebSearchOptions options) {
    return std::make_shared<QwenSubscriptionWebSearchBackend>(std::move(options));
}

} // namespace core::tools::web
