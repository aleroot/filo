#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../auth/MiniMaxClientIdentity.hpp"
#include "../llm/ProviderDefinition.hpp"
#include "../llm/protocols/MiniMaxProtocol.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::web {

namespace {

constexpr std::size_t kMaxSearchBytes = 4 * 1024 * 1024;

[[nodiscard]] bool host_in(std::string_view host, const std::vector<std::string>& domains) {
    return std::ranges::any_of(domains, [&](const std::string& entry) {
        const auto domain = detail::normalize_search_domain(entry);
        return !domain.empty()
            && (host == domain
                || (host.size() > domain.size() && host.ends_with(domain)
                    && host[host.size() - domain.size() - 1] == '.'));
    });
}

/// The plan search API has no domain filters, so they are applied to its hits.
[[nodiscard]] bool passes_filters(std::string_view url, const DomainFilters& filters) {
    const auto host = detail::normalize_search_domain(url);
    if (!filters.allowed_domains.empty() && !host_in(host, filters.allowed_domains)) return false;
    return !host_in(host, filters.blocked_domains);
}

[[nodiscard]] std::optional<std::string> subscription_key(const core::llm::ProviderMetadata& metadata) {
    const auto auth = metadata.credential_source->get_auth();
    for (const auto& [name, value] : auth.headers) {
        if (core::utils::ascii::iequals(name, "x-api-key") && !value.empty()) return value;
    }
    return std::nullopt;
}

class MiniMaxWebSearchBackend final : public IWebSearchBackend {
public:
    explicit MiniMaxWebSearchBackend(MiniMaxWebSearchOptions options) : options_(std::move(options)) {}

    [[nodiscard]] std::string_view name() const noexcept override {
        return "minimax-plan-search";
    }

    /// Only the official M Plan endpoint: its Subscription Key is the search
    /// credential, and a custom proxy's key must never be sent to MiniMax.
    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata || !metadata->credential_source
            || !core::llm::is_official_minimax_base_url(metadata->base_url)) {
            return false;
        }
        const auto* definition = core::llm::find_builtin_provider_definition(metadata->provider_name);
        return definition && definition->registry_provider == "minimax"
            && definition->billing_kind == core::auth::BillingKind::Subscription;
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata || !metadata->credential_source) {
            return std::unexpected("MiniMax search requires the M Plan provider with a Subscription Key.");
        }
        const auto key = subscription_key(*metadata);
        if (!key) return std::unexpected("MiniMax search requires a Subscription Key; run `filo --auth minimax`.");

        cpr::Header headers{{"Authorization", "Bearer " + *key}};
        core::auth::minimax_cli::apply_identity_headers(headers);
        std::string body;
        const cpr::Response response = cpr::Post(
            cpr::Url{options_.search_url}, headers,
            cpr::Body{minimax_web_search_request_json(request)},
            cpr::Timeout{options_.timeout_ms}, cpr::Redirect{false},
            cpr::WriteCallback{[&](std::string_view bytes, intptr_t) {
                if (bytes.size() > kMaxSearchBytes - body.size()) return false;
                body.append(bytes);
                return !context.cancellation_requested || !context.cancellation_requested();
            }});
        if (context.cancellation_requested && context.cancellation_requested()) {
            return std::unexpected("MiniMax search was cancelled.");
        }
        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected("MiniMax search request failed: " + response.error.message);
        }
        if (response.status_code != 200) {
            return std::unexpected(
                core::llm::protocols::describe_minimax_error(static_cast<int>(response.status_code), body));
        }
        return parse_minimax_web_search_response(body, request);
    }

private:
    MiniMaxWebSearchOptions options_;
};

} // namespace

std::string minimax_web_search_request_json(const SearchRequest& request) {
    core::utils::JsonWriter writer(128);
    {
        auto _root = writer.object();
        writer.kv_str("q", request.query);
    }
    return std::move(writer).take();
}

std::expected<SearchResponse, std::string>
parse_minimax_web_search_response(std::string_view body, const SearchRequest& request) {
    // Failures arrive as HTTP 200 with a non-zero base_resp.status_code.
    if (core::llm::protocols::minimax_error_code(body) != 0) {
        return std::unexpected(core::llm::protocols::describe_minimax_error(200, body));
    }
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::unexpected("MiniMax search response was not valid JSON.");
    }
    SearchResponse out{.backend = "minimax-plan-search"};
    simdjson::dom::array organic;
    if (doc["organic"].get_array().get(organic) != simdjson::SUCCESS) return out;
    for (const auto element : organic) {
        if (static_cast<int>(out.results.size()) >= request.limit) break;
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        SearchHit hit{
            .title = core::utils::json::first_string_field_or_empty(object, {"title"}),
            .url = core::utils::json::first_string_field_or_empty(object, {"link", "url"}),
            .snippet = core::utils::json::first_string_field_or_empty(object, {"snippet"}),
        };
        if (!passes_filters(hit.url, request.domains)) continue;
        // The publication date is the only freshness signal the API returns.
        if (const auto date = core::utils::json::first_string_field_or_empty(object, {"date"}); !date.empty()) {
            hit.snippet = hit.snippet.empty() ? date : date + " — " + hit.snippet;
        }
        detail::add_hit_if_present(out, std::move(hit));
    }
    return out;
}

std::shared_ptr<IWebSearchBackend> make_minimax_web_search_backend(MiniMaxWebSearchOptions options) {
    return std::make_shared<MiniMaxWebSearchBackend>(std::move(options));
}

} // namespace core::tools::web
