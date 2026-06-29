#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../auth/KimiOAuthFlow.hpp"
#include "../utils/JsonWriter.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <format>
#include <memory>
#include <string>
#include <string_view>

namespace core::tools::web {

namespace {

constexpr int kSearchTimeoutMs = 60000;

[[nodiscard]] bool is_kimi_coding_endpoint(
    const core::llm::ProviderMetadata& metadata) {
    return metadata.api_type == core::config::ApiType::Kimi
        && detail::contains_case_insensitive(metadata.base_url, "api.kimi.com/coding");
}

void add_auth_headers(cpr::Header& headers, const core::auth::AuthInfo& auth) {
    for (const auto& [key, value] : auth.headers) {
        headers[key] = value;
    }
}

[[nodiscard]] cpr::Header kimi_headers(const core::auth::AuthInfo& auth,
                                       std::string_view tool_call_id) {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
        {"User-Agent", core::auth::KimiOAuthFlow::getUserAgent()},
    };

    std::string device_id;
    if (const auto it = auth.properties.find("device_id");
        it != auth.properties.end()) {
        device_id = it->second;
    }
    for (const auto& [key, value] : core::auth::KimiOAuthFlow::getCommonHeaders(device_id)) {
        headers[key] = value;
    }
    add_auth_headers(headers, auth);
    if (!tool_call_id.empty()) {
        headers["X-Msh-Tool-Call-Id"] = std::string(tool_call_id);
    }
    return headers;
}

[[nodiscard]] std::string search_payload(const SearchRequest& request) {
    core::utils::JsonWriter writer(512);
    {
        auto _root = writer.object();
        writer.kv_str("text_query", request.query).comma()
              .kv_num("limit", request.limit).comma()
              .kv_bool("enable_page_crawling", request.include_page_content).comma()
              .kv_num("timeout_seconds", 30);
    }
    return std::move(writer).take();
}

void parse_search_array(SearchResponse& response, simdjson::dom::array array) {
    for (auto element : array) {
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        detail::parse_search_hit_object(response, object);
    }
}

void parse_results(SearchResponse& response, simdjson::dom::element doc) {
    simdjson::dom::array array;
    if (doc["search_results"].get_array().get(array) == simdjson::SUCCESS) {
        parse_search_array(response, array);
    }
    if (doc["results"].get_array().get(array) == simdjson::SUCCESS) {
        parse_search_array(response, array);
    }
    if (doc["data"].get_array().get(array) == simdjson::SUCCESS) {
        parse_search_array(response, array);
    }

    std::string_view answer;
    if (doc["answer"].get(answer) == simdjson::SUCCESS
        || doc["summary"].get(answer) == simdjson::SUCCESS) {
        response.answer = std::string(answer);
    }
}

class KimiWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "kimi-coding-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && is_kimi_coding_endpoint(*metadata)
            && metadata->credential_source != nullptr;
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source) {
            return std::unexpected("Kimi search requires an active HTTP provider with credentials.");
        }
        if (!request.domains.allowed_domains.empty()
            || !request.domains.blocked_domains.empty()) {
            return std::unexpected(
                "Kimi search does not support allowed_domains or blocked_domains filters. "
                "Remove the filters or use an Anthropic/Claude provider for domain-filtered search.");
        }

        const auto auth = metadata->credential_source->get_auth();
        const cpr::Response response = cpr::Post(
            cpr::Url{detail::append_path(metadata->base_url, "/search")},
            kimi_headers(auth, context.tool_call_id),
            cpr::Body{search_payload(request)},
            cpr::Timeout{kSearchTimeoutMs});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected("Kimi search request failed: " + response.error.message);
        }
        if (response.status_code < 200 || response.status_code >= 300) {
            return std::unexpected(std::format(
                "Kimi search failed with HTTP {}: {}",
                response.status_code,
                response.text.substr(0, 1000)));
        }

        simdjson::dom::parser parser;
        simdjson::padded_string padded(response.text);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            return std::unexpected("Kimi search response was not valid JSON.");
        }

        SearchResponse out{.backend = std::string(name())};
        parse_results(out, doc);
        return out;
    }
};

} // namespace

std::shared_ptr<IWebSearchBackend> make_kimi_web_search_backend() {
    return std::make_shared<KimiWebSearchBackend>();
}

} // namespace core::tools::web
