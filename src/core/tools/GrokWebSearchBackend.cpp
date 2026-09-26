#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../auth/XaiGrokClientIdentity.hpp"
#include "../llm/protocols/GrokBuildEndpoint.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::web {

namespace {

// The reference client does not lower reasoning effort, so this matches
// Filo's Grok stream inactivity budget rather than the 60s used by backends
// that do not run a reasoning model.
constexpr int kSearchTimeoutMs = 600000;
constexpr std::size_t kMaxDomainFilters = 5;
constexpr int kMaxOutputTokens = 8192;
constexpr std::string_view kEmptySearchAnswer = "No search results found.";

[[nodiscard]] bool is_grok_search_host(std::string_view base_url) {
    using core::llm::protocols::grok_build::is_official_proxy;
    using core::llm::protocols::grok_build::is_public_api;
    return is_public_api(base_url) || is_official_proxy(base_url);
}

[[nodiscard]] std::string responses_url(std::string_view base_url) {
    std::string base = core::utils::str::trim_trailing_slashes(base_url);
    if (core::utils::str::to_lower_ascii_copy(base).ends_with("/responses")) {
        return base;
    }
    return base + "/responses";
}

[[nodiscard]] std::expected<std::vector<std::string>, std::string>
normalized_domains(const std::vector<std::string>& domains,
                   std::string_view label) {
    std::vector<std::string> out;
    out.reserve(std::min(domains.size(), kMaxDomainFilters));
    for (const auto& domain : domains) {
        std::string normalized = detail::normalize_search_domain(domain);
        if (normalized.empty()) continue;
        if (std::ranges::find(out, normalized) != out.end()) continue;
        if (out.size() >= kMaxDomainFilters) {
            return std::unexpected(std::format(
                "xAI web search accepts at most {} {}.",
                kMaxDomainFilters,
                label));
        }
        out.push_back(std::move(normalized));
    }
    return out;
}

void add_url_hit(SearchResponse& response, std::string url) {
    if (url.empty()) return;
    SearchHit hit;
    hit.url = std::move(url);
    detail::add_hit_if_present(response, std::move(hit));
}

void parse_output_text(SearchResponse& response,
                       const simdjson::dom::object& content) {
    const auto type = core::utils::json::first_string_field_or_empty(content, {"type"});
    if (type != "output_text") return;

    std::string_view text;
    if (content["text"].get(text) == simdjson::SUCCESS && !text.empty()) {
        if (!response.answer.empty()) response.answer += "\n\n";
        response.answer += text;
    }

    simdjson::dom::array annotations;
    if (content["annotations"].get_array().get(annotations) != simdjson::SUCCESS) return;
    for (auto annotation_element : annotations) {
        simdjson::dom::object annotation;
        if (annotation_element.get_object().get(annotation) != simdjson::SUCCESS) continue;
        // grok-build extract_citations keeps Annotation::UrlCitation only.
        if (core::utils::json::first_string_field_or_empty(annotation, {"type"})
            != "url_citation") {
            continue;
        }
        detail::parse_search_hit_object(response, annotation);
    }
}

void parse_message_item(SearchResponse& response,
                        const simdjson::dom::object& item) {
    simdjson::dom::array content;
    if (item["content"].get_array().get(content) != simdjson::SUCCESS) return;
    for (auto content_element : content) {
        simdjson::dom::object content_object;
        if (content_element.get_object().get(content_object) != simdjson::SUCCESS) continue;
        parse_output_text(response, content_object);
    }
}

void parse_citation_array(SearchResponse& response, simdjson::dom::array citations) {
    for (auto element : citations) {
        std::string_view url;
        if (element.get(url) == simdjson::SUCCESS) {
            add_url_hit(response, std::string(url));
            continue;
        }
        simdjson::dom::object object;
        if (element.get_object().get(object) == simdjson::SUCCESS) {
            detail::parse_search_hit_object(response, object);
        }
    }
}

void parse_output_array(SearchResponse& response, simdjson::dom::array output) {
    for (auto item_element : output) {
        simdjson::dom::object item;
        if (item_element.get_object().get(item) != simdjson::SUCCESS) continue;
        if (core::utils::json::first_string_field_or_empty(item, {"type"}) == "message") {
            parse_message_item(response, item);
        }
    }
}

[[nodiscard]] std::string read_error_message(simdjson::dom::element doc) {
    simdjson::dom::object error_object;
    if (doc["error"].get_object().get(error_object) == simdjson::SUCCESS) {
        const std::string message = core::utils::json::first_string_field_or_empty(
            error_object, {"message", "detail", "error"});
        if (!message.empty()) return message;
    }
    std::string_view message;
    if (doc["message"].get(message) == simdjson::SUCCESS && !message.empty()) {
        return std::string(message);
    }
    if (doc["error"].get(message) == simdjson::SUCCESS && !message.empty()) {
        return std::string(message);
    }
    return {};
}

[[nodiscard]] cpr::Header grok_headers(const core::auth::AuthInfo& auth,
                                       std::string_view base_url) {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
    };
    for (const auto& [key, value] : auth.headers) {
        headers[key] = value;
    }

    // grok-build inject_proxy_headers: every web-search request carries the
    // client identity, and cli-chat-proxy also requires its auth middleware
    // headers. Chat-turn headers (x-grok-conv-id and friends) are not part of
    // that side call.
    if (core::llm::protocols::grok_build::is_official_proxy(base_url)) {
        core::auth::xai_grok::apply_proxy_identity_headers(headers);
    } else {
        core::auth::xai_grok::remove_proxy_identity_headers(headers);
        if (core::llm::protocols::grok_build::is_public_api(base_url)) {
            headers["x-grok-client-identifier"] =
                std::string(core::auth::xai_grok::kClientIdentifier);
            headers["x-grok-client-version"] =
                std::string(core::auth::xai_grok::kClientVersion);
        }
    }
    return headers;
}

class GrokWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "grok-responses-web-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && metadata->credential_source != nullptr
            && is_grok_search_host(metadata->base_url);
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source) {
            return std::unexpected(
                "Grok web search requires an active Grok provider with credentials.");
        }
        if (!is_grok_search_host(metadata->base_url)) {
            return std::unexpected(
                "Grok web search is only available for api.x.ai or the Grok account session.");
        }

        // Same rule as the other provider backends: search as the model the
        // user selected. grok-4.7 is a valid web_search model; pinning the
        // side call to grok-4.6 was grok-build's separate default, not an API
        // limit.
        const std::string model = !context.model_name.empty()
            ? context.model_name
            : metadata->default_model;
        auto payload = grok_web_search_request_json(request, model);
        if (!payload.has_value()) return std::unexpected(payload.error());

        const auto auth = metadata->credential_source->get_auth();
        const cpr::Response response = cpr::Post(
            cpr::Url{responses_url(metadata->base_url)},
            grok_headers(auth, metadata->base_url),
            cpr::Body{std::move(*payload)},
            cpr::Timeout{kSearchTimeoutMs});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected(
                "Grok web search request failed: " + response.error.message);
        }

        simdjson::dom::parser parser;
        simdjson::padded_string padded(response.text);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            if (response.status_code < 200 || response.status_code >= 300) {
                return std::unexpected(std::format(
                    "Grok web search failed with HTTP {}: {}",
                    response.status_code,
                    response.text.substr(0, 1000)));
            }
            return std::unexpected("Grok web search response was not valid JSON.");
        }

        if (response.status_code < 200 || response.status_code >= 300) {
            std::string message = read_error_message(doc);
            if (response.status_code == 401) {
                if (!message.empty()) message += " ";
                message += "For a Grok account session, run 'filo --auth grok' again. "
                           "For API access, check XAI_API_KEY.";
            }
            return std::unexpected(std::format(
                "Grok web search failed with HTTP {}: {}",
                response.status_code,
                message.empty() ? response.text.substr(0, 1000) : message));
        }

        auto parsed = parse_grok_web_search_response(response.text, request.limit);
        if (!parsed.has_value()) return parsed;
        parsed->backend = std::string(name());
        return parsed;
    }
};

} // namespace

std::expected<std::string, std::string>
grok_web_search_request_json(const SearchRequest& request,
                             std::string_view model) {
    if (model.empty()) {
        return std::unexpected("Grok web search requires an active model.");
    }

    auto allowed = normalized_domains(request.domains.allowed_domains, "allowed_domains");
    if (!allowed.has_value()) return std::unexpected(allowed.error());
    auto blocked = normalized_domains(request.domains.blocked_domains, "blocked_domains");
    if (!blocked.has_value()) return std::unexpected(blocked.error());
    // xAI rejects a web_search tool that sets both lists. grok-build's
    // WebSearchOptions::validate enforces the same rule, capped at 5.
    if (!allowed->empty() && !blocked->empty()) {
        return std::unexpected(
            "xAI web search accepts allowed_domains or blocked_domains, not both.");
    }

    // Wire body matches WebSearchClient::build_request_json plus
    // web_search_sampling_config: raw query as input, store false, temperature
    // 0.1, top_p 0.95, max_output_tokens 8192. No reasoning override and no
    // stream flag — the reference omits both.
    core::utils::JsonWriter writer(2048);
    {
        auto _root = writer.object();
        writer.kv_str("model", model).comma()
              .kv_bool("store", false).comma()
              .kv_float("temperature", 0.1).comma()
              .kv_float("top_p", 0.95).comma()
              .kv_num("max_output_tokens", kMaxOutputTokens).comma()
              .kv_str("input", request.query);
        writer.comma().key("tools");
        {
            auto _tools = writer.array();
            {
                auto _tool = writer.object();
                writer.kv_str("type", "web_search");
                if (!allowed->empty() || !blocked->empty()) {
                    writer.comma().key("filters");
                    {
                        auto _filters = writer.object();
                        if (!allowed->empty()) {
                            detail::write_string_array(writer, "allowed_domains", *allowed);
                        } else {
                            detail::write_string_array(writer, "excluded_domains", *blocked);
                        }
                    }
                }
            }
        }
    }
    return std::move(writer).take();
}

std::expected<SearchResponse, std::string>
parse_grok_web_search_response(std::string_view body, int limit) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::unexpected("Grok web search response was not valid JSON.");
    }

    SearchResponse response;
    simdjson::dom::array output;
    if (doc["output"].get_array().get(output) == simdjson::SUCCESS) {
        parse_output_array(response, output);
    } else if (doc["response"]["output"].get_array().get(output) == simdjson::SUCCESS) {
        parse_output_array(response, output);
    }

    if (response.answer.empty()) {
        std::string_view output_text;
        if (doc["output_text"].get(output_text) == simdjson::SUCCESS && !output_text.empty()) {
            response.answer = std::string(output_text);
        } else if (doc["response"]["output_text"].get(output_text) == simdjson::SUCCESS
                   && !output_text.empty()) {
            response.answer = std::string(output_text);
        }
    }

    simdjson::dom::array citations;
    if (doc["citations"].get_array().get(citations) == simdjson::SUCCESS) {
        parse_citation_array(response, citations);
    } else if (doc["response"]["citations"].get_array().get(citations) == simdjson::SUCCESS) {
        parse_citation_array(response, citations);
    }

    if (limit > 0 && response.results.size() > static_cast<std::size_t>(limit)) {
        response.results.resize(static_cast<std::size_t>(limit));
    }
    // WebSearchClient::search uses output_text().unwrap_or("No search results found.").
    if (response.answer.empty()) {
        response.answer = std::string(kEmptySearchAnswer);
    }
    return response;
}

std::shared_ptr<IWebSearchBackend> make_grok_web_search_backend() {
    return std::make_shared<GrokWebSearchBackend>();
}

} // namespace core::tools::web
