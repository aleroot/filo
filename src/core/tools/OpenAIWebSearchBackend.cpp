#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../llm/ProviderClientIdentity.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::web {

namespace {

constexpr int kSearchTimeoutMs = 60000;
constexpr std::size_t kMaxDomainFilters = 100;

[[nodiscard]] std::string normalized_base_url(std::string_view base_url) {
    return core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_trailing_slashes(base_url));
}

[[nodiscard]] bool is_openai_responses_endpoint(
    const core::llm::ProviderMetadata& metadata) {
    if (metadata.api_type != core::config::ApiType::OpenAI) return false;

    const std::string normalized = normalized_base_url(metadata.base_url);
    return normalized == "https://api.openai.com"
        || normalized == "https://api.openai.com/v1"
        || normalized == "https://api.openai.com/v1/responses"
        || normalized == "https://chatgpt.com/backend-api/codex"
        || normalized == "https://chatgpt.com/backend-api/codex/responses";
}

[[nodiscard]] bool is_codex_backend(std::string_view base_url) {
    return core::utils::str::contains_case_insensitive(
        base_url,
        "chatgpt.com/backend-api/codex");
}

[[nodiscard]] std::string responses_url(std::string_view base_url) {
    std::string base = core::utils::str::trim_trailing_slashes(base_url);
    const std::string lower = core::utils::str::to_lower_ascii_copy(base);
    if (lower == "https://api.openai.com") {
        return base + "/v1/responses";
    }
    if (lower.ends_with("/responses")) {
        return base;
    }
    return base + "/responses";
}

void add_auth_headers(cpr::Header& headers, const core::auth::AuthInfo& auth) {
    for (const auto& [key, value] : auth.headers) {
        headers[key] = value;
    }
}

[[nodiscard]] std::string request_id_for(const ToolInvocationContext& context) {
    if (!context.session_context.session_id.empty()) {
        return context.session_context.session_id;
    }
    if (!context.tool_call_id.empty()) {
        return context.tool_call_id;
    }
    return "filo";
}

[[nodiscard]] cpr::Header openai_headers(
    const core::auth::AuthInfo& auth,
    const core::llm::ProviderMetadata& metadata,
    const ToolInvocationContext& context) {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
    };
    add_auth_headers(headers, auth);

    if (const auto it = auth.properties.find("account_id");
        it != auth.properties.end() && !it->second.empty()
        && !detail::find_header_case_insensitive(headers, "chatgpt-account-id").has_value()) {
        headers["chatgpt-account-id"] = it->second;
    }

    if (is_codex_backend(metadata.base_url)) {
        const std::string request_id = request_id_for(context);
        headers["x-client-request-id"] = request_id;
        headers["x-codex-installation-id"] =
            metadata.client_identity_source->installation_id();
        headers["x-codex-window-id"] = request_id + ":0";
        headers["x-responsesapi-include-timing-metrics"] = "true";
    }

    return headers;
}

[[nodiscard]] std::string normalize_domain(std::string_view value) {
    std::string domain = core::utils::str::trim_ascii_copy(value);
    if (domain.empty()) return {};

    const std::string lowered = core::utils::str::to_lower_ascii_copy(domain);
    if (lowered.starts_with("http://")) {
        domain.erase(0, std::string("http://").size());
    } else if (lowered.starts_with("https://")) {
        domain.erase(0, std::string("https://").size());
    }

    if (const std::size_t at = domain.rfind('@'); at != std::string::npos) {
        domain.erase(0, at + 1);
    }
    if (const std::size_t end = domain.find_first_of("/?#"); end != std::string::npos) {
        domain.resize(end);
    }
    if (!domain.empty() && domain.front() != '[') {
        if (const std::size_t port = domain.find(':'); port != std::string::npos) {
            domain.resize(port);
        }
    }
    while (!domain.empty() && domain.back() == '.') {
        domain.pop_back();
    }
    return core::utils::str::to_lower_ascii_copy(domain);
}

[[nodiscard]] std::vector<std::string> normalized_domains(
    const std::vector<std::string>& domains) {
    std::vector<std::string> out;
    out.reserve(std::min(domains.size(), kMaxDomainFilters));
    for (const auto& domain : domains) {
        if (out.size() >= kMaxDomainFilters) break;
        std::string normalized = normalize_domain(domain);
        if (normalized.empty()) continue;
        if (std::ranges::find(out, normalized) != out.end()) continue;
        out.push_back(std::move(normalized));
    }
    return out;
}

void write_string_array(core::utils::JsonWriter& writer,
                        std::string_view key,
                        const std::vector<std::string>& values) {
    writer.key(key);
    {
        auto _array = writer.array();
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0) writer.comma();
            writer.str(values[i]);
        }
    }
}

void write_domain_filters(core::utils::JsonWriter& writer,
                          const SearchRequest& request) {
    const auto allowed = normalized_domains(request.domains.allowed_domains);
    const auto blocked = normalized_domains(request.domains.blocked_domains);
    if (allowed.empty() && blocked.empty()) return;

    writer.comma().key("filters");
    {
        auto _filters = writer.object();
        bool first = true;
        if (!allowed.empty()) {
            write_string_array(writer, "allowed_domains", allowed);
            first = false;
        }
        if (!blocked.empty()) {
            if (!first) writer.comma();
            write_string_array(writer, "blocked_domains", blocked);
        }
    }
}

[[nodiscard]] std::string search_prompt(const SearchRequest& request) {
    std::string prompt = "Perform a web search for this query and return concise "
                         "findings with source URLs. Limit the answer to at most ";
    prompt += std::to_string(request.limit);
    prompt += " cited sources.\n\nQuery: ";
    prompt += request.query;
    if (request.include_page_content) {
        prompt += "\n\nUse enough page context to make the citations useful.";
    }
    return prompt;
}

[[nodiscard]] std::string search_payload(const SearchRequest& request,
                                         std::string_view model) {
    core::utils::JsonWriter writer(2048);
    {
        auto _root = writer.object();
        writer.kv_str("model", model).comma()
              .kv_bool("stream", false).comma()
              .kv_bool("store", false).comma()
              .kv_str("input", search_prompt(request)).comma()
              .kv_str("tool_choice", "required").comma()
              .key("tools");
        {
            auto _tools = writer.array();
            {
                auto _tool = writer.object();
                writer.kv_str("type", "web_search").comma()
                      .kv_str("search_context_size",
                              request.include_page_content ? "high" : "medium").comma()
                      .kv_bool("external_web_access", true);
                write_domain_filters(writer, request);
            }
        }
        writer.comma().key("include");
        {
            auto _include = writer.array();
            writer.str("web_search_call.action.sources");
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::optional<int64_t> read_i64(
    const simdjson::dom::object& object,
    std::string_view key) {
    int64_t value = 0;
    if (object[key].get(value) == simdjson::SUCCESS) return value;
    return std::nullopt;
}

void parse_url_citation(SearchResponse& response,
                        const simdjson::dom::object& annotation,
                        std::string_view text) {
    const auto type = core::utils::json::first_string_field_or_empty(annotation, {"type"});
    if (type != "url_citation") return;

    SearchHit hit{
        .title = core::utils::json::first_string_field_or_empty(annotation, {"title"}),
        .url = core::utils::json::first_string_field_or_empty(annotation, {"url"}),
    };

    const auto start = read_i64(annotation, "start_index");
    const auto end = read_i64(annotation, "end_index");
    if (start.has_value() && end.has_value()
        && *start >= 0 && *end > *start
        && static_cast<std::size_t>(*end) <= text.size()) {
        hit.snippet = std::string(text.substr(
            static_cast<std::size_t>(*start),
            static_cast<std::size_t>(*end - *start)));
    }

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
        parse_url_citation(response, annotation, text);
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

void parse_source_array(SearchResponse& response,
                        simdjson::dom::array sources) {
    for (auto source_element : sources) {
        simdjson::dom::object source_object;
        if (source_element.get_object().get(source_object) != simdjson::SUCCESS) continue;
        detail::parse_search_hit_object(response, source_object);
    }
}

void parse_web_search_call(SearchResponse& response,
                           const simdjson::dom::object& item) {
    simdjson::dom::array results;
    if (item["results"].get_array().get(results) == simdjson::SUCCESS) {
        parse_source_array(response, results);
    }

    simdjson::dom::object action;
    if (item["action"].get_object().get(action) != simdjson::SUCCESS) return;

    simdjson::dom::array sources;
    if (action["sources"].get_array().get(sources) == simdjson::SUCCESS) {
        parse_source_array(response, sources);
    }
}

void parse_output_array(SearchResponse& response,
                        simdjson::dom::array output) {
    for (auto item_element : output) {
        simdjson::dom::object item;
        if (item_element.get_object().get(item) != simdjson::SUCCESS) continue;

        const auto type = core::utils::json::first_string_field_or_empty(item, {"type"});
        if (type == "message") {
            parse_message_item(response, item);
        } else if (type == "web_search_call") {
            parse_web_search_call(response, item);
        }
    }
}

void parse_results(SearchResponse& response,
                   simdjson::dom::element doc,
                   int limit) {
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

    if (limit > 0 && response.results.size() > static_cast<std::size_t>(limit)) {
        response.results.resize(static_cast<std::size_t>(limit));
    }
}

[[nodiscard]] std::string read_error_message(simdjson::dom::element doc) {
    auto from_error_object = [](simdjson::dom::object error_object) -> std::string {
        return core::utils::json::first_string_field_or_empty(error_object, {"message", "detail", "error"});
    };

    simdjson::dom::object error_object;
    if (doc["error"].get_object().get(error_object) == simdjson::SUCCESS) {
        if (std::string message = from_error_object(error_object); !message.empty()) {
            return message;
        }
    }
    if (doc["response"]["error"].get_object().get(error_object) == simdjson::SUCCESS) {
        if (std::string message = from_error_object(error_object); !message.empty()) {
            return message;
        }
    }

    std::string_view message;
    if (doc["message"].get(message) == simdjson::SUCCESS && !message.empty()) {
        return std::string(message);
    }
    return {};
}

class OpenAIWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "openai-responses-web-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value()
            || !is_openai_responses_endpoint(*metadata)
            || metadata->credential_source == nullptr) {
            return false;
        }
        if (is_codex_backend(metadata->base_url)
            && metadata->client_identity_source == nullptr) {
            return false;
        }
        return true;
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source) {
            return std::unexpected(
                "OpenAI web search requires an active OpenAI/Codex provider with credentials.");
        }
        if (!is_openai_responses_endpoint(*metadata)) {
            return std::unexpected(
                "OpenAI web search is only available for api.openai.com or the Codex backend.");
        }
        if (is_codex_backend(metadata->base_url) && !metadata->client_identity_source) {
            return std::unexpected(
                "Codex web search requires a provider client identity source.");
        }

        const std::string model = !context.model_name.empty()
            ? context.model_name
            : metadata->default_model;
        if (model.empty()) {
            return std::unexpected("OpenAI web search requires an active model.");
        }

        const auto auth = metadata->credential_source->get_auth();
        const cpr::Response response = cpr::Post(
            cpr::Url{responses_url(metadata->base_url)},
            openai_headers(auth, *metadata, context),
            cpr::Body{search_payload(request, model)},
            cpr::Timeout{kSearchTimeoutMs});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected(
                "OpenAI web search request failed: " + response.error.message);
        }

        simdjson::dom::parser parser;
        simdjson::padded_string padded(response.text);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            if (response.status_code < 200 || response.status_code >= 300) {
                return std::unexpected(std::format(
                    "OpenAI web search failed with HTTP {}: {}",
                    response.status_code,
                    response.text.substr(0, 1000)));
            }
            return std::unexpected("OpenAI web search response was not valid JSON.");
        }

        if (response.status_code < 200 || response.status_code >= 300) {
            const std::string message = read_error_message(doc);
            return std::unexpected(std::format(
                "OpenAI web search failed with HTTP {}: {}",
                response.status_code,
                message.empty() ? response.text.substr(0, 1000) : message));
        }

        SearchResponse out{.backend = std::string(name())};
        parse_results(out, doc, request.limit);
        if (out.answer.empty() && out.results.empty()) {
            return std::unexpected(
                "OpenAI web search response did not contain output text or citations.");
        }
        return out;
    }
};

} // namespace

std::shared_ptr<IWebSearchBackend> make_openai_web_search_backend() {
    return std::make_shared<OpenAIWebSearchBackend>();
}

} // namespace core::tools::web
