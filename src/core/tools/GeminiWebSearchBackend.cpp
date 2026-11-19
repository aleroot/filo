#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/GoogleAntigravityIdentity.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../llm/AntigravityModelTraits.hpp"
#include "../llm/protocols/GeminiProtocol.hpp"
#include "../utils/UriUtils.hpp"
#include "../utils/Uuid.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <format>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::tools::web {

namespace {

constexpr int kSearchTimeoutMs = 60000;
constexpr int kCatalogTimeoutMs = 15000;
constexpr int kRedirectTimeoutMs = 8000;
constexpr int kMaxOutputTokens = 4096;

[[nodiscard]] bool is_antigravity_provider(const core::llm::ProviderMetadata& metadata) {
    return metadata.service_id == core::llm::antigravity::kServiceId;
}

/// An API-key (Gemini API) session: the Gemini api_type on anything but the
/// Antigravity account, which speaks the Cloud Code Assist envelope instead.
[[nodiscard]] bool is_gemini_api_provider(const core::llm::ProviderMetadata& metadata) {
    return metadata.api_type == core::config::ApiType::Gemini
        && !is_antigravity_provider(metadata);
}

[[nodiscard]] cpr::Header antigravity_headers(const core::auth::AuthInfo& auth) {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"User-Agent", core::auth::antigravity::user_agent()},
    };
    for (const auto& [key, value] : auth.headers) headers[key] = value;
    return headers;
}

[[nodiscard]] std::string project_of(const core::auth::AuthInfo& auth) {
    const auto it = auth.properties.find("project_id");
    return it != auth.properties.end() ? it->second : std::string{};
}

[[nodiscard]] bool host_matches(std::string_view host, std::string_view domain) {
    if (domain.empty()) return false;
    if (host == domain) return true;
    return host.size() > domain.size()
        && host.ends_with(domain)
        && host[host.size() - domain.size() - 1] == '.';
}

[[nodiscard]] std::vector<std::string> normalized_domains(const std::vector<std::string>& values) {
    std::vector<std::string> out;
    for (const auto& value : values) {
        if (auto domain = detail::normalize_search_domain(value); !domain.empty()) {
            out.push_back(std::move(domain));
        }
    }
    return out;
}

/// The grounding API returns every source as a `vertexaisearch` redirect; the
/// hub follows it to show the real page (its "web search redirect" fetch).
/// A failed lookup keeps the redirect, which still resolves when opened.
[[nodiscard]] std::string resolve_redirect(const std::string& url) {
    const auto host = core::utils::uri::extract_http_host(url);
    if (!host || !core::utils::ascii::iequals(*host, "vertexaisearch.cloud.google.com")) {
        return url;
    }
    const cpr::Response response = cpr::Head(
        cpr::Url{url},
        cpr::Redirect{false},
        cpr::Timeout{kRedirectTimeoutMs});
    if (response.error.code != cpr::ErrorCode::OK
        || response.status_code < 300 || response.status_code >= 400) {
        return url;
    }
    const auto location = detail::find_header_case_insensitive(response.header, "Location");
    return location && !location->empty() ? *location : url;
}

/// `webSearchModelIds[0]` of the account's catalog, which is how the hub picks
/// its search model (it fails with "no web search models available" without
/// one). Cached per process once read.
[[nodiscard]] std::string web_search_model(const std::string& base_url,
                                           const cpr::Header& headers) {
    static std::mutex mutex;
    static std::string cached;
    {
        std::scoped_lock lock(mutex);
        if (!cached.empty()) return cached;
    }
    const cpr::Response response = cpr::Post(
        cpr::Url{detail::append_path(base_url, "/v1internal:fetchAvailableModels")},
        headers,
        cpr::Body{"{}"},
        cpr::Timeout{kCatalogTimeoutMs});
    if (response.error.code != cpr::ErrorCode::OK
        || response.status_code < 200 || response.status_code >= 300) {
        return std::string(kAntigravityDefaultWebSearchModel);
    }
    std::string model = antigravity_web_search_model(response.text);
    std::scoped_lock lock(mutex);
    cached = model;
    return model;
}

/// The query, with the domain filters as search operators: grounding ignores
/// `googleSearch.excludeDomains` (observed on the Antigravity backend), so
/// both backends send them this way and enforce them again on the resolved
/// source URLs.
[[nodiscard]] std::string grounded_search_prompt(const SearchRequest& request) {
    std::string query = request.query;
    const auto allowed = normalized_domains(request.domains.allowed_domains);
    for (std::size_t i = 0; i < allowed.size(); ++i) {
        query += i == 0 ? " (" : " OR ";
        query += "site:" + allowed[i];
        if (i + 1 == allowed.size()) query += ')';
    }
    for (const auto& blocked : normalized_domains(request.domains.blocked_domains)) {
        query += " -site:" + blocked;
    }
    return "Search the web with Google Search for the following query and answer from the "
           "results. Be concise and factual, and keep the details a developer would need. "
           "Query: " + query;
}

/// A GenerateContentRequest whose only tool is Google Search grounding.
/// `tool_key` is the field spelling of each API: the Cloud Code Assist
/// envelope uses `googleSearch`, the public Gemini API documents `google_search`.
void write_grounded_request(core::utils::JsonWriter& writer,
                            const std::string& prompt,
                            std::string_view tool_key) {
    auto _request = writer.object();
    writer.key("contents");
    {
        auto _contents = writer.array();
        auto _content = writer.object();
        writer.kv_str("role", "user").comma().key("parts");
        auto _parts = writer.array();
        auto _part = writer.object();
        writer.kv_str("text", prompt);
    }
    writer.comma().key("tools");
    {
        auto _tools = writer.array();
        auto _tool = writer.object();
        writer.key(tool_key);
        auto _search = writer.object();
    }
    writer.comma().key("generationConfig");
    {
        auto _config = writer.object();
        writer.kv_num("maxOutputTokens", kMaxOutputTokens);
    }
}

[[nodiscard]] std::vector<std::string> resolve_redirects(const std::vector<std::string>& urls) {
    std::vector<std::future<std::string>> pending;
    pending.reserve(urls.size());
    for (const auto& url : urls) {
        pending.push_back(std::async(std::launch::async, resolve_redirect, url));
    }
    std::vector<std::string> resolved;
    resolved.reserve(urls.size());
    for (auto& future : pending) resolved.push_back(future.get());
    return resolved;
}

class AntigravityWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "antigravity-google-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && metadata->credential_source != nullptr
            && is_antigravity_provider(*metadata);
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source
            || !is_antigravity_provider(*metadata)) {
            return std::unexpected(
                "Antigravity web search requires an active Google account (Antigravity) provider.");
        }

        const auto auth = metadata->credential_source->get_auth();
        const cpr::Header headers = antigravity_headers(auth);
        const std::string model = web_search_model(metadata->base_url, headers);
        const cpr::Response response = cpr::Post(
            cpr::Url{detail::append_path(metadata->base_url, "/v1internal:generateContent")},
            headers,
            cpr::Body{antigravity_web_search_request_json(request, model, project_of(auth))},
            cpr::Timeout{kSearchTimeoutMs});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected(
                "Antigravity web search request failed: " + response.error.message);
        }
        if (response.status_code < 200 || response.status_code >= 300) {
            return std::unexpected(std::format(
                "Antigravity web search failed with HTTP {}: {}",
                response.status_code,
                response.text.substr(0, 1000)));
        }

        auto parsed = parse_antigravity_web_search_response(
            response.text, request, resolve_redirects);
        if (parsed.has_value()) parsed->backend = std::string(name());
        return parsed;
    }
};

/// Gemini API (API key) search: one `generateContent` turn on the session
/// model with only the `google_search` grounding tool, parsed like the
/// Antigravity reply (same GenerateContentResponse, without the envelope).
class GeminiApiWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "gemini-google-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && metadata->credential_source != nullptr
            && is_gemini_api_provider(*metadata);
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source
            || !is_gemini_api_provider(*metadata)) {
            return std::unexpected(
                "Gemini web search requires an active Gemini API provider with credentials.");
        }

        // Same rule as the other provider backends: search as the model the
        // user selected.
        const std::string model = !context.model_name.empty()
            ? context.model_name
            : metadata->default_model;
        if (model.empty()) {
            return std::unexpected("Gemini web search needs a model; none is selected.");
        }

        // The API key travels as the `x-goog-api-key` header rather than the
        // `?key=` the chat stream uses, so no URL here can carry it into an
        // error message.
        const auto auth = metadata->credential_source->get_auth();
        cpr::Header headers{{"Content-Type", "application/json"}};
        for (const auto& [key, value] : auth.headers) headers[key] = value;
        std::unordered_map<std::string, std::string> query;
        for (const auto& [key, value] : auth.query_params) {
            if (key == "key") headers["x-goog-api-key"] = value;
            else query.emplace(key, value);
        }

        const cpr::Response response = cpr::Post(
            cpr::Url{core::utils::uri::append_query_params(
                gemini_web_search_url(metadata->base_url, model), query)},
            headers,
            cpr::Body{gemini_web_search_request_json(request)},
            cpr::Timeout{kSearchTimeoutMs});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected("Gemini web search request failed: " + response.error.message);
        }
        if (response.status_code < 200 || response.status_code >= 300) {
            std::string message;
            simdjson::dom::parser parser;
            simdjson::padded_string padded(response.text);
            std::string_view text;
            if (parser.parse(padded)["error"]["message"].get(text) == simdjson::SUCCESS) {
                message = std::string(text);
            }
            return std::unexpected(std::format(
                "Gemini web search failed with HTTP {} on {}: {}",
                response.status_code,
                model,
                message.empty() ? response.text.substr(0, 1000) : message));
        }

        auto parsed = parse_antigravity_web_search_response(
            response.text, request, resolve_redirects);
        if (parsed.has_value()) parsed->backend = std::string(name());
        return parsed;
    }
};

} // namespace

std::string antigravity_web_search_model(std::string_view catalog_body) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(catalog_body);
    simdjson::dom::array ids;
    if (parser.parse(padded)["webSearchModelIds"].get(ids) == simdjson::SUCCESS) {
        for (auto id : ids) {
            std::string_view value;
            if (id.get(value) == simdjson::SUCCESS && !value.empty()) return std::string(value);
        }
    }
    return std::string(kAntigravityDefaultWebSearchModel);
}

std::string antigravity_web_search_request_json(const SearchRequest& request,
                                                std::string_view model,
                                                std::string_view project) {
    const std::string prompt = grounded_search_prompt(request);
    core::utils::JsonWriter writer(1024 + prompt.size());
    {
        auto _root = writer.object();
        if (!project.empty()) writer.kv_str("project", project).comma();
        writer.kv_str("requestId", "agent/" + core::utils::random_uuid_v4()).comma()
              .key("request");
        write_grounded_request(writer, prompt, "googleSearch");
        writer.comma()
              .kv_str("model", model).comma()
              .kv_str("userAgent", "antigravity").comma()
              .kv_str("requestType", "agent");
    }
    return std::move(writer).take();
}

std::string gemini_web_search_request_json(const SearchRequest& request) {
    const std::string prompt = grounded_search_prompt(request);
    core::utils::JsonWriter writer(512 + prompt.size());
    write_grounded_request(writer, prompt, "google_search");
    return std::move(writer).take();
}

std::string gemini_web_search_url(std::string_view base_url, std::string_view model) {
    return core::utils::str::trim_trailing_slashes(std::string(base_url))
         + "/v1beta/models/"
         + core::llm::protocols::normalize_requested_gemini_model(model)
         + ":generateContent";
}

std::expected<SearchResponse, std::string>
parse_antigravity_web_search_response(std::string_view body,
                                      const SearchRequest& request,
                                      const RedirectResolver& resolve) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::unexpected("Gemini web search response was not valid JSON.");
    }
    simdjson::dom::element response;
    if (doc["response"].get(response) != simdjson::SUCCESS) response = doc;

    SearchResponse out;
    simdjson::dom::array candidates;
    simdjson::dom::object candidate;
    if (response["candidates"].get(candidates) != simdjson::SUCCESS
        || candidates.at(0).get(candidate) != simdjson::SUCCESS) {
        out.answer = "No search results found.";
        return out;
    }

    simdjson::dom::array parts;
    if (candidate["content"]["parts"].get(parts) == simdjson::SUCCESS) {
        for (auto part : parts) {
            bool thought = false;
            if (part["thought"].get(thought) == simdjson::SUCCESS && thought) continue;
            std::string_view text;
            if (part["text"].get(text) == simdjson::SUCCESS) out.answer += text;
        }
    }

    // One hit per grounding chunk; its snippet is the answer text the
    // groundingSupports attribute to it.
    std::vector<SearchHit> hits;
    simdjson::dom::object grounding;
    if (candidate["groundingMetadata"].get(grounding) == simdjson::SUCCESS) {
        simdjson::dom::array chunks;
        if (grounding["groundingChunks"].get(chunks) == simdjson::SUCCESS) {
            for (auto chunk : chunks) {
                SearchHit hit;
                std::string_view value;
                if (chunk["web"]["uri"].get(value) == simdjson::SUCCESS) hit.url = std::string(value);
                if (chunk["web"]["title"].get(value) == simdjson::SUCCESS) hit.title = std::string(value);
                hits.push_back(std::move(hit));
            }
        }
        simdjson::dom::array supports;
        if (grounding["groundingSupports"].get(supports) == simdjson::SUCCESS) {
            for (auto support : supports) {
                std::string_view text;
                simdjson::dom::array indices;
                if (support["segment"]["text"].get(text) != simdjson::SUCCESS
                    || support["groundingChunkIndices"].get(indices) != simdjson::SUCCESS) {
                    continue;
                }
                for (auto index : indices) {
                    int64_t i = -1;
                    if (index.get(i) != simdjson::SUCCESS || i < 0
                        || static_cast<std::size_t>(i) >= hits.size()) {
                        continue;
                    }
                    auto& snippet = hits[static_cast<std::size_t>(i)].snippet;
                    if (snippet.find(text) != std::string::npos) continue;
                    if (!snippet.empty()) snippet += ' ';
                    snippet += text;
                }
            }
        }
    }

    std::vector<std::string> urls;
    urls.reserve(hits.size());
    for (const auto& hit : hits) urls.push_back(hit.url);
    if (resolve) {
        auto resolved = resolve(urls);
        if (resolved.size() == hits.size()) {
            for (std::size_t i = 0; i < hits.size(); ++i) hits[i].url = std::move(resolved[i]);
        }
    }

    const auto allowed = normalized_domains(request.domains.allowed_domains);
    const auto blocked = normalized_domains(request.domains.blocked_domains);
    const int limit = request.limit > 0 ? request.limit : 5;
    for (auto& hit : hits) {
        if (static_cast<int>(out.results.size()) >= limit) break;
        const std::string host = core::utils::str::to_lower_ascii_copy(
            core::utils::uri::extract_http_host(hit.url).value_or(std::string_view{}));
        const auto matches = [&](const std::string& domain) { return host_matches(host, domain); };
        if (!allowed.empty() && std::ranges::none_of(allowed, matches)) continue;
        if (std::ranges::any_of(blocked, matches)) continue;
        detail::add_hit_if_present(out, std::move(hit));
    }
    if (out.answer.empty() && out.results.empty()) out.answer = "No search results found.";
    return out;
}

std::shared_ptr<IWebSearchBackend> make_antigravity_web_search_backend() {
    return std::make_shared<AntigravityWebSearchBackend>();
}

std::shared_ptr<IWebSearchBackend> make_gemini_web_search_backend() {
    return std::make_shared<GeminiApiWebSearchBackend>();
}

} // namespace core::tools::web
