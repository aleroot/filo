#include "WebBackendAdapters.hpp"

#include "McpHttpToolCall.hpp"
#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

namespace core::tools::web {

namespace {

constexpr int kZaiWebTimeoutMs = 60000;
constexpr int kReaderMaxChars = 200000;
constexpr std::string_view kMcpProtocolVersion = "2024-11-05";

[[nodiscard]] bool is_zai_endpoint(const core::llm::ProviderMetadata& metadata) {
    const auto host = core::utils::uri::extract_http_host(metadata.base_url);
    if (!host.has_value() || !core::utils::ascii::iequals(*host, "api.z.ai")) {
        return false;
    }
    return metadata.api_type == core::config::ApiType::OpenAI
        || metadata.api_type == core::config::ApiType::Anthropic;
}

[[nodiscard]] std::expected<void, std::string> validate_fetch_target(std::string_view url) {
    const auto host = core::utils::uri::extract_http_host(url);
    if (!host.has_value() || host->empty()) {
        return std::unexpected("URL must be a valid http:// or https:// URL without credentials.");
    }
    if (const auto policy_error = core::tools::policy::enforce_url_policy(
            names::kFetchUrl,
            url)) {
        return std::unexpected("Tool policy blocked URL: " + *policy_error);
    }
    return {};
}

[[nodiscard]] std::string mcp_api_url(std::string base_url, std::string_view server) {
    base_url = core::utils::str::trim_trailing_slashes(base_url);
    if (base_url.ends_with("/chat/completions")) {
        base_url.resize(base_url.size() - std::string_view("/chat/completions").size());
    }
    const std::array<std::string_view, 3> suffixes{
        "/api/coding/paas/v4",
        "/api/paas/v4",
        "/api/anthropic",
    };
    for (const auto suffix : suffixes) {
        if (base_url.size() >= suffix.size()
            && base_url.compare(base_url.size() - suffix.size(), suffix.size(), suffix) == 0) {
            base_url.resize(base_url.size() - suffix.size());
            break;
        }
    }
    return base_url + "/api/mcp/" + std::string(server) + "/mcp";
}

/// `web_search_prime` arguments.
[[nodiscard]] std::string mcp_search_arguments(const SearchRequest& request) {
    core::utils::JsonWriter writer(256);
    {
        auto _args = writer.object();
        writer.kv_str("search_query", request.query).comma()
              .kv_str("content_size", request.limit > 10 ? "high" : "medium").comma()
              .kv_str("location", "us");
        if (!request.domains.allowed_domains.empty()) {
            writer.comma().kv_str(
                "search_domain_filter",
                request.domains.allowed_domains.front());
        }
    }
    return std::move(writer).take();
}

/// `webReader` arguments.
[[nodiscard]] std::string mcp_reader_arguments(const FetchRequest& request) {
    core::utils::JsonWriter writer(256);
    {
        auto _args = writer.object();
        writer.kv_str("url", request.url).comma()
              .kv_num("timeout", 20).comma()
              .kv_str("return_format", "markdown").comma()
              .kv_bool("retain_images", false);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string unwrap_json_string_literal(std::string text) {
    std::string_view trimmed = core::utils::str::trim_ascii_view(text);
    if (!trimmed.starts_with('"')) return text;

    std::string wrapped = "{\"value\":";
    wrapped.append(trimmed.data(), trimmed.size());
    wrapped.push_back('}');

    simdjson::dom::parser parser;
    simdjson::padded_string padded(wrapped);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return text;
    std::string_view value;
    if (doc["value"].get(value) != simdjson::SUCCESS) return text;
    return std::string(value);
}

[[nodiscard]] std::expected<std::string, std::string>
call_zai_mcp_tool(const ToolInvocationContext& context,
                  core::auth::ICredentialSource& credentials,
                  std::string url,
                  std::string tool,
                  std::string arguments,
                  std::string_view service) {
    auto text = detail::call_mcp_http_tool({
        .url = std::move(url),
        .tool = std::move(tool),
        .arguments_json = std::move(arguments),
        .protocol_version = kMcpProtocolVersion,
        .credentials = &credentials,
        .timeout_ms = kZaiWebTimeoutMs,
        .cancelled = context.cancellation_requested,
    });
    if (!text) return std::unexpected(detail::describe_mcp_failure(text.error(), service));
    if (text->empty()) {
        return std::unexpected(std::string(service) + " response did not contain text content.");
    }
    return std::move(*text);
}

void parse_text_content(SearchResponse& response, std::string_view text) {
    if (text.empty()) return;
    if (!response.answer.empty()) response.answer += "\n\n";
    response.answer.append(text.data(), text.size());
}

void parse_text_content(FetchResponse& response, std::string_view text) {
    if (text.empty()) return;
    if (!response.text.empty()) response.text += "\n\n";
    response.text.append(text.data(), text.size());
}

void parse_search_array(SearchResponse& response, simdjson::dom::array array) {
    for (auto element : array) {
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        detail::parse_search_hit_object(response, object);
    }
}

void parse_citations(SearchResponse& response, simdjson::dom::array citations) {
    for (auto element : citations) {
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        detail::parse_search_hit_object(response, object);
    }
}

void parse_content_array(SearchResponse& response, simdjson::dom::array content) {
    for (auto element : content) {
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        if (const auto text = core::utils::json::first_string_field_or_empty(object, {"text", "content"});
            !text.empty()) {
            parse_text_content(response, text);
        }
        simdjson::dom::array citations;
        if (object["citations"].get_array().get(citations) == simdjson::SUCCESS) {
            parse_citations(response, citations);
        }
    }
}

void parse_message_content(SearchResponse& response, simdjson::dom::object message) {
    std::string_view content_text;
    if (message["content"].get(content_text) == simdjson::SUCCESS) {
        parse_text_content(response, content_text);
    }

    simdjson::dom::array content_array;
    if (message["content"].get_array().get(content_array) == simdjson::SUCCESS) {
        parse_content_array(response, content_array);
    }

    simdjson::dom::array citations;
    if (message["citations"].get_array().get(citations) == simdjson::SUCCESS) {
        parse_citations(response, citations);
    }
}

void parse_search_response(SearchResponse& response, simdjson::dom::element doc) {
    simdjson::dom::array root_array;
    if (doc.get_array().get(root_array) == simdjson::SUCCESS) {
        parse_search_array(response, root_array);
        return;
    }

    std::string_view answer;
    if (doc["answer"].get(answer) == simdjson::SUCCESS
        || doc["summary"].get(answer) == simdjson::SUCCESS) {
        parse_text_content(response, answer);
    }

    simdjson::dom::array array;
    if (doc["search_results"].get_array().get(array) == simdjson::SUCCESS) {
        parse_search_array(response, array);
    }
    if (doc["search_result"].get_array().get(array) == simdjson::SUCCESS) {
        parse_search_array(response, array);
    }
    if (doc["results"].get_array().get(array) == simdjson::SUCCESS) {
        parse_search_array(response, array);
    }
    if (doc["citations"].get_array().get(array) == simdjson::SUCCESS) {
        parse_citations(response, array);
    }

    simdjson::dom::array choices;
    if (doc["choices"].get_array().get(choices) != simdjson::SUCCESS) return;
    for (auto choice : choices) {
        simdjson::dom::object choice_obj;
        if (choice.get_object().get(choice_obj) != simdjson::SUCCESS) continue;
        simdjson::dom::object message;
        if (choice_obj["message"].get_object().get(message) == simdjson::SUCCESS) {
            parse_message_content(response, message);
        }
    }
}

void parse_reader_result(FetchResponse& response, simdjson::dom::object result) {
    if (const auto title = core::utils::json::first_string_field_or_empty(result, {"title"});
        response.title.empty() && !title.empty()) {
        response.title = title;
    }
    if (const auto url = core::utils::json::first_string_field_or_empty(result, {"url"});
        response.final_url.empty() && !url.empty()) {
        response.final_url = url;
    }
    if (const auto text = core::utils::json::first_string_field_or_empty(
            result, {"content", "description", "text", "markdown"});
        !text.empty()) {
        parse_text_content(response, text);
    }
}

void parse_reader_content(FetchResponse& response, simdjson::dom::array content) {
    for (auto element : content) {
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        if (const auto title = core::utils::json::first_string_field_or_empty(object, {"title", "document_title"});
            response.title.empty() && !title.empty()) {
            response.title = title;
        }
        if (const auto url = core::utils::json::first_string_field_or_empty(object, {"url", "source_url"});
            response.final_url.empty() && !url.empty()) {
            response.final_url = url;
        }
        if (const auto text = core::utils::json::first_string_field_or_empty(
                object, {"text", "content", "markdown", "summary"});
            !text.empty()) {
            parse_text_content(response, text);
        }
    }
}

void parse_fetch_response(FetchResponse& response, simdjson::dom::element doc) {
    simdjson::dom::object root;
    if (doc.get_object().get(root) == simdjson::SUCCESS) {
        response.title = core::utils::json::first_string_field_or_empty(root, {"title", "name"});
        response.final_url = core::utils::json::first_string_field_or_empty(root, {"url", "final_url", "source_url"});
    }

    std::string_view text;
    if (doc["content"].get(text) == simdjson::SUCCESS
        || doc["text"].get(text) == simdjson::SUCCESS
        || doc["markdown"].get(text) == simdjson::SUCCESS
        || doc["answer"].get(text) == simdjson::SUCCESS
        || doc["summary"].get(text) == simdjson::SUCCESS) {
        parse_text_content(response, text);
    }

    simdjson::dom::array content;
    if (doc["content"].get_array().get(content) == simdjson::SUCCESS
        || doc["results"].get_array().get(content) == simdjson::SUCCESS) {
        parse_reader_content(response, content);
    }

    simdjson::dom::object reader_result;
    if (doc["reader_result"].get_object().get(reader_result) == simdjson::SUCCESS) {
        parse_reader_result(response, reader_result);
    }

    simdjson::dom::array choices;
    if (doc["choices"].get_array().get(choices) != simdjson::SUCCESS) return;
    for (auto choice : choices) {
        simdjson::dom::object choice_obj;
        if (choice.get_object().get(choice_obj) != simdjson::SUCCESS) continue;
        simdjson::dom::object message;
        if (choice_obj["message"].get_object().get(message) != simdjson::SUCCESS) continue;

        std::string_view message_text;
        if (message["content"].get(message_text) == simdjson::SUCCESS) {
            parse_text_content(response, message_text);
        }

        simdjson::dom::array message_content;
        if (message["content"].get_array().get(message_content) == simdjson::SUCCESS) {
            parse_reader_content(response, message_content);
        }
    }
}

class ZaiWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "zai-native-web-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && is_zai_endpoint(*metadata)
            && metadata->credential_source != nullptr;
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source) {
            return std::unexpected("Z.ai web search requires an active Z.ai provider with credentials.");
        }
        if (request.domains.allowed_domains.size() > 1) {
            return std::unexpected(
                "Z.ai web search supports at most one allowed_domains entry. "
                "Use a single domain or switch to an Anthropic/Claude provider for multi-domain search.");
        }
        if (!request.domains.blocked_domains.empty()) {
            return std::unexpected(
                "Z.ai web search does not support blocked_domains filters. "
                "Remove the filter or use an Anthropic/Claude provider for domain-filtered search.");
        }

        const auto mcp_text = call_zai_mcp_tool(
            context, *metadata->credential_source,
            mcp_api_url(metadata->base_url, "web_search_prime"), "web_search_prime",
            mcp_search_arguments(request), "Z.ai web search");
        if (!mcp_text.has_value()) return std::unexpected(mcp_text.error());

        simdjson::dom::parser parser;
        simdjson::padded_string padded(unwrap_json_string_literal(*mcp_text));
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            return std::unexpected("Z.ai web search response was not valid JSON.");
        }

        SearchResponse out{.backend = std::string(name())};
        parse_search_response(out, doc);
        if (out.answer.empty() && out.results.empty()) {
            return std::unexpected(
                "Z.ai web search response did not contain text, citations, or results.");
        }
        return out;
    }
};

class ZaiWebFetchBackend final : public IWebFetchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "zai-native-url-reader";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && is_zai_endpoint(*metadata)
            && metadata->credential_source != nullptr;
    }

    [[nodiscard]] std::expected<FetchResponse, std::string>
    fetch(const FetchRequest& request,
          const ToolInvocationContext& context) const override {
        if (auto valid = validate_fetch_target(request.url); !valid) {
            return std::unexpected(valid.error());
        }

        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source) {
            return std::unexpected("Z.ai URL reader requires an active Z.ai provider with credentials.");
        }

        const auto mcp_text = call_zai_mcp_tool(
            context, *metadata->credential_source,
            mcp_api_url(metadata->base_url, "web_reader"), "webReader",
            mcp_reader_arguments(request), "Z.ai URL reader");
        if (!mcp_text.has_value()) return std::unexpected(mcp_text.error());

        simdjson::dom::parser parser;
        simdjson::padded_string padded(unwrap_json_string_literal(*mcp_text));
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            return std::unexpected("Z.ai URL reader response was not valid JSON.");
        }

        FetchResponse out{
            .final_url = request.url,
            .content_type = "text/markdown",
            .status_code = 200,
        };
        parse_fetch_response(out, doc);
        if (out.text.size() > static_cast<std::size_t>(std::clamp(
                request.max_bytes, 1024, kReaderMaxChars))) {
            out.text.resize(static_cast<std::size_t>(std::clamp(
                request.max_bytes, 1024, kReaderMaxChars)));
            out.truncated = true;
        }
        if (out.final_url.empty()) out.final_url = request.url;
        if (out.text.empty()) {
            return std::unexpected("Z.ai URL reader response did not contain readable text.");
        }
        return out;
    }
};

} // namespace

std::shared_ptr<IWebSearchBackend> make_zai_web_search_backend() {
    return std::make_shared<ZaiWebSearchBackend>();
}

std::shared_ptr<IWebFetchBackend> make_zai_web_fetch_backend() {
    return std::make_shared<ZaiWebFetchBackend>();
}

} // namespace core::tools::web
