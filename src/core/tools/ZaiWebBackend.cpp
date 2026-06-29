#include "WebBackendAdapters.hpp"

#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace core::tools::web {

namespace {

constexpr int kZaiWebTimeoutMs = 60000;
constexpr int kReaderMaxChars = 200000;
constexpr std::string_view kMcpProtocolVersion = "2024-11-05";

[[nodiscard]] bool is_zai_endpoint(const core::llm::ProviderMetadata& metadata) {
    const auto host = core::utils::uri::extract_http_host(metadata.base_url);
    return metadata.api_type == core::config::ApiType::OpenAI
        && host.has_value()
        && core::utils::ascii::iequals(*host, "api.z.ai");
}

void add_auth_headers(cpr::Header& headers, const core::auth::AuthInfo& auth) {
    for (const auto& [key, value] : auth.headers) {
        headers[key] = value;
    }
}

[[nodiscard]] cpr::Header zai_mcp_headers(const core::auth::AuthInfo& auth,
                                          std::string_view session_id = {}) {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept", "application/json, text/event-stream"},
    };
    add_auth_headers(headers, auth);
    if (!session_id.empty()) {
        headers["Mcp-Session-Id"] = std::string(session_id);
    }
    return headers;
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
    const std::array<std::string_view, 2> suffixes{
        "/api/coding/paas/v4",
        "/api/paas/v4",
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

[[nodiscard]] std::string mcp_initialize_payload() {
    return std::format(
        R"({{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"{}","capabilities":{{}},"clientInfo":{{"name":"filo","version":"0.1.0"}}}}}})",
        kMcpProtocolVersion);
}

[[nodiscard]] std::string mcp_initialized_payload() {
    return R"({"jsonrpc":"2.0","method":"notifications/initialized","params":{}})";
}

[[nodiscard]] std::string mcp_search_payload(const SearchRequest& request) {
    core::utils::JsonWriter writer(512);
    {
        auto _root = writer.object();
        writer.kv_str("jsonrpc", "2.0").comma()
              .kv_num("id", 3).comma()
              .kv_str("method", "tools/call").comma()
              .key("params");
        {
            auto _params = writer.object();
            writer.kv_str("name", "web_search_prime").comma()
                  .key("arguments");
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
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string mcp_reader_payload(const FetchRequest& request) {
    core::utils::JsonWriter writer(512);
    {
        auto _root = writer.object();
        writer.kv_str("jsonrpc", "2.0").comma()
              .kv_num("id", 3).comma()
              .kv_str("method", "tools/call").comma()
              .key("params");
        {
            auto _params = writer.object();
            writer.kv_str("name", "webReader").comma()
                  .key("arguments");
            {
                auto _args = writer.object();
                writer.kv_str("url", request.url).comma()
                      .kv_num("timeout", 20).comma()
                      .kv_str("return_format", "markdown").comma()
                      .kv_bool("retain_images", false);
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::optional<std::string_view>
find_header_case_insensitive(const cpr::Header& headers, std::string_view key) {
    for (const auto& [name, value] : headers) {
        if (core::utils::ascii::iequals(name, key)) return value;
    }
    return std::nullopt;
}

[[nodiscard]] std::expected<std::string, std::string>
extract_mcp_text_payload(std::string_view body) {
    std::string data_payload;
    std::size_t pos = 0;
    while (pos < body.size()) {
        const std::size_t line_end = body.find('\n', pos);
        std::string_view line = body.substr(
            pos,
            line_end == std::string_view::npos ? body.size() - pos : line_end - pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line.starts_with("data:")) {
            std::string_view data = line.substr(5);
            data = core::utils::str::trim_ascii_view(data);
            if (!data.empty()) {
                if (!data_payload.empty()) data_payload.push_back('\n');
                data_payload.append(data.data(), data.size());
            }
        }
        if (line_end == std::string_view::npos) break;
        pos = line_end + 1;
    }
    if (data_payload.empty()) {
        data_payload.assign(body.data(), body.size());
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(data_payload);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::unexpected("Z.ai MCP response was not valid JSON-RPC.");
    }

    simdjson::dom::object error;
    if (doc["error"].get_object().get(error) == simdjson::SUCCESS) {
        std::string_view message;
        if (error["message"].get(message) == simdjson::SUCCESS) {
            return std::unexpected(std::string(message));
        }
        return std::unexpected("Z.ai MCP tool call failed.");
    }

    simdjson::dom::object result;
    if (doc["result"].get_object().get(result) != simdjson::SUCCESS) {
        return std::unexpected("Z.ai MCP response did not contain a result.");
    }

    bool is_error = false;
    [[maybe_unused]] auto _ = result["isError"].get(is_error);

    std::string text;
    simdjson::dom::array content;
    if (result["content"].get_array().get(content) == simdjson::SUCCESS) {
        for (auto element : content) {
            simdjson::dom::object item;
            if (element.get_object().get(item) != simdjson::SUCCESS) continue;
            std::string_view item_text;
            if (item["text"].get(item_text) == simdjson::SUCCESS && !item_text.empty()) {
                if (!text.empty()) text += "\n\n";
                text.append(item_text.data(), item_text.size());
            }
        }
    }

    if (is_error) {
        return std::unexpected(text.empty() ? "Z.ai MCP tool call failed." : text);
    }
    if (text.empty()) {
        return std::unexpected("Z.ai MCP response did not contain text content.");
    }
    return text;
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
call_zai_mcp_tool(const core::auth::AuthInfo& auth,
                  std::string_view url,
                  std::string_view call_payload,
                  std::string_view tool_label) {
    const cpr::Response init = cpr::Post(
        cpr::Url{std::string(url)},
        zai_mcp_headers(auth),
        cpr::Body{mcp_initialize_payload()},
        cpr::Timeout{kZaiWebTimeoutMs});
    if (init.error.code != cpr::ErrorCode::OK) {
        return std::unexpected(std::format(
            "Z.ai {} MCP initialize request failed: {}",
            tool_label,
            init.error.message));
    }
    if (init.status_code < 200 || init.status_code >= 300) {
        return std::unexpected(std::format(
            "Z.ai {} MCP initialize failed with HTTP {}: {}",
            tool_label,
            init.status_code,
            init.text.substr(0, 1000)));
    }

    const auto session_id = find_header_case_insensitive(init.header, "mcp-session-id");
    if (!session_id.has_value() || session_id->empty()) {
        return std::unexpected(std::format(
            "Z.ai {} MCP initialize response did not include mcp-session-id.",
            tool_label));
    }

    (void)cpr::Post(
        cpr::Url{std::string(url)},
        zai_mcp_headers(auth, *session_id),
        cpr::Body{mcp_initialized_payload()},
        cpr::Timeout{kZaiWebTimeoutMs});

    const cpr::Response response = cpr::Post(
        cpr::Url{std::string(url)},
        zai_mcp_headers(auth, *session_id),
        cpr::Body{std::string(call_payload)},
        cpr::Timeout{kZaiWebTimeoutMs});
    if (response.error.code != cpr::ErrorCode::OK) {
        return std::unexpected(std::format(
            "Z.ai {} MCP tool request failed: {}",
            tool_label,
            response.error.message));
    }
    if (response.status_code < 200 || response.status_code >= 300) {
        return std::unexpected(std::format(
            "Z.ai {} MCP tool failed with HTTP {}: {}",
            tool_label,
            response.status_code,
            response.text.substr(0, 1000)));
    }
    return extract_mcp_text_payload(response.text);
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
        if (const auto text = detail::read_first_string(object, {"text", "content"});
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
    if (const auto title = detail::read_first_string(result, {"title"});
        response.title.empty() && !title.empty()) {
        response.title = title;
    }
    if (const auto url = detail::read_first_string(result, {"url"});
        response.final_url.empty() && !url.empty()) {
        response.final_url = url;
    }
    if (const auto text = detail::read_first_string(
            result, {"content", "description", "text", "markdown"});
        !text.empty()) {
        parse_text_content(response, text);
    }
}

void parse_reader_content(FetchResponse& response, simdjson::dom::array content) {
    for (auto element : content) {
        simdjson::dom::object object;
        if (element.get_object().get(object) != simdjson::SUCCESS) continue;
        if (const auto title = detail::read_first_string(object, {"title", "document_title"});
            response.title.empty() && !title.empty()) {
            response.title = title;
        }
        if (const auto url = detail::read_first_string(object, {"url", "source_url"});
            response.final_url.empty() && !url.empty()) {
            response.final_url = url;
        }
        if (const auto text = detail::read_first_string(
                object, {"text", "content", "markdown", "summary"});
            !text.empty()) {
            parse_text_content(response, text);
        }
    }
}

void parse_fetch_response(FetchResponse& response, simdjson::dom::element doc) {
    simdjson::dom::object root;
    if (doc.get_object().get(root) == simdjson::SUCCESS) {
        response.title = detail::read_first_string(root, {"title", "name"});
        response.final_url = detail::read_first_string(root, {"url", "final_url", "source_url"});
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

        const auto auth = metadata->credential_source->get_auth();
        const auto mcp_text = call_zai_mcp_tool(
            auth,
            mcp_api_url(metadata->base_url, "web_search_prime"),
            mcp_search_payload(request),
            "web search");
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

        const auto auth = metadata->credential_source->get_auth();
        const auto mcp_text = call_zai_mcp_tool(
            auth,
            mcp_api_url(metadata->base_url, "web_reader"),
            mcp_reader_payload(request),
            "URL reader");
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
