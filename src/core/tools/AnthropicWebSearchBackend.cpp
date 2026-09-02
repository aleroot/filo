#include "WebBackendAdapters.hpp"

#include "WebBackendSupport.hpp"
#include "../auth/ICredentialSource.hpp"
#include "../llm/AnthropicCompatibility.hpp"
#include "../llm/protocols/AnthropicProtocol.hpp"
#include "../utils/JsonWriter.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::tools::web {

namespace {

constexpr int kSearchTimeoutMs = 60000;
constexpr int kMaxSearchUses = 8;

constexpr std::string_view kWebSearchBeta = "web-search-2025-03-05";
constexpr std::string_view kBillingHeader =
    core::llm::anthropic::kBillingHeader;

[[nodiscard]] bool is_anthropic_endpoint(
    const core::llm::ProviderMetadata& metadata) {
    return metadata.api_type == core::config::ApiType::Anthropic
        && !metadata.base_url.empty();
}

void write_string_array(core::utils::JsonWriter& writer,
                        std::string_view key,
                        const std::vector<std::string>& values) {
    writer.key(key);
    {
        auto _arr = writer.array();
        bool first = true;
        for (const auto& value : values) {
            if (value.empty()) continue;
            if (!first) writer.comma();
            first = false;
            writer.str(value);
        }
    }
}

[[nodiscard]] std::string search_payload(const SearchRequest& request,
                                         std::string_view model) {
    core::utils::JsonWriter writer(2048);
    {
        auto _root = writer.object();
        writer.kv_str("model", model).comma()
              .kv_bool("stream", false).comma()
              .kv_num("max_tokens", 2048).comma()
              .kv_str("system",
                      "x-anthropic-billing-header: "
                          + std::string(kBillingHeader)
                          + "\n\nYou are an assistant for performing a web search tool use. "
                            "Return concise findings and preserve source URLs.").comma()
              .key("messages");
        {
            auto _messages = writer.array();
            {
                auto _message = writer.object();
                writer.kv_str("role", "user").comma()
                      .kv_str("content", "Call the web_search tool to search for the following query. "
                          "Base your answer on the search results and cite source URLs. Query: "
                          + request.query);
            }
        }
        writer.comma().key("tools");
        {
            auto _tools = writer.array();
            {
                auto _tool = writer.object();
                writer.kv_str("type", "web_search_20250305").comma()
                      .kv_str("name", "web_search").comma()
                      .kv_num("max_uses", kMaxSearchUses);
                if (!request.domains.allowed_domains.empty()) {
                    writer.comma();
                    write_string_array(writer, "allowed_domains", request.domains.allowed_domains);
                }
                if (!request.domains.blocked_domains.empty()) {
                    writer.comma();
                    write_string_array(writer, "blocked_domains", request.domains.blocked_domains);
                }
            }
        }
        writer.comma().key("tool_choice");
        {
            auto _choice = writer.object();
            if (core::llm::anthropic::rejects_forced_tool_choice(model)) {
                writer.kv_str("type", "auto");
            } else {
                writer.kv_str("type", "tool").comma()
                      .kv_str("name", "web_search");
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] cpr::Header anthropic_headers(
    core::auth::AuthInfo auth,
    const core::llm::protocols::AnthropicProtocol& protocol) {
    auto& beta = auth.headers["anthropic-beta"];
    if (!beta.empty()) beta += ',';
    beta += kWebSearchBeta;
    if (auth.headers.contains("Authorization")) {
        auth.properties["oauth"] = "1";
    }
    auto headers = protocol.build_headers(auth);
    headers["Accept"] = "application/json";
    return headers;
}

void parse_text_block(SearchResponse& response,
                      const simdjson::dom::object& block) {
    if (auto text = core::utils::json::optional_string_field(block, "text"); text.has_value() && !text->empty()) {
        if (!response.answer.empty()) response.answer += "\n\n";
        response.answer += *text;
    }

    simdjson::dom::array citations;
    if (block["citations"].get_array().get(citations) != simdjson::SUCCESS) return;

    for (auto citation : citations) {
        simdjson::dom::object object;
        if (citation.get_object().get(object) != simdjson::SUCCESS) continue;
        SearchHit hit{
            .title = core::utils::json::first_string_field_or_empty(object, {"title", "document_title"}),
            .url = core::utils::json::first_string_field_or_empty(object, {"url", "source_url"}),
            .snippet = core::utils::json::first_string_field_or_empty(object, {"cited_text", "snippet"}),
        };
        detail::add_hit_if_present(response, std::move(hit));
    }
}

void parse_web_result_block(SearchResponse& response,
                            const simdjson::dom::object& block) {
    simdjson::dom::array content;
    if (block["content"].get_array().get(content) != simdjson::SUCCESS) return;

    for (auto item : content) {
        simdjson::dom::object object;
        if (item.get_object().get(object) != simdjson::SUCCESS) continue;
        detail::parse_search_hit_object(response, object);
    }
}

[[nodiscard]] std::optional<std::string> parse_results(
    SearchResponse& response, simdjson::dom::element doc) {
    simdjson::dom::array content;
    if (doc["content"].get_array().get(content) != simdjson::SUCCESS) {
        return "Claude web search response is missing content.";
    }

    bool searched = false;
    for (auto element : content) {
        simdjson::dom::object block;
        if (element.get_object().get(block) != simdjson::SUCCESS) continue;

        const auto type = core::utils::json::first_string_field_or_empty(block, {"type"});
        if (type == "text") {
            parse_text_block(response, block);
        } else if (type == "web_search_tool_result") {
            simdjson::dom::array results;
            if (block["content"].get_array().get(results) != simdjson::SUCCESS) {
                simdjson::dom::object error;
                if (block["content"].get_object().get(error) == simdjson::SUCCESS) {
                    return "Claude web search failed: "
                        + core::utils::json::first_string_field_or_empty(error, {"error_code", "type"});
                }
                return "Claude web search returned an invalid tool result.";
            }
            searched = true;
            parse_web_result_block(response, block);
        }
    }
    // With automatic tool choice the model can answer without searching.
    // Never present that answer as retrieved evidence.
    if (!searched) return "Claude did not complete a web search. Please retry the search.";
    return std::nullopt;
}

class AnthropicWebSearchBackend final : public IWebSearchBackend {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "anthropic-native-web-search";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        return metadata.has_value()
            && is_anthropic_endpoint(*metadata)
            && metadata->credential_source != nullptr;
    }

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request,
           const ToolInvocationContext& context) const override {
        const auto metadata = detail::metadata_for(context);
        if (!metadata.has_value() || !metadata->credential_source) {
            return std::unexpected(
                "Claude web search requires an active Anthropic provider with credentials.");
        }

        std::string model = !context.model_name.empty()
            ? context.model_name
            : metadata->default_model;
        if (model.empty()) {
            return std::unexpected("Claude web search requires an active model.");
        }

        core::llm::ChatRequest normalized;
        normalized.model = std::move(model);
        core::llm::protocols::AnthropicProtocol protocol;
        protocol.prepare_request(normalized);
        model = std::move(normalized.model);

        const auto auth = metadata->credential_source->get_auth();
        const cpr::Response response = cpr::Post(
            cpr::Url{detail::append_path(metadata->base_url, "/v1/messages")},
            anthropic_headers(auth, protocol),
            cpr::Body{search_payload(request, model)},
            cpr::Timeout{kSearchTimeoutMs});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected("Claude web search request failed: " + response.error.message);
        }
        if (response.status_code < 200 || response.status_code >= 300) {
            return std::unexpected(std::format(
                "Claude web search failed with HTTP {}: {}",
                response.status_code,
                response.text.substr(0, 1000)));
        }

        simdjson::dom::parser parser;
        simdjson::padded_string padded(response.text);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            return std::unexpected("Claude web search response was not valid JSON.");
        }

        SearchResponse out{.backend = std::string(name())};
        if (const auto error = parse_results(out, doc)) {
            return std::unexpected(*error);
        }
        return out;
    }
};

} // namespace

std::shared_ptr<IWebSearchBackend> make_anthropic_web_search_backend() {
    return std::make_shared<AnthropicWebSearchBackend>();
}

} // namespace core::tools::web
