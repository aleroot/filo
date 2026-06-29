#include "WebSearchTool.hpp"

#include "ToolNames.hpp"
#include "WebAccess.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>

namespace core::tools {

namespace {

[[nodiscard]] std::string error_json(std::string_view message) {
    core::utils::JsonWriter writer(256);
    {
        auto _root = writer.object();
        writer.kv_str("error", message);
    }
    return std::move(writer).take();
}

void read_string_array(simdjson::dom::element doc,
                       std::string_view key,
                       std::vector<std::string>& out) {
    simdjson::dom::array array;
    if (doc[key].get_array().get(array) != simdjson::SUCCESS) return;

    for (auto item : array) {
        std::string_view value;
        if (item.get(value) == simdjson::SUCCESS && !value.empty()) {
            out.emplace_back(value);
        }
    }
}

[[nodiscard]] std::expected<web::SearchRequest, std::string>
parse_search_request(std::string_view json_args) {
    simdjson::dom::parser parser;
    const std::string args = json_args.empty() ? std::string("{}") : std::string(json_args);
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::unexpected("Arguments must be a JSON object.");
    }

    std::string_view query;
    if (doc["query"].get(query) != simdjson::SUCCESS || query.empty()) {
        return std::unexpected("Missing required string argument 'query'.");
    }

    web::SearchRequest request;
    request.query = std::string(query);

    int64_t limit = request.limit;
    if (doc["limit"].get(limit) == simdjson::SUCCESS) {
        request.limit = static_cast<int>(std::clamp<int64_t>(limit, 1, 20));
    }

    bool include_page_content = false;
    if (doc["include_page_content"].get(include_page_content) == simdjson::SUCCESS
        || doc["include_content"].get(include_page_content) == simdjson::SUCCESS) {
        request.include_page_content = include_page_content;
    }

    read_string_array(doc, "allowed_domains", request.domains.allowed_domains);
    read_string_array(doc, "blocked_domains", request.domains.blocked_domains);
    return request;
}

} // namespace

ToolDefinition WebSearchTool::get_definition() const {
    return {
        .name = std::string(names::kWebSearch),
        .title = "Web Search",
        .description =
            "Search the web through the active provider's native search backend. "
            "Use this when current or external information is needed and cite URLs "
            "from the returned results.",
        .parameters = {
            {
                .name = "query",
                .type = "string",
                .description = "Search query.",
                .required = true,
            },
            {
                .name = "limit",
                .type = "integer",
                .description = "Maximum number of search results to return, from 1 to 20.",
                .required = false,
            },
            {
                .name = "include_page_content",
                .type = "boolean",
                .description = "Ask the backend to include crawled page content when supported.",
                .required = false,
            },
            {
                .name = "allowed_domains",
                .type = "array",
                .description = "Optional provider-supported allow-list of domains.",
                .required = false,
                .items_schema = R"({"type":"string"})",
            },
            {
                .name = "blocked_domains",
                .type = "array",
                .description = "Optional provider-supported block-list of domains.",
                .required = false,
                .items_schema = R"({"type":"string"})",
            },
        },
        .annotations = {
            .read_only_hint = true,
            .destructive_hint = false,
            .idempotent_hint = false,
            .open_world_hint = true,
        },
    };
}

std::string WebSearchTool::execute(const std::string& json_args,
                                   const core::context::SessionContext& context) {
    return execute(json_args, ToolInvocationContext{.session_context = context});
}

std::string WebSearchTool::execute(const std::string& json_args,
                                   const ToolInvocationContext& invocation) {
    auto request = parse_search_request(json_args);
    if (!request) {
        return error_json(request.error());
    }

    auto response = web::WebAccess::instance().search(*request, invocation);
    if (!response) {
        return error_json(response.error());
    }
    return web::search_response_to_json(*response);
}

} // namespace core::tools
