#include "WebFetchTool.hpp"

#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "WebAccess.hpp"
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

[[nodiscard]] std::expected<web::FetchRequest, std::string>
parse_fetch_request(std::string_view json_args) {
    simdjson::dom::parser parser;
    const std::string args = json_args.empty() ? std::string("{}") : std::string(json_args);
    simdjson::padded_string padded(args);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::unexpected("Arguments must be a JSON object.");
    }

    std::string_view url;
    if (doc["url"].get(url) != simdjson::SUCCESS || url.empty()) {
        return std::unexpected("Missing required string argument 'url'.");
    }

    web::FetchRequest request;
    request.url = std::string(url);

    int64_t max_bytes = request.max_bytes;
    if (doc["max_bytes"].get(max_bytes) == simdjson::SUCCESS) {
        request.max_bytes = static_cast<int>(
            std::clamp<int64_t>(max_bytes, 1024, 2 * 1024 * 1024));
    }
    return request;
}

} // namespace

ToolDefinition WebFetchTool::get_definition() const {
    return {
        .name = std::string(names::kFetchUrl),
        .title = "Fetch URL",
        .description =
            "Fetch a known http or https URL and return extracted text content for analysis. "
            "Use web_search first when the URL is not already known.",
        .parameters = {
            {
                .name = "url",
                .type = "string",
                .description = "HTTP or HTTPS URL to fetch. Credentials in URLs are rejected.",
                .required = true,
            },
            {
                .name = "max_bytes",
                .type = "integer",
                .description = "Maximum response bytes to process, from 1024 to 2097152.",
                .required = false,
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

std::string WebFetchTool::execute(const std::string& json_args,
                                  const core::context::SessionContext& context) {
    return execute(json_args, ToolInvocationContext{.session_context = context});
}

std::string WebFetchTool::execute(const std::string& json_args,
                                  const ToolInvocationContext& invocation) {
    auto request = parse_fetch_request(json_args);
    if (!request) {
        return error_json(request.error());
    }
    if (const auto policy_error = core::tools::policy::enforce_url_policy(
            names::kFetchUrl,
            request->url)) {
        return error_json("Tool policy blocked URL: " + *policy_error);
    }

    auto response = web::WebAccess::instance().fetch(*request, invocation);
    if (!response) {
        return error_json(response.error());
    }
    return web::fetch_response_to_json(*response);
}

} // namespace core::tools
