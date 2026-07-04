#pragma once

#include "WebAccess.hpp"

#include "../llm/LLMProvider.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace core::tools::web::detail {

[[nodiscard]] inline std::optional<core::llm::ProviderMetadata>
metadata_for(const ToolInvocationContext& context) {
    if (!context.provider) return std::nullopt;
    return context.provider->metadata();
}

[[nodiscard]] inline std::string append_path(std::string base_url,
                                             std::string_view suffix) {
    base_url = core::utils::str::trim_trailing_slashes(base_url);
    base_url += suffix;
    return base_url;
}

inline void add_hit_if_present(SearchResponse& response, SearchHit hit) {
    if (hit.url.empty() && hit.title.empty() && hit.snippet.empty() && hit.content.empty()) {
        return;
    }
    if (!hit.url.empty()) {
        const auto duplicate = std::ranges::any_of(
            response.results,
            [&](const SearchHit& existing) { return existing.url == hit.url; });
        if (duplicate) return;
    }
    response.results.push_back(std::move(hit));
}

inline void parse_search_hit_object(SearchResponse& response,
                                    const simdjson::dom::object& object) {
    SearchHit hit{
        .title = core::utils::json::first_string_field_or_empty(object, {"title", "name"}),
        .url = core::utils::json::first_string_field_or_empty(object, {"url", "link", "source_url"}),
        .snippet = core::utils::json::first_string_field_or_empty(
            object, {"snippet", "summary", "description", "cited_text"}),
        .content = core::utils::json::first_string_field_or_empty(
            object, {"content", "markdown", "text"}),
    };
    add_hit_if_present(response, std::move(hit));
}

[[nodiscard]] inline std::optional<std::string> find_header_case_insensitive(
    const cpr::Header& headers,
    std::string_view name) {
    for (const auto& [key, value] : headers) {
        if (core::utils::ascii::iequals(key, name)) {
            return value;
        }
    }
    return std::nullopt;
}

} // namespace core::tools::web::detail
