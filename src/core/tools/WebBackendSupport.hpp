#pragma once

#include "WebAccess.hpp"

#include "../llm/LLMProvider.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <initializer_list>
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

[[nodiscard]] inline bool contains_case_insensitive(std::string_view haystack,
                                                    std::string_view needle) {
    return core::utils::str::to_lower_ascii_copy(haystack)
        .find(core::utils::str::to_lower_ascii_copy(needle)) != std::string::npos;
}

[[nodiscard]] inline std::string append_path(std::string base_url,
                                             std::string_view suffix) {
    base_url = core::utils::str::trim_trailing_slashes(base_url);
    base_url += suffix;
    return base_url;
}

[[nodiscard]] inline std::optional<std::string> read_string(
    const simdjson::dom::object& object,
    std::string_view key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::nullopt;
}

[[nodiscard]] inline std::string read_first_string(
    const simdjson::dom::object& object,
    std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        if (auto value = read_string(object, key); value.has_value()) {
            return *value;
        }
    }
    return {};
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
        .title = read_first_string(object, {"title", "name"}),
        .url = read_first_string(object, {"url", "link", "source_url"}),
        .snippet = read_first_string(object, {"snippet", "summary", "description", "cited_text"}),
        .content = read_first_string(object, {"content", "markdown", "text"}),
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
