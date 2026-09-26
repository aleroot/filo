#pragma once

#include "WebAccess.hpp"

#include "../llm/LLMProvider.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
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

// Host form of a search-filter entry. OpenAI and xAI both want a bare domain;
// models often pass a full URL.
[[nodiscard]] inline std::string normalize_search_domain(std::string_view value) {
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

inline void write_string_array(core::utils::JsonWriter& writer,
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

} // namespace core::tools::web::detail
