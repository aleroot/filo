#pragma once

#include "WebAccess.hpp"

#include <memory>

namespace core::tools::web {

[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_kimi_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_anthropic_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_openai_web_search_backend();
[[nodiscard]] std::string openai_web_search_request_json(
    const SearchRequest& request, std::string_view model, bool codex);
[[nodiscard]] std::expected<SearchResponse, std::string> parse_openai_web_search_response(
    std::string_view body, int limit, bool streaming);
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_grok_web_search_backend();
struct DashScopeWebSearchOptions {
    std::string model = {}; // Empty selects qwen3.8-flash; explicit models can opt Coding Plan in.
    int timeout_ms = 120000;
    bool web_extractor = true;
};
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_dashscope_web_search_backend(
    DashScopeWebSearchOptions options = {});

// Shared by GrokWebSearchBackend and its tests. xAI web search is a Responses
// API turn whose only tool is `web_search`. `model` is the session model
// (grok-4.7 searches as grok-4.7); this is not a hosted tool on the main turn.
[[nodiscard]] std::expected<std::string, std::string>
grok_web_search_request_json(const SearchRequest& request,
                              std::string_view model);
[[nodiscard]] std::expected<SearchResponse, std::string>
parse_grok_web_search_response(std::string_view body, int limit);
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_zai_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebFetchBackend> make_zai_web_fetch_backend();
[[nodiscard]] std::shared_ptr<IWebFetchBackend> make_direct_web_fetch_backend();

} // namespace core::tools::web
