#pragma once

#include "WebAccess.hpp"
#include "../llm/MiniMaxEndpoints.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace core::tools::web {

[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_kimi_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_anthropic_web_search_backend();
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_openai_web_search_backend();
[[nodiscard]] std::string openai_web_search_request_json(
    const SearchRequest& request,
    std::string_view model,
    bool codex,
    std::string_view session_id = {});
// ChatGPT Codex base, with or without a trailing /responses, joined the way
// codex-rs Provider::url_for_path joins "alpha/search".
[[nodiscard]] std::string openai_codex_search_url(std::string_view base_url);
[[nodiscard]] std::expected<SearchResponse, std::string> parse_openai_web_search_response(
    std::string_view body, int limit, bool streaming);
[[nodiscard]] std::expected<SearchResponse, std::string> parse_openai_codex_search_response(
    std::string_view body, int limit);
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_grok_web_search_backend();
struct DashScopeWebSearchOptions {
    std::string model = {}; // Empty selects qwen3.8-flash; explicit models can opt Coding Plan in.
    int timeout_ms = 120000;
    bool web_extractor = true;
};
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_dashscope_web_search_backend(
    DashScopeWebSearchOptions options = {});

// Qwen Subscription search: the gateway's MCP `web_search` tool, called with
// the account session (Qwen Code tools/web-search-subscription.ts). The URL is
// injectable for tests only; production always targets the official host.
struct QwenSubscriptionWebSearchOptions {
    std::string search_url = {};
    int timeout_ms = 30000;
};
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_qwen_subscription_web_search_backend(
    QwenSubscriptionWebSearchOptions options = {});

// Shared by GrokWebSearchBackend and its tests. xAI web search is a Responses
// API turn whose only tool is `web_search`. `model` is the session model
// (grok-4.7 searches as grok-4.7); this is not a hosted tool on the main turn.
[[nodiscard]] std::expected<std::string, std::string>
grok_web_search_request_json(const SearchRequest& request,
                              std::string_view model);
[[nodiscard]] std::expected<SearchResponse, std::string>
parse_grok_web_search_response(std::string_view body, int limit);
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_zai_web_search_backend();

// MiniMax M Plan search (`POST /v1/coding_plan/search`, the endpoint
// MiniMax-AI/cli uses). The URL is injectable for tests only; production
// always targets the official host.
struct MiniMaxWebSearchOptions {
    std::string search_url = std::string(core::llm::kMiniMaxSearchUrl);
    int timeout_ms = 30000;
};
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_minimax_web_search_backend(
    MiniMaxWebSearchOptions options = {});
struct MimoWebSearchOptions {
    int timeout_ms = 30000;
};
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_mimo_web_search_backend(
    MimoWebSearchOptions options = {});
[[nodiscard]] std::string minimax_web_search_request_json(const SearchRequest& request);
[[nodiscard]] std::expected<SearchResponse, std::string>
parse_minimax_web_search_response(std::string_view body, const SearchRequest& request);
// Antigravity (Google account) search, as the hub's own search_web tool does
// it: one Cloud Code Assist `generateContent` turn on the catalog's
// `webSearchModelIds[0]` with only the `googleSearch` grounding tool; the
// grounding chunks become the hits.
inline constexpr std::string_view kAntigravityDefaultWebSearchModel = "gemini-3.5-flash-lite";
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_antigravity_web_search_backend();
[[nodiscard]] std::string antigravity_web_search_model(std::string_view catalog_body);
[[nodiscard]] std::string antigravity_web_search_request_json(const SearchRequest& request,
                                                              std::string_view model,
                                                              std::string_view project);
// Gemini API (API key) search: the same grounded turn on the public API,
// `POST {base}/v1beta/models/{model}:generateContent` with `google_search`,
// as the session model. The reply parses with
// parse_antigravity_web_search_response (it accepts the bare response).
[[nodiscard]] std::shared_ptr<IWebSearchBackend> make_gemini_web_search_backend();
[[nodiscard]] std::string gemini_web_search_request_json(const SearchRequest& request);
[[nodiscard]] std::string gemini_web_search_url(std::string_view base_url, std::string_view model);
// Maps the grounding redirect URLs to the pages they point at (same order and size).
using RedirectResolver =
    std::function<std::vector<std::string>(const std::vector<std::string>&)>;
[[nodiscard]] std::expected<SearchResponse, std::string>
parse_antigravity_web_search_response(std::string_view body,
                                      const SearchRequest& request,
                                      const RedirectResolver& resolve = {});
[[nodiscard]] std::shared_ptr<IWebFetchBackend> make_zai_web_fetch_backend();
[[nodiscard]] std::shared_ptr<IWebFetchBackend> make_direct_web_fetch_backend();

} // namespace core::tools::web
