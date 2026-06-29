#include "WebAccess.hpp"

#include "../utils/JsonWriter.hpp"

#include <format>
#include <utility>

namespace core::tools::web {

WebAccess& WebAccess::instance() {
    static WebAccess access = make_default_web_access();
    return access;
}

WebAccess::WebAccess(std::vector<std::shared_ptr<IWebSearchBackend>> search_backends,
                     std::vector<std::shared_ptr<IWebFetchBackend>> fetch_backends)
    : search_backends_(std::move(search_backends))
    , fetch_backends_(std::move(fetch_backends)) {}

std::expected<SearchResponse, std::string>
WebAccess::search(const SearchRequest& request,
                  const ToolInvocationContext& context) const {
    for (const auto& backend : search_backends_) {
        if (!backend || !backend->supports(context)) continue;
        return backend->search(request, context);
    }

    std::string provider = context.provider_name.empty() ? "<unknown>" : context.provider_name;
    return std::unexpected(std::format(
        "No web search backend is available for provider '{}'. "
        "Configure a provider-native web search backend, or use fetch_url when "
        "you already have a URL.",
        provider));
}

std::expected<FetchResponse, std::string>
WebAccess::fetch(const FetchRequest& request,
                 const ToolInvocationContext& context) const {
    for (const auto& backend : fetch_backends_) {
        if (!backend || !backend->supports(context)) continue;
        return backend->fetch(request, context);
    }
    return std::unexpected("No URL fetch backend is available.");
}

std::string search_response_to_json(const SearchResponse& response) {
    core::utils::JsonWriter writer(4096);
    {
        auto _root = writer.object();
        writer.kv_str("backend", response.backend).comma()
              .kv_str("answer", response.answer).comma()
              .key("results");
        {
            auto _results = writer.array();
            bool first = true;
            for (const auto& result : response.results) {
                if (!first) writer.comma();
                first = false;
                auto _item = writer.object();
                writer.kv_str("title", result.title).comma()
                      .kv_str("url", result.url).comma()
                      .kv_str("snippet", result.snippet).comma()
                      .kv_str("content", result.content);
            }
        }
    }
    return std::move(writer).take();
}

std::string fetch_response_to_json(const FetchResponse& response) {
    core::utils::JsonWriter writer(4096);
    {
        auto _root = writer.object();
        writer.kv_str("url", response.final_url).comma()
              .kv_num("status_code", response.status_code).comma()
              .kv_str("content_type", response.content_type).comma()
              .kv_str("title", response.title).comma()
              .kv_bool("truncated", response.truncated).comma()
              .kv_str("content", response.text);
    }
    return std::move(writer).take();
}

} // namespace core::tools::web
