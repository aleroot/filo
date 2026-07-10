#include "WebAccess.hpp"

#include "../utils/JsonWriter.hpp"

#include <format>
#include <utility>

namespace core::tools::web {

namespace {

// Keep individual web-tool results useful without letting a successful fetch
// consume an unbounded share of the model context. The transport budget lives
// in FetchRequest; this separate budget applies after text extraction.
constexpr std::size_t kMaxReturnedContentBytes = 64 * 1024;

[[nodiscard]] std::size_t utf8_prefix_boundary(std::string_view text,
                                                std::size_t max_bytes) noexcept {
    if (text.size() <= max_bytes) return text.size();

    // Keep a valid UTF-8 prefix valid when the byte budget lands in the middle
    // of a multi-byte code point. Malformed source text is left untouched;
    // this only avoids introducing a new malformed suffix while truncating.
    std::size_t lead = max_bytes;
    while (lead > 0
           && (static_cast<unsigned char>(text[lead - 1]) & 0xC0U) == 0x80U) {
        --lead;
    }
    if (lead == 0) return max_bytes;
    --lead;

    const unsigned char first = static_cast<unsigned char>(text[lead]);
    const std::size_t sequence_length = (first & 0x80U) == 0 ? 1
        : (first & 0xE0U) == 0xC0U ? 2
        : (first & 0xF0U) == 0xE0U ? 3
        : (first & 0xF8U) == 0xF0U ? 4
        : 0;
    if (sequence_length == 0 || max_bytes - lead >= sequence_length) {
        return max_bytes;
    }
    return lead;
}

} // namespace

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
    const bool content_truncated = response.text.size() > kMaxReturnedContentBytes;
    const std::string_view content = content_truncated
        ? std::string_view(response.text).substr(
            0,
            utf8_prefix_boundary(response.text, kMaxReturnedContentBytes))
        : std::string_view(response.text);

    core::utils::JsonWriter writer(4096);
    {
        auto _root = writer.object();
        writer.kv_str("url", response.final_url).comma()
              .kv_num("status_code", response.status_code).comma()
              .kv_str("content_type", response.content_type).comma()
              .kv_str("title", response.title).comma()
              .kv_bool("truncated", response.truncated || content_truncated).comma()
              .kv_str("content", content);
    }
    return std::move(writer).take();
}

} // namespace core::tools::web
