#include "WebBackendAdapters.hpp"

#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "WebBackendSupport.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"

#include <cpr/cpr.h>

#include <algorithm>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace core::tools::web {

namespace {

constexpr int kFetchTimeoutMs = 60000;
constexpr int kMaxFetchBytes = 2 * 1024 * 1024;
constexpr int kMaxRedirects = 5;

// A fetched page is up to kMaxFetchBytes. Every step below rewrites it in
// place, since none of them makes text longer: no lowered copy of the page per
// pass, and no erase()/replace() per match, which shifted the rest of the page
// each time and took seconds on entity- or script-heavy pages.

[[nodiscard]] std::string normalize_space(std::string text) {
    core::utils::str::collapse_ascii_whitespace(text);
    return text;
}

/// Replaces every @p from with the shorter @p to, left to right.
void replace_all(std::string& text, std::string_view from, std::string_view to) {
    std::size_t write = 0;
    std::size_t read = 0;
    for (std::size_t match = text.find(from); ; match = text.find(from, read)) {
        const std::size_t keep_end = match == std::string::npos ? text.size() : match;
        std::memmove(text.data() + write, text.data() + read, keep_end - read);
        write += keep_end - read;
        if (match == std::string::npos) break;
        std::memcpy(text.data() + write, to.data(), to.size());
        write += to.size();
        read = match + from.size();
    }
    text.resize(write);
}

[[nodiscard]] std::string decode_html_entities(std::string text) {
    replace_all(text, "&amp;", "&");
    replace_all(text, "&lt;", "<");
    replace_all(text, "&gt;", ">");
    replace_all(text, "&quot;", "\"");
    replace_all(text, "&#39;", "'");
    replace_all(text, "&nbsp;", " ");
    return text;
}

void erase_tag_blocks(std::string& html,
                      std::string_view open_tag,
                      std::string_view close_tag) {
    const core::utils::str::CaseInsensitiveAsciiSearcher open(open_tag);
    const core::utils::str::CaseInsensitiveAsciiSearcher close(close_tag);
    std::size_t write = 0;
    std::size_t read = 0;
    for (std::size_t begin = open.find(html); ; begin = open.find(html, read)) {
        const std::size_t keep_end = begin == std::string::npos ? html.size() : begin;
        std::memmove(html.data() + write, html.data() + read, keep_end - read);
        write += keep_end - read;
        if (begin == std::string::npos) break;
        const auto end = close.find(html, begin);
        read = end == std::string::npos ? html.size() : end + close_tag.size();
    }
    html.resize(write);
}

[[nodiscard]] std::string extract_title(std::string_view html) {
    const auto start = core::utils::str::CaseInsensitiveAsciiSearcher("<title").find(html);
    if (start == std::string::npos) return {};
    const auto start_close = html.find('>', start);
    if (start_close == std::string::npos) return {};
    const auto end =
        core::utils::str::CaseInsensitiveAsciiSearcher("</title>").find(html, start_close + 1);
    if (end == std::string::npos) return {};
    return normalize_space(decode_html_entities(
        std::string(html.substr(start_close + 1, end - start_close - 1))));
}

[[nodiscard]] std::string html_to_text(std::string html) {
    erase_tag_blocks(html, "<script", "</script>");
    erase_tag_blocks(html, "<style", "</style>");
    erase_tag_blocks(html, "<noscript", "</noscript>");

    // Tags become single spaces, so the text never outgrows the page.
    std::size_t write = 0;
    bool in_tag = false;
    for (const char ch : html) {
        if (ch == '<') {
            in_tag = true;
            html[write++] = ' ';
            continue;
        }
        if (ch == '>') {
            in_tag = false;
            html[write++] = ' ';
            continue;
        }
        if (!in_tag) html[write++] = ch;
    }
    html.resize(write);
    return normalize_space(decode_html_entities(std::move(html)));
}

[[nodiscard]] std::expected<void, std::string> validate_http_url(std::string_view url) {
    const auto host = core::utils::uri::extract_http_host(url);
    if (!host.has_value() || host->empty()) {
        return std::unexpected("URL must be a valid http:// or https:// URL without credentials.");
    }
    return {};
}

[[nodiscard]] std::expected<void, std::string> validate_fetch_target(std::string_view url, std::string_view operation) {
    if (auto valid = validate_http_url(url); !valid) {
        return std::unexpected(valid.error());
    }
    if (const auto policy_error = core::tools::policy::enforce_url_policy(
            names::kFetchUrl,
            url)) {
        return std::unexpected("Tool policy blocked URL: " + *policy_error);
    }
    if (!operation.empty()) {
        if (auto error = core::tools::policy::enforce_url_policy(operation, url))
            return std::unexpected("Tool policy blocked URL: " + *error);
    }
    return {};
}

[[nodiscard]] bool is_redirect_status(long status_code) noexcept {
    return status_code == 301
        || status_code == 302
        || status_code == 303
        || status_code == 307
        || status_code == 308;
}

[[nodiscard]] std::optional<std::string> scheme_for(std::string_view url) {
    if (core::utils::ascii::istarts_with(url, "https://")) return "https";
    if (core::utils::ascii::istarts_with(url, "http://")) return "http";
    return std::nullopt;
}

[[nodiscard]] std::string origin_for(std::string_view url) {
    const auto scheme = scheme_for(url);
    const auto host = core::utils::uri::extract_http_host(url);
    if (!scheme.has_value() || !host.has_value()) return {};

    const std::size_t authority_start = scheme->size() + 3;
    const std::size_t authority_end = url.find_first_of("/?#", authority_start);
    return std::string(url.substr(0, authority_end == std::string_view::npos
        ? url.size()
        : authority_end));
}

[[nodiscard]] std::string directory_base_for(std::string_view url) {
    std::string origin = origin_for(url);
    if (origin.empty()) return {};

    const std::size_t path_start = origin.size();
    const std::size_t query_start = url.find_first_of("?#", path_start);
    const std::string_view path = url.substr(
        path_start,
        query_start == std::string_view::npos ? std::string_view::npos : query_start - path_start);
    const std::size_t slash = path.rfind('/');
    if (slash == std::string_view::npos) return origin + "/";
    return origin + std::string(path.substr(0, slash + 1));
}

[[nodiscard]] std::expected<std::string, std::string> resolve_redirect_url(
    std::string_view current_url,
    std::string_view location_header) {
    const std::string location = core::utils::str::trim_ascii_copy(location_header);
    if (location.empty()) {
        return std::unexpected("Redirect response did not include a Location header.");
    }

    if (core::utils::ascii::istarts_with(location, "http://")
        || core::utils::ascii::istarts_with(location, "https://")) {
        return location;
    }

    if (location.starts_with("//")) {
        const auto scheme = scheme_for(current_url);
        if (!scheme.has_value()) return std::unexpected("Redirect target could not be resolved.");
        return *scheme + ":" + location;
    }

    if (location.starts_with('/')) {
        const std::string origin = origin_for(current_url);
        if (origin.empty()) return std::unexpected("Redirect target could not be resolved.");
        return origin + location;
    }

    const std::string base = directory_base_for(current_url);
    if (base.empty()) return std::unexpected("Redirect target could not be resolved.");
    return base + location;
}

class DirectWebFetchBackend final : public IWebFetchBackend {
public:
    [[nodiscard]] bool preserves_bytes() const noexcept override { return true; }
    [[nodiscard]] std::string_view name() const noexcept override {
        return "direct-http-fetch";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext&) const override {
        return true;
    }

    [[nodiscard]] std::expected<FetchResponse, std::string>
    fetch(const FetchRequest& request,
          const ToolInvocationContext& context) const override {
        std::string current_url = request.url;
        if (auto valid = validate_fetch_target(current_url, request.policy_tool); !valid) {
            return std::unexpected(valid.error());
        }

        const int max_bytes = std::clamp(request.max_bytes, 1024, kMaxFetchBytes);
        std::string body;
        body.reserve(static_cast<std::size_t>(std::min(max_bytes, 64 * 1024)));
        bool truncated = false;

        cpr::Response response;
        for (int redirect_count = 0; redirect_count <= kMaxRedirects; ++redirect_count) {
            if (context.cancellation_requested && context.cancellation_requested())
                return std::unexpected("Request cancelled.");
            body.clear();
            truncated = false;
            response = cpr::Download(
                cpr::WriteCallback{[&body, &truncated, &context, max_bytes](
                                       std::string_view data,
                                       intptr_t) {
                    if (context.cancellation_requested && context.cancellation_requested()) return false;
                    const std::size_t remaining =
                        static_cast<std::size_t>(max_bytes) - body.size();
                    if (data.size() > remaining) {
                        body.append(data.substr(0, remaining));
                        truncated = true;
                        return false;
                    }
                    body.append(data);
                    return true;
                }},
                cpr::Url{current_url},
                cpr::Header{
                    {"Accept", "text/markdown, text/html, text/plain, */*"},
                    {"User-Agent", "filo/1.0 (+https://github.com/alessio/filo)"},
                },
                cpr::Redirect{false},
                cpr::ProgressCallback{[&context](cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t,
                    cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t, intptr_t) {
                    return !context.cancellation_requested || !context.cancellation_requested();
                }},
                cpr::Timeout{kFetchTimeoutMs});

            // Returning false from the write callback intentionally aborts
            // libcurl once the bounded buffer is full.  That is a successful
            // partial retrieval: retain the captured prefix and surface it
            // below with FetchResponse::truncated instead of converting the
            // page into a tool error.
            if (!truncated && response.error.code != cpr::ErrorCode::OK) {
                return std::unexpected("Fetch request failed: " + response.error.message);
            }

            if (!is_redirect_status(response.status_code)) {
                break;
            }

            if (redirect_count == kMaxRedirects) {
                return std::unexpected("Fetch failed: too many redirects.");
            }

            const auto location =
                detail::find_header_case_insensitive(response.header, "location");
            if (!location.has_value()) {
                return std::unexpected("Redirect response did not include a Location header.");
            }

            auto next_url = resolve_redirect_url(current_url, *location);
            if (!next_url) {
                return std::unexpected(next_url.error());
            }
            if (auto valid = validate_fetch_target(*next_url, request.policy_tool); !valid) {
                return std::unexpected(valid.error());
            }
            current_url = std::move(*next_url);
        }

        if (response.status_code < 200 || response.status_code >= 300) {
            return std::unexpected(std::format(
                "Fetch failed with HTTP {}: {}",
                response.status_code,
                body.substr(0, 1000)));
        }

        const std::string content_type =
            detail::find_header_case_insensitive(response.header, "content-type").value_or("");

        const std::string preview = body.substr(0, std::min<std::size_t>(body.size(), 512));
        const bool looks_html = core::utils::str::contains_case_insensitive(content_type, "text/html")
            || core::utils::str::contains_case_insensitive(preview, "<html")
            || core::utils::str::contains_case_insensitive(preview, "<!doctype html");

        FetchResponse out{
            .final_url = response.url.str().empty() ? current_url : response.url.str(),
            .content_type = content_type,
            .title = looks_html ? extract_title(body) : std::string{},
            .text = looks_html ? html_to_text(std::move(body))
                : request.preserve_bytes ? std::move(body) : normalize_space(std::move(body)),
            .status_code = response.status_code,
            .truncated = truncated,
        };
        return out;
    }
};

} // namespace

std::shared_ptr<IWebFetchBackend> make_direct_web_fetch_backend() {
    return std::make_shared<DirectWebFetchBackend>();
}

} // namespace core::tools::web
