#include "WebBackendAdapters.hpp"

#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "WebBackendSupport.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"

#include <cpr/cpr.h>

#include <algorithm>
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

[[nodiscard]] std::string normalize_space(std::string text) {
    return core::utils::str::collapse_ascii_whitespace_copy(text);
}

void replace_all(std::string& text, std::string_view from, std::string_view to) {
    std::size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
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
    std::string lowered = core::utils::str::to_lower_ascii_copy(html);
    std::size_t pos = 0;
    while ((pos = lowered.find(open_tag, pos)) != std::string::npos) {
        const auto end = lowered.find(close_tag, pos);
        const auto erase_end = end == std::string::npos
            ? html.size()
            : end + close_tag.size();
        html.erase(pos, erase_end - pos);
        lowered.erase(pos, erase_end - pos);
    }
}

[[nodiscard]] std::string extract_title(std::string_view html) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(html);
    const auto start = lowered.find("<title");
    if (start == std::string::npos) return {};
    const auto start_close = lowered.find('>', start);
    if (start_close == std::string::npos) return {};
    const auto end = lowered.find("</title>", start_close + 1);
    if (end == std::string::npos) return {};
    return normalize_space(decode_html_entities(
        std::string(html.substr(start_close + 1, end - start_close - 1))));
}

[[nodiscard]] std::string html_to_text(std::string html) {
    erase_tag_blocks(html, "<script", "</script>");
    erase_tag_blocks(html, "<style", "</style>");
    erase_tag_blocks(html, "<noscript", "</noscript>");

    std::string text;
    text.reserve(html.size());
    bool in_tag = false;
    for (const char ch : html) {
        if (ch == '<') {
            in_tag = true;
            text.push_back(' ');
            continue;
        }
        if (ch == '>') {
            in_tag = false;
            text.push_back(' ');
            continue;
        }
        if (!in_tag) text.push_back(ch);
    }
    return normalize_space(decode_html_entities(std::move(text)));
}

[[nodiscard]] std::expected<void, std::string> validate_http_url(std::string_view url) {
    const auto host = core::utils::uri::extract_http_host(url);
    if (!host.has_value() || host->empty()) {
        return std::unexpected("URL must be a valid http:// or https:// URL without credentials.");
    }
    return {};
}

[[nodiscard]] std::expected<void, std::string> validate_fetch_target(std::string_view url) {
    if (auto valid = validate_http_url(url); !valid) {
        return std::unexpected(valid.error());
    }
    if (const auto policy_error = core::tools::policy::enforce_url_policy(
            names::kFetchUrl,
            url)) {
        return std::unexpected("Tool policy blocked URL: " + *policy_error);
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
    [[nodiscard]] std::string_view name() const noexcept override {
        return "direct-http-fetch";
    }

    [[nodiscard]] bool supports(const ToolInvocationContext&) const override {
        return true;
    }

    [[nodiscard]] std::expected<FetchResponse, std::string>
    fetch(const FetchRequest& request,
          const ToolInvocationContext&) const override {
        std::string current_url = request.url;
        if (auto valid = validate_fetch_target(current_url); !valid) {
            return std::unexpected(valid.error());
        }

        const int max_bytes = std::clamp(request.max_bytes, 1024, kMaxFetchBytes);
        std::string body;
        body.reserve(static_cast<std::size_t>(std::min(max_bytes, 64 * 1024)));
        bool truncated = false;

        cpr::Response response;
        for (int redirect_count = 0; redirect_count <= kMaxRedirects; ++redirect_count) {
            body.clear();
            truncated = false;
            response = cpr::Download(
                cpr::WriteCallback{[&body, &truncated, max_bytes](
                                       std::string_view data,
                                       intptr_t) {
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
                cpr::Timeout{kFetchTimeoutMs});

            if (truncated) {
                return std::unexpected(std::format(
                    "Fetch response exceeded max_bytes limit of {} bytes.",
                    max_bytes));
            }

            if (response.error.code != cpr::ErrorCode::OK) {
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
            if (auto valid = validate_fetch_target(*next_url); !valid) {
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
            .text = looks_html ? html_to_text(std::move(body)) : normalize_space(std::move(body)),
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
