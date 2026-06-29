#pragma once

#include "Tool.hpp"

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::web {

struct DomainFilters {
    std::vector<std::string> allowed_domains;
    std::vector<std::string> blocked_domains;
};

struct SearchRequest {
    std::string query;
    int limit = 5;
    bool include_page_content = false;
    DomainFilters domains;
};

struct SearchHit {
    std::string title;
    std::string url;
    std::string snippet;
    std::string content;
};

struct SearchResponse {
    std::string backend;
    std::string answer;
    std::vector<SearchHit> results;
};

struct FetchRequest {
    std::string url;
    int max_bytes = 200000;
};

struct FetchResponse {
    std::string final_url;
    std::string content_type;
    std::string title;
    std::string text;
    long status_code = 0;
    bool truncated = false;
};

class IWebSearchBackend {
public:
    virtual ~IWebSearchBackend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool supports(const ToolInvocationContext& context) const = 0;
    [[nodiscard]] virtual std::expected<SearchResponse, std::string>
    search(const SearchRequest& request, const ToolInvocationContext& context) const = 0;
};

class IWebFetchBackend {
public:
    virtual ~IWebFetchBackend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool supports(const ToolInvocationContext& context) const = 0;
    [[nodiscard]] virtual std::expected<FetchResponse, std::string>
    fetch(const FetchRequest& request, const ToolInvocationContext& context) const = 0;
};

class WebAccess {
public:
    WebAccess(std::vector<std::shared_ptr<IWebSearchBackend>> search_backends,
              std::vector<std::shared_ptr<IWebFetchBackend>> fetch_backends);

    static WebAccess& instance();

    [[nodiscard]] std::expected<SearchResponse, std::string>
    search(const SearchRequest& request, const ToolInvocationContext& context) const;

    [[nodiscard]] std::expected<FetchResponse, std::string>
    fetch(const FetchRequest& request, const ToolInvocationContext& context) const;

private:
    std::vector<std::shared_ptr<IWebSearchBackend>> search_backends_;
    std::vector<std::shared_ptr<IWebFetchBackend>> fetch_backends_;
};

[[nodiscard]] WebAccess make_default_web_access();

[[nodiscard]] std::string search_response_to_json(const SearchResponse& response);
[[nodiscard]] std::string fetch_response_to_json(const FetchResponse& response);

} // namespace core::tools::web
