#pragma once

/**
 * @file McpHttpToolCall.hpp
 * @brief One-shot MCP tool call over Streamable HTTP, for provider-hosted web
 * services that expose a single tool behind an MCP endpoint (Z.ai search and
 * reader, Qwen Subscription search).
 *
 * The exchange is the minimal session the spec allows: `initialize`, then
 * `notifications/initialized`, then `tools/call`. The session id the server
 * assigns and, from protocol 2025-06-18 on, the negotiated version travel on
 * every later request. Auth headers are read from the credential source per
 * request and refreshed once on HTTP 401. Callers own their error wording.
 */

#include "../auth/ICredentialSource.hpp"

#include <expected>
#include <functional>
#include <string>
#include <string_view>

namespace core::tools::web::detail {

struct McpHttpToolCall {
    std::string url;
    std::string tool;
    /// JSON object passed as `arguments`.
    std::string arguments_json;
    /// Version requested at `initialize`; the server's answer wins.
    std::string_view protocol_version;
    core::auth::ICredentialSource* credentials = nullptr;
    int timeout_ms = 30000;
    std::function<bool()> cancelled = {};
};

struct McpHttpFailure {
    enum class Kind {
        Cancelled,
        Transport,       ///< `detail` is the transport error.
        Http,            ///< `status` and `detail` (the response body) are set.
        InvalidResponse, ///< No JSON-RPC answer, or one over the size limit.
        Rpc,             ///< JSON-RPC error; `detail` is its message.
        Tool,            ///< `isError` result; `detail` is the tool's text.
    };
    Kind kind = Kind::InvalidResponse;
    std::string_view stage; ///< "initialize" or "tools/call".
    long status = 0;
    std::string detail;
};

/// Joined text content of the tool result; empty when it returned no text.
[[nodiscard]] std::expected<std::string, McpHttpFailure> call_mcp_http_tool(
    const McpHttpToolCall& call);

/// "<service> MCP <stage> failed with HTTP 500: <body>" and similar.
[[nodiscard]] std::string describe_mcp_failure(const McpHttpFailure& failure,
                                               std::string_view service);

} // namespace core::tools::web::detail
