#include "McpHttpToolCall.hpp"

#include "WebBackendSupport.hpp"
#include "../llm/protocols/SseUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "core/version/Version.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <format>
#include <optional>
#include <utility>

namespace core::tools::web::detail {
namespace {

constexpr std::size_t kMaxResponseBytes = 4 * 1024 * 1024;
constexpr std::size_t kMaxQuotedBodyBytes = 1000;
/// First protocol revision that defines the MCP-Protocol-Version header.
constexpr std::string_view kVersionHeaderSince = "2025-06-18";

/// The JSON-RPC answer with @p id, delivered either as JSON or as an SSE stream.
[[nodiscard]] std::optional<std::string> rpc_message(std::string_view body, int id) {
    const auto matches = [id](std::string_view json) {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(json);
        simdjson::dom::object object;
        int64_t message_id = 0;
        return parser.parse(padded).get(object) == simdjson::SUCCESS
            && object["id"].get(message_id) == simdjson::SUCCESS && message_id == id;
    };
    const std::string_view trimmed = core::utils::str::trim_ascii_view(body);
    if (trimmed.starts_with('{')) {
        if (matches(trimmed)) return std::string(trimmed);
        return std::nullopt;
    }
    std::size_t start = 0;
    while (start < body.size()) {
        std::size_t end = body.find("\n\n", start);
        std::size_t delimiter = 2;
        if (const auto crlf = body.find("\r\n\r\n", start);
            crlf != std::string_view::npos && (end == std::string_view::npos || crlf < end)) {
            end = crlf;
            delimiter = 4;
        }
        const std::string_view frame = body.substr(
            start, end == std::string_view::npos ? std::string_view::npos : end - start);
        core::llm::protocols::sse::ParsedEventView event;
        if (core::llm::protocols::sse::parse_event_payload(frame, event, false)
            && matches(event.data)) {
            return std::string(event.data);
        }
        if (end == std::string_view::npos) break;
        start = end + delimiter;
    }
    return std::nullopt;
}

class Session {
public:
    explicit Session(const McpHttpToolCall& call) : call_(call) {}

    [[nodiscard]] std::expected<std::string, McpHttpFailure> request(
        int id, std::string_view method, std::string_view params) {
        core::utils::JsonWriter writer(128 + params.size());
        {
            auto _root = writer.object();
            writer.kv_str("jsonrpc", "2.0").comma().kv_num("id", id).comma()
                  .kv_str("method", method).comma().kv_raw("params", params);
        }
        auto response = send(method, std::move(writer).take());
        if (!response) return std::unexpected(std::move(response.error()));
        auto message = rpc_message(response->text, id);
        if (!message) {
            return std::unexpected(McpHttpFailure{.kind = McpHttpFailure::Kind::InvalidResponse,
                                                  .stage = method});
        }
        return std::move(*message);
    }

    /// Best effort: a session the server did not accept fails at the next call.
    void notify(std::string_view method) {
        core::utils::JsonWriter writer(96);
        {
            auto _root = writer.object();
            writer.kv_str("jsonrpc", "2.0").comma().kv_str("method", method);
        }
        (void)send(method, std::move(writer).take());
    }

    void negotiated(std::string_view version) {
        if (version >= kVersionHeaderSince) protocol_version_ = std::string(version);
    }

private:
    [[nodiscard]] bool cancelled() const { return call_.cancelled && call_.cancelled(); }

    [[nodiscard]] cpr::Response post(const std::string& body) {
        cpr::Header headers{{"Content-Type", "application/json"},
                            {"Accept", "application/json, text/event-stream"},
                            {"User-Agent", std::string(core::version::user_agent)}};
        if (call_.credentials) {
            for (const auto& [key, value] : call_.credentials->get_auth().headers) {
                headers[key] = value;
            }
        }
        if (!session_id_.empty()) headers["Mcp-Session-Id"] = session_id_;
        if (!protocol_version_.empty()) headers["MCP-Protocol-Version"] = protocol_version_;
        return cpr::Post(
            cpr::Url{call_.url}, std::move(headers), cpr::Body{body},
            cpr::Timeout{call_.timeout_ms}, cpr::Redirect{false},
            cpr::ProgressCallback{[this](cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t,
                                         cpr::cpr_pf_arg_t, cpr::cpr_pf_arg_t, intptr_t) {
                return !cancelled();
            }});
    }

    [[nodiscard]] std::expected<cpr::Response, McpHttpFailure> send(
        std::string_view stage, const std::string& body) {
        cpr::Response response = post(body);
        if (response.status_code == 401 && call_.credentials
            && call_.credentials->refresh_on_auth_failure()) {
            response = post(body);
        }
        using Kind = McpHttpFailure::Kind;
        if (cancelled()) return std::unexpected(McpHttpFailure{.kind = Kind::Cancelled, .stage = stage});
        if (response.error.code != cpr::ErrorCode::OK) {
            return std::unexpected(McpHttpFailure{
                .kind = Kind::Transport, .stage = stage, .detail = response.error.message});
        }
        if (response.text.size() > kMaxResponseBytes) {
            return std::unexpected(McpHttpFailure{.kind = Kind::InvalidResponse, .stage = stage});
        }
        if (response.status_code < 200 || response.status_code >= 300) {
            return std::unexpected(McpHttpFailure{.kind = Kind::Http, .stage = stage,
                                                  .status = response.status_code,
                                                  .detail = std::move(response.text)});
        }
        if (const auto id = find_header_case_insensitive(response.header, "mcp-session-id")) {
            session_id_ = *id;
        }
        return response;
    }

    const McpHttpToolCall& call_;
    std::string session_id_;
    std::string protocol_version_;
};

[[nodiscard]] std::string initialize_params(std::string_view protocol_version) {
    core::utils::JsonWriter writer(256);
    {
        auto _root = writer.object();
        writer.kv_str("protocolVersion", protocol_version).comma().key("capabilities");
        { auto _capabilities = writer.object(); }
        writer.comma().key("clientInfo");
        auto _client = writer.object();
        writer.kv_str("name", core::version::product_name).comma()
              .kv_str("version", core::version::value);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string_view negotiated_version(simdjson::dom::parser& parser,
                                                  const std::string& message) {
    std::string_view version;
    if (parser.parse(message)["result"]["protocolVersion"].get(version) != simdjson::SUCCESS) {
        return {};
    }
    return version;
}

[[nodiscard]] std::expected<std::string, McpHttpFailure> tool_text(const std::string& message) {
    using Kind = McpHttpFailure::Kind;
    constexpr std::string_view kStage = "tools/call";
    simdjson::dom::parser parser;
    simdjson::dom::object root;
    if (parser.parse(message).get(root) != simdjson::SUCCESS) {
        return std::unexpected(McpHttpFailure{.kind = Kind::InvalidResponse, .stage = kStage});
    }
    if (simdjson::dom::object error; root["error"].get(error) == simdjson::SUCCESS) {
        std::string_view text;
        if (error["message"].get(text) != simdjson::SUCCESS) text = {};
        return std::unexpected(McpHttpFailure{.kind = Kind::Rpc, .stage = kStage,
                                              .detail = std::string(text)});
    }
    simdjson::dom::object result;
    if (root["result"].get(result) != simdjson::SUCCESS) {
        return std::unexpected(McpHttpFailure{.kind = Kind::InvalidResponse, .stage = kStage});
    }
    std::string text;
    if (simdjson::dom::array content; result["content"].get(content) == simdjson::SUCCESS) {
        for (const auto block : content) {
            std::string_view type;
            if (block["type"].get(type) != simdjson::SUCCESS) type = "text";
            std::string_view item;
            if (type != "text" || block["text"].get(item) != simdjson::SUCCESS || item.empty()) {
                continue;
            }
            if (!text.empty()) text += "\n\n";
            text += item;
        }
    }
    if (bool is_error = false; result["isError"].get(is_error) == simdjson::SUCCESS && is_error) {
        return std::unexpected(McpHttpFailure{.kind = Kind::Tool, .stage = kStage,
                                              .detail = std::move(text)});
    }
    return text;
}

} // namespace

std::expected<std::string, McpHttpFailure> call_mcp_http_tool(const McpHttpToolCall& call) {
    Session session(call);
    auto initialized = session.request(1, "initialize", initialize_params(call.protocol_version));
    if (!initialized) return std::unexpected(std::move(initialized.error()));
    simdjson::dom::parser parser;
    session.negotiated(negotiated_version(parser, *initialized));
    session.notify("notifications/initialized");

    core::utils::JsonWriter params(64 + call.tool.size() + call.arguments_json.size());
    {
        auto _root = params.object();
        params.kv_str("name", call.tool).comma().kv_raw("arguments", call.arguments_json);
    }
    auto called = session.request(2, "tools/call", std::move(params).take());
    if (!called) return std::unexpected(std::move(called.error()));
    return tool_text(*called);
}

std::string describe_mcp_failure(const McpHttpFailure& failure, std::string_view service) {
    using Kind = McpHttpFailure::Kind;
    switch (failure.kind) {
    case Kind::Cancelled:
        return std::format("{} cancelled.", service);
    case Kind::Transport:
        return std::format("{} MCP {} request failed: {}", service, failure.stage, failure.detail);
    case Kind::Http:
        return std::format("{} MCP {} failed with HTTP {}: {}", service, failure.stage,
                           failure.status,
                           std::string_view(failure.detail).substr(0, kMaxQuotedBodyBytes));
    case Kind::InvalidResponse:
        return std::format("{} MCP {} returned an invalid response.", service, failure.stage);
    case Kind::Rpc:
    case Kind::Tool:
        return failure.detail.empty()
            ? std::format("{} MCP tool call failed.", service)
            : std::format("{} MCP tool call failed: {}", service, failure.detail);
    }
    return std::format("{} MCP tool call failed.", service);
}

} // namespace core::tools::web::detail
