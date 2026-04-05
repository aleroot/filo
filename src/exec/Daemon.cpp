#include "Daemon.hpp"
#include "RequestReplayCache.hpp"
#include <httplib.h>
#include <format>
#include "../core/config/ConfigManager.hpp"
#include "../core/agent/Agent.hpp"
#include "../core/llm/ProviderManager.hpp"
#include "../core/llm/ProviderFactory.hpp"
#include "../core/mcp/McpDispatcher.hpp"
#include "../core/tools/ToolManager.hpp"
#include "../core/tools/ShellTool.hpp"
#include "../core/context/SessionContext.hpp"
#include "../core/utils/JsonUtils.hpp"
#include "../core/utils/JsonWriter.hpp"
#include "../core/utils/UriUtils.hpp"
#include "../core/workspace/Workspace.hpp"
#include "../core/logging/Logger.hpp"
#include <simdjson.h>
#include <array>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <system_error>
#include <string_view>
#include <unordered_map>

namespace exec::daemon {

namespace {
std::mutex g_server_mutex;
std::shared_ptr<httplib::Server> g_svr;
} // namespace

void stop_server() {
    std::shared_ptr<httplib::Server> server;
    {
        std::lock_guard<std::mutex> lock(g_server_mutex);
        server = g_svr;
    }
    if (server) server->stop();
}

static void init_providers() {
    auto& config_manager = core::config::ConfigManager::get_instance();
    const auto& config = config_manager.get_config();
    auto& provider_manager = core::llm::ProviderManager::get_instance();
    for (const auto& [name, pconfig] : config.providers) {
        if (auto provider = core::llm::ProviderFactory::create_provider(name, pconfig))
            provider_manager.register_provider(name, provider);
    }
}

namespace {

constexpr std::string_view kJsonRpcParseErrorPrefix{
    R"({"jsonrpc":"2.0","error":{"code":-32700)"};
constexpr std::string_view kJsonRpcInvalidRequestPrefix{
    R"({"jsonrpc":"2.0","error":{"code":-32600)"};

[[nodiscard]] detail::RequestReplayCache& replay_cache() {
    static detail::RequestReplayCache cache;
    return cache;
}

using core::utils::JsonWriter;
namespace uri = core::utils::uri;

struct ResponseId {
    enum class Kind {
        none,
        integer,
        string,
    };

    Kind kind = Kind::none;
    int64_t integer_value = 0;
    std::string string_value;
};

struct ParsedJsonRpcMessage {
    enum class Kind {
        request,
        notification,
        response,
    };

    Kind kind = Kind::request;
    ResponseId id;
    std::string method;
};

struct PendingServerResponse {
    PendingServerResponse()
        : future(promise.get_future().share()) {}

    std::promise<std::string> promise;
    std::shared_future<std::string> future;
};

struct HttpSessionState {
    std::string protocol_version;
    bool client_ready = false;
    bool client_supports_roots = false;
    bool client_supports_roots_list_changed = false;
    bool roots_dirty = false;
    bool roots_refresh_in_progress = false;
    std::optional<core::workspace::WorkspaceSnapshot> cached_workspace;
    std::uint64_t workspace_version = 0;
    std::unordered_map<std::string, std::shared_ptr<PendingServerResponse>> pending_server_responses;
    std::uint64_t next_server_request_id = 1;
    std::deque<std::chrono::steady_clock::time_point> recent_tool_calls;
    std::chrono::steady_clock::time_point last_used{std::chrono::steady_clock::now()};
    std::condition_variable roots_cv;
    std::mutex mutex;
};

std::mutex g_http_sessions_mutex;
std::unordered_map<std::string, std::shared_ptr<HttpSessionState>> g_http_sessions;

constexpr std::size_t kMaxToolCallsPerWindow = 512;
constexpr auto kToolRateWindow = std::chrono::minutes{1};

void write_response_id(JsonWriter& w, const ResponseId& id) {
    w.key("id");
    switch (id.kind) {
        case ResponseId::Kind::integer:
            w.number(id.integer_value);
            break;
        case ResponseId::Kind::string:
            w.str(id.string_value);
            break;
        case ResponseId::Kind::none:
            w.null_val();
            break;
    }
}

[[nodiscard]] std::string make_jsonrpc_error_body(const ResponseId& id,
                                                  int code,
                                                  std::string_view message) {
    JsonWriter w(128 + message.size());
    {
        auto _obj = w.object();
        w.kv_str("jsonrpc", "2.0").comma();
        write_response_id(w, id);
        w.comma().key("error");
        {
            auto _err = w.object();
            w.kv_num("code", code).comma().kv_str("message", message);
        }
    }
    return std::move(w).take();
}

[[nodiscard]] std::optional<ParsedJsonRpcMessage>
parse_jsonrpc_message(std::string_view json_body) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return std::nullopt;

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return std::nullopt;

    std::string_view jsonrpc;
    if (root["jsonrpc"].get_string().get(jsonrpc) != simdjson::SUCCESS || jsonrpc != "2.0") {
        return std::nullopt;
    }

    ParsedJsonRpcMessage parsed;

    simdjson::ondemand::value id_value;
    if (root["id"].get(id_value) == simdjson::SUCCESS) {
        simdjson::ondemand::json_type id_type;
        if (id_value.type().get(id_type) != simdjson::SUCCESS) return std::nullopt;

        if (id_type == simdjson::ondemand::json_type::number) {
            int64_t id_num = 0;
            if (id_value.get_int64().get(id_num) != simdjson::SUCCESS) return std::nullopt;
            parsed.id.kind = ResponseId::Kind::integer;
            parsed.id.integer_value = id_num;
        } else if (id_type == simdjson::ondemand::json_type::string) {
            std::string_view id_str;
            if (id_value.get_string().get(id_str) != simdjson::SUCCESS) return std::nullopt;
            parsed.id.kind = ResponseId::Kind::string;
            parsed.id.string_value.assign(id_str);
        } else {
            return std::nullopt;
        }
    }

    std::string_view method;
    if (root["method"].get_string().get(method) == simdjson::SUCCESS) {
        parsed.method.assign(method);
        parsed.kind = parsed.id.kind == ResponseId::Kind::none
            ? ParsedJsonRpcMessage::Kind::notification
            : ParsedJsonRpcMessage::Kind::request;
        return parsed;
    }

    simdjson::ondemand::value ignored;
    if (root["result"].get(ignored) == simdjson::SUCCESS
        || root["error"].get(ignored) == simdjson::SUCCESS) {
        parsed.kind = ParsedJsonRpcMessage::Kind::response;
        return parsed;
    }

    return std::nullopt;
}

[[nodiscard]] std::optional<std::string>
extract_initialize_protocol_version(std::string_view response_body) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(response_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return std::nullopt;

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return std::nullopt;

    simdjson::ondemand::object result;
    if (root["result"].get_object().get(result) != simdjson::SUCCESS) return std::nullopt;

    std::string_view version;
    if (result["protocolVersion"].get_string().get(version) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    return std::string(version);
}

[[nodiscard]] std::string to_lower_ascii(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] std::string response_id_key(const ResponseId& id) {
    switch (id.kind) {
        case ResponseId::Kind::integer:
            return std::format("i:{}", id.integer_value);
        case ResponseId::Kind::string:
            return std::string("s:") + id.string_value;
        case ResponseId::Kind::none:
            return {};
    }
    return {};
}

[[nodiscard]] std::string string_response_id_key(std::string_view id) {
    return std::string("s:") + std::string(id);
}

[[nodiscard]] std::filesystem::path normalize_path(const std::filesystem::path& path) {
    std::error_code ec;
    auto normalized = std::filesystem::weakly_canonical(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    ec.clear();
    normalized = std::filesystem::absolute(path, ec);
    if (!ec) {
        return normalized.lexically_normal();
    }

    return path.lexically_normal();
}

[[nodiscard]] std::optional<std::filesystem::path>
parse_local_file_uri(std::string_view uri_value, std::string& error_out) {
    if (!to_lower_ascii(uri_value).starts_with("file://")) {
        error_out = "Only file:// root URIs are supported";
        return std::nullopt;
    }

    std::string_view rest = uri_value.substr(7);
    std::string_view encoded_path;
    if (rest.starts_with('/')) {
        encoded_path = rest;
    } else {
        const auto slash = rest.find('/');
        const std::string_view authority =
            slash == std::string_view::npos ? rest : rest.substr(0, slash);
        const auto lowered_authority = to_lower_ascii(authority);
        if (!authority.empty() && lowered_authority != "localhost") {
            error_out = "Unsupported file URI authority";
            return std::nullopt;
        }
        encoded_path = slash == std::string_view::npos ? std::string_view{"/"} : rest.substr(slash);
    }

    if (encoded_path.find('?') != std::string_view::npos
        || encoded_path.find('#') != std::string_view::npos) {
        error_out = "File URI must not include query or fragment";
        return std::nullopt;
    }

    std::string decoded_path;
    if (!uri::percent_decode(encoded_path, decoded_path)) {
        error_out = "Malformed percent-encoding in file URI";
        return std::nullopt;
    }

    if (decoded_path.empty() || decoded_path.find('\0') != std::string::npos) {
        error_out = "Invalid file URI path";
        return std::nullopt;
    }

    return normalize_path(std::filesystem::path(decoded_path));
}

[[nodiscard]] std::pair<bool, bool>
extract_initialize_roots_capability(std::string_view json_body) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return {false, false};

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return {false, false};

    simdjson::ondemand::object params;
    if (root["params"].get_object().get(params) != simdjson::SUCCESS) return {false, false};

    simdjson::ondemand::object capabilities;
    if (params["capabilities"].get_object().get(capabilities) != simdjson::SUCCESS) {
        return {false, false};
    }

    simdjson::ondemand::object roots;
    if (capabilities["roots"].get_object().get(roots) != simdjson::SUCCESS) {
        return {false, false};
    }

    bool list_changed = false;
    [[maybe_unused]] const auto ignored = roots["listChanged"].get(list_changed);
    return {true, list_changed};
}

[[nodiscard]] bool request_needs_authoritative_workspace(const ParsedJsonRpcMessage& parsed) {
    return parsed.kind == ParsedJsonRpcMessage::Kind::request
        && (parsed.method == "tools/call"
            || parsed.method == "resources/list"
            || parsed.method == "resources/read");
}

[[nodiscard]] std::string build_sse_message(std::string_view payload) {
    std::string event;
    event.reserve(payload.size() + 24);
    event += "event: message\n";
    event += "data: ";
    event += payload;
    event += "\n\n";
    return event;
}

[[nodiscard]] std::string build_roots_list_request_body(std::string_view request_id) {
    JsonWriter w(96);
    {
        auto _obj = w.object();
        w.kv_str("jsonrpc", "2.0").comma()
            .kv_str("id", request_id).comma()
            .kv_str("method", "roots/list");
    }
    return std::move(w).take();
}

[[nodiscard]] std::optional<core::workspace::WorkspaceSnapshot>
parse_roots_list_response(std::string_view response_body,
                          std::string_view expected_request_id,
                          std::uint64_t workspace_version,
                          std::string& error_out) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(response_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
        error_out = "Client returned invalid JSON for roots/list";
        return std::nullopt;
    }

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        error_out = "Client returned invalid JSON-RPC response for roots/list";
        return std::nullopt;
    }

    std::string_view jsonrpc;
    if (root["jsonrpc"].get_string().get(jsonrpc) != simdjson::SUCCESS || jsonrpc != "2.0") {
        error_out = "Client roots/list response used an invalid JSON-RPC version";
        return std::nullopt;
    }

    std::string_view response_id;
    if (root["id"].get_string().get(response_id) != simdjson::SUCCESS
        || response_id != expected_request_id) {
        error_out = "Client returned an unexpected roots/list response id";
        return std::nullopt;
    }

    simdjson::ondemand::object error;
    if (root["error"].get_object().get(error) == simdjson::SUCCESS) {
        std::string_view message;
        if (error["message"].get_string().get(message) == simdjson::SUCCESS && !message.empty()) {
            error_out = std::string(message);
        } else {
            error_out = "Client rejected roots/list";
        }
        return std::nullopt;
    }

    simdjson::ondemand::object result;
    if (root["result"].get_object().get(result) != simdjson::SUCCESS) {
        error_out = "Client roots/list response was missing 'result'";
        return std::nullopt;
    }

    simdjson::ondemand::array roots;
    if (result["roots"].get_array().get(roots) != simdjson::SUCCESS) {
        error_out = "Client roots/list response was missing 'roots'";
        return std::nullopt;
    }

    core::workspace::WorkspaceSnapshot snapshot;
    snapshot.enforce = true;
    snapshot.version = workspace_version;

    bool first_root = true;
    for (auto root_item : roots) {
        simdjson::ondemand::object root_object;
        if (root_item.get_object().get(root_object) != simdjson::SUCCESS) {
            error_out = "Client roots/list entry was not an object";
            return std::nullopt;
        }

        std::string_view uri_value;
        if (root_object["uri"].get_string().get(uri_value) != simdjson::SUCCESS) {
            error_out = "Client roots/list entry was missing 'uri'";
            return std::nullopt;
        }

        std::string uri_error;
        auto parsed_path = parse_local_file_uri(uri_value, uri_error);
        if (!parsed_path.has_value()) {
            error_out = std::format("Invalid root URI '{}': {}",
                                    std::string(uri_value),
                                    uri_error);
            return std::nullopt;
        }

        if (first_root) {
            snapshot.primary = *parsed_path;
            first_root = false;
        } else {
            snapshot.additional.push_back(*parsed_path);
        }
    }

    return snapshot;
}

[[nodiscard]] std::optional<std::string> parse_origin_host(std::string_view origin) {
    // Strictly allow only http(s) origins. Reject opaque/special origins.
    const auto lower = to_lower_ascii(origin);
    std::size_t pos = std::string_view::npos;
    if (lower.starts_with("http://")) {
        pos = 7;
    } else if (lower.starts_with("https://")) {
        pos = 8;
    } else {
        return std::nullopt;
    }

    std::string_view rest = origin.substr(pos);
    if (rest.empty()) return std::nullopt;

    // Reject userinfo section.
    const std::size_t at = rest.find('@');
    const std::size_t slash = rest.find('/');
    if (at != std::string_view::npos
        && (slash == std::string_view::npos || at < slash)) {
        return std::nullopt;
    }

    const std::size_t authority_end = rest.find('/');
    std::string_view authority = (authority_end == std::string_view::npos)
        ? rest
        : rest.substr(0, authority_end);
    if (authority.empty()) return std::nullopt;

    // IPv6 literal
    if (authority.front() == '[') {
        const std::size_t end = authority.find(']');
        if (end == std::string_view::npos) return std::nullopt;
        return to_lower_ascii(authority.substr(0, end + 1));
    }

    const std::size_t colon = authority.find(':');
    if (colon == std::string_view::npos) {
        return to_lower_ascii(authority);
    }
    return to_lower_ascii(authority.substr(0, colon));
}

[[nodiscard]] bool is_loopback_host(std::string_view host) {
    const auto lower = to_lower_ascii(host);
    return lower == "localhost"
        || lower == "127.0.0.1"
        || lower == "::1"
        || lower == "[::1]";
}

[[nodiscard]] bool is_origin_allowed(const httplib::Request& req,
                                     std::string_view listen_host) {
    if (!req.has_header("Origin")) return true;

    const std::string origin = req.get_header_value("Origin");
    const auto origin_host = parse_origin_host(origin);
    if (!origin_host.has_value()) return false;
    if (is_loopback_host(*origin_host)) return true;

    const auto bind_host = to_lower_ascii(listen_host);
    if (bind_host != "0.0.0.0" && bind_host != "::" && !bind_host.empty()) {
        return *origin_host == bind_host;
    }
    return false;
}

[[nodiscard]] std::optional<std::string>
extract_request_id_for_replay(std::string_view json_body) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_body);
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return std::nullopt;

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return std::nullopt;

    simdjson::ondemand::value id_value;
    if (root["id"].get(id_value) != simdjson::SUCCESS) return std::nullopt;

    simdjson::ondemand::json_type id_type;
    if (id_value.type().get(id_type) != simdjson::SUCCESS) return std::nullopt;

    if (id_type == simdjson::ondemand::json_type::number) {
        int64_t id_num = 0;
        if (id_value.get_int64().get(id_num) != simdjson::SUCCESS) return std::nullopt;
        return std::format("i:{}", id_num);
    }
    if (id_type == simdjson::ondemand::json_type::string) {
        std::string_view id_str;
        if (id_value.get_string().get(id_str) != simdjson::SUCCESS) return std::nullopt;
        return std::string("s:") + std::string(id_str);
    }
    return std::nullopt;
}

[[nodiscard]] uint64_t fnv1a64(std::string_view value) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= static_cast<uint64_t>(ch);
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::optional<std::string> build_replay_key(const httplib::Request& req) {
    const auto request_id = extract_request_id_for_replay(req.body);
    if (!request_id.has_value()) return std::nullopt;

    if (!req.has_header("MCP-Session-Id")) return std::nullopt;
    const std::string session_id = req.get_header_value("MCP-Session-Id");
    if (session_id.empty()) return std::nullopt;

    const std::string session_key = detail::RequestReplayCache::session_token(session_id);

    const uint64_t h1 = fnv1a64(req.body);
    const uint64_t h2 = static_cast<uint64_t>(std::hash<std::string_view>{}(req.body));
    return std::format(
        "{}|{}|{:016x}{:016x}|{:08x}",
        session_key,
        *request_id,
        h1,
        h2,
        static_cast<unsigned>(req.body.size()));
}

[[nodiscard]] bool has_required_accept_header(const httplib::Request& req) {
    if (!req.has_header("Accept")) return false;

    const std::string accept = to_lower_ascii(req.get_header_value("Accept"));
    return accept.find("application/json") != std::string::npos
        && accept.find("text/event-stream") != std::string::npos;
}

[[nodiscard]] bool has_supported_protocol_header(const httplib::Request& req) {
    if (!req.has_header("MCP-Protocol-Version")) return true;
    const std::string version = req.get_header_value("MCP-Protocol-Version");
    return version == "2024-11-05" || version == "2025-11-25";
}

[[nodiscard]] std::pair<std::string, std::shared_ptr<HttpSessionState>>
create_http_session(std::string protocol_version) {
    static std::random_device rd;
    static constexpr std::string_view kHex = "0123456789abcdef";

    std::lock_guard<std::mutex> lock(g_http_sessions_mutex);
    for (;;) {
        std::array<unsigned char, 16> bytes{};
        for (auto& byte : bytes) {
            byte = static_cast<unsigned char>(rd());
        }

        std::string session_id;
        session_id.reserve(bytes.size() * 2);
        for (const unsigned char byte : bytes) {
            session_id.push_back(kHex[byte >> 4]);
            session_id.push_back(kHex[byte & 0x0f]);
        }

        if (g_http_sessions.contains(session_id)) continue;

        auto state = std::make_shared<HttpSessionState>();
        state->protocol_version = std::move(protocol_version);
        g_http_sessions.emplace(session_id, state);
        return {session_id, std::move(state)};
    }
}

[[nodiscard]] std::shared_ptr<HttpSessionState> find_http_session(std::string_view session_id) {
    std::lock_guard<std::mutex> lock(g_http_sessions_mutex);
    auto it = g_http_sessions.find(std::string(session_id));
    if (it == g_http_sessions.end()) return {};
    return it->second;
}

[[nodiscard]] bool erase_http_session(std::string_view session_id) {
    std::lock_guard<std::mutex> lock(g_http_sessions_mutex);
    return g_http_sessions.erase(std::string(session_id)) > 0;
}

void clear_http_sessions() {
    std::lock_guard<std::mutex> lock(g_http_sessions_mutex);
    g_http_sessions.clear();
}

[[nodiscard]] bool consume_tool_call_budget(HttpSessionState& session) {
    const auto now = std::chrono::steady_clock::now();
    while (!session.recent_tool_calls.empty()
           && now - session.recent_tool_calls.front() > kToolRateWindow) {
        session.recent_tool_calls.pop_front();
    }
    if (session.recent_tool_calls.size() >= kMaxToolCallsPerWindow) {
        return false;
    }
    session.recent_tool_calls.push_back(now);
    return true;
}

void mark_session_workspace_dirty(HttpSessionState& session) {
    session.roots_dirty = true;
    session.cached_workspace.reset();
    session.roots_cv.notify_all();
}

void apply_session_workspace(HttpSessionState& session,
                             core::workspace::WorkspaceSnapshot snapshot) {
    snapshot.version = ++session.workspace_version;
    session.cached_workspace = std::move(snapshot);
    session.roots_dirty = false;
    session.roots_refresh_in_progress = false;
    session.roots_cv.notify_all();
}

void fail_session_workspace_refresh(HttpSessionState& session) {
    session.cached_workspace.reset();
    session.roots_dirty = true;
    session.roots_refresh_in_progress = false;
    session.roots_cv.notify_all();
}

[[nodiscard]] std::shared_ptr<PendingServerResponse>
register_pending_server_response(HttpSessionState& session,
                                 std::string request_id) {
    auto pending = std::make_shared<PendingServerResponse>();
    session.pending_server_responses.emplace(
        string_response_id_key(request_id),
        pending);
    return pending;
}

[[nodiscard]] std::shared_ptr<PendingServerResponse>
take_pending_server_response(HttpSessionState& session,
                             std::string_view response_id) {
    auto it = session.pending_server_responses.find(std::string(response_id));
    if (it == session.pending_server_responses.end()) {
        return {};
    }

    auto pending = it->second;
    session.pending_server_responses.erase(it);
    return pending;
}

[[nodiscard]] detail::McpDispatchResult dispatch_mcp_payload(
    const std::string& body,
    const std::optional<core::workspace::WorkspaceSnapshot>& workspace_override,
    std::string_view session_id) {
    detail::McpDispatchResult result;
    const auto snapshot = workspace_override.value_or(
        core::workspace::Workspace::get_instance().snapshot());
    auto session_context = core::context::make_session_context(
        snapshot,
        core::context::SessionTransport::mcp_http,
        std::string(session_id));

    result.body = core::mcp::McpDispatcher::get_instance().dispatch(body, session_context);
    if (result.body.empty()) {
        result.status = 202;
        return result;
    }
    if (result.body.starts_with(kJsonRpcParseErrorPrefix)
        || result.body.starts_with(kJsonRpcInvalidRequestPrefix)) {
        result.status = 400;
    }
    return result;
}

void write_mcp_result(httplib::Response& res, const detail::McpDispatchResult& result) {
    res.status = result.status;
    if (!result.body.empty()) {
        res.set_content(result.body, "application/json");
    }
}

void set_cors_headers(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE");
    res.set_header("Access-Control-Allow-Headers",
                   "Content-Type, Accept, Authorization, MCP-Protocol-Version, MCP-Session-Id, Last-Event-ID");
    res.set_header("Access-Control-Expose-Headers", "MCP-Session-Id");
    res.set_header("Access-Control-Max-Age", "86400");
    res.set_header("Vary", "Origin");
    if (req.has_header("Origin")) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
    }
}

[[nodiscard]] bool enforce_origin_policy(const httplib::Request& req,
                                         httplib::Response& res,
                                         std::string_view listen_host) {
    if (is_origin_allowed(req, listen_host)) return true;

    res.status = 403;
    res.set_content(
        R"({"jsonrpc":"2.0","error":{"code":-32001,"message":"Forbidden origin"},"id":null})",
        "application/json");
    return false;
}

void handle_daemon_exception(const httplib::Request& /*req*/,
                             httplib::Response& res,
                             std::exception_ptr ep) {
    std::string message = "Unknown internal server error";
    try {
        if (ep) std::rethrow_exception(ep);
    } catch (const std::exception& e) {
        message = e.what();
    }
    core::logging::error("Daemon request handler exception: {}", message);
    res.status = 500;
    res.set_content(
        std::format(
            R"({{"error":"Internal server error: {}"}})",
            core::utils::escape_json_string(message)),
        "application/json");
}

void handle_ping(const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok","message":"Filo daemon is running."})",
                    "application/json");
}

void handle_mcp_options(const std::string& host,
                        const httplib::Request& req,
                        httplib::Response& res) {
    if (!enforce_origin_policy(req, res, host)) return;
    set_cors_headers(req, res);
    res.status = 204;
}

void handle_mcp_get(const std::string& host,
                    const httplib::Request& req,
                    httplib::Response& res) {
    if (!enforce_origin_policy(req, res, host)) return;
    set_cors_headers(req, res);
    res.status = 405;
    res.set_header("Allow", "POST, GET, OPTIONS, DELETE");
    res.set_content(R"({"error":"Method Not Allowed — use POST /mcp"})",
                    "application/json");
}

void handle_mcp_delete(const std::string& host,
                       const httplib::Request& req,
                       httplib::Response& res) {
    if (!enforce_origin_policy(req, res, host)) return;
    set_cors_headers(req, res);

    if (!req.has_header("MCP-Session-Id")) {
        res.status = 400;
        res.set_content(
            R"({"error":"Missing MCP-Session-Id header"})",
            "application/json");
        return;
    }

    const std::string session_id = req.get_header_value("MCP-Session-Id");
    if (session_id.empty()) {
        res.status = 400;
        res.set_content(
            R"({"error":"Empty MCP-Session-Id header"})",
            "application/json");
        return;
    }

    if (!erase_http_session(session_id)) {
        res.status = 404;
        res.set_content(
            R"({"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null})",
            "application/json");
        return;
    }

    replay_cache().clear_session(session_id);
    core::tools::ShellTool::clear_mcp_session(session_id);
    res.status = 204;
}

void handle_mcp_post(const std::string& host,
                     const httplib::Request& req,
                     httplib::Response& res) {
    if (!enforce_origin_policy(req, res, host)) return;
    set_cors_headers(req, res);

    if (req.body.empty()) {
        res.status = 400;
        res.set_content(R"({"error":"Empty request body"})", "application/json");
        return;
    }

    if (!has_required_accept_header(req)) {
        res.status = 406;
        res.set_content(
            R"({"error":"Invalid Accept header. MCP clients must advertise both application/json and text/event-stream."})",
            "application/json");
        return;
    }

    if (!has_supported_protocol_header(req)) {
        res.status = 400;
        res.set_content(
            R"({"error":"Unsupported MCP-Protocol-Version header. Supported: 2025-11-25, 2024-11-05."})",
            "application/json");
        return;
    }

    const auto parsed = parse_jsonrpc_message(req.body);
    std::shared_ptr<HttpSessionState> session;
    std::string session_id;
    std::optional<core::workspace::WorkspaceSnapshot> workspace_override;
    bool should_refresh_roots = false;

    if (parsed.has_value()) {
        const bool has_session_header = req.has_header("MCP-Session-Id");
        if (!has_session_header) {
            if (parsed->kind != ParsedJsonRpcMessage::Kind::request
                || parsed->method != "initialize") {
                res.status = 400;
                res.set_content(
                    make_jsonrpc_error_body(parsed->id, -32002,
                                            "Missing MCP-Session-Id header. Send initialize first."),
                    "application/json");
                return;
            }

            auto init_result = dispatch_mcp_payload(req.body, std::nullopt, {});
            if (init_result.status == 200 && !init_result.body.empty()) {
                if (auto version = extract_initialize_protocol_version(init_result.body)) {
                    auto [new_session_id, new_session] = create_http_session(std::move(*version));
                    const auto [supports_roots, supports_roots_list_changed] =
                        extract_initialize_roots_capability(req.body);
                    {
                        std::lock_guard<std::mutex> lock(new_session->mutex);
                        new_session->client_supports_roots = supports_roots;
                        new_session->client_supports_roots_list_changed =
                            supports_roots_list_changed;
                        new_session->roots_dirty = supports_roots;
                    }
                    res.set_header("MCP-Session-Id", new_session_id);
                }
            }
            write_mcp_result(res, init_result);
            return;
        }

        session_id = req.get_header_value("MCP-Session-Id");
        if (session_id.empty()) {
            res.status = 400;
            res.set_content(
                make_jsonrpc_error_body(parsed->id, -32002, "Empty MCP-Session-Id header"),
                "application/json");
            return;
        }

        session = find_http_session(session_id);
        if (!session) {
            res.status = 404;
            res.set_content(
                make_jsonrpc_error_body(parsed->id, -32001, "Session not found"),
                "application/json");
            return;
        }

        if (parsed->kind == ParsedJsonRpcMessage::Kind::request
            && parsed->method == "initialize") {
            res.status = 400;
            res.set_content(
                make_jsonrpc_error_body(parsed->id, -32600,
                                        "Initialize requests must not include MCP-Session-Id"),
                "application/json");
            return;
        }

        if (parsed->kind == ParsedJsonRpcMessage::Kind::response) {
            std::shared_ptr<PendingServerResponse> pending;
            {
                std::lock_guard<std::mutex> lock(session->mutex);
                pending = take_pending_server_response(*session, response_id_key(parsed->id));
            }

            if (pending) {
                try {
                    pending->promise.set_value(req.body);
                } catch (...) {
                }
            }
            res.status = 202;
            return;
        }

        {
            std::unique_lock<std::mutex> lock(session->mutex);
            session->last_used = std::chrono::steady_clock::now();

            if (req.has_header("MCP-Protocol-Version")) {
                const std::string version = req.get_header_value("MCP-Protocol-Version");
                if (version != session->protocol_version) {
                    res.status = 400;
                    res.set_content(
                        make_jsonrpc_error_body(
                            parsed->id,
                            -32002,
                            std::format("MCP-Protocol-Version mismatch. Expected {}.",
                                        session->protocol_version)),
                        "application/json");
                    return;
                }
            }

            if (parsed->kind == ParsedJsonRpcMessage::Kind::notification
                && parsed->method == "notifications/roots/list_changed") {
                mark_session_workspace_dirty(*session);
                lock.unlock();
                core::tools::ShellTool::clear_mcp_session(session_id);
                lock.lock();
            }

            if (!session->client_ready) {
                if (parsed->kind == ParsedJsonRpcMessage::Kind::notification
                    && parsed->method == "notifications/initialized") {
                    session->client_ready = true;
                } else if (!(parsed->kind == ParsedJsonRpcMessage::Kind::request
                             && parsed->method == "ping")) {
                    res.status = 400;
                    res.set_content(
                        make_jsonrpc_error_body(
                            parsed->id,
                            -32002,
                            "Session not ready. Send notifications/initialized first."),
                        "application/json");
                    return;
                }
            }

            if (parsed->kind == ParsedJsonRpcMessage::Kind::request
                && parsed->method == "tools/call"
                && !consume_tool_call_budget(*session)) {
                res.status = 429;
                res.set_content(
                    make_jsonrpc_error_body(parsed->id, -32002,
                                            "Tool invocation rate limit exceeded"),
                    "application/json");
                return;
            }

            if (request_needs_authoritative_workspace(*parsed) && session->client_supports_roots) {
                while (session->roots_refresh_in_progress) {
                    session->roots_cv.wait(lock, [&]() {
                        return !session->roots_refresh_in_progress;
                    });
                }

                if (session->cached_workspace.has_value() && !session->roots_dirty) {
                    workspace_override = *session->cached_workspace;
                } else {
                    session->roots_refresh_in_progress = true;
                    should_refresh_roots = true;
                }
            } else if (request_needs_authoritative_workspace(*parsed)
                       && session->cached_workspace.has_value()) {
                workspace_override = *session->cached_workspace;
            }
        }
    }

    const auto replay_key = build_replay_key(req);
    if (!replay_key.has_value()) {
        detail::McpDispatchResult execution_result;
        try {
            execution_result = dispatch_mcp_payload(req.body, workspace_override, session_id);
        } catch (const std::exception& e) {
            execution_result.status = 500;
            execution_result.body = std::format(
                R"({{"error":"Internal server error: {}"}})",
                core::utils::escape_json_string(e.what()));
        } catch (...) {
            execution_result.status = 500;
            execution_result.body = R"({"error":"Internal server error"})";
        }
        write_mcp_result(res, execution_result);
        return;
    }

    auto decision = replay_cache().begin(*replay_key);
    if (decision.kind == detail::RequestReplayCache::DecisionKind::replay_completed) {
        write_mcp_result(res, decision.replay_result);
        return;
    }
    if (decision.kind == detail::RequestReplayCache::DecisionKind::wait_inflight) {
        write_mcp_result(res, replay_cache().wait_for(decision.inflight));
        return;
    }

    if (should_refresh_roots && session) {
        std::string roots_request_id;
        std::shared_ptr<PendingServerResponse> pending_roots_response;
        std::uint64_t next_workspace_version = 0;
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            roots_request_id = std::format("roots-{}", session->next_server_request_id++);
            pending_roots_response = register_pending_server_response(*session, roots_request_id);
            next_workspace_version = session->workspace_version + 1;
        }

        const ResponseId request_id = parsed->id;
        const std::string roots_request_body = build_roots_list_request_body(roots_request_id);
        const std::string original_request_body = req.body;
        const std::string replay_key_value = *replay_key;
        auto inflight = decision.inflight;
        auto session_ptr = session;

        res.status = 200;
        res.set_chunked_content_provider(
            "text/event-stream",
            [session_ptr,
             session_id,
             request_id,
             roots_request_id,
             roots_request_body,
             original_request_body,
             pending_roots_response,
             next_workspace_version,
             replay_key_value,
             inflight,
             sent = false](size_t, httplib::DataSink& sink) mutable -> bool {
                if (sent) {
                    return true;
                }
                sent = true;

                auto finish_and_stream =
                    [&](const detail::McpDispatchResult& result, bool cache_result) -> bool {
                        const std::string payload = build_sse_message(result.body);
                        const bool wrote = sink.write(payload.data(), payload.size());
                        sink.done();
                        replay_cache().finish(replay_key_value, inflight, result, cache_result);
                        return wrote;
                    };

                const std::string roots_request_event = build_sse_message(roots_request_body);
                if (!sink.write(roots_request_event.data(), roots_request_event.size())) {
                    {
                        std::lock_guard<std::mutex> lock(session_ptr->mutex);
                        session_ptr->pending_server_responses.erase(
                            string_response_id_key(roots_request_id));
                        fail_session_workspace_refresh(*session_ptr);
                    }
                    detail::McpDispatchResult error_result{
                        .status = 200,
                        .body = make_jsonrpc_error_body(
                            request_id,
                            -32603,
                            "Failed to stream roots/list request to client"),
                    };
                    replay_cache().finish(replay_key_value, inflight, error_result, false);
                    return false;
                }

                try {
                    if (pending_roots_response->future.wait_for(std::chrono::seconds{15})
                        != std::future_status::ready) {
                        {
                            std::lock_guard<std::mutex> lock(session_ptr->mutex);
                            session_ptr->pending_server_responses.erase(
                                string_response_id_key(roots_request_id));
                            fail_session_workspace_refresh(*session_ptr);
                        }
                        core::tools::ShellTool::clear_mcp_session(session_id);
                        detail::McpDispatchResult timeout_result{
                            .status = 200,
                            .body = make_jsonrpc_error_body(
                                request_id,
                                -32002,
                                "Timed out waiting for roots/list response"),
                        };
                        return finish_and_stream(timeout_result, false);
                    }

                    std::string roots_response_body = pending_roots_response->future.get();
                    std::string roots_error;
                    auto snapshot = parse_roots_list_response(
                        roots_response_body,
                        roots_request_id,
                        next_workspace_version,
                        roots_error);
                    if (!snapshot.has_value()) {
                        {
                            std::lock_guard<std::mutex> lock(session_ptr->mutex);
                            fail_session_workspace_refresh(*session_ptr);
                        }
                        core::tools::ShellTool::clear_mcp_session(session_id);
                        detail::McpDispatchResult invalid_roots_result{
                            .status = 200,
                            .body = make_jsonrpc_error_body(
                                request_id,
                                -32002,
                                roots_error),
                        };
                        return finish_and_stream(invalid_roots_result, false);
                    }

                    bool clear_shell = false;
                    std::optional<core::workspace::WorkspaceSnapshot> resolved_workspace;
                    {
                        std::lock_guard<std::mutex> lock(session_ptr->mutex);
                        clear_shell = !session_ptr->cached_workspace.has_value()
                            || session_ptr->cached_workspace->primary != snapshot->primary
                            || session_ptr->cached_workspace->additional != snapshot->additional
                            || session_ptr->cached_workspace->enforce != snapshot->enforce;
                        apply_session_workspace(*session_ptr, *snapshot);
                        resolved_workspace = session_ptr->cached_workspace;
                    }

                    if (clear_shell) {
                        core::tools::ShellTool::clear_mcp_session(session_id);
                    }

                    const auto final_result = dispatch_mcp_payload(
                        original_request_body,
                        resolved_workspace,
                        session_id);
                    return finish_and_stream(
                        final_result,
                        final_result.status == 200 || final_result.status == 202);
                } catch (const std::exception& e) {
                    {
                        std::lock_guard<std::mutex> lock(session_ptr->mutex);
                        fail_session_workspace_refresh(*session_ptr);
                    }
                    core::tools::ShellTool::clear_mcp_session(session_id);
                    detail::McpDispatchResult error_result{
                        .status = 200,
                        .body = make_jsonrpc_error_body(
                            request_id,
                            -32603,
                            std::format("Failed to refresh workspace roots: {}",
                                        e.what())),
                    };
                    return finish_and_stream(error_result, false);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(session_ptr->mutex);
                        fail_session_workspace_refresh(*session_ptr);
                    }
                    core::tools::ShellTool::clear_mcp_session(session_id);
                    detail::McpDispatchResult error_result{
                        .status = 200,
                        .body = make_jsonrpc_error_body(
                            request_id,
                            -32603,
                            "Failed to refresh workspace roots"),
                    };
                    return finish_and_stream(error_result, false);
                }
            });
        return;
    }

    detail::McpDispatchResult execution_result;
    try {
        execution_result = dispatch_mcp_payload(req.body, workspace_override, session_id);
    } catch (const std::exception& e) {
        execution_result.status = 500;
        execution_result.body = std::format(
            R"({{"error":"Internal server error: {}"}})",
            core::utils::escape_json_string(e.what()));
    } catch (...) {
        execution_result.status = 500;
        execution_result.body = R"({"error":"Internal server error"})";
    }

    const bool cache_result = execution_result.status == 200 || execution_result.status == 202;
    replay_cache().finish(*replay_key, decision.inflight, execution_result, cache_result);
    write_mcp_result(res, execution_result);
}

void handle_api_chat(const httplib::Request& req, httplib::Response& res) {
    simdjson::ondemand::parser parser;
    simdjson::padded_string padded_body(req.body);
    simdjson::ondemand::document doc;

    if (parser.iterate(padded_body).get(doc) != simdjson::SUCCESS) {
        res.status = 400;
        res.set_content(R"({"error":"Invalid JSON"})", "application/json");
        return;
    }

    std::string_view prompt;
    if (doc["prompt"].get_string().get(prompt) != simdjson::SUCCESS) {
        res.status = 400;
        res.set_content(R"({"error":"Missing 'prompt' field"})", "application/json");
        return;
    }

    auto& config_manager = core::config::ConfigManager::get_instance();
    const auto& config = config_manager.get_config();
    auto& provider_manager = core::llm::ProviderManager::get_instance();

    std::shared_ptr<core::llm::LLMProvider> llm_provider;
    try {
        llm_provider = provider_manager.get_provider(config.default_provider);
    } catch (const std::exception& e) {
        res.status = 500;
        std::string body = std::format(
            R"({{"error":"Failed to initialize default provider: {}"}})",
            core::utils::escape_json_string(e.what()));
        res.set_content(body, "application/json");
        return;
    }

    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        llm_provider,
        tool_manager,
        core::context::make_session_context(
            core::workspace::Workspace::get_instance().snapshot(),
            core::context::SessionTransport::unspecified));

    std::string final_response;
    agent->send_message(
        std::string(prompt),
        [&final_response](const std::string& text) { final_response += text; },
        [](const std::string& /*name*/, const std::string& /*args*/) {},
        []() {});

    res.set_content(
        std::format(R"({{"response":"{}"}})", core::utils::escape_json_string(final_response)),
        "application/json");
}

} // namespace

void run_server(int port, const std::string& host) {
    auto svr = std::make_shared<httplib::Server>();
    {
        std::lock_guard<std::mutex> lock(g_server_mutex);
        g_svr = svr;
    }

    init_providers();
    clear_http_sessions();

    // Pre-construct the McpDispatcher singleton so tools are registered once,
    // at startup, before any request arrives.  Concurrent HTTP handler threads
    // only ever READ the ToolManager map — they never write it.
    auto& dispatcher = core::mcp::McpDispatcher::get_instance();
    (void)dispatcher;   // suppress unused-variable warning

    // Local command proxy is expected to be stable under transient disconnects.
    svr->set_tcp_nodelay(true);
    svr->set_read_timeout(std::chrono::seconds{30});
    svr->set_write_timeout(std::chrono::seconds{120});
    svr->set_keep_alive_timeout(20);
    svr->set_keep_alive_max_count(200);
    svr->set_payload_max_length(4 * 1024 * 1024);

    svr->set_exception_handler(handle_daemon_exception);

    // -------------------------------------------------------------------------
    // Health check
    // -------------------------------------------------------------------------
    svr->Get("/ping", handle_ping);

    // -------------------------------------------------------------------------
    // MCP Streamable-HTTP transport
    //
    // Spec §Transport: servers MUST return HTTP 405 for wrong method on /mcp.
    // -------------------------------------------------------------------------

    const auto host_copy = std::string(host);
    svr->Options("/mcp", std::bind_front(handle_mcp_options, host_copy));
    svr->Get("/mcp", std::bind_front(handle_mcp_get, host_copy));
    svr->Post("/mcp", std::bind_front(handle_mcp_post, host_copy));
    svr->Delete("/mcp", std::bind_front(handle_mcp_delete, host_copy));
    svr->Post("/api/chat", handle_api_chat);

    if (!svr->bind_to_port(host, port)) {
        core::logging::error("Filo daemon failed to bind to {}:{}.", host, port);
        std::lock_guard<std::mutex> lock(g_server_mutex);
        if (g_svr == svr) g_svr.reset();
        return;
    }

    core::logging::info("Filo daemon listening on {}:{}.", host, port);
    core::logging::info("MCP endpoint : http://{}:{}/mcp", host, port);
    core::logging::info("Health check : http://{}:{}/ping", host, port);

    const bool listen_ok = svr->listen_after_bind();
    if (!listen_ok) {
        core::logging::error("Filo daemon stopped with a listen error on {}:{}.", host, port);
    } else {
        core::logging::info("Filo daemon stopped.");
    }

    {
        std::lock_guard<std::mutex> lock(g_server_mutex);
        if (g_svr == svr) g_svr.reset();
    }
}

} // namespace exec::daemon
