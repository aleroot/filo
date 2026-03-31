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
#include "../core/utils/JsonUtils.hpp"
#include "../core/logging/Logger.hpp"
#include <simdjson.h>
#include <cctype>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

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

[[nodiscard]] std::string to_lower_ascii(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
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

[[nodiscard]] detail::McpDispatchResult dispatch_mcp_payload(const std::string& body) {
    detail::McpDispatchResult result;
    result.body = core::mcp::McpDispatcher::get_instance().dispatch(body);
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

    const auto replay_key = build_replay_key(req);
    if (!replay_key.has_value()) {
        write_mcp_result(res, dispatch_mcp_payload(req.body));
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

    detail::McpDispatchResult execution_result;
    try {
        std::optional<core::tools::ShellTool::ScopedMcpSessionContext> session_scope;
        if (req.has_header("MCP-Session-Id")) {
            std::string session_id = req.get_header_value("MCP-Session-Id");
            if (!session_id.empty()) {
                session_scope.emplace(core::tools::ShellTool::scoped_mcp_session(
                    std::move(session_id)));
            }
        }
        execution_result = dispatch_mcp_payload(req.body);
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
    auto agent = std::make_shared<core::agent::Agent>(llm_provider, tool_manager);

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
