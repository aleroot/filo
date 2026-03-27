#include "Daemon.hpp"
#include <httplib.h>
#include <format>
#include "../core/config/ConfigManager.hpp"
#include "../core/agent/Agent.hpp"
#include "../core/llm/ProviderManager.hpp"
#include "../core/llm/ProviderFactory.hpp"
#include "../core/mcp/McpDispatcher.hpp"
#include "../core/tools/ToolManager.hpp"
#include "../core/utils/JsonUtils.hpp"
#include <simdjson.h>
#include <cctype>
#include <optional>

namespace exec::daemon {

static httplib::Server* g_svr = nullptr;

void stop_server() {
    if (g_svr) g_svr->stop();
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

[[nodiscard]] bool has_required_accept_header(const httplib::Request& req) {
    if (!req.has_header("Accept")) return false;

    const std::string accept = to_lower_ascii(req.get_header_value("Accept"));
    if (accept.find("*/*") != std::string::npos) return true;

    return accept.find("application/json") != std::string::npos
        && accept.find("text/event-stream") != std::string::npos;
}

[[nodiscard]] bool has_supported_protocol_header(const httplib::Request& req) {
    if (!req.has_header("MCP-Protocol-Version")) return true;
    const std::string version = req.get_header_value("MCP-Protocol-Version");
    return version == "2024-11-05" || version == "2025-11-25";
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

} // namespace

void run_server(int port, const std::string& host) {
    httplib::Server svr;
    g_svr = &svr;

    init_providers();

    // Pre-construct the McpDispatcher singleton so tools are registered once,
    // at startup, before any request arrives.  Concurrent HTTP handler threads
    // only ever READ the ToolManager map — they never write it.
    auto& dispatcher = core::mcp::McpDispatcher::get_instance();
    (void)dispatcher;   // suppress unused-variable warning

    // -------------------------------------------------------------------------
    // Health check
    // -------------------------------------------------------------------------
    svr.Get("/ping", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok","message":"Filo daemon is running."})",
                        "application/json");
    });

    // -------------------------------------------------------------------------
    // MCP Streamable-HTTP transport
    //
    // Spec §Transport: servers MUST return HTTP 405 for wrong method on /mcp.
    // -------------------------------------------------------------------------

    // OPTIONS /mcp — CORS preflight
    svr.Options("/mcp", [host](const httplib::Request& req, httplib::Response& res) {
        if (!enforce_origin_policy(req, res, host)) return;
        set_cors_headers(req, res);
        res.status = 204;
    });

    // GET /mcp — spec requires 405 (not 404)
    svr.Get("/mcp", [host](const httplib::Request& req, httplib::Response& res) {
        if (!enforce_origin_policy(req, res, host)) return;
        set_cors_headers(req, res);
        res.status = 405;
        res.set_header("Allow", "POST, GET, OPTIONS");
        res.set_content(R"({"error":"Method Not Allowed — use POST /mcp"})",
                        "application/json");
    });

    // POST /mcp — main MCP endpoint
    //
    //   200 application/json  – for requests (messages that carry an "id")
    //   202 Accepted          – for notifications / responses (no reply body)
    //   400 application/json  – body is empty or not valid JSON
    svr.Post("/mcp", [host](const httplib::Request& req, httplib::Response& res) {
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

        std::string response = core::mcp::McpDispatcher::get_instance().dispatch(req.body);

        if (response.empty()) {
            // JSON-RPC notification / response payload.
            res.status = 202;
        } else {
            if (response.starts_with(kJsonRpcParseErrorPrefix)
                || response.starts_with(kJsonRpcInvalidRequestPrefix)) {
                res.status = 400;
            }
            res.set_content(response, "application/json");
        }
    });

    // -------------------------------------------------------------------------
    // REST chat endpoint (To be implemented)
    // -------------------------------------------------------------------------
    svr.Post("/api/chat", [](const httplib::Request& req, httplib::Response& res) {
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
                R"({{"error":"Failed to initialize default provider: {}"}}))",
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
    });

    svr.listen(host, port);
    g_svr = nullptr;
}

} // namespace exec::daemon
