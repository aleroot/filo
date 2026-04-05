#include "Server.hpp"
#include "../core/mcp/McpDispatcher.hpp"
#include "../core/utils/JsonWriter.hpp"
#include <simdjson.h>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <atomic>

namespace exec {
namespace mcp {

namespace {

using core::utils::JsonWriter;

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

enum class SessionPhase {
    awaiting_initialize,
    awaiting_initialized,
    running,
};

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

void emit_response(std::string_view response) {
    if (response.empty()) return;
    std::println(stdout, "{}", response);
    std::fflush(stdout);
}

} // namespace

static std::atomic<bool> is_running{true};

void stop_server() {
    is_running = false;
}

void run_server() {
    is_running = true;
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Trigger skill registration on the dispatcher singleton before the loop so
    // the first tools/list response is not delayed by one-time initialisation.
    auto& dispatcher = core::mcp::McpDispatcher::get_instance();
    SessionPhase phase = SessionPhase::awaiting_initialize;

    std::string line;
    while (is_running && std::getline(std::cin, line)) {
        if (line.empty()) continue;

        const auto parsed = parse_jsonrpc_message(line);
        if (parsed.has_value()) {
            if (parsed->kind == ParsedJsonRpcMessage::Kind::response) {
                continue;
            }

            if (phase == SessionPhase::awaiting_initialize) {
                if (!(parsed->kind == ParsedJsonRpcMessage::Kind::request
                      && parsed->method == "initialize")) {
                    if (parsed->kind == ParsedJsonRpcMessage::Kind::request) {
                        emit_response(make_jsonrpc_error_body(
                            parsed->id,
                            -32002,
                            "Initialize must be the first request on a stdio MCP session."));
                    }
                    continue;
                }

                std::string response = dispatcher.dispatch(line);
                if (!response.empty()) {
                    phase = SessionPhase::awaiting_initialized;
                    emit_response(response);
                }
                continue;
            }

            if (phase == SessionPhase::awaiting_initialized) {
                if (parsed->kind == ParsedJsonRpcMessage::Kind::notification
                    && parsed->method == "notifications/initialized") {
                    dispatcher.dispatch(line);
                    phase = SessionPhase::running;
                    continue;
                }

                if (parsed->kind == ParsedJsonRpcMessage::Kind::request) {
                    const int error_code = parsed->method == "initialize" ? -32600 : -32002;
                    const std::string_view message = parsed->method == "initialize"
                        ? "Initialize has already completed for this stdio MCP session."
                        : "Session not ready. Send notifications/initialized first.";
                    if (parsed->method == "ping") {
                        emit_response(dispatcher.dispatch(line));
                    } else {
                        emit_response(make_jsonrpc_error_body(parsed->id, error_code, message));
                    }
                }
                continue;
            }

            if (parsed->kind == ParsedJsonRpcMessage::Kind::request
                && parsed->method == "initialize") {
                emit_response(make_jsonrpc_error_body(
                    parsed->id,
                    -32600,
                    "Initialize has already completed for this stdio MCP session."));
                continue;
            }
        }

        emit_response(dispatcher.dispatch(line));
    }
}

} // namespace mcp
} // namespace exec
