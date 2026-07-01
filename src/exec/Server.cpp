#include "Server.hpp"
#include "../core/context/SessionContext.hpp"
#include "../core/mcp/McpDispatcher.hpp"
#include "../core/mcp/McpRoots.hpp"
#include "../core/tools/ShellTool.hpp"
#include "../core/utils/JsonWriter.hpp"
#include "../core/workspace/Workspace.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <atomic>
#include <format>
#include <vector>

#if !defined(_WIN32)
#include <poll.h>
#include <unistd.h>
#endif

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

struct QueuedStdioRequest {
    std::string body;
    ResponseId id;
};

struct PendingStdioRootsRefresh {
    std::string request_id;
    std::uint64_t workspace_version = 0;
    std::chrono::steady_clock::time_point deadline;
    std::vector<QueuedStdioRequest> queued_requests;
};

struct StdioWorkspaceState {
    bool client_supports_roots = false;
    bool roots_dirty = false;
    std::uint64_t workspace_version = 0;
    std::uint64_t next_roots_request_id = 1;
    std::optional<core::workspace::WorkspaceSnapshot> cached_workspace;
    std::optional<PendingStdioRootsRefresh> pending_roots_refresh;
};

constexpr auto kStdioRootsListTimeout = std::chrono::seconds{15};

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

[[nodiscard]] bool response_id_equals(const ResponseId& id, std::string_view expected) {
    return id.kind == ResponseId::Kind::string && id.string_value == expected;
}

[[nodiscard]] bool request_needs_authoritative_workspace(const ParsedJsonRpcMessage& parsed) {
    return parsed.kind == ParsedJsonRpcMessage::Kind::request
        && core::mcp::is_workspace_sensitive_request(parsed.method);
}

void emit_response(std::string_view response) {
    if (response.empty()) return;
    std::println(stdout, "{}", response);
    std::fflush(stdout);
}

class StdioLineReader {
public:
    enum class Status {
        line,
        timeout,
        eof,
    };

    struct Result {
        Status status;
        std::string line{};
    };

    [[nodiscard]] Result read_until(
        const std::optional<std::chrono::steady_clock::time_point>& deadline)
    {
        if (auto line = take_buffered_line()) {
            return {.status = Status::line, .line = std::move(*line)};
        }

#if defined(_WIN32)
        (void)deadline;
        std::string line;
        if (!std::getline(std::cin, line)) {
            return {.status = Status::eof};
        }
        trim_cr(line);
        return {.status = Status::line, .line = std::move(line)};
#else
        while (true) {
            int timeout_ms = -1;
            if (deadline.has_value()) {
                const auto now = std::chrono::steady_clock::now();
                if (now >= *deadline) {
                    return {.status = Status::timeout};
                }
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(*deadline - now);
                timeout_ms = static_cast<int>(std::max<std::int64_t>(1, remaining.count()));
            }

            pollfd input{
                .fd = STDIN_FILENO,
                .events = POLLIN,
                .revents = 0,
            };
            const int ready = ::poll(&input, 1, timeout_ms);
            if (ready == 0) {
                return {.status = Status::timeout};
            }
            if (ready < 0) {
                if (errno == EINTR) continue;
                return {.status = Status::eof};
            }
            if ((input.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0
                && (input.revents & POLLIN) == 0) {
                return {.status = Status::eof};
            }

            char chunk[4096];
            const ssize_t bytes_read = ::read(STDIN_FILENO, chunk, sizeof(chunk));
            if (bytes_read == 0) {
                if (buffer_offset_ < buffer_.size()) {
                    std::string line = buffer_.substr(buffer_offset_);
                    buffer_.clear();
                    buffer_offset_ = 0;
                    trim_cr(line);
                    return {.status = Status::line, .line = std::move(line)};
                }
                return {.status = Status::eof};
            }
            if (bytes_read < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                return {.status = Status::eof};
            }

            buffer_.append(chunk, static_cast<std::size_t>(bytes_read));
            if (auto line = take_buffered_line()) {
                return {.status = Status::line, .line = std::move(*line)};
            }
        }
#endif
    }

private:
    [[nodiscard]] std::optional<std::string> take_buffered_line() {
        const auto newline = buffer_.find('\n', buffer_offset_);
        if (newline == std::string::npos) {
            return std::nullopt;
        }
        std::string line = buffer_.substr(buffer_offset_, newline - buffer_offset_);
        buffer_offset_ = newline + 1;
        compact_buffer_if_needed();
        trim_cr(line);
        return line;
    }

    static void trim_cr(std::string& line) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
    }

    std::string buffer_;
    std::size_t buffer_offset_ = 0;

    void compact_buffer_if_needed() {
        if (buffer_offset_ == 0) return;
        if (buffer_offset_ == buffer_.size()) {
            buffer_.clear();
            buffer_offset_ = 0;
            return;
        }
        if (buffer_offset_ < 4096 || buffer_offset_ * 2 < buffer_.size()) return;
        buffer_.erase(0, buffer_offset_);
        buffer_offset_ = 0;
    }
};

void apply_cached_workspace_context(StdioWorkspaceState& state,
                                    core::context::SessionContext& session_context) {
    if (!state.cached_workspace.has_value()) return;
    session_context = core::context::make_session_context(
        *state.cached_workspace,
        core::context::SessionTransport::mcp_stdio,
        "stdio");
}

void mark_stdio_roots_dirty(StdioWorkspaceState& state) {
    state.roots_dirty = true;
    state.cached_workspace.reset();
    core::tools::ShellTool::clear_mcp_session("stdio");
}

void queue_workspace_request_for_roots(StdioWorkspaceState& state,
                                       const ParsedJsonRpcMessage& parsed,
                                       std::string_view body) {
    if (!state.pending_roots_refresh.has_value()) {
        auto& pending = state.pending_roots_refresh.emplace();
        pending.request_id = std::format("roots-{}", state.next_roots_request_id++);
        pending.workspace_version = state.workspace_version + 1;
        pending.deadline = std::chrono::steady_clock::now() + kStdioRootsListTimeout;
        emit_response(core::mcp::build_roots_list_request_body(pending.request_id));
    }

    state.pending_roots_refresh->queued_requests.push_back(
        {.body = std::string(body), .id = parsed.id});
}

void fail_pending_roots_refresh(StdioWorkspaceState& state, std::string_view message) {
    if (!state.pending_roots_refresh.has_value()) return;

    auto pending = std::move(*state.pending_roots_refresh);
    state.pending_roots_refresh.reset();
    state.roots_dirty = true;
    state.cached_workspace.reset();
    core::tools::ShellTool::clear_mcp_session("stdio");

    for (const auto& request : pending.queued_requests) {
        emit_response(make_jsonrpc_error_body(request.id, -32002, message));
    }
}

void dispatch_pending_roots_requests(StdioWorkspaceState& state,
                                     core::context::SessionContext& session_context,
                                     core::mcp::McpDispatcher& dispatcher) {
    if (!state.pending_roots_refresh.has_value()) return;

    auto pending = std::move(*state.pending_roots_refresh);
    state.pending_roots_refresh.reset();

    for (const auto& request : pending.queued_requests) {
        emit_response(dispatcher.dispatch(request.body, session_context));
    }
}

void handle_roots_list_response(StdioWorkspaceState& state,
                                core::context::SessionContext& session_context,
                                core::mcp::McpDispatcher& dispatcher,
                                std::string_view body) {
    if (!state.pending_roots_refresh.has_value()) return;

    const auto workspace_version = state.pending_roots_refresh->workspace_version;
    std::string roots_error;
    auto snapshot = core::mcp::parse_roots_list_response(
        body,
        state.pending_roots_refresh->request_id,
        workspace_version,
        roots_error);
    if (!snapshot.has_value()) {
        fail_pending_roots_refresh(state, roots_error);
        return;
    }

    const bool workspace_changed =
        !state.cached_workspace.has_value()
        || state.cached_workspace->primary != snapshot->primary
        || state.cached_workspace->additional != snapshot->additional
        || state.cached_workspace->enforce != snapshot->enforce;

    state.workspace_version = workspace_version;
    state.cached_workspace = *snapshot;
    state.roots_dirty = false;
    apply_cached_workspace_context(state, session_context);

    if (workspace_changed) {
        core::tools::ShellTool::clear_mcp_session("stdio");
    }

    dispatch_pending_roots_requests(state, session_context, dispatcher);
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
    auto session_context = core::context::make_session_context(
        core::workspace::Workspace::get_instance().snapshot(),
        core::context::SessionTransport::mcp_stdio,
        "stdio");
    StdioWorkspaceState workspace_state;
    SessionPhase phase = SessionPhase::awaiting_initialize;
    StdioLineReader reader;

    while (is_running) {
        std::optional<std::chrono::steady_clock::time_point> read_deadline;
        if (workspace_state.pending_roots_refresh.has_value()) {
            read_deadline = workspace_state.pending_roots_refresh->deadline;
        }

        auto read = reader.read_until(read_deadline);
        if (read.status == StdioLineReader::Status::timeout) {
            fail_pending_roots_refresh(
                workspace_state,
                "Timed out waiting for roots/list response");
            continue;
        }
        if (read.status == StdioLineReader::Status::eof) {
            fail_pending_roots_refresh(
                workspace_state,
                "Client disconnected while waiting for roots/list response");
            break;
        }

        std::string line = std::move(read.line);
        if (line.empty()) continue;

        const auto parsed = parse_jsonrpc_message(line);
        if (parsed.has_value()) {
            if (parsed->kind == ParsedJsonRpcMessage::Kind::response) {
                if (workspace_state.pending_roots_refresh.has_value()
                    && response_id_equals(parsed->id,
                                          workspace_state.pending_roots_refresh->request_id)) {
                    handle_roots_list_response(
                        workspace_state,
                        session_context,
                        dispatcher,
                        line);
                }
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

                std::string response = dispatcher.dispatch(line, session_context);
                if (!response.empty()) {
                    workspace_state.client_supports_roots =
                        core::mcp::extract_roots_capability_from_initialize(line).supported;
                    workspace_state.roots_dirty = workspace_state.client_supports_roots;
                    phase = SessionPhase::awaiting_initialized;
                    emit_response(response);
                }
                continue;
            }

            if (phase == SessionPhase::awaiting_initialized) {
                if (parsed->kind == ParsedJsonRpcMessage::Kind::notification
                    && parsed->method == "notifications/initialized") {
                    dispatcher.dispatch(line, session_context);
                    phase = SessionPhase::running;
                    continue;
                }

                if (parsed->kind == ParsedJsonRpcMessage::Kind::request) {
                    const int error_code = parsed->method == "initialize" ? -32600 : -32002;
                    const std::string_view message = parsed->method == "initialize"
                        ? "Initialize has already completed for this stdio MCP session."
                        : "Session not ready. Send notifications/initialized first.";
                    if (parsed->method == "ping") {
                        emit_response(dispatcher.dispatch(line, session_context));
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

            if (parsed->kind == ParsedJsonRpcMessage::Kind::notification
                && parsed->method == "notifications/roots/list_changed") {
                mark_stdio_roots_dirty(workspace_state);
                continue;
            }

            if (request_needs_authoritative_workspace(*parsed)) {
                if (workspace_state.client_supports_roots
                    && (!workspace_state.cached_workspace.has_value()
                        || workspace_state.roots_dirty
                        || workspace_state.pending_roots_refresh.has_value())) {
                    queue_workspace_request_for_roots(workspace_state, *parsed, line);
                    continue;
                }

                if (workspace_state.client_supports_roots
                    && workspace_state.cached_workspace.has_value()) {
                    apply_cached_workspace_context(workspace_state, session_context);
                }
            }
        }

        emit_response(dispatcher.dispatch(line, session_context));
    }
}

} // namespace mcp
} // namespace exec
