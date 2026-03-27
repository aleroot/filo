#include "McpClientSession.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <cpr/cpr.h>

// POSIX
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <format>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>

extern char** environ;  // POSIX: the current environment

namespace core::mcp {

// ===========================================================================
// JSON-RPC helpers (used by both transports)
// ===========================================================================

namespace {

// Build a JSON-RPC 2.0 request string (with id).
[[nodiscard]] std::string make_jsonrpc_request(int id,
                                                std::string_view method,
                                                std::string_view params_json) {
    std::string s;
    s.reserve(256);
    s += R"({"jsonrpc":"2.0","id":)";
    s += std::to_string(id);
    s += R"(,"method":")";
    s += method;
    s += '"';
    if (!params_json.empty() && params_json != "{}") {
        s += R"(,"params":)";
        s += params_json;
    }
    s += "}\n";
    return s;
}

// Build a JSON-RPC 2.0 notification string (no id — no response expected).
[[nodiscard]] std::string make_jsonrpc_notification(std::string_view method,
                                                     std::string_view params_json) {
    std::string s;
    s += R"({"jsonrpc":"2.0","method":")";
    s += method;
    s += '"';
    if (!params_json.empty() && params_json != "{}") {
        s += R"(,"params":)";
        s += params_json;
    }
    s += "}\n";
    return s;
}

// Extract the "result" field from a JSON-RPC response, or throw on error.
[[nodiscard]] std::string extract_result(std::string_view json) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error("MCP: invalid JSON-RPC response");
    }

    // Check for error
    simdjson::dom::object err_obj;
    if (doc["error"].get(err_obj) == simdjson::SUCCESS) {
        std::string_view msg;
        [[maybe_unused]] const auto err = err_obj["message"].get(msg);
        throw std::runtime_error(std::string("MCP error: ") + std::string(msg));
    }

    // Return the raw "result" value as a string for the caller to parse
    simdjson::dom::element result;
    if (doc["result"].get(result) != simdjson::SUCCESS) {
        throw std::runtime_error("MCP: missing 'result' in response");
    }
    return std::string(simdjson::to_string(result));
}

// Build the tools/call params JSON.
[[nodiscard]] std::string make_tools_call_params(const std::string& tool_name,
                                                  const std::string& arguments_json) {
    std::string s;
    s += R"({"name":")";
    s += core::utils::escape_json_string(tool_name);
    s += R"(","arguments":)";
    // arguments_json is already a valid JSON object string
    s += arguments_json.empty() ? "{}" : arguments_json;
    s += '}';
    return s;
}

// Build the initialize params JSON.
[[nodiscard]] std::string make_initialize_params() {
    return R"({"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"filo","version":"0.1.0"}})";
}

} // namespace

// ===========================================================================
// parse_tools_list — converts tools/list JSON result to McpToolDef vector.
// ===========================================================================

std::vector<McpToolDef> parse_tools_list(std::string_view json_result) {
    std::vector<McpToolDef> tools;

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return tools;

    simdjson::dom::array tools_arr;
    if (doc["tools"].get(tools_arr) != simdjson::SUCCESS) return tools;

    for (simdjson::dom::element t : tools_arr) {
        McpToolDef def;

        std::string_view name_v;
        if (t["name"].get(name_v) != simdjson::SUCCESS) continue;
        def.name = std::string(name_v);

        std::string_view desc_v;
        if (t["description"].get(desc_v) == simdjson::SUCCESS) {
            def.description = std::string(desc_v);
        }

        // Parse inputSchema (JSON Schema format)
        simdjson::dom::object schema;
        if (t["inputSchema"].get(schema) == simdjson::SUCCESS) {
            simdjson::dom::object props;
            if (schema["properties"].get(props) == simdjson::SUCCESS) {
                // Collect required field names
                std::vector<std::string> required_names;
                simdjson::dom::array req_arr;
                if (schema["required"].get(req_arr) == simdjson::SUCCESS) {
                    for (simdjson::dom::element r : req_arr) {
                        std::string_view rv;
                        if (r.get(rv) == simdjson::SUCCESS) {
                            required_names.emplace_back(rv);
                        }
                    }
                }

                for (auto [key, val] : props) {
                    McpToolParameter param;
                    param.name = std::string(key);

                    std::string_view type_v;
                    if (val["type"].get(type_v) == simdjson::SUCCESS)
                        param.type = std::string(type_v);
                    else
                        param.type = "string";

                    std::string_view pdesc_v;
                    if (val["description"].get(pdesc_v) == simdjson::SUCCESS)
                        param.description = std::string(pdesc_v);

                    param.required = std::ranges::find(required_names, param.name)
                                     != required_names.end();
                    def.parameters.push_back(std::move(param));
                }
            }
        }

        tools.push_back(std::move(def));
    }
    return tools;
}

// ===========================================================================
// StdioMcpSession
// ===========================================================================

StdioMcpSession::StdioMcpSession(const core::config::McpServerConfig& config) {
    if (config.command.empty()) {
        throw std::runtime_error("MCP stdio session: 'command' is empty");
    }

    // Create bidirectional pipes
    //   pipe_to_child : parent writes → child reads (child stdin)
    //   pipe_from_child: child writes → parent reads (child stdout)
    int pipe_to_child[2]   = {-1, -1};
    int pipe_from_child[2] = {-1, -1};

    if (pipe(pipe_to_child) != 0 || pipe(pipe_from_child) != 0) {
        throw std::runtime_error(std::string("MCP: pipe() failed: ") + strerror(errno));
    }

    // Build argv for posix_spawn
    std::vector<std::string> argv_strings;
    argv_strings.push_back(config.command);
    for (const auto& a : config.args) argv_strings.push_back(a);

    std::vector<char*> argv_ptrs;
    for (auto& s : argv_strings) argv_ptrs.push_back(s.data());
    argv_ptrs.push_back(nullptr);

    // Build envp — start with the parent environment, add extras
    std::vector<std::string> env_strings;
    for (char** e = environ; *e; ++e) env_strings.emplace_back(*e);
    for (const auto& kv : config.env) env_strings.push_back(kv);
    std::vector<char*> env_ptrs;
    for (auto& s : env_strings) env_ptrs.push_back(s.data());
    env_ptrs.push_back(nullptr);

    // Set up file actions for posix_spawn
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    // Redirect child stdin ← pipe_to_child[0]
    posix_spawn_file_actions_adddup2(&actions, pipe_to_child[0],   STDIN_FILENO);
    // Redirect child stdout → pipe_from_child[1]
    posix_spawn_file_actions_adddup2(&actions, pipe_from_child[1], STDOUT_FILENO);
    // Close the original pipe ends in the child (after dup2)
    posix_spawn_file_actions_addclose(&actions, pipe_to_child[0]);
    posix_spawn_file_actions_addclose(&actions, pipe_from_child[1]);
    // Close the parent ends in the child
    posix_spawn_file_actions_addclose(&actions, pipe_to_child[1]);
    posix_spawn_file_actions_addclose(&actions, pipe_from_child[0]);

    pid_t pid = -1;
    int rc = posix_spawnp(&pid, config.command.c_str(), &actions, nullptr,
                          argv_ptrs.data(), env_ptrs.data());
    posix_spawn_file_actions_destroy(&actions);

    // Close child-side ends in the parent
    close(pipe_to_child[0]);
    close(pipe_from_child[1]);

    if (rc != 0) {
        close(pipe_to_child[1]);
        close(pipe_from_child[0]);
        throw std::runtime_error(std::string("MCP: posix_spawnp('") + config.command + "') failed: " + strerror(rc));
    }

    child_pid_ = pid;
    write_fd_  = pipe_to_child[1];
    read_fd_   = pipe_from_child[0];
    running_.store(true, std::memory_order_release);

    // Start the reader thread
    reader_thread_ = std::thread(&StdioMcpSession::reader_loop, this);
}

StdioMcpSession::~StdioMcpSession() {
    shutdown();
}

void StdioMcpSession::shutdown() noexcept {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;

    // Close write end — signals EOF to the child
    if (write_fd_ != -1) { close(write_fd_); write_fd_ = -1; }

    // Reject all pending promises
    {
        std::lock_guard lock(pending_mutex_);
        for (auto& [id, prom] : pending_) {
            try { prom.set_exception(
                    std::make_exception_ptr(
                        std::runtime_error("MCP session shut down"))); }
            catch (...) {}
        }
        pending_.clear();
    }

    const pid_t pid = child_pid_;
    if (pid != -1) {
        // Ask the child to terminate so the reader thread observes EOF promptly.
        kill(pid, SIGTERM);
    }

    // Close read end — causes reader_loop to exit
    if (read_fd_ != -1) { close(read_fd_); read_fd_ = -1; }

    // The reader thread may still touch object state while unwinding; always join
    // to avoid detached access after this session is destroyed.
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    // Reap the child process. If TERM did not stop it yet, force kill.
    if (pid != -1) {
        int status = 0;
        pid_t waited = -1;
        do {
            waited = waitpid(pid, &status, WNOHANG);
        } while (waited == -1 && errno == EINTR);

        if (waited == 0) {
            kill(pid, SIGKILL);
            do {
                waited = waitpid(pid, &status, 0);
            } while (waited == -1 && errno == EINTR);
        }
    }
    child_pid_ = -1;
}

// Reader loop — runs on its own thread.
// Reads newline-delimited JSON-RPC responses and resolves pending promises.
void StdioMcpSession::reader_loop() {
    std::string line_buf;
    char ch;

    while (running_.load(std::memory_order_acquire)) {
        ssize_t n = read(read_fd_, &ch, 1);
        if (n <= 0) break;  // EOF or error → session is ending

        if (ch == '\n') {
            if (line_buf.empty()) continue;

            // Parse the JSON-RPC id and dispatch to the waiting promise
            {
                thread_local simdjson::dom::parser parser;
                simdjson::padded_string ps(line_buf);
                simdjson::dom::element doc;
                if (parser.parse(ps).get(doc) == simdjson::SUCCESS) {
                    int64_t id_v = -1;
                    if (doc["id"].get(id_v) == simdjson::SUCCESS) {
                        std::lock_guard lock(pending_mutex_);
                        auto it = pending_.find(static_cast<int>(id_v));
                        if (it != pending_.end()) {
                            it->second.set_value(line_buf);
                            pending_.erase(it);
                        }
                    }
                    // Notifications (no id) are silently ignored.
                }
            }

            line_buf.clear();
        } else {
            line_buf += ch;
        }
    }

    // On exit, reject any remaining promises
    std::lock_guard lock(pending_mutex_);
    for (auto& [id, prom] : pending_) {
        try { prom.set_exception(
                std::make_exception_ptr(
                    std::runtime_error("MCP: read_fd closed unexpectedly"))); }
        catch (...) {}
    }
    pending_.clear();
}

std::string StdioMcpSession::send_request(std::string_view method,
                                           std::string_view params_json) {
    const int id = next_id_.fetch_add(1, std::memory_order_relaxed);
    std::string msg = make_jsonrpc_request(id, method, params_json);

    std::promise<std::string> prom;
    auto fut = prom.get_future();

    {
        std::lock_guard lock(pending_mutex_);
        pending_.emplace(id, std::move(prom));
    }

    {
        std::lock_guard wlock(write_mutex_);
        const char* p = msg.data();
        size_t remaining = msg.size();
        while (remaining > 0) {
            ssize_t written = write(write_fd_, p, remaining);
            if (written <= 0) {
                // Remove the pending promise to avoid leaking
                std::lock_guard lock(pending_mutex_);
                pending_.erase(id);
                throw std::runtime_error("MCP: write() to child stdin failed");
            }
            p         += written;
            remaining -= static_cast<size_t>(written);
        }
    }

    // Block until reader_loop resolves the promise
    std::string raw_response = fut.get();
    return extract_result(raw_response);
}

void StdioMcpSession::send_notification(std::string_view method,
                                         std::string_view params_json) {
    std::string msg = make_jsonrpc_notification(method, params_json);
    std::lock_guard wlock(write_mutex_);
    const char* p = msg.data();
    size_t remaining = msg.size();
    while (remaining > 0) {
        ssize_t written = write(write_fd_, p, remaining);
        if (written <= 0) break;
        p         += written;
        remaining -= static_cast<size_t>(written);
    }
}

std::vector<McpToolDef> StdioMcpSession::initialize() {
    // 1. initialize handshake
    [[maybe_unused]] const auto init_result =
        send_request("initialize", make_initialize_params());
    // 2. notifications/initialized (no response expected)
    send_notification("notifications/initialized");
    // 3. tools/list
    std::string tools_result = send_request("tools/list", "");
    return parse_tools_list(tools_result);
}

std::string StdioMcpSession::call_tool(const std::string& tool_name,
                                        const std::string& arguments_json) {
    std::string params = make_tools_call_params(tool_name, arguments_json);
    std::string result = send_request("tools/call", params);

    // MCP result format: {"content":[{"type":"text","text":"..."}],"isError":false}
    // Flatten it into a simple JSON string for the skill result.
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return result;

    bool is_error = false;
    [[maybe_unused]] const auto err = doc["isError"].get(is_error);

    std::string text_out;
    simdjson::dom::array content;
    if (doc["content"].get(content) == simdjson::SUCCESS) {
        for (simdjson::dom::element part : content) {
            std::string_view ptype;
            if (part["type"].get(ptype) != simdjson::SUCCESS) continue;
            if (ptype == "text") {
                std::string_view t;
                if (part["text"].get(t) == simdjson::SUCCESS) {
                    if (!text_out.empty()) text_out += '\n';
                    text_out += t;
                }
            }
        }
    }

    if (is_error) {
        return std::format(R"({{"error":"{}"}})",
                           core::utils::escape_json_string(text_out));
    }
    return std::format(R"({{"result":"{}"}})",
                       core::utils::escape_json_string(text_out));
}

// ===========================================================================
// HttpMcpSession
// ===========================================================================

HttpMcpSession::HttpMcpSession(const core::config::McpServerConfig& config)
    : url_(config.url) {
    if (url_.empty()) {
        throw std::runtime_error("MCP HTTP session: 'url' is empty");
    }
}

std::string HttpMcpSession::post_json(const std::string& body,
                                       bool include_protocol_header) {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "application/json, text/event-stream"}
    };
    if (include_protocol_header && !protocol_version_.empty()) {
        headers["MCP-Protocol-Version"] = protocol_version_;
    }

    cpr::Response r = cpr::Post(
        cpr::Url{url_},
        headers,
        cpr::Body{body}
    );
    if (r.error.code != cpr::ErrorCode::OK) {
        throw std::runtime_error("MCP HTTP: " + r.error.message);
    }
    if (r.status_code < 200 || r.status_code >= 300) {
        throw std::runtime_error(
            std::format("MCP HTTP {}: {}", r.status_code, r.text));
    }
    return r.text;
}

void HttpMcpSession::update_negotiated_protocol_version(std::string_view initialize_result) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(initialize_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return;

    std::string_view version;
    if (doc["protocolVersion"].get(version) != simdjson::SUCCESS) return;
    if (!version.empty()) {
        protocol_version_ = std::string(version);
    }
}

std::string HttpMcpSession::send_request(std::string_view method,
                                          std::string_view params_json) {
    const int id = next_id_.fetch_add(1, std::memory_order_relaxed);
    std::string body = make_jsonrpc_request(id, method, params_json);
    // Strip the trailing newline for HTTP bodies
    if (!body.empty() && body.back() == '\n') body.pop_back();

    std::string raw;
    {
        std::lock_guard lock(request_mutex_);
        raw = post_json(body);
    }
    return extract_result(raw);
}

std::vector<McpToolDef> HttpMcpSession::initialize() {
    const auto init_result = send_request("initialize", make_initialize_params());
    update_negotiated_protocol_version(init_result);

    // HTTP transport: send initialized notification as a fire-and-forget POST
    std::string notif = make_jsonrpc_notification("notifications/initialized", "");
    if (!notif.empty() && notif.back() == '\n') notif.pop_back();
    try {
        std::lock_guard lock(request_mutex_);
        [[maybe_unused]] const auto notif_result = post_json(notif);
    } catch (...) {}  // notifications may not get a response — ignore errors

    std::string tools_result = send_request("tools/list", "");
    return parse_tools_list(tools_result);
}

std::string HttpMcpSession::call_tool(const std::string& tool_name,
                                       const std::string& arguments_json) {
    std::string params = make_tools_call_params(tool_name, arguments_json);
    std::string result = send_request("tools/call", params);

    // Same content-block flattening as the stdio session
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return result;

    bool is_error = false;
    [[maybe_unused]] const auto err = doc["isError"].get(is_error);

    std::string text_out;
    simdjson::dom::array content;
    if (doc["content"].get(content) == simdjson::SUCCESS) {
        for (simdjson::dom::element part : content) {
            std::string_view ptype;
            if (part["type"].get(ptype) != simdjson::SUCCESS) continue;
            if (ptype == "text") {
                std::string_view t;
                if (part["text"].get(t) == simdjson::SUCCESS) {
                    if (!text_out.empty()) text_out += '\n';
                    text_out += t;
                }
            }
        }
    }

    if (is_error) {
        return std::format(R"({{"error":"{}"}})",
                           core::utils::escape_json_string(text_out));
    }
    return std::format(R"({{"result":"{}"}})",
                       core::utils::escape_json_string(text_out));
}

// ===========================================================================
// Factory
// ===========================================================================

std::unique_ptr<IMcpClientSession>
make_mcp_session(const core::config::McpServerConfig& config) {
    if (config.transport == "http") {
        return std::make_unique<HttpMcpSession>(config);
    }
    return std::make_unique<StdioMcpSession>(config);
}

} // namespace core::mcp
