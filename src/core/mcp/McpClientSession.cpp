#include "McpClientSession.hpp"
#include "McpOAuth.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "core/net/NetworkTraffic.hpp"
#include "core/version/Version.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/StringUtils.hpp"
#include <simdjson.h>
#include <cpr/cpr.h>

// POSIX
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>

#include <cerrno>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <format>
#include <array>
#include <iostream>
#include <sstream>
#include <chrono>
#include <thread>
#include <iterator>
#include <utility>
#include <ranges>
#include <cstdint>
#include <limits>

extern char** environ;  // POSIX: the current environment

namespace core::mcp {

// ===========================================================================
// JSON-RPC helpers (used by both transports)
// ===========================================================================

namespace {

// Close both ends of a pipe pair and mark them invalid. Safe to call twice.
void close_pipe_pair(int (&fds)[2]) noexcept {
    for (int& fd : fds) {
        if (fd != -1) {
            ::close(fd);
            fd = -1;
        }
    }
}

// Mark a descriptor close-on-exec so spawned children never inherit it.
void set_cloexec(int fd) noexcept {
    if (fd == -1) return;
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags != -1) {
        (void)::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }
}

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

[[nodiscard]] std::string make_cancelled_notification(int request_id,
                                                       std::string_view reason) {
    return make_jsonrpc_notification(
        "notifications/cancelled",
        std::format(
            R"({{"requestId":{},"reason":"{}"}})",
            request_id,
            core::utils::escape_json_string(reason)));
}

[[nodiscard]] std::string format_timeout(std::chrono::milliseconds timeout) {
    if (timeout.count() % 1000 == 0) {
        return std::format("{}s", timeout.count() / 1000);
    }
    return std::format("{}ms", timeout.count());
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

[[nodiscard]] bool looks_like_json_rpc_message(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return false;
    return text[first] == '{' || text[first] == '[';
}

[[nodiscard]] bool is_json_rpc_response_message(std::string_view json) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return false;

    // Responses carry an id plus result or error. Notifications/requests do not.
    simdjson::dom::element id_elem;
    if (doc["id"].get(id_elem) != simdjson::SUCCESS) return false;
    simdjson::dom::element result_or_error;
    if (doc["result"].get(result_or_error) == simdjson::SUCCESS) return true;
    if (doc["error"].get(result_or_error) == simdjson::SUCCESS) return true;
    return false;
}

[[nodiscard]] std::string trim_ascii_ws_copy(std::string_view text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(begin, end - begin + 1));
}

// Parse Streamable HTTP SSE payloads into the last JSON-RPC response message.
[[nodiscard]] std::string extract_json_rpc_from_sse(std::string_view sse_body) {
    std::string last_response;
    std::string current_data;
    bool have_data = false;

    auto flush_event = [&]() {
        if (!have_data) {
            current_data.clear();
            return;
        }
        const std::string payload = trim_ascii_ws_copy(current_data);
        current_data.clear();
        have_data = false;
        if (payload.empty() || payload == "[DONE]") return;
        if (!looks_like_json_rpc_message(payload)) return;
        if (is_json_rpc_response_message(payload)) {
            last_response = payload;
        }
    };

    std::size_t pos = 0;
    while (pos <= sse_body.size()) {
        const auto next = sse_body.find('\n', pos);
        const auto line_view = (next == std::string_view::npos)
            ? sse_body.substr(pos)
            : sse_body.substr(pos, next - pos);
        pos = (next == std::string_view::npos) ? sse_body.size() + 1 : next + 1;

        std::string_view line = line_view;
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        if (line.empty()) {
            flush_event();
            continue;
        }
        if (line.starts_with(':')) {
            // Comment / keep-alive
            continue;
        }
        if (line.starts_with("data:")) {
            std::string_view data = line.substr(5);
            if (!data.empty() && data.front() == ' ') data.remove_prefix(1);
            if (have_data) current_data.push_back('\n');
            current_data.append(data);
            have_data = true;
        }
        // Ignore event:/id:/retry: fields for response extraction.
    }
    flush_event();

    if (last_response.empty()) {
        throw std::runtime_error(
            "MCP HTTP: SSE response did not contain a JSON-RPC result/error");
    }
    return last_response;
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

// Build the resources/read params JSON.
[[nodiscard]] std::string make_resource_read_params(const std::string& uri) {
    return std::format(
        R"({{"uri":"{}"}})",
        core::utils::escape_json_string(uri));
}

// Build the prompts/get params JSON.
[[nodiscard]] std::string make_prompt_get_params(const std::string& prompt_name,
                                                 const std::string& arguments_json) {
    std::string s;
    s += R"({"name":")";
    s += core::utils::escape_json_string(prompt_name);
    s += '"';
    if (!arguments_json.empty() && arguments_json != "{}") {
        s += R"(,"arguments":)";
        s += arguments_json;
    }
    s += '}';
    return s;
}

[[nodiscard]] std::string make_list_params_with_cursor(
    const std::optional<std::string>& cursor)
{
    if (!cursor.has_value() || cursor->empty()) return {};
    return std::format(
        R"({{"cursor":"{}"}})",
        core::utils::escape_json_string(*cursor));
}

// Build the initialize params JSON.
[[nodiscard]] std::string make_initialize_params(bool enable_sampling) {
    if (enable_sampling) {
        return std::format(
            R"({{"protocolVersion":"2025-11-25","capabilities":{{"sampling":{{"tools":{{}}}}}},"clientInfo":{{"name":"filo","version":"{}"}}}})",
            core::version::value);
    }
    return std::format(
        R"({{"protocolVersion":"2025-11-25","capabilities":{{}},"clientInfo":{{"name":"filo","version":"{}"}}}})",
        core::version::value);
}

struct ParsedServerCapabilities {
    bool has_tools = true;
    bool tools_advertised = false;
    bool has_resources = false;
    bool has_prompts = false;
};

[[nodiscard]] ParsedServerCapabilities parse_server_capabilities(
    std::string_view initialize_result)
{
    ParsedServerCapabilities out;
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(initialize_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return out;

    simdjson::dom::object capabilities;
    if (doc["capabilities"].get(capabilities) != simdjson::SUCCESS) return out;

    simdjson::dom::element elem;
    out.tools_advertised = capabilities["tools"].get(elem) == simdjson::SUCCESS;
    out.has_resources = capabilities["resources"].get(elem) == simdjson::SUCCESS;
    out.has_prompts = capabilities["prompts"].get(elem) == simdjson::SUCCESS;
    if (out.tools_advertised) {
        out.has_tools = true;
    } else if (out.has_resources || out.has_prompts) {
        // If newer capability metadata is present and tools is absent, default
        // to "no tools" but allow a compatibility probe in initialize().
        out.has_tools = false;
    } else {
        // Backward compatibility: older servers may return an empty capabilities
        // object but still support tools/list.
        out.has_tools = true;
    }
    return out;
}

[[nodiscard]] std::optional<std::string> parse_next_cursor(std::string_view json_result) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return std::nullopt;

    std::string_view cursor;
    if (doc["nextCursor"].get(cursor) == simdjson::SUCCESS && !cursor.empty()) {
        return std::string(cursor);
    }
    return std::nullopt;
}

[[nodiscard]] std::string make_jsonrpc_success_response(
    std::string_view id_json,
    std::string_view result_json)
{
    std::string s;
    s.reserve(128 + result_json.size());
    s += R"({"jsonrpc":"2.0","id":)";
    s += id_json;
    s += R"(,"result":)";
    s += result_json;
    s += "}\n";
    return s;
}

[[nodiscard]] std::string make_jsonrpc_error_response(
    std::string_view id_json,
    int code,
    std::string_view message)
{
    std::string s;
    s.reserve(192 + message.size());
    s += R"({"jsonrpc":"2.0","id":)";
    s += id_json;
    s += R"(,"error":{"code":)";
    s += std::to_string(code);
    s += R"(,"message":")";
    s += core::utils::escape_json_string(message);
    s += R"("}})";
    s += '\n';
    return s;
}

[[nodiscard]] std::string flatten_tool_result(std::string_view result_json) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(result_json);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        return std::string(result_json);
    }

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

[[nodiscard]] bool write_all_fd(int fd, std::string_view data) {
    const char* ptr = data.data();
    std::size_t remaining = data.size();
    while (remaining > 0) {
        const ssize_t written = ::write(fd, ptr, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (written == 0) return false;
        ptr += written;
        remaining -= static_cast<std::size_t>(written);
    }
    return true;
}

[[nodiscard]] std::optional<std::string> find_header_case_insensitive(
    const cpr::Header& headers,
    std::string_view key)
{
    const std::string lowered_key = core::utils::str::to_lower_ascii_copy(key);
    for (const auto& [header_name, header_value] : headers) {
        if (core::utils::str::to_lower_ascii_copy(header_name) == lowered_key) {
            return header_value;
        }
    }
    return std::nullopt;
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

        std::string_view title_v;
        if (t["title"].get(title_v) == simdjson::SUCCESS) {
            def.title = std::string(title_v);
        }

        std::string_view desc_v;
        if (t["description"].get(desc_v) == simdjson::SUCCESS) {
            def.description = std::string(desc_v);
        }

        // Parse inputSchema (JSON Schema format)
        simdjson::dom::object schema;
        if (t["inputSchema"].get(schema) == simdjson::SUCCESS) {
            def.input_schema = simdjson::to_string(schema);

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
                    param.schema = simdjson::to_string(val);

                    std::string_view type_v;
                    if (val["type"].get(type_v) == simdjson::SUCCESS)
                        param.type = std::string(type_v);
                    else
                        param.type = "string";

                    std::string_view pdesc_v;
                    if (val["description"].get(pdesc_v) == simdjson::SUCCESS)
                        param.description = std::string(pdesc_v);

                    simdjson::dom::element items;
                    if (val["items"].get(items) == simdjson::SUCCESS) {
                        param.items_schema = simdjson::to_string(items);
                    }

                    param.required = std::ranges::find(required_names, param.name)
                                     != required_names.end();
                    def.parameters.push_back(std::move(param));
                }
            }
        }

        simdjson::dom::element output_schema;
        if (t["outputSchema"].get(output_schema) == simdjson::SUCCESS) {
            def.output_schema = simdjson::to_string(output_schema);
        }

        simdjson::dom::object annotations;
        if (t["annotations"].get(annotations) == simdjson::SUCCESS) {
            [[maybe_unused]] const auto read_only_result =
                annotations["readOnlyHint"].get(def.annotations.read_only_hint);
            [[maybe_unused]] const auto destructive_result =
                annotations["destructiveHint"].get(def.annotations.destructive_hint);
            [[maybe_unused]] const auto idempotent_result =
                annotations["idempotentHint"].get(def.annotations.idempotent_hint);
            [[maybe_unused]] const auto open_world_result =
                annotations["openWorldHint"].get(def.annotations.open_world_hint);
        }

        tools.push_back(std::move(def));
    }
    return tools;
}

std::vector<McpResourceDef> parse_resources_list(std::string_view json_result) {
    std::vector<McpResourceDef> resources;

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return resources;

    simdjson::dom::array resources_arr;
    if (doc["resources"].get(resources_arr) != simdjson::SUCCESS) return resources;

    for (simdjson::dom::element r : resources_arr) {
        McpResourceDef def;

        std::string_view uri_v;
        if (r["uri"].get(uri_v) != simdjson::SUCCESS) continue;
        def.uri = std::string(uri_v);

        std::string_view name_v;
        if (r["name"].get(name_v) == simdjson::SUCCESS) {
            def.name = std::string(name_v);
        }

        std::string_view title_v;
        if (r["title"].get(title_v) == simdjson::SUCCESS) {
            def.title = std::string(title_v);
        }

        std::string_view desc_v;
        if (r["description"].get(desc_v) == simdjson::SUCCESS) {
            def.description = std::string(desc_v);
        }

        std::string_view mime_v;
        if (r["mimeType"].get(mime_v) == simdjson::SUCCESS) {
            def.mime_type = std::string(mime_v);
        }

        resources.push_back(std::move(def));
    }

    return resources;
}

std::vector<McpResourceTemplateDef> parse_resource_templates_list(
    std::string_view json_result)
{
    std::vector<McpResourceTemplateDef> templates;

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return templates;

    simdjson::dom::array templates_arr;
    if (doc["resourceTemplates"].get(templates_arr) != simdjson::SUCCESS) return templates;

    for (simdjson::dom::element t : templates_arr) {
        McpResourceTemplateDef def;

        std::string_view uri_template_v;
        if (t["uriTemplate"].get(uri_template_v) != simdjson::SUCCESS) continue;
        def.uri_template = std::string(uri_template_v);

        std::string_view name_v;
        if (t["name"].get(name_v) == simdjson::SUCCESS) {
            def.name = std::string(name_v);
        }

        std::string_view title_v;
        if (t["title"].get(title_v) == simdjson::SUCCESS) {
            def.title = std::string(title_v);
        }

        std::string_view desc_v;
        if (t["description"].get(desc_v) == simdjson::SUCCESS) {
            def.description = std::string(desc_v);
        }

        std::string_view mime_v;
        if (t["mimeType"].get(mime_v) == simdjson::SUCCESS) {
            def.mime_type = std::string(mime_v);
        }

        templates.push_back(std::move(def));
    }

    return templates;
}

std::vector<McpPromptDef> parse_prompts_list(std::string_view json_result) {
    std::vector<McpPromptDef> prompts;

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_result);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return prompts;

    simdjson::dom::array prompts_arr;
    if (doc["prompts"].get(prompts_arr) != simdjson::SUCCESS) return prompts;

    for (simdjson::dom::element p : prompts_arr) {
        McpPromptDef def;

        std::string_view name_v;
        if (p["name"].get(name_v) != simdjson::SUCCESS) continue;
        def.name = std::string(name_v);

        std::string_view title_v;
        if (p["title"].get(title_v) == simdjson::SUCCESS) {
            def.title = std::string(title_v);
        }

        std::string_view desc_v;
        if (p["description"].get(desc_v) == simdjson::SUCCESS) {
            def.description = std::string(desc_v);
        }

        simdjson::dom::array arguments_arr;
        if (p["arguments"].get(arguments_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element arg : arguments_arr) {
                McpPromptArgumentDef arg_def;
                std::string_view arg_name_v;
                if (arg["name"].get(arg_name_v) != simdjson::SUCCESS) continue;
                arg_def.name = std::string(arg_name_v);

                std::string_view arg_title_v;
                if (arg["title"].get(arg_title_v) == simdjson::SUCCESS) {
                    arg_def.title = std::string(arg_title_v);
                }

                std::string_view arg_desc_v;
                if (arg["description"].get(arg_desc_v) == simdjson::SUCCESS) {
                    arg_def.description = std::string(arg_desc_v);
                }

                [[maybe_unused]] const auto required_result =
                    arg["required"].get(arg_def.required);
                def.arguments.push_back(std::move(arg_def));
            }
        }

        prompts.push_back(std::move(def));
    }

    return prompts;
}

// ===========================================================================
// StdioMcpSession
// ===========================================================================

StdioMcpSession::StdioMcpSession(const core::config::McpServerConfig& config,
                                 McpSamplingCallback sampling_callback,
                                 std::chrono::milliseconds request_timeout)
    : server_name_(config.name)
    , sampling_callback_(std::move(sampling_callback))
    , request_timeout_(request_timeout.count() > 0
                           ? request_timeout
                           : std::chrono::seconds(60)) {
    if (config.command.empty()) {
        throw std::runtime_error("MCP stdio session: 'command' is empty");
    }
    if (!core::landrun::LandrunSettings::instance().permits(
            core::landrun::LandrunCapability::unsandboxed_child_process)) {
        throw std::runtime_error(
            "landrun secure mode refuses unsandboxed stdio MCP servers; "
            "use a remote MCP transport or explicitly start Filo with --sandbox off");
    }
    client_sampling_enabled_.store(
        static_cast<bool>(sampling_callback_),
        std::memory_order_release);

    // Create bidirectional pipes
    //   pipe_to_child : parent writes → child reads (child stdin)
    //   pipe_from_child: child writes → parent reads (child stdout)
    int pipe_to_child[2]   = {-1, -1};
    int pipe_from_child[2] = {-1, -1};

    // Shutdown wake pipe. Created before the child pipes so a failure here can
    // never strand an already-spawned server process.
    if (pipe(wake_pipe_) != 0) {
        throw std::runtime_error(std::string("MCP: pipe() failed: ") + strerror(errno));
    }
    // Never leak the wake pipe into the spawned server.
    set_cloexec(wake_pipe_[0]);
    set_cloexec(wake_pipe_[1]);

    if (pipe(pipe_to_child) != 0 || pipe(pipe_from_child) != 0) {
        const int err = errno;
        // Close every descriptor opened so far: if the first pipe() succeeded
        // and the second failed, the first pair would otherwise leak.
        close_pipe_pair(pipe_to_child);
        close_pipe_pair(pipe_from_child);
        close_pipe_pair(wake_pipe_);
        throw std::runtime_error(std::string("MCP: pipe() failed: ") + strerror(err));
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
        close_pipe_pair(wake_pipe_);
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
    transport_closed_.store(true, std::memory_order_release);

    // Close write end — signals EOF to the child
    {
        std::lock_guard wlock(write_mutex_);
        close_write_fd_locked();
    }

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

    // Wake the reader out of poll(). The read fd deliberately stays open until
    // the thread has been joined: closing it here would both race with the
    // reader's own use of it and risk the descriptor number being recycled by
    // another thread's open()/socket() before the reader noticed, making it
    // read from an unrelated file.
    if (wake_pipe_[1] != -1) {
        const char token = 'x';
        ssize_t written = 0;
        do {
            written = ::write(wake_pipe_[1], &token, 1);
        } while (written < 0 && errno == EINTR);
    }

    // The reader thread may still touch object state while unwinding; always join
    // to avoid detached access after this session is destroyed.
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    // Safe now: the only thread that observes these descriptors has exited.
    if (read_fd_ != -1) { close(read_fd_); read_fd_ = -1; }
    close_pipe_pair(wake_pipe_);

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

void StdioMcpSession::close_write_fd_locked() noexcept {
    if (write_fd_ != -1) {
        close(write_fd_);
        write_fd_ = -1;
    }
}

bool StdioMcpSession::write_message(std::string_view message) {
    std::lock_guard wlock(write_mutex_);
    if (transport_closed_.load(std::memory_order_acquire) || write_fd_ == -1) {
        return false;
    }

    if (write_all_fd(write_fd_, message)) {
        return true;
    }

    transport_closed_.store(true, std::memory_order_release);
    close_write_fd_locked();
    return false;
}

// Reader loop — runs on its own thread.
// Reads newline-delimited JSON-RPC responses and resolves pending promises.
void StdioMcpSession::reader_loop() {
    auto write_response = [this](std::string&& payload) {
        [[maybe_unused]] const bool ok = write_message(payload);
    };

    auto dispatch_line = [this, &write_response](std::string_view line) {
        thread_local simdjson::dom::parser parser;
        simdjson::padded_string ps(line);
        simdjson::dom::element doc;
        if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return;

        std::string_view method;
        if (doc["method"].get(method) == simdjson::SUCCESS) {
            simdjson::dom::element id_elem;
            if (doc["id"].get(id_elem) != simdjson::SUCCESS) {
                // Notification from the server: no response expected.
                return;
            }

            const std::string id_json = simdjson::to_string(id_elem);
            simdjson::dom::element params_elem;
            std::string params_json{"{}"};
            if (doc["params"].get(params_elem) == simdjson::SUCCESS) {
                params_json = std::string(simdjson::to_string(params_elem));
            }

            if (method == "sampling/createMessage") {
                if (!sampling_callback_) {
                    write_response(make_jsonrpc_error_response(id_json, -32601, "Method not found"));
                    return;
                }

                try {
                    const std::string sampling_result =
                        sampling_callback_(server_name_, params_json);
                    write_response(make_jsonrpc_success_response(id_json, sampling_result));
                } catch (const std::exception& e) {
                    write_response(make_jsonrpc_error_response(
                        id_json,
                        -32603,
                        std::string("sampling/createMessage failed: ") + e.what()));
                } catch (...) {
                    write_response(make_jsonrpc_error_response(
                        id_json,
                        -32603,
                        "sampling/createMessage failed"));
                }
                return;
            }

            write_response(make_jsonrpc_error_response(id_json, -32601, "Method not found"));
            return;
        }

        int64_t id_v = -1;
        if (doc["id"].get(id_v) != simdjson::SUCCESS) return;

        std::lock_guard lock(pending_mutex_);
        auto it = pending_.find(static_cast<int>(id_v));
        if (it != pending_.end()) {
            it->second.set_value(std::string(line));
            pending_.erase(it);
        }
    };

    std::string buffered;
    buffered.reserve(8192);
    std::array<char, 4096> chunk{};

    while (running_.load(std::memory_order_acquire)) {
        // Wait for either payload data or a shutdown signal. poll() rather than
        // a bare blocking read() is what makes shutdown deterministic: POSIX
        // does not guarantee that closing a descriptor wakes another thread
        // already blocked on it.
        struct pollfd fds[2] = {
            {.fd = read_fd_,      .events = POLLIN, .revents = 0},
            {.fd = wake_pipe_[0], .events = POLLIN, .revents = 0},
        };
        if (::poll(fds, 2, -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (fds[1].revents != 0) break;  // shutdown requested
        if ((fds[0].revents & (POLLERR | POLLNVAL)) != 0) break;
        // POLLHUP still needs a read() to drain whatever the child left behind.
        if ((fds[0].revents & (POLLIN | POLLHUP)) == 0) continue;

        ssize_t n = ::read(read_fd_, chunk.data(), chunk.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;  // EOF → session is ending

        buffered.append(chunk.data(), static_cast<std::size_t>(n));
        std::size_t line_start = 0;
        while (true) {
            const std::size_t newline = buffered.find('\n', line_start);
            if (newline == std::string::npos) break;

            std::string_view line{buffered.data() + line_start, newline - line_start};
            if (!line.empty() && line.back() == '\r') {
                line.remove_suffix(1);
            }
            if (!line.empty()) dispatch_line(line);
            line_start = newline + 1;
        }

        if (line_start > 0) {
            buffered.erase(0, line_start);
        }
    }

    transport_closed_.store(true, std::memory_order_release);
    {
        std::lock_guard wlock(write_mutex_);
        close_write_fd_locked();
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
                                           std::string_view params_json,
                                           std::chrono::milliseconds timeout) {
    const int id = next_id_.fetch_add(1, std::memory_order_relaxed);
    std::string msg = make_jsonrpc_request(id, method, params_json);

    std::promise<std::string> prom;
    auto fut = prom.get_future();

    {
        std::lock_guard lock(pending_mutex_);
        pending_.emplace(id, std::move(prom));
    }

    {
        if (!write_message(msg)) {
            // Remove the pending promise to avoid leaking
            std::lock_guard lock(pending_mutex_);
            pending_.erase(id);
            throw std::runtime_error("MCP: write() to child stdin failed");
        }
    }

    if (fut.wait_for(timeout) != std::future_status::ready) {
        {
            std::lock_guard lock(pending_mutex_);
            pending_.erase(id);
        }

        const std::string reason = std::format(
            "Filo timed out waiting for MCP method '{}' after {}",
            method,
            format_timeout(timeout));
        const std::string cancellation = make_cancelled_notification(id, reason);
        {
            [[maybe_unused]] const bool ok = write_message(cancellation);
        }

        throw std::runtime_error(std::format(
            "MCP: request '{}' timed out after {}",
            method,
            format_timeout(timeout)));
    }

    std::string raw_response = fut.get();
    return extract_result(raw_response);
}

void StdioMcpSession::send_notification(std::string_view method,
                                         std::string_view params_json) {
    std::string msg = make_jsonrpc_notification(method, params_json);
    if (!write_message(msg)) {
        throw std::runtime_error("MCP: write() to child stdin failed");
    }
}

void StdioMcpSession::update_server_capabilities(std::string_view initialize_result) {
    const auto parsed = parse_server_capabilities(initialize_result);
    server_supports_tools_.store(parsed.has_tools, std::memory_order_release);
    server_tools_advertised_.store(parsed.tools_advertised, std::memory_order_release);
    server_supports_resources_.store(parsed.has_resources, std::memory_order_release);
    server_supports_prompts_.store(parsed.has_prompts, std::memory_order_release);
}

std::vector<McpToolDef> StdioMcpSession::initialize() {
    // 1. initialize handshake
    const auto init_result =
        send_request("initialize", make_initialize_params(supports_sampling()), request_timeout_);
    update_server_capabilities(init_result);

    // 2. notifications/initialized (no response expected)
    send_notification("notifications/initialized");
    if (!server_supports_tools_.load(std::memory_order_acquire)) {
        if (!server_tools_advertised_.load(std::memory_order_acquire)) {
            try {
                std::string tools_result = send_request("tools/list", "", request_timeout_);
                server_supports_tools_.store(true, std::memory_order_release);
                return parse_tools_list(tools_result);
            } catch (...) {
                // Compatibility path: server omitted tools capability and also
                // rejected tools/list. Keep tools disabled, continue with
                // resources/prompts if available.
            }
        }
        return {};
    }

    // 3. tools/list
    std::string tools_result = send_request("tools/list", "", request_timeout_);
    return parse_tools_list(tools_result);
}

std::string StdioMcpSession::call_tool(const std::string& tool_name,
                                        const std::string& arguments_json) {
    if (!server_supports_tools_.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            std::format("MCP server '{}' does not advertise tools capability", server_name_));
    }
    std::string params = make_tools_call_params(tool_name, arguments_json);
    std::string result = send_request("tools/call", params, request_timeout_);
    return flatten_tool_result(result);
}

std::vector<McpResourceDef> StdioMcpSession::list_resources() {
    if (!supports_resources()) return {};

    std::vector<McpResourceDef> resources;
    std::optional<std::string> cursor;
    do {
        const std::string result = send_request(
            "resources/list",
            make_list_params_with_cursor(cursor),
            request_timeout_);
        auto page = parse_resources_list(result);
        resources.insert(resources.end(),
                         std::make_move_iterator(page.begin()),
                         std::make_move_iterator(page.end()));
        cursor = parse_next_cursor(result);
    } while (cursor.has_value());

    return resources;
}

std::vector<McpResourceTemplateDef> StdioMcpSession::list_resource_templates() {
    if (!supports_resources()) return {};

    std::vector<McpResourceTemplateDef> templates;
    std::optional<std::string> cursor;
    do {
        const std::string result = send_request(
            "resources/templates/list",
            make_list_params_with_cursor(cursor),
            request_timeout_);
        auto page = parse_resource_templates_list(result);
        templates.insert(templates.end(),
                         std::make_move_iterator(page.begin()),
                         std::make_move_iterator(page.end()));
        cursor = parse_next_cursor(result);
    } while (cursor.has_value());

    return templates;
}

std::string StdioMcpSession::read_resource(const std::string& uri) {
    if (!supports_resources()) {
        throw std::runtime_error(
            std::format("MCP server '{}' does not advertise resources capability", server_name_));
    }
    return send_request("resources/read", make_resource_read_params(uri), request_timeout_);
}

std::vector<McpPromptDef> StdioMcpSession::list_prompts() {
    if (!supports_prompts()) return {};

    std::vector<McpPromptDef> prompts;
    std::optional<std::string> cursor;
    do {
        const std::string result = send_request(
            "prompts/list",
            make_list_params_with_cursor(cursor),
            request_timeout_);
        auto page = parse_prompts_list(result);
        prompts.insert(prompts.end(),
                       std::make_move_iterator(page.begin()),
                       std::make_move_iterator(page.end()));
        cursor = parse_next_cursor(result);
    } while (cursor.has_value());

    return prompts;
}

std::string StdioMcpSession::get_prompt(const std::string& prompt_name,
                                         const std::string& arguments_json) {
    if (!supports_prompts()) {
        throw std::runtime_error(
            std::format("MCP server '{}' does not advertise prompts capability", server_name_));
    }
    return send_request(
        "prompts/get",
        make_prompt_get_params(prompt_name, arguments_json),
        request_timeout_);
}

// ===========================================================================
// HttpMcpSession
// ===========================================================================

HttpMcpSession::HttpMcpSession(const core::config::McpServerConfig& config,
                               McpSamplingCallback sampling_callback,
                               std::chrono::milliseconds request_timeout,
                               std::shared_ptr<McpHttpAuth> auth)
    : url_(config.url)
    , server_name_(config.name)
    , auth_(std::move(auth))
    , request_timeout_(request_timeout.count() > 0
                           ? request_timeout
                           : mcp_request_timeout(config, kDefaultMcpHttpTimeout))
    , sampling_callback_(std::move(sampling_callback)) {
    if (url_.empty()) {
        throw std::runtime_error("MCP HTTP session: 'url' is empty");
    }

    // When auth_ owns Authorization, keep only non-Authorization extras here.
    // Without auth_, expand all configured headers (static API keys, etc.).
    extra_headers_.reserve(config.headers.size());
    for (const auto& [name, value] : config.headers) {
        if (name.empty()) continue;
        if (auth_) {
            std::string lower;
            lower.reserve(name.size());
            for (unsigned char c : name) {
                lower.push_back(static_cast<char>(std::tolower(c)));
            }
            if (lower == "authorization") continue;
        }
        extra_headers_.emplace_back(name, expand_mcp_env_placeholders(value));
    }
}

HttpMcpSession::~HttpMcpSession() {
    shutdown();
}

cpr::Header HttpMcpSession::build_request_headers(bool include_protocol_header) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "application/json, text/event-stream"}
    };
    if (include_protocol_header && !protocol_version_.empty()) {
        headers["MCP-Protocol-Version"] = protocol_version_;
    }
    if (!session_id_.empty()) {
        headers["MCP-Session-Id"] = session_id_;
    }
    for (const auto& [name, value] : extra_headers_) {
        headers[name] = value;
    }
    // Dynamic OAuth / static Authorization from the auth provider (may refresh).
    if (auth_) {
        for (const auto& [name, value] : auth_->request_headers()) {
            headers[name] = value;
        }
    }
    return headers;
}

HttpMcpSession::HttpJsonResponse HttpMcpSession::post_json_once(
    const std::string& body,
    bool include_protocol_header) {
    cpr::Header headers = build_request_headers(include_protocol_header);

    // Clamp to int32 for cpr; request_timeout_ is already non-negative.
    const auto timeout_ms = request_timeout_.count() > 0
        ? static_cast<int32_t>(std::min<std::int64_t>(
              request_timeout_.count(),
              static_cast<std::int64_t>(std::numeric_limits<int32_t>::max())))
        : static_cast<int32_t>(kDefaultMcpHttpTimeout.count());

    cpr::Response r = cpr::Post(
        cpr::Url{url_},
        headers,
        cpr::Body{body},
        cpr::Timeout{timeout_ms}
    );
    const bool request_observed =
        r.error.code == cpr::ErrorCode::OK || r.status_code != 0 || !r.text.empty();
    core::net::NetworkTrafficStats::get_instance().record({
        .bytes_sent = request_observed
            ? static_cast<uint64_t>(body.size()) + core::net::estimated_http_header_bytes(headers)
            : 0,
        .bytes_received = static_cast<uint64_t>(r.text.size())
            + core::net::estimated_http_header_bytes(r.header),
    });
    if (r.error.code != cpr::ErrorCode::OK) {
        throw std::runtime_error("MCP HTTP: " + r.error.message);
    }
    return HttpJsonResponse{
        .status_code = r.status_code,
        .body = std::move(r.text),
        .headers = std::move(r.header),
    };
}

HttpMcpSession::HttpJsonResponse HttpMcpSession::post_json(const std::string& body,
                                                           bool include_protocol_header) {
    auto response = post_json_once(body, include_protocol_header);

    // Mid-session OAuth recovery: one forced refresh + single retry on 401.
    if (response.status_code == 401 && auth_ && auth_->can_refresh()) {
        if (auth_->refresh_after_unauthorized()) {
            response = post_json_once(body, include_protocol_header);
        }
    }

    // 202 Accepted is valid for pure notifications under Streamable HTTP.
    if (response.status_code < 200 || response.status_code >= 300) {
        std::string detail = response.body;
        if (detail.size() > 512) {
            detail.resize(512);
            detail += "…";
        }
        if (response.status_code == 401 || response.status_code == 403) {
            throw std::runtime_error(std::format(
                "MCP HTTP {}: authentication failed for '{}'. "
                "Run `/mcp login {}` or set Authorization via headers "
                "(e.g. Bearer API key) and reconnect.",
                response.status_code,
                server_name_,
                server_name_));
        }
        throw std::runtime_error(
            std::format("MCP HTTP {}: {}", response.status_code, detail));
    }
    return response;
}

void HttpMcpSession::update_server_capabilities(std::string_view initialize_result) {
    const auto parsed = parse_server_capabilities(initialize_result);
    server_supports_tools_.store(parsed.has_tools, std::memory_order_release);
    server_tools_advertised_.store(parsed.tools_advertised, std::memory_order_release);
    server_supports_resources_.store(parsed.has_resources, std::memory_order_release);
    server_supports_prompts_.store(parsed.has_prompts, std::memory_order_release);
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
        raw = post_json(body).body;
    }
    return extract_result(normalize_streamable_http_response_body(raw));
}

std::vector<McpToolDef> HttpMcpSession::initialize() {
    std::string init_body = make_jsonrpc_request(
        next_id_.fetch_add(1, std::memory_order_relaxed),
        "initialize",
        make_initialize_params(false));
    if (!init_body.empty() && init_body.back() == '\n') init_body.pop_back();

    HttpJsonResponse init_response;
    {
        std::lock_guard lock(request_mutex_);
        session_id_.clear();
        init_response = post_json(init_body, /*include_protocol_header=*/false);
        if (const auto session_header =
                find_header_case_insensitive(init_response.headers, "MCP-Session-Id")) {
            session_id_ = *session_header;
        }
    }

    const auto init_result = extract_result(
        normalize_streamable_http_response_body(init_response.body));
    update_server_capabilities(init_result);
    update_negotiated_protocol_version(init_result);

    // HTTP transport: send initialized notification as a fire-and-forget POST
    std::string notif = make_jsonrpc_notification("notifications/initialized", "");
    if (!notif.empty() && notif.back() == '\n') notif.pop_back();
    try {
        std::lock_guard lock(request_mutex_);
        [[maybe_unused]] const auto notif_result = post_json(notif);
    } catch (...) {}  // notifications may not get a response — ignore errors

    if (!server_supports_tools_.load(std::memory_order_acquire)) {
        if (!server_tools_advertised_.load(std::memory_order_acquire)) {
            try {
                const std::string tools_result = send_request("tools/list", "");
                server_supports_tools_.store(true, std::memory_order_release);
                return parse_tools_list(tools_result);
            } catch (...) {
                // Compatibility path: server omitted tools capability and also
                // rejected tools/list. Keep tools disabled, continue with
                // resources/prompts if available.
            }
        }
        return {};
    }

    const std::string tools_result = send_request("tools/list", "");
    return parse_tools_list(tools_result);
}

void HttpMcpSession::shutdown() noexcept {
    std::lock_guard lock(request_mutex_);
    if (session_id_.empty()) return;

    try {
        cpr::Header headers = build_request_headers(/*include_protocol_header=*/true);
        [[maybe_unused]] const auto response = cpr::Delete(
            cpr::Url{url_},
            headers,
            cpr::Timeout{static_cast<int32_t>(request_timeout_.count())});
    } catch (...) {}

    session_id_.clear();
}

std::string HttpMcpSession::call_tool(const std::string& tool_name,
                                       const std::string& arguments_json) {
    if (!server_supports_tools_.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            std::format("MCP server '{}' does not advertise tools capability", server_name_));
    }

    std::string params = make_tools_call_params(tool_name, arguments_json);
    std::string result = send_request("tools/call", params);
    return flatten_tool_result(result);
}

std::vector<McpResourceDef> HttpMcpSession::list_resources() {
    if (!supports_resources()) return {};

    std::vector<McpResourceDef> resources;
    std::optional<std::string> cursor;
    do {
        const std::string result =
            send_request("resources/list", make_list_params_with_cursor(cursor));
        auto page = parse_resources_list(result);
        resources.insert(resources.end(),
                         std::make_move_iterator(page.begin()),
                         std::make_move_iterator(page.end()));
        cursor = parse_next_cursor(result);
    } while (cursor.has_value());

    return resources;
}

std::vector<McpResourceTemplateDef> HttpMcpSession::list_resource_templates() {
    if (!supports_resources()) return {};

    std::vector<McpResourceTemplateDef> templates;
    std::optional<std::string> cursor;
    do {
        const std::string result = send_request(
            "resources/templates/list",
            make_list_params_with_cursor(cursor));
        auto page = parse_resource_templates_list(result);
        templates.insert(templates.end(),
                         std::make_move_iterator(page.begin()),
                         std::make_move_iterator(page.end()));
        cursor = parse_next_cursor(result);
    } while (cursor.has_value());

    return templates;
}

std::string HttpMcpSession::read_resource(const std::string& uri) {
    if (!supports_resources()) {
        throw std::runtime_error(
            std::format("MCP server '{}' does not advertise resources capability", server_name_));
    }
    return send_request("resources/read", make_resource_read_params(uri));
}

std::vector<McpPromptDef> HttpMcpSession::list_prompts() {
    if (!supports_prompts()) return {};

    std::vector<McpPromptDef> prompts;
    std::optional<std::string> cursor;
    do {
        const std::string result =
            send_request("prompts/list", make_list_params_with_cursor(cursor));
        auto page = parse_prompts_list(result);
        prompts.insert(prompts.end(),
                       std::make_move_iterator(page.begin()),
                       std::make_move_iterator(page.end()));
        cursor = parse_next_cursor(result);
    } while (cursor.has_value());

    return prompts;
}

std::string HttpMcpSession::get_prompt(const std::string& prompt_name,
                                        const std::string& arguments_json) {
    if (!supports_prompts()) {
        throw std::runtime_error(
            std::format("MCP server '{}' does not advertise prompts capability", server_name_));
    }
    return send_request("prompts/get", make_prompt_get_params(prompt_name, arguments_json));
}

// ===========================================================================
// Streamable HTTP helpers (public)
// ===========================================================================

std::string expand_mcp_env_placeholders(std::string_view value) {
    std::string out;
    out.reserve(value.size());

    for (std::size_t i = 0; i < value.size();) {
        if (value[i] != '$') {
            out.push_back(value[i++]);
            continue;
        }

        // Escape: $$ -> $
        if (i + 1 < value.size() && value[i + 1] == '$') {
            out.push_back('$');
            i += 2;
            continue;
        }

        std::string_view name;
        std::size_t consume = 0;
        if (i + 1 < value.size() && value[i + 1] == '{') {
            const auto close = value.find('}', i + 2);
            if (close == std::string_view::npos) {
                out.push_back(value[i++]);
                continue;
            }
            name = value.substr(i + 2, close - (i + 2));
            consume = close - i + 1;
        } else {
            std::size_t j = i + 1;
            while (j < value.size()) {
                const char c = value[j];
                const bool ok =
                    (c >= 'A' && c <= 'Z') ||
                    (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') ||
                    c == '_';
                if (!ok) break;
                ++j;
            }
            if (j == i + 1) {
                out.push_back(value[i++]);
                continue;
            }
            name = value.substr(i + 1, j - (i + 1));
            consume = j - i;
        }

        if (name.empty()) {
            out.push_back(value[i++]);
            continue;
        }

        const std::string name_str(name);
        if (const char* env = std::getenv(name_str.c_str())) {
            out += env;
        }
        // Missing env vars expand to empty, matching common shell behavior for optional secrets.
        i += consume;
    }

    return out;
}

std::string normalize_streamable_http_response_body(std::string_view body) {
    const std::string trimmed = trim_ascii_ws_copy(body);
    if (trimmed.empty()) {
        throw std::runtime_error("MCP HTTP: empty response body");
    }

    if (looks_like_json_rpc_message(trimmed)) {
        return trimmed;
    }

    // Streamable HTTP allows text/event-stream responses for requests.
    // Heuristic: treat bodies with SSE data: lines as SSE even if Content-Type
    // was not preserved by the transport layer.
    if (trimmed.find("data:") != std::string::npos) {
        return extract_json_rpc_from_sse(trimmed);
    }

    throw std::runtime_error("MCP HTTP: unrecognized response body format");
}

// ===========================================================================
// Factory
// ===========================================================================

std::unique_ptr<IMcpClientSession>
make_mcp_session(const core::config::McpServerConfig& config,
                 McpSamplingCallback sampling_callback,
                 std::string_view config_dir) {
    if (config.transport == "http") {
        auto auth = make_mcp_http_auth(config, config_dir);
        const auto timeout = mcp_request_timeout(config, kDefaultMcpHttpTimeout);
        return std::make_unique<HttpMcpSession>(
            config,
            std::move(sampling_callback),
            timeout,
            std::move(auth));
    }
    const auto timeout = mcp_request_timeout(config, kDefaultMcpStdioTimeout);
    return std::make_unique<StdioMcpSession>(
        config,
        std::move(sampling_callback),
        timeout);
}

} // namespace core::mcp
