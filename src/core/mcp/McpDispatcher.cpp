#include "McpDispatcher.hpp"
#include "../tools/ToolManager.hpp"
#include "../tools/ShellTool.hpp"
#include "../tools/ApplyPatchTool.hpp"
#include "../tools/FileSearchTool.hpp"
#include "../tools/ReadFileTool.hpp"
#include "../tools/WriteFileTool.hpp"
#include "../tools/ListDirectoryTool.hpp"
#include "../tools/ReplaceTool.hpp"
#include "../tools/GrepSearchTool.hpp"
#include "../tools/SearchReplaceTool.hpp"
#include "../tools/DeleteFileTool.hpp"
#include "../tools/MoveFileTool.hpp"
#include "../tools/CreateDirectoryTool.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>
#include <string_view>
#include <array>
#include <format>
#include <optional>
#include <ranges>

namespace core::mcp {

using core::utils::JsonWriter;

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// MCP protocol versions this server understands, in preference order.
/// The first entry is returned when the client requests an unknown version.
constexpr std::array<std::string_view, 2> kSupportedProtocolVersions{
    "2025-11-25",  ///< Current version — includes structuredContent in tools/call
    "2024-11-05",  ///< Legacy version
};

/// Default version announced and used when no matching version is negotiated.
constexpr std::string_view kDefaultProtocolVersion = kSupportedProtocolVersions.front();

/// Human-readable instructions embedded in the initialize response.
/// Lampo's local model reads these to understand the server's capabilities and
/// preferred usage patterns without trial-and-error.
constexpr std::string_view kServerInstructions =
    "filo-mcp exposes filesystem and shell tools running on the user's local machine "
    "outside Lampo's sandbox. "
    "Always use absolute paths for unambiguous file access. "
    "Prefer search_replace or apply_patch for surgical code edits over write_file, "
    "which overwrites the entire file. "
    "Check exit_code in run_terminal_command results: 0 = success, non-zero = failure. "
    "The 'previous_content' field in write_file responses enables diff display without "
    "an extra read_file round-trip.";

namespace {

// ---------------------------------------------------------------------------
// Request-ID helpers
// ---------------------------------------------------------------------------

/**
 * @brief Typed container for a JSON-RPC 2.0 request @c id.
 *
 * MCP 2025-11-25 §3.1 restricts request IDs to integers and strings.
 * Null, boolean, float, object, and array IDs are invalid.
 *
 * @c Kind::none means no @c id field was present — the message is a
 * @em notification and must not receive a response.
 */
struct RequestId {
    enum class Kind {
        none,    ///< No id field — message is a notification
        integer, ///< id is an integer (int64)
        string,  ///< id is a string
        invalid, ///< id has an illegal type — triggers -32600 Invalid Request
    };

    Kind    kind          = Kind::none;
    int64_t integer_value = 0;
    std::string string_value;
};

/// @returns @c true if @p version is in the server's supported list.
[[nodiscard]] bool is_supported_protocol_version(std::string_view version) {
    return std::ranges::find(kSupportedProtocolVersions, version) != kSupportedProtocolVersions.end();
}

/**
 * @brief Parses the @c id field from an already-opened JSON-RPC object.
 *
 * Uses simdjson ondemand — the caller must keep @p root alive for the
 * duration of this function.
 *
 * @param root  The top-level JSON object of the incoming request.
 * @return A @c RequestId describing the id's kind and value.
 */
[[nodiscard]] RequestId parse_request_id(simdjson::ondemand::object& root) {
    RequestId id;

    simdjson::ondemand::value id_val;
    if (root["id"].get(id_val) != simdjson::SUCCESS) {
        return id;  // Kind::none — notification
    }

    simdjson::ondemand::json_type type;
    if (id_val.type().get(type) != simdjson::SUCCESS) {
        id.kind = RequestId::Kind::invalid;
        return id;
    }

    if (type == simdjson::ondemand::json_type::number) {
        int64_t num = 0;
        if (id_val.get_int64().get(num) == simdjson::SUCCESS) {
            id.kind          = RequestId::Kind::integer;
            id.integer_value = num;
        } else {
            id.kind = RequestId::Kind::invalid;  // float — not allowed
        }
        return id;
    }

    if (type == simdjson::ondemand::json_type::string) {
        std::string_view sv;
        if (id_val.get_string().get(sv) == simdjson::SUCCESS) {
            id.kind = RequestId::Kind::string;
            id.string_value.assign(sv);
        } else {
            id.kind = RequestId::Kind::invalid;
        }
        return id;
    }

    // null, bool, object, array — all invalid per JSON-RPC 2.0 §4
    id.kind = RequestId::Kind::invalid;
    return id;
}

/**
 * @brief Writes the @c "id" key-value pair into @p w using the correct JSON type.
 *
 * @c Kind::none and @c Kind::invalid both produce @c null so the spec's
 * requirement of including the id even in error responses is satisfied.
 *
 * @param w   JsonWriter that already has an object scope open.
 * @param id  The parsed request id.
 */
void write_response_id(JsonWriter& w, const RequestId& id) {
    w.key("id");
    switch (id.kind) {
        case RequestId::Kind::integer:
            w.number(id.integer_value);
            break;
        case RequestId::Kind::string:
            w.str(id.string_value);
            break;
        case RequestId::Kind::none:
        case RequestId::Kind::invalid:
            w.null_val();
            break;
    }
}

/**
 * @brief Selects the best protocol version to use for this session.
 *
 * If the client's requested version is in @c kSupportedProtocolVersions
 * it is echoed back (exact match wins).  Otherwise the server defaults to
 * @c kDefaultProtocolVersion (the newest supported version) per the MCP
 * 2025-11-25 negotiation rules.
 *
 * @param root  Top-level JSON object of the @c initialize request.
 * @return The negotiated protocol version string.
 */
[[nodiscard]] std::string negotiate_protocol_version(simdjson::ondemand::object& root) {
    simdjson::ondemand::object params;
    if (root["params"].get_object().get(params) != simdjson::SUCCESS) {
        return std::string(kDefaultProtocolVersion);
    }

    std::string_view requested;
    if (params["protocolVersion"].get_string().get(requested) != simdjson::SUCCESS) {
        return std::string(kDefaultProtocolVersion);
    }

    if (is_supported_protocol_version(requested)) {
        return std::string(requested);
    }
    return std::string(kDefaultProtocolVersion);
}

/**
 * @brief Checks whether a simdjson DOM value matches a JSON Schema type token.
 *
 * Used during argument validation in @c tools/call.  Unknown type tokens
 * (e.g. vendor extensions) are accepted to avoid false rejections.
 *
 * @param value          The parsed argument value.
 * @param expected_type  A JSON Schema primitive type string.
 * @return @c true if the value's type matches @p expected_type.
 */
[[nodiscard]] bool matches_declared_type(const simdjson::dom::element& value,
                                         std::string_view expected_type) {
    const auto type = value.type();
    if (expected_type == "string")  return type == simdjson::dom::element_type::STRING;
    if (expected_type == "boolean") return type == simdjson::dom::element_type::BOOL;
    if (expected_type == "object")  return type == simdjson::dom::element_type::OBJECT;
    if (expected_type == "array")   return type == simdjson::dom::element_type::ARRAY;
    if (expected_type == "integer") {
        return type == simdjson::dom::element_type::INT64
            || type == simdjson::dom::element_type::UINT64;
    }
    if (expected_type == "number") {
        return type == simdjson::dom::element_type::INT64
            || type == simdjson::dom::element_type::UINT64
            || type == simdjson::dom::element_type::DOUBLE;
    }
    return true;  // unknown type token — don't reject
}

/**
 * @brief Validates the @c arguments object against a tool's declared parameter schema.
 *
 * Checks that:
 * - @p args_json is valid JSON and parses to an object.
 * - All required parameters are present.
 * - Present parameters have the declared type.
 *
 * @param def       The tool definition providing the parameter schema.
 * @param args_json Raw JSON string of the @c arguments field (may be @c "{}").
 * @return @c std::nullopt on success, or an error message string on failure.
 */
[[nodiscard]] std::optional<std::string> validate_tool_arguments(
    const core::tools::ToolDefinition& def,
    std::string_view args_json)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(args_json).get(doc) != simdjson::SUCCESS) {
        return "Invalid params: 'arguments' must be valid JSON";
    }

    simdjson::dom::object args;
    if (doc.get(args) != simdjson::SUCCESS) {
        return "Invalid params: 'arguments' must be a JSON object";
    }

    for (const auto& param : def.parameters) {
        simdjson::dom::element value;
        const bool has_value = args[param.name].get(value) == simdjson::SUCCESS;
        if (!has_value) {
            if (param.required) {
                return std::format("Invalid params: missing required argument '{}'", param.name);
            }
            continue;
        }
        if (!matches_declared_type(value, param.type)) {
            return std::format("Invalid params: argument '{}' must be of type '{}'",
                               param.name, param.type);
        }
    }

    return std::nullopt;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// JSON-RPC envelope builders
// ---------------------------------------------------------------------------

/**
 * @brief Wraps @p result in a JSON-RPC 2.0 success response.
 *
 * @param id      The echoed request id.
 * @param result  A pre-serialised JSON value string for the @c result field.
 * @return The complete JSON-RPC response string.
 */
static std::string make_response(const RequestId& id, std::string_view result) {
    JsonWriter w(64 + result.size());
    {
        auto _obj = w.object();
        w.kv_str("jsonrpc", "2.0").comma();
        write_response_id(w, id);
        w.comma().kv_raw("result", result);
    }
    return std::move(w).take();
}

/**
 * @brief Builds a JSON-RPC 2.0 error response.
 *
 * @param id   The echoed request id (or null if parsing failed before id was read).
 * @param code Standard JSON-RPC error code (e.g. -32601).
 * @param msg  Human-readable error message.
 * @return The complete JSON-RPC response string.
 */
static std::string make_error(const RequestId& id, int code, std::string_view msg) {
    JsonWriter w(128);
    {
        auto _obj = w.object();
        w.kv_str("jsonrpc", "2.0").comma();
        write_response_id(w, id);
        w.comma().key("error");
        {
            auto _err = w.object();
            w.kv_num("code", code).comma().kv_str("message", msg);
        }
    }
    return std::move(w).take();
}

/// Pre-built parse-error response (id is null — used before the id is parsed).
static constexpr std::string_view kParseError{
    R"({"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error"},"id":null})"};

/// Pre-built invalid-request response (id is null — used before the id is parsed).
static constexpr std::string_view kInvalidRequest{
    R"({"jsonrpc":"2.0","error":{"code":-32600,"message":"Invalid Request"},"id":null})"};

// ---------------------------------------------------------------------------
// tools/list result (built once, cached as a static)
// ---------------------------------------------------------------------------

/**
 * @brief Returns the cached JSON object for a @c tools/list result.
 *
 * The tool list is static — tools are registered at startup and never change.
 * The result is built once on first call (C++11 magic-static guarantee) and
 * returned by const reference on every subsequent call.
 *
 * ### Emitted tool object fields (MCP 2025-11-25)
 * - @c name        — programmatic identifier
 * - @c title       — human-readable display name (optional, aids Lampo UI)
 * - @c description — prose description for the LLM
 * - @c inputSchema — JSON Schema object built from @c ToolDefinition::parameters
 * - @c annotations — behavioral hints (@c readOnlyHint, @c destructiveHint,
 *                    @c idempotentHint, @c openWorldHint)
 *
 * @return A const reference to the cached JSON result string.
 */
static const std::string& tools_list_result() {
    static const std::string result = [] {
        auto& sm    = core::tools::ToolManager::get_instance();
        auto  tools = sm.get_all_tools();

        JsonWriter w(8192);
        {
            auto _root = w.object();
            w.key("tools");
            {
                auto _arr = w.array();
                bool first_tool = true;

                for (const auto& tool : tools) {
                    if (!first_tool) w.comma();
                    first_tool = false;

                    const auto& def = tool.function;
                    {
                        auto _tool = w.object();

                        // name (required), title (optional display name for Lampo)
                        w.kv_str("name", def.name).comma()
                         .kv_str("title", def.title).comma()
                         .kv_str("description", def.description).comma()
                         .key("inputSchema");

                        // inputSchema — JSON Schema object
                        {
                            auto _schema = w.object();
                            w.kv_str("type", "object").comma().key("properties");
                            {
                                auto _props = w.object();
                                bool first_prop = true;
                                for (const auto& p : def.parameters) {
                                    if (!first_prop) w.comma();
                                    first_prop = false;
                                    w.key(p.name);
                                    {
                                        auto _p = w.object();
                                        w.kv_str("type", p.type).comma()
                                         .kv_str("description", p.description);
                                    }
                                }
                            }

                            // Collect required-parameter names.
                            // Tools have at most ~5 params; capacity 8 is safe.
                            struct {
                                std::array<std::string_view, 8> buf{};
                                std::size_t count = 0;
                                void push_back(std::string_view sv) { if (count < 8) buf[count++] = sv; }
                                auto begin() const { return buf.begin(); }
                                auto end()   const { return buf.begin() + static_cast<std::ptrdiff_t>(count); }
                            } required;
                            for (const auto& p : def.parameters)
                                if (p.required) required.push_back(p.name);

                            w.comma().key("required");
                            {
                                auto _req = w.array();
                                bool first_req = true;
                                for (auto name : required) {
                                    if (!first_req) w.comma();
                                    first_req = false;
                                    w.str(name);
                                }
                            }
                        }

                        // annotations — behavioral hints for the MCP client (MCP 2025-11-25 §Tools)
                        // Always emit all four fields so Lampo can rely on their presence.
                        w.comma().key("annotations");
                        {
                            auto _ann = w.object();
                            const auto& ann = def.annotations;
                            w.kv_bool("readOnlyHint",   ann.read_only_hint).comma()
                             .kv_bool("destructiveHint", ann.destructive_hint).comma()
                             .kv_bool("idempotentHint",  ann.idempotent_hint).comma()
                             .kv_bool("openWorldHint",   ann.open_world_hint);
                        }
                    }
                }
            }
        }
        return std::move(w).take();
    }();
    return result;
}

// ---------------------------------------------------------------------------
// Singleton lifecycle
// ---------------------------------------------------------------------------

McpDispatcher& McpDispatcher::get_instance() {
    static McpDispatcher instance;
    return instance;
}

McpDispatcher::McpDispatcher() { register_tools(); }

/**
 * @brief Registers all built-in filo tools into the @c ToolManager.
 *
 * Called exactly once from the constructor.  After this point the
 * @c ToolManager map is read-only and safe for concurrent HTTP threads.
 */
void McpDispatcher::register_tools() {
    auto& sm = core::tools::ToolManager::get_instance();
    sm.register_tool(std::make_shared<core::tools::ShellTool>());
    sm.register_tool(std::make_shared<core::tools::ApplyPatchTool>());
    sm.register_tool(std::make_shared<core::tools::FileSearchTool>());
    sm.register_tool(std::make_shared<core::tools::ReadFileTool>());
    sm.register_tool(std::make_shared<core::tools::WriteFileTool>());
    sm.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    sm.register_tool(std::make_shared<core::tools::ReplaceTool>());
    sm.register_tool(std::make_shared<core::tools::GrepSearchTool>());
    sm.register_tool(std::make_shared<core::tools::SearchReplaceTool>());
    sm.register_tool(std::make_shared<core::tools::DeleteFileTool>());
    sm.register_tool(std::make_shared<core::tools::MoveFileTool>());
    sm.register_tool(std::make_shared<core::tools::CreateDirectoryTool>());
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::string McpDispatcher::dispatch(const std::string& json_request) {
    if (json_request.empty())
        return std::string(kParseError);

    // Each call owns its own parser — ondemand parsers are not thread-safe and
    // must not be shared across concurrent HTTP handler threads.
    simdjson::ondemand::parser parser;
    simdjson::padded_string    padded(json_request);
    simdjson::ondemand::document req;

    if (parser.iterate(padded).get(req) != simdjson::SUCCESS)
        return std::string(kParseError);

    // Force real validation: ondemand::iterate() is lazy and returns SUCCESS
    // even for non-JSON text; get_object() triggers the actual parse.
    simdjson::ondemand::object root;
    if (req.get_object().get(root) != simdjson::SUCCESS)
        return std::string(kParseError);

    // --- id ---
    // Parse before jsonrpc/method so error responses always echo the id back.
    const RequestId id = parse_request_id(root);
    if (id.kind == RequestId::Kind::invalid)
        return std::string(kInvalidRequest);

    // --- jsonrpc ---
    std::string_view jsonrpc;
    if (root["jsonrpc"].get_string().get(jsonrpc) != simdjson::SUCCESS || jsonrpc != "2.0")
        return make_error(id, -32600, "Invalid Request");

    // --- method ---
    std::string_view method;
    if (root["method"].get_string().get(method) != simdjson::SUCCESS)
        return make_error(id, -32600, "Invalid Request");

    // Notifications (no id) must not receive a response.
    // This covers: notifications/initialized, notifications/cancelled, and any
    // future notifications the client may send (unknown ones are silently ignored).
    if (id.kind == RequestId::Kind::none) return {};

    // -----------------------------------------------------------------------
    // Method dispatch
    // -----------------------------------------------------------------------

    // --- initialize ---
    if (method == "initialize") {
        JsonWriter rw(512);
        {
            auto _res = rw.object();

            rw.kv_str("protocolVersion", negotiate_protocol_version(root)).comma()
              .key("capabilities");
            {
                auto _cap = rw.object();
                rw.key("tools");
                {
                    auto _tools = rw.object();
                    // listChanged: false — tool list is static; clients need not poll.
                    rw.kv_bool("listChanged", false);
                }
            }

            rw.comma().key("serverInfo");
            {
                auto _si = rw.object();
                rw.kv_str("name", "filo-mcp").comma()
                  .kv_str("version", "0.1.0");
            }

            // instructions (MCP 2025-11-25, optional) — guidance for the local model
            // on how to use this server effectively.
            rw.comma().kv_str("instructions", kServerInstructions);
        }
        return make_response(id, rw.view());

    // --- tools/list ---
    } else if (method == "tools/list") {
        // tools_list_result() is cached — built once, returned by const ref.
        // The cursor/pagination fields are not needed: our tool list is static
        // and fits comfortably in a single response.
        return make_response(id, tools_list_result());

    // --- tools/call ---
    } else if (method == "tools/call") {
        simdjson::ondemand::object params;
        if (root["params"].get_object().get(params) != simdjson::SUCCESS)
            return make_error(id, -32602, "Invalid params: missing 'params' object");

        std::string_view name;
        if (params["name"].get_string().get(name) != simdjson::SUCCESS)
            return make_error(id, -32602, "Invalid params: missing 'name'");

        // 'arguments' is optional per MCP spec — absent is treated as {}.
        std::string args_json = "{}";
        simdjson::ondemand::value arguments_value;
        if (params["arguments"].get(arguments_value) == simdjson::SUCCESS) {
            simdjson::ondemand::json_type arg_type;
            if (arguments_value.type().get(arg_type) != simdjson::SUCCESS
                || arg_type != simdjson::ondemand::json_type::object) {
                return make_error(id, -32602, "Invalid params: 'arguments' must be an object");
            }
            simdjson::ondemand::object arguments;
            std::string_view raw_json;
            if (arguments_value.get_object().get(arguments) != simdjson::SUCCESS
                || arguments.raw_json().get(raw_json) != simdjson::SUCCESS) {
                return make_error(id, -32602, "Invalid params: could not parse 'arguments'");
            }
            args_json = std::string(raw_json);
        }

        auto& sm = core::tools::ToolManager::get_instance();
        auto tool_def = sm.get_tool_definition(std::string(name));
        if (!tool_def.has_value())
            return make_error(id, -32602, std::format("Unknown tool: {}", name));

        if (auto validation_error = validate_tool_arguments(*tool_def, args_json))
            return make_error(id, -32602, *validation_error);

        std::string result = sm.execute_tool(std::string(name), args_json);

        // Detect error responses via a cheap prefix check.
        // CONTRACT: every tool that signals failure must return a JSON object
        // whose FIRST key is "error" — {"error": "..."}.  Success responses
        // must start with any other key.  This prefix check is O(1) and avoids
        // a full re-parse of potentially large file-content results.
        const bool is_error = result.starts_with(R"({"error")") ||
                              result.starts_with(R"({ "error")");

        JsonWriter rw(result.size() + 128);
        {
            auto _res = rw.object();
            rw.key("content");
            {
                auto _arr = rw.array();
                {
                    auto _item = rw.object();
                    rw.kv_str("type", "text").comma()
                      .kv_str("text", result);  // properly escaped via JsonWriter
                }
            }
            rw.comma().kv_bool("isError", is_error);
            // structuredContent (MCP 2025-11-25): embed the raw tool JSON alongside
            // the escaped text block so clients can access structured fields directly
            // without re-parsing the escaped string.  Lampo should prefer this field
            // over parsing content[0].text.
            rw.comma().kv_raw("structuredContent", result);
        }
        return make_response(id, rw.view());

    // --- ping ---
    } else if (method == "ping") {
        // Either party may ping the other; receiver must respond with empty result.
        return make_response(id, "{}");

    // --- unknown method ---
    } else {
        return make_error(id, -32601, "Method not found");
    }
}

} // namespace core::mcp
