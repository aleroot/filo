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
#include "../tools/GetWorkspaceConfigTool.hpp"
#include "../tools/SkillLoader.hpp"
#include "../workspace/Workspace.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/Base64.hpp"
#include "../utils/MimeUtils.hpp"
#include "../utils/UriUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <array>
#include <format>
#include <optional>
#include <ranges>
#include <system_error>
#include <vector>

namespace core::mcp {

using core::utils::JsonWriter;
using core::utils::Base64;
namespace ascii = core::utils::ascii;
namespace mime = core::utils::mime;
namespace uri = core::utils::uri;

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

constexpr std::size_t kMaxResourceBytes = 1024 * 1024; // 1 MiB
constexpr std::size_t kMaxDirectoryListingBytes = 256 * 1024; // 256 KiB
constexpr std::size_t kMaxDirectoryEntries = 4096;
constexpr int kResourceNotFoundCode = -32002;
constexpr int kResourceInternalErrorCode = -32603;

[[nodiscard]] std::string path_to_file_uri(const std::filesystem::path& path) {
    std::error_code ec;
    auto abs = std::filesystem::absolute(path, ec);
    auto normalized = (ec ? path : abs).lexically_normal();
    std::string generic = normalized.generic_string();
    if (generic.empty() || generic.front() != '/') generic.insert(generic.begin(), '/');
    return "file://" + uri::percent_encode_uri_path(generic);
}

[[nodiscard]] std::optional<std::filesystem::path> parse_file_uri(std::string_view uri,
                                                                  std::string& error_out) {
    if (!ascii::istarts_with(uri, "file://")) {
        error_out = "Invalid params: only file:// URIs are supported";
        return std::nullopt;
    }

    std::string_view rest = uri.substr(7);
    std::string_view encoded_path;
    if (rest.starts_with('/')) {
        encoded_path = rest;
    } else {
        const auto slash = rest.find('/');
        const std::string_view authority =
            slash == std::string_view::npos ? rest : rest.substr(0, slash);
        if (!authority.empty() && !ascii::iequals(authority, "localhost")) {
            error_out = "Invalid params: unsupported file URI authority";
            return std::nullopt;
        }
        encoded_path = slash == std::string_view::npos ? std::string_view{"/"} : rest.substr(slash);
    }

    // file:// URIs for local filesystem paths must not include query/fragment.
    if (encoded_path.find('?') != std::string_view::npos
        || encoded_path.find('#') != std::string_view::npos) {
        error_out = "Invalid params: file URI must not include query or fragment";
        return std::nullopt;
    }

    std::string decoded_path;
    if (!uri::percent_decode(encoded_path, decoded_path)) {
        error_out = "Invalid params: malformed percent-encoding in URI";
        return std::nullopt;
    }
    if (decoded_path.empty()) {
        error_out = "Invalid params: URI path is empty";
        return std::nullopt;
    }
    if (decoded_path.find('\0') != std::string::npos) {
        error_out = "Invalid params: URI path contains NUL byte";
        return std::nullopt;
    }
    return std::filesystem::path(decoded_path);
}

[[nodiscard]] bool is_likely_binary(std::string_view data) {
    if (data.empty()) return false;
    std::size_t control_count = 0;
    for (const unsigned char ch : data) {
        if (ch == 0) return true;
        const bool is_control = ch < 0x20 && ch != '\n' && ch != '\r' && ch != '\t' && ch != '\f';
        if (is_control) ++control_count;
    }
    return (control_count * 100) / data.size() > 5;
}

struct ResourceFileReadResult {
    std::string data;
    bool truncated = false;
};

struct PromptTemplate {
    std::string name;
    std::string description;
    std::string body;
    bool accepts_arguments = false;
};

[[nodiscard]] std::optional<ResourceFileReadResult> read_resource_file_limited(
    const std::filesystem::path& path,
    std::string& error_out)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        error_out = std::format("Failed to open file: {}", path.string());
        return std::nullopt;
    }

    ResourceFileReadResult result;
    result.data.reserve(kMaxResourceBytes + 1);
    std::array<char, 4096> chunk{};

    while (ifs) {
        ifs.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto n = ifs.gcount();
        if (n <= 0) break;

        const auto available = kMaxResourceBytes - result.data.size();
        const auto count = static_cast<std::size_t>(n);
        if (count > available) {
            result.data.append(chunk.data(), available);
            result.truncated = true;
            break;
        }
        result.data.append(chunk.data(), count);
    }

    if (ifs.bad()) {
        error_out = std::format("Failed to read file: {}", path.string());
        return std::nullopt;
    }

    return result;
}

[[nodiscard]] std::string expand_prompt_arguments(std::string_view body,
                                                 std::string_view arguments) {
    constexpr std::string_view kPlaceholder = "$ARGUMENTS";
    std::string result(body);
    std::size_t pos = 0;
    while ((pos = result.find(kPlaceholder, pos)) != std::string::npos) {
        result.replace(pos, kPlaceholder.size(), arguments);
        pos += arguments.size();
    }
    return result;
}

[[nodiscard]] std::vector<PromptTemplate> discover_prompt_templates() {
    std::vector<PromptTemplate> prompts;

    for (const auto& root : core::tools::SkillLoader::default_search_paths()) {
        if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
            continue;
        }

        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (!entry.is_directory()) continue;

            auto manifest = core::tools::SkillLoader::parse_manifest(entry.path());
            if (!manifest.has_value()) continue;
            if (!manifest->enabled || manifest->type != core::tools::SkillType::Prompt) {
                continue;
            }

            PromptTemplate prompt{
                .name = manifest->name,
                .description = manifest->description,
                .body = manifest->body,
                .accepts_arguments = manifest->body.find("$ARGUMENTS") != std::string::npos,
            };

            const auto existing = std::ranges::find(prompts, prompt.name, &PromptTemplate::name);
            if (existing != prompts.end()) {
                *existing = std::move(prompt);
            } else {
                prompts.push_back(std::move(prompt));
            }
        }
    }

    std::ranges::sort(prompts, {}, &PromptTemplate::name);
    return prompts;
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

[[nodiscard]] std::string build_parameter_schema_json(const core::tools::ToolParameter& parameter) {
    if (!parameter.schema.empty()) {
        return parameter.schema;
    }

    JsonWriter w(128 + parameter.description.size() + parameter.items_schema.size());
    {
        auto _schema = w.object();
        w.kv_str("type", parameter.type);
        if (!parameter.description.empty()) {
            w.comma().kv_str("description", parameter.description);
        }
        if (parameter.type == "array" && !parameter.items_schema.empty()) {
            w.comma().kv_raw("items", parameter.items_schema);
        }
    }
    return std::move(w).take();
}

[[nodiscard]] std::string build_input_schema_json(const core::tools::ToolDefinition& def) {
    if (!def.input_schema.empty()) {
        return def.input_schema;
    }

    JsonWriter w(512 + def.parameters.size() * 128);
    {
        auto _schema = w.object();
        w.kv_str("type", "object").comma().key("properties");
        {
            auto _props = w.object();
            bool first_prop = true;
            for (const auto& parameter : def.parameters) {
                if (!first_prop) w.comma();
                first_prop = false;
                w.kv_raw(parameter.name, build_parameter_schema_json(parameter));
            }
        }

        w.comma().key("required");
        {
            auto _req = w.array();
            bool first_req = true;
            for (const auto& parameter : def.parameters) {
                if (!parameter.required) continue;
                if (!first_req) w.comma();
                first_req = false;
                w.str(parameter.name);
            }
        }
    }
    return std::move(w).take();
}

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
                         .kv_raw("inputSchema", build_input_schema_json(def));

                        if (!def.output_schema.empty()) {
                            w.comma().kv_raw("outputSchema", def.output_schema);
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
    sm.register_tool(std::make_shared<core::tools::GetWorkspaceConfigTool>());
}

namespace {

struct RpcError {
    int code = -32603;
    std::string message;
};

template <typename T>
using RpcExpected = std::expected<T, RpcError>;

struct ParsedToolCallRequest {
    std::string name;
    std::string arguments_json{"{}"};
};

struct ParsedPromptGetRequest {
    std::string name;
    std::string arguments;
};

[[nodiscard]] std::string make_rpc_error(const RequestId& id, const RpcError& error) {
    return make_error(id, error.code, error.message);
}

[[nodiscard]] std::expected<void, RpcError> parse_params_object(
    simdjson::ondemand::object& root,
    simdjson::ondemand::object& params_out)
{
    if (root["params"].get_object().get(params_out) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'params' object"});
    }
    return {};
}

[[nodiscard]] std::string build_initialize_result(simdjson::ondemand::object& root) {
    const bool has_prompts = !discover_prompt_templates().empty();
    JsonWriter rw(1024);
    {
        auto _res = rw.object();

        rw.kv_str("protocolVersion", negotiate_protocol_version(root)).comma()
            .key("capabilities");
        {
            auto _cap = rw.object();
            rw.key("tools");
            {
                auto _tools = rw.object();
                rw.kv_bool("listChanged", false);
            }
            rw.comma().key("resources");
            {
                auto _res_obj = rw.object();
                rw.kv_bool("subscribe", false).comma()
                    .kv_bool("listChanged", false);
            }
            if (has_prompts) {
                rw.comma().key("prompts");
                {
                    auto _prompts = rw.object();
                    rw.kv_bool("listChanged", false);
                }
            }
        }

        rw.comma().key("serverInfo");
        {
            auto _si = rw.object();
            rw.kv_str("name", "filo-mcp").comma()
                .kv_str("version", "0.1.0");
        }

        auto& ws = core::workspace::Workspace::get_instance();
        std::string dynamic_instructions = std::string(kServerInstructions);

        if (ws.is_enforced()) {
            dynamic_instructions += "\n\nALLOWED WORKSPACE FOLDERS (strict enforcement enabled):\n";
            dynamic_instructions += "- Primary: " + ws.get_primary().string() + "\n";
            for (const auto& add : ws.get_additional()) {
                dynamic_instructions += "- Additional: " + add.string() + "\n";
            }
            dynamic_instructions += "\nUse filesystem tools for strict path-bounded operations. "
                                    "run_terminal_command remains open-world and can access paths "
                                    "outside these folders if the invoked command does so.";
        } else {
            dynamic_instructions += "\n\nWORKSPACE: Working directory is " + ws.get_primary().string()
                + " (enforcement disabled; system-wide path access is allowed).";
        }
        rw.comma().kv_str("instructions", dynamic_instructions);
    }
    return std::move(rw).take();
}

[[nodiscard]] std::string build_resources_templates_list_result() {
    JsonWriter rw(128);
    {
        auto _res = rw.object();
        rw.key("resourceTemplates");
        {
            auto _arr = rw.array();
        }
    }
    return std::move(rw).take();
}

[[nodiscard]] std::string build_prompts_list_result(
    const std::vector<PromptTemplate>& prompts)
{
    JsonWriter rw(512 + prompts.size() * 256);
    {
        auto _res = rw.object();
        rw.key("prompts");
        {
            auto _arr = rw.array();
            bool first_prompt = true;
            for (const auto& prompt : prompts) {
                if (!first_prompt) rw.comma();
                first_prompt = false;

                auto _prompt = rw.object();
                rw.kv_str("name", prompt.name);
                if (!prompt.description.empty()) {
                    rw.comma().kv_str("description", prompt.description);
                }
                if (prompt.accepts_arguments) {
                    rw.comma().key("arguments");
                    {
                        auto _args = rw.array();
                        auto _arg = rw.object();
                        rw.kv_str("name", "arguments").comma()
                            .kv_str("description",
                                    "Text substituted for $ARGUMENTS in the prompt body.")
                            .comma()
                            .kv_bool("required", false);
                    }
                }
            }
        }
    }
    return std::move(rw).take();
}

[[nodiscard]] std::string build_resources_list_result() {
    auto& ws = core::workspace::Workspace::get_instance();
    JsonWriter rw(2048);
    {
        auto _res = rw.object();
        rw.key("resources");
        {
            auto _arr = rw.array();

            auto add_resource = [&](const std::filesystem::path& path,
                                    std::string_view name,
                                    std::string_view desc) {
                auto _item = rw.object();
                rw.kv_str("uri", path_to_file_uri(path)).comma()
                    .kv_str("name", name).comma()
                    .kv_str("description", desc).comma()
                    .kv_str("mimeType", "application/x-directory");
            };

            bool first = true;
            if (!ws.get_primary().empty()) {
                add_resource(
                    ws.get_primary(),
                    "Primary Workspace",
                    "The main working directory for this session.");
                first = false;
            }

            for (std::size_t i = 0; i < ws.get_additional().size(); ++i) {
                if (!first) rw.comma();
                first = false;
                add_resource(
                    ws.get_additional()[i],
                    std::format("Additional Workspace {}", i + 1),
                    "An additional directory allowed for this session.");
            }
        }
    }
    return std::move(rw).take();
}

[[nodiscard]] RpcExpected<std::filesystem::path> parse_resources_read_path(
    simdjson::ondemand::object& root)
{
    simdjson::ondemand::object params;
    if (auto params_result = parse_params_object(root, params); !params_result) {
        return std::unexpected(params_result.error());
    }

    std::string_view uri_sv;
    if (params["uri"].get_string().get(uri_sv) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'uri'"});
    }

    std::string uri_parse_error;
    auto path_opt = parse_file_uri(uri_sv, uri_parse_error);
    if (!path_opt.has_value()) {
        return std::unexpected(RpcError{-32602, std::move(uri_parse_error)});
    }
    return *path_opt;
}

[[nodiscard]] RpcExpected<std::string> build_directory_listing(
    const std::filesystem::path& path)
{
    std::error_code it_ec;
    std::vector<std::string> lines;
    lines.reserve(128);
    std::size_t listing_bytes = 0;
    bool listing_truncated = false;

    for (std::filesystem::directory_iterator it(
             path,
             std::filesystem::directory_options::skip_permission_denied,
             it_ec),
         end;
         it != end;
         it.increment(it_ec))
    {
        if (it_ec) break;

        std::error_code type_ec;
        const bool is_directory = it->is_directory(type_ec);
        if (type_ec) continue;

        std::uintmax_t size = 0;
        if (!is_directory) {
            std::error_code size_ec;
            if (it->is_regular_file(size_ec) && !size_ec) {
                size = it->file_size(size_ec);
                if (size_ec) size = 0;
            }
        }

        std::string line = std::format(
            "{}\t{}\t{}",
            is_directory ? "dir" : "file",
            it->path().filename().string(),
            size);
        if (lines.size() >= kMaxDirectoryEntries
            || listing_bytes + line.size() + 1 > kMaxDirectoryListingBytes) {
            listing_truncated = true;
            break;
        }

        listing_bytes += line.size() + 1;
        lines.push_back(std::move(line));
    }

    if (it_ec) {
        return std::unexpected(
            RpcError{
                kResourceInternalErrorCode,
                std::format("Failed to list directory '{}': {}", path.string(), it_ec.message())});
    }

    std::ranges::sort(lines);
    std::string listing;
    listing.reserve(std::min(listing_bytes + 64, kMaxDirectoryListingBytes + 64));
    for (std::size_t i = 0; i < lines.size(); ++i) {
        listing += lines[i];
        if (i + 1 < lines.size()) listing.push_back('\n');
    }
    if (listing_truncated) {
        if (!listing.empty()) listing += '\n';
        listing += "... [TRUNCATED DIRECTORY LISTING] ...";
    }
    return listing;
}

[[nodiscard]] RpcExpected<std::string> build_resources_read_result(
    const std::filesystem::path& path)
{
    auto& ws = core::workspace::Workspace::get_instance();
    if (!ws.is_path_allowed(path)) {
        return std::unexpected(
            RpcError{-32001, "Access denied: path is outside the allowed workspace"});
    }

    std::error_code status_ec;
    const auto status = std::filesystem::status(path, status_ec);
    if (status_ec || status.type() == std::filesystem::file_type::not_found) {
        return std::unexpected(
            RpcError{kResourceNotFoundCode, std::format("Resource not found: {}", path.string())});
    }

    JsonWriter rw(4096);
    {
        auto _res = rw.object();
        rw.key("contents");
        {
            auto _arr = rw.array();
            auto _item = rw.object();
            rw.kv_str("uri", path_to_file_uri(path)).comma();

            if (status.type() == std::filesystem::file_type::directory) {
                rw.kv_str("mimeType", "application/x-directory").comma();
                auto listing = build_directory_listing(path);
                if (!listing) return std::unexpected(listing.error());
                rw.kv_str("text", *listing);
            } else if (status.type() == std::filesystem::file_type::regular) {
                std::string read_error;
                auto read_result = read_resource_file_limited(path, read_error);
                if (!read_result.has_value()) {
                    return std::unexpected(
                        RpcError{kResourceInternalErrorCode, std::move(read_error)});
                }

                const bool binary = is_likely_binary(read_result->data);
                rw.kv_str("mimeType", mime::guess_type(path, binary)).comma();

                if (binary) {
                    if (read_result->truncated) {
                        return std::unexpected(RpcError{
                            kResourceInternalErrorCode,
                            std::format("Binary resource too large: {} (max {} bytes)",
                                        path.string(),
                                        kMaxResourceBytes)});
                    }
                    rw.kv_str("blob", Base64::encode(read_result->data));
                } else {
                    if (read_result->truncated) {
                        read_result->data += "\n\n... [TRUNCATED DUE TO SIZE] ...";
                    }
                    rw.kv_str("text", read_result->data);
                }
            } else {
                return std::unexpected(
                    RpcError{kResourceInternalErrorCode,
                             std::format("Unsupported resource type for '{}'", path.string())});
            }
        }
    }
    return std::move(rw).take();
}

[[nodiscard]] RpcExpected<ParsedToolCallRequest> parse_tool_call_request(
    simdjson::ondemand::object& root)
{
    simdjson::ondemand::object params;
    if (auto params_result = parse_params_object(root, params); !params_result) {
        return std::unexpected(params_result.error());
    }

    std::string_view name;
    if (params["name"].get_string().get(name) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'name'"});
    }

    ParsedToolCallRequest parsed;
    parsed.name = std::string(name);

    simdjson::ondemand::value arguments_value;
    if (params["arguments"].get(arguments_value) == simdjson::SUCCESS) {
        simdjson::ondemand::json_type arg_type;
        if (arguments_value.type().get(arg_type) != simdjson::SUCCESS
            || arg_type != simdjson::ondemand::json_type::object) {
            return std::unexpected(
                RpcError{-32602, "Invalid params: 'arguments' must be an object"});
        }

        simdjson::ondemand::object arguments;
        std::string_view raw_json;
        if (arguments_value.get_object().get(arguments) != simdjson::SUCCESS
            || arguments.raw_json().get(raw_json) != simdjson::SUCCESS) {
            return std::unexpected(
                RpcError{-32602, "Invalid params: could not parse 'arguments'"});
        }
        parsed.arguments_json = std::string(raw_json);
    }

    return parsed;
}

[[nodiscard]] RpcExpected<ParsedPromptGetRequest> parse_prompt_get_request(
    simdjson::ondemand::object& root)
{
    simdjson::ondemand::object params;
    if (auto params_result = parse_params_object(root, params); !params_result) {
        return std::unexpected(params_result.error());
    }

    std::string_view name;
    if (params["name"].get_string().get(name) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'name'"});
    }

    ParsedPromptGetRequest parsed{.name = std::string(name)};

    simdjson::ondemand::value arguments_value;
    if (params["arguments"].get(arguments_value) == simdjson::SUCCESS) {
        simdjson::ondemand::json_type arguments_type;
        if (arguments_value.type().get(arguments_type) != simdjson::SUCCESS
            || arguments_type != simdjson::ondemand::json_type::object) {
            return std::unexpected(
                RpcError{-32602, "Invalid params: 'arguments' must be an object"});
        }

        simdjson::ondemand::object arguments;
        if (arguments_value.get_object().get(arguments) != simdjson::SUCCESS) {
            return std::unexpected(
                RpcError{-32602, "Invalid params: could not parse 'arguments'"});
        }

        std::string_view replacement_text;
        if (arguments["arguments"].get_string().get(replacement_text) == simdjson::SUCCESS) {
            parsed.arguments = std::string(replacement_text);
        } else {
            simdjson::ondemand::value maybe_argument_value;
            if (arguments["arguments"].get(maybe_argument_value) == simdjson::SUCCESS) {
                return std::unexpected(
                    RpcError{-32602,
                             "Invalid params: prompt argument 'arguments' must be a string"});
            }
        }
    }

    return parsed;
}

[[nodiscard]] RpcExpected<std::string> build_tool_call_result(
    const ParsedToolCallRequest& request)
{
    auto& sm = core::tools::ToolManager::get_instance();
    auto tool_def = sm.get_tool_definition(request.name);
    if (!tool_def.has_value()) {
        return std::unexpected(
            RpcError{-32602, std::format("Unknown tool: {}", request.name)});
    }

    if (auto validation_error = validate_tool_arguments(*tool_def, request.arguments_json)) {
        return std::unexpected(RpcError{-32602, *validation_error});
    }

    std::string tool_result = sm.execute_tool(request.name, request.arguments_json);
    const bool is_error = tool_result.starts_with(R"({"error")")
        || tool_result.starts_with(R"({ "error")");

    JsonWriter rw(tool_result.size() + 128);
    {
        auto _res = rw.object();
        rw.key("content");
        {
            auto _arr = rw.array();
            auto _item = rw.object();
            rw.kv_str("type", "text").comma()
                .kv_str("text", tool_result);
        }
        rw.comma().kv_bool("isError", is_error);
        if (!is_error) {
            rw.comma().kv_raw("structuredContent", tool_result);
        }
    }
    return std::move(rw).take();
}

[[nodiscard]] RpcExpected<std::string> build_prompt_get_result(
    const ParsedPromptGetRequest& request)
{
    const auto prompts = discover_prompt_templates();
    const auto it = std::ranges::find(prompts, request.name, &PromptTemplate::name);
    if (it == prompts.end()) {
        return std::unexpected(
            RpcError{-32602, std::format("Unknown prompt: {}", request.name)});
    }

    const std::string expanded_body =
        it->accepts_arguments ? expand_prompt_arguments(it->body, request.arguments)
                              : it->body;

    JsonWriter rw(256 + expanded_body.size() + it->description.size());
    {
        auto _res = rw.object();
        if (!it->description.empty()) {
            rw.kv_str("description", it->description).comma();
        }
        rw.key("messages");
        {
            auto _messages = rw.array();
            auto _message = rw.object();
            rw.kv_str("role", "user").comma().key("content");
            {
                auto _content = rw.object();
                rw.kv_str("type", "text").comma()
                    .kv_str("text", expanded_body);
            }
        }
    }
    return std::move(rw).take();
}

using MethodHandler = std::string (*)(const RequestId&, simdjson::ondemand::object&);

[[nodiscard]] std::string handle_initialize(
    const RequestId& id,
    simdjson::ondemand::object& root)
{
    return make_response(id, build_initialize_result(root));
}

[[nodiscard]] std::string handle_resources_templates_list(
    const RequestId& id,
    simdjson::ondemand::object&)
{
    return make_response(id, build_resources_templates_list_result());
}

[[nodiscard]] std::string handle_prompts_list(
    const RequestId& id,
    simdjson::ondemand::object&)
{
    return make_response(id, build_prompts_list_result(discover_prompt_templates()));
}

[[nodiscard]] std::string handle_prompts_get(
    const RequestId& id,
    simdjson::ondemand::object& root)
{
    auto request = parse_prompt_get_request(root);
    if (!request) return make_rpc_error(id, request.error());

    auto result = build_prompt_get_result(*request);
    if (!result) return make_rpc_error(id, result.error());

    return make_response(id, *result);
}

[[nodiscard]] std::string handle_resources_list(
    const RequestId& id,
    simdjson::ondemand::object&)
{
    return make_response(id, build_resources_list_result());
}

[[nodiscard]] std::string handle_resources_read(
    const RequestId& id,
    simdjson::ondemand::object& root)
{
    auto path = parse_resources_read_path(root);
    if (!path) return make_rpc_error(id, path.error());

    auto result = build_resources_read_result(*path);
    if (!result) return make_rpc_error(id, result.error());

    return make_response(id, *result);
}

[[nodiscard]] std::string handle_tools_list(
    const RequestId& id,
    simdjson::ondemand::object&)
{
    return make_response(id, tools_list_result());
}

[[nodiscard]] std::string handle_tools_call(
    const RequestId& id,
    simdjson::ondemand::object& root)
{
    auto request = parse_tool_call_request(root);
    if (!request) return make_rpc_error(id, request.error());

    auto result = build_tool_call_result(*request);
    if (!result) return make_rpc_error(id, result.error());

    return make_response(id, *result);
}

[[nodiscard]] std::string handle_ping(
    const RequestId& id,
    simdjson::ondemand::object&)
{
    return make_response(id, "{}");
}

struct MethodRoute {
    std::string_view name;
    MethodHandler handler;
};

constexpr std::array<MethodRoute, 9> kMethodRoutes{{
    {"initialize", &handle_initialize},
    {"prompts/list", &handle_prompts_list},
    {"prompts/get", &handle_prompts_get},
    {"resources/templates/list", &handle_resources_templates_list},
    {"resources/list", &handle_resources_list},
    {"resources/read", &handle_resources_read},
    {"tools/list", &handle_tools_list},
    {"tools/call", &handle_tools_call},
    {"ping", &handle_ping},
}};

[[nodiscard]] std::string route_method(
    std::string_view method,
    const RequestId& id,
    simdjson::ondemand::object& root)
{
    const auto it = std::ranges::find(kMethodRoutes, method, &MethodRoute::name);
    if (it == kMethodRoutes.end()) {
        return make_error(id, -32601, "Method not found");
    }
    return it->handler(id, root);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::string McpDispatcher::dispatch(const std::string& json_request) {
    if (json_request.empty()) return std::string(kParseError);

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_request);
    simdjson::ondemand::document request_doc;
    if (parser.iterate(padded).get(request_doc) != simdjson::SUCCESS) {
        return std::string(kParseError);
    }

    simdjson::ondemand::object root;
    if (request_doc.get_object().get(root) != simdjson::SUCCESS) {
        return std::string(kParseError);
    }

    const RequestId id = parse_request_id(root);
    if (id.kind == RequestId::Kind::invalid) {
        return std::string(kInvalidRequest);
    }

    std::string_view jsonrpc;
    if (root["jsonrpc"].get_string().get(jsonrpc) != simdjson::SUCCESS || jsonrpc != "2.0") {
        return make_error(id, -32600, "Invalid Request");
    }

    std::string_view method;
    if (root["method"].get_string().get(method) != simdjson::SUCCESS) {
        return make_error(id, -32600, "Invalid Request");
    }

    if (id.kind == RequestId::Kind::none) {
        return {};
    }

    return route_method(method, id, root);
}

} // namespace core::mcp
