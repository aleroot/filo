#include "McpDispatcher.hpp"
#include "McpTaskManager.hpp"
#include "RemoteActivity.hpp"
#include "ToolCallResult.hpp"
#include "../context/SessionContext.hpp"
#include "../tools/BuiltinToolRegistry.hpp"
#include "../tools/ToolManager.hpp"
#include "../tools/ShellTool.hpp"
#include "../tools/ApplyPatchTool.hpp"
#include "../tools/FileSearchTool.hpp"
#include "../tools/ReadTool.hpp"
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
#include "../tools/TaskTool.hpp"
#include "../tools/SkillRegistry.hpp"
#include "../tools/ActivateSkillTool.hpp"
#include "../workspace/Workspace.hpp"
#include "../workspace/MentionIndex.hpp"
#include "../workspace/PathVisibility.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/Base64.hpp"
#include "../utils/MimeUtils.hpp"
#include "../utils/UriUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "core/version/Version.hpp"
#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <array>
#include <atomic>
#include <format>
#include <optional>
#include <ranges>
#include <system_error>
#include <unordered_set>
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
constexpr std::array<std::string_view, 3> kSupportedProtocolVersions{
    "2026-07-28",  ///< Stateless MCP
    "2025-11-25",  ///< Legacy Streamable HTTP
    "2024-11-05",  ///< Legacy version
};

/// Default version announced and used when no matching version is negotiated.
constexpr std::string_view kLegacyProtocolVersion = "2025-11-25";
constexpr std::string_view kDefaultProtocolVersion = kLegacyProtocolVersion;
constexpr const char* kTasksExtensionIdentifier = "io.modelcontextprotocol/tasks";

/// Cache-control hints for list-style results (MCP 2026-07-28 §Caching).
///
/// ttlMs 0 means "immediately stale": clients MAY re-fetch the result every
/// time it is needed. Filo's tool catalog is registration-fixed and never
/// changes within a process, so a positive TTL is the spec-intended signal
/// that lets clients (Lampo resolves tooling per user message) keep their
/// cached catalog instead of re-fetching the full payload on every request.
/// Workspace- and disk-derived lists use a shorter TTL so edits stay visible;
/// resources/read intentionally keeps ttlMs 0 because file contents must
/// always be served fresh.
constexpr int64_t kStaticListTtlMs    = 3'600'000; ///< 1 hour — registration-fixed lists
constexpr int64_t kDerivedListTtlMs   = 60'000;    ///< 1 minute — workspace/disk-derived lists

/// Human-readable instructions returned by discovery and legacy initialize.
constexpr std::string_view kServerInstructions =
    "filo-mcp provides local coding tools in the configured workspace. "
    "Paths may be absolute or relative to the active workspace. "
    "Prefer file_search or grep_search before read, and use line slices for large files. "
    "read.question asks the configured reader worker for cited evidence; use exact reads before editing. "
    "Prefer search_replace for exact edits or apply_patch for diffs; write_file replaces a whole file. "
    "Check run_terminal_command.exit_code after shell calls. "
    "write_file.previous_content supports diff display without another read. "
    "delegate_task start/resume run in the background: task-capable clients receive an MCP "
    "task handle; other clients receive the completed tool result.";

namespace {

// ---------------------------------------------------------------------------
// Request-ID helpers
// ---------------------------------------------------------------------------

/**
 * @brief Typed container for a JSON-RPC 2.0 request @c id.
 *
 * MCP restricts request IDs to integers and strings.
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

struct InitializeInfo {
    std::string protocol_version;
};

[[nodiscard]] bool has_tasks_extension_capability(
    simdjson::ondemand::object& capabilities)
{
    simdjson::ondemand::object extensions;
    if (capabilities["extensions"].get_object().get(extensions) != simdjson::SUCCESS) {
        return false;
    }

    simdjson::ondemand::value tasks_extension;
    return extensions[kTasksExtensionIdentifier].get(tasks_extension) == simdjson::SUCCESS;
}

/**
 * @brief Parses initialize params and selects the best protocol version.
 */
[[nodiscard]] InitializeInfo parse_initialize_info(simdjson::ondemand::object& root) {
    InitializeInfo info{.protocol_version = std::string(kDefaultProtocolVersion)};

    simdjson::ondemand::object params;
    if (root["params"].get_object().get(params) != simdjson::SUCCESS) {
        return info;
    }

    std::string_view requested;
    if (params["protocolVersion"].get_string().get(requested) == simdjson::SUCCESS
        && is_supported_protocol_version(requested)) {
        info.protocol_version = std::string(requested);
    }

    return info;
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
                // Some client adapters expose an MCP tool through a generic
                // string-valued `arguments` field and then forward that field
                // unchanged. On the MCP wire this becomes
                // params.arguments.arguments instead of placing the declared
                // tool fields directly in params.arguments. Do not guess at or
                // execute JSON-like strings (the server also exposes mutating
                // tools), but make this integration error unmistakable.
                const bool tool_declares_arguments = std::ranges::any_of(
                    def.parameters,
                    [](const auto& candidate) {
                        return candidate.name == "arguments";
                    });
                simdjson::dom::element nested_arguments;
                if (!tool_declares_arguments
                    && args["arguments"].get(nested_arguments) == simdjson::SUCCESS
                    && nested_arguments.type()
                        == simdjson::dom::element_type::STRING) {
                    return std::format(
                        "Invalid params: received a nested string at "
                        "params.arguments.arguments; pass '{}' directly under "
                        "params.arguments",
                        param.name);
                }
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

/// Resource template completed by `completion/complete` so clients can offer an
/// `@`-style workspace file picker. `{+path}` uses RFC 6570 reserved expansion
/// because completion values are absolute paths and must keep their separators.
constexpr std::string_view kFileResourceTemplate = "file:///{+path}";
constexpr std::string_view kFileResourceTemplateArgument = "path";
/// MCP caps a completion result at 100 values.
constexpr std::size_t kMaxCompletionValues = 100;
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
[[nodiscard]] std::string modernize_result(std::string_view result) {
    if (result.size() < 2 || result.front() != '{' || result.back() != '}') {
        return std::string(result);
    }

    const std::string_view inner = result.substr(1, result.size() - 2);
    JsonWriter w(result.size() + 160);
    {
        auto object = w.object();
        w.key("_meta");
        {
            auto meta = w.object();
            w.key("io.modelcontextprotocol/serverInfo");
            {
                auto server = w.object();
                w.kv_str("name", "filo-mcp").comma()
                    .kv_str("version", core::version::value);
            }
        }
        if (!inner.starts_with(R"("resultType":)")) {
            w.comma().kv_str("resultType", "complete");
        }
        if (!inner.empty()) {
            w.comma().raw(inner);
        }
    }
    return std::move(w).take();
}

static std::string make_response(const RequestId& id,
                                 std::string_view result,
                                 McpProtocolMode mode) {
    const std::string modern_result = mode == McpProtocolMode::stateless
        ? modernize_result(result)
        : std::string(result);
    JsonWriter w(64 + modern_result.size());
    {
        auto _obj = w.object();
        w.kv_str("jsonrpc", "2.0").comma();
        write_response_id(w, id);
        w.comma().kv_raw("result", modern_result);
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

[[nodiscard]] std::string make_missing_task_capability_error(const RequestId& id) {
    JsonWriter w(256);
    {
        auto object = w.object();
        w.kv_str("jsonrpc", "2.0").comma();
        write_response_id(w, id);
        w.comma().key("error");
        {
            auto error = w.object();
            w.kv_num("code", -32021).comma()
                .kv_str("message", "Missing required client capability.").comma()
                .key("data");
            {
                auto data = w.object();
                w.key("requiredCapabilities");
                {
                    auto capabilities = w.object();
                    w.key("extensions");
                    {
                        auto extensions = w.object();
                        w.key(kTasksExtensionIdentifier);
                        { auto tasks = w.object(); }
                    }
                }
            }
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

        // Absent "required" already means "no constraints" in JSON Schema;
        // an empty array is redundant wire bytes on every catalog fetch.
        bool has_required = false;
        for (const auto& parameter : def.parameters) {
            if (parameter.required) { has_required = true; break; }
        }
        if (has_required) {
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
 * ### Emitted tool object fields (MCP 2025-11-25 / 2026-07-28)
 * - @c name        — programmatic identifier
 * - @c title       — human-readable display name (optional, aids client UIs)
 * - @c description — prose description for the LLM
 * - @c inputSchema — JSON Schema object built from @c ToolDefinition::parameters
 * - @c annotations — behavioral hints; only @em non-default hints are emitted
 *                    (@c readOnlyHint, @c destructiveHint, @c idempotentHint,
 *                    @c openWorldHint — spec defaults are false, true, false,
 *                    true respectively, and clients apply them themselves)
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
            // The catalog is registration-fixed (listChanged: false); a positive
            // TTL lets stateless clients cache it instead of re-fetching per use.
            w.kv_num("ttlMs", kStaticListTtlMs).comma()
                .kv_str("cacheScope", "private").comma()
                .key("tools");
            {
                auto _arr = w.array();
                bool first_tool = true;

                for (const auto& tool : tools) {
                    if (!first_tool) w.comma();
                    first_tool = false;

                    const auto& def = tool.function;
                    {
                        auto _tool = w.object();

                        w.kv_str("name", def.name).comma()
                         .kv_str("title", def.title).comma()
                         .kv_str("description", def.description).comma()
                         .kv_raw("inputSchema", build_input_schema_json(def));

                        if (!def.output_schema.empty()) {
                            w.comma().kv_raw("outputSchema", def.output_schema);
                        }

                        // Emit only hints that differ from the spec defaults
                        // (readOnly=false, destructive=true, idempotent=false,
                        // openWorld=true). Clients apply the defaults for absent
                        // keys, so redundant false/true entries are pure wire
                        // waste. A tool whose hints are all defaults omits the
                        // annotations object entirely.
                        const auto& ann = def.annotations;
                        const bool destructive_non_default = !ann.destructive_hint;
                        const bool open_world_non_default  = !ann.open_world_hint;
                        if (ann.read_only_hint || ann.idempotent_hint
                            || destructive_non_default || open_world_non_default) {
                            w.comma().key("annotations");
                            {
                                auto _ann = w.object();
                                bool first_hint = true;
                                const auto write_hint = [&](std::string_view key,
                                                            bool value) {
                                    if (!first_hint) w.comma();
                                    first_hint = false;
                                    w.kv_bool(key, value);
                                };
                                if (ann.read_only_hint) {
                                    write_hint("readOnlyHint", true);
                                }
                                if (ann.idempotent_hint) {
                                    write_hint("idempotentHint", true);
                                }
                                if (destructive_non_default) {
                                    write_hint("destructiveHint", false);
                                }
                                if (open_world_non_default) {
                                    write_hint("openWorldHint", false);
                                }
                            }
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
    core::tools::register_builtin_tools(sm, core::tools::mcp_builtin_tool_options());
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
    bool client_supports_tasks = false;
};

struct ParsedPromptGetRequest {
    std::string name;
    std::string arguments{};
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

[[nodiscard]] std::string build_initialize_result(simdjson::ondemand::object& root,
                                                  const core::context::SessionContext& context) {
    const bool has_prompts = !discover_prompt_templates().empty();
    const InitializeInfo initialize = parse_initialize_info(root);
    JsonWriter rw(1024);
    {
        auto _res = rw.object();

        rw.kv_str("protocolVersion", initialize.protocol_version).comma()
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
            rw.comma().key("completions");
            { auto _completions = rw.object(); }
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
                .kv_str("version", core::version::value);
        }

        const auto& workspace = context.workspace_view();
        std::string dynamic_instructions = std::string(kServerInstructions);

        if (workspace.enforce()) {
            dynamic_instructions += "\n\nALLOWED WORKSPACE FOLDERS (strict enforcement enabled):\n";
            dynamic_instructions += "- Primary: " + workspace.primary().string() + "\n";
            for (const auto& add : workspace.additional()) {
                dynamic_instructions += "- Additional: " + add.string() + "\n";
            }
            const auto& scratch = workspace.scratch();
            if (!scratch.empty()) {
                dynamic_instructions +=
                    "\nSCRATCH DIRECTORIES (in scope for every filesystem tool):\n";
                for (const auto& scratch_root : scratch.readable_roots()) {
                    const bool writable = scratch.allows_write(scratch_root);
                    dynamic_instructions += "- " + scratch_root.string()
                        + (writable ? " (read/write)\n" : " (read-only)\n");
                }
            }
            dynamic_instructions += "\nUse filesystem tools for strict path-bounded operations. "
                                    "run_terminal_command remains open-world and can access paths "
                                    "outside these folders if the invoked command does so.";
        } else {
            dynamic_instructions += "\n\nWORKSPACE: Working directory is " + workspace.primary().string()
                + " (enforcement disabled; system-wide path access is allowed).";
        }
        rw.comma().kv_str("instructions", dynamic_instructions);
    }
    return std::move(rw).take();
}

[[nodiscard]] std::string build_discover_result() {
    const bool has_prompts = !discover_prompt_templates().empty();
    JsonWriter rw(1024);
    {
        auto result = rw.object();
        rw.key("supportedVersions");
        {
            auto versions = rw.array();
            for (std::size_t index = 0; index < kSupportedProtocolVersions.size(); ++index) {
                if (index != 0) rw.comma();
                rw.str(kSupportedProtocolVersions[index]);
            }
        }
        rw.comma().key("capabilities");
        {
            auto capabilities = rw.object();
            rw.key("tools");
            { auto tools = rw.object(); }
            rw.comma().key("resources");
            { auto resources = rw.object(); }
            rw.comma().key("completions");
            { auto completions = rw.object(); }
            if (has_prompts) {
                rw.comma().key("prompts");
                auto prompts = rw.object();
            }
            rw.comma().key("extensions");
            {
                auto extensions = rw.object();
                rw.key(kTasksExtensionIdentifier);
                auto tasks = rw.object();
            }
        }
        rw.comma().kv_str("instructions", kServerInstructions)
            .comma().kv_num("ttlMs", kStaticListTtlMs)
            .comma().kv_str("cacheScope", "private");
    }
    return std::move(rw).take();
}

[[nodiscard]] std::string build_resources_templates_list_result() {
    JsonWriter rw(128);
    {
        auto _res = rw.object();
        // Registration-fixed content: the single workspace file template whose
        // `path` argument is completed by completion/complete.
        rw.kv_num("ttlMs", kStaticListTtlMs).comma()
            .kv_str("cacheScope", "private").comma()
            .key("resourceTemplates");
        {
            auto _arr = rw.array();
            auto _item = rw.object();
            rw.kv_str("uriTemplate", kFileResourceTemplate).comma()
                .kv_str("name", "Workspace File").comma()
                .kv_str("description",
                        "Any file or directory visible inside the session workspace. "
                        "Complete the 'path' argument to search the workspace by name.");
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
        // Prompt skills are re-discovered from disk on each call; a short TTL
        // keeps client caches cheap without hiding skill edits for long.
        rw.kv_num("ttlMs", kDerivedListTtlMs).comma()
            .kv_str("cacheScope", "private").comma()
            .key("prompts");
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

[[nodiscard]] std::string build_resources_list_result(
    const core::context::SessionContext& context) {
    const auto& workspace = context.workspace_view();
    JsonWriter rw(2048);
    {
        auto _res = rw.object();
        // Workspace-derived (roots may change mid-connection): short TTL.
        rw.kv_num("ttlMs", kDerivedListTtlMs).comma()
            .kv_str("cacheScope", "private").comma()
            .key("resources");
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
            if (!workspace.primary().empty()) {
                add_resource(
                    workspace.primary(),
                    "Primary Workspace",
                    "The main working directory for this session.");
                first = false;
            }

            for (std::size_t i = 0; i < workspace.additional().size(); ++i) {
                if (!first) rw.comma();
                first = false;
                add_resource(
                    workspace.additional()[i],
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
    const std::filesystem::path& path,
    const core::context::SessionContext& context)
{
    std::vector<std::string> lines;
    lines.reserve(128);
    std::size_t listing_bytes = 0;
    bool listing_truncated = false;

    for (const auto& entry : core::workspace::collect_visible_directory_entries(
             path,
             context)) {
        std::error_code type_ec;
        const bool is_directory = entry.is_directory(type_ec);
        if (type_ec) continue;

        std::uintmax_t size = 0;
        if (!is_directory) {
            std::error_code size_ec;
            if (entry.is_regular_file(size_ec) && !size_ec) {
                size = entry.file_size(size_ec);
                if (size_ec) size = 0;
            }
        }

        std::string line = std::format(
            "{}\t{}\t{}",
            is_directory ? "dir" : "file",
            entry.path().filename().string(),
            size);
        if (lines.size() >= kMaxDirectoryEntries
            || listing_bytes + line.size() + 1 > kMaxDirectoryListingBytes) {
            listing_truncated = true;
            break;
        }

        listing_bytes += line.size() + 1;
        lines.push_back(std::move(line));
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
    const std::filesystem::path& path,
    const core::context::SessionContext& context)
{
    if (!context.allows_read(path)) {
        return std::unexpected(
            RpcError{-32001, "Access denied: path is outside the allowed workspace"});
    }

    auto visibility_context = context;
    if (!visibility_context.path_visibility) {
        const core::workspace::AgentIgnorePathVisibilityFactory visibility_factory;
        visibility_context.path_visibility =
            std::make_shared<core::workspace::PathVisibility>(
                visibility_factory.for_context(visibility_context));
    }
    if (const auto hidden_reason =
            visibility_context.path_visibility->hidden_reason(path.string(), path)) {
        return std::unexpected(RpcError{-32001, *hidden_reason});
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
        // File contents must always be served fresh: ttlMs 0 (immediately
        // stale) is intentional here, unlike the static catalog lists above.
        rw.kv_num("ttlMs", 0).comma()
            .kv_str("cacheScope", "private").comma()
            .key("contents");
        {
            auto _arr = rw.array();
            auto _item = rw.object();
            rw.kv_str("uri", path_to_file_uri(path)).comma();

            if (status.type() == std::filesystem::file_type::directory) {
                rw.kv_str("mimeType", "application/x-directory").comma();
                auto listing = build_directory_listing(path, visibility_context);
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

    simdjson::ondemand::object meta;
    if (params["_meta"].get_object().get(meta) == simdjson::SUCCESS) {
        simdjson::ondemand::object client_capabilities;
        if (meta["io.modelcontextprotocol/clientCapabilities"]
                .get_object()
                .get(client_capabilities) == simdjson::SUCCESS) {
            parsed.client_supports_tasks =
                has_tasks_extension_capability(client_capabilities);
        }
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

/// Reports the lifecycle of one inbound HTTP tool call to the remote-activity
/// hub. Guarantees exactly one terminal report — even when tool execution
/// throws — so the activity feed can never leak a phantom "running" entry.
/// A no-op for non-HTTP transports.
class RemoteToolCallReporter {
public:
    RemoteToolCallReporter(const core::context::SessionContext& context,
                           std::string_view tool_name,
                           std::string_view arguments_json) {
        if (context.transport != core::context::SessionTransport::mcp_http) return;
        activity_id_ = RemoteActivityHub::get_instance().tool_started(
            context.session_id,
            tool_name,
            arguments_json);
    }

    RemoteToolCallReporter(const RemoteToolCallReporter&) = delete;
    RemoteToolCallReporter& operator=(const RemoteToolCallReporter&) = delete;

    ~RemoteToolCallReporter() {
        if (activity_id_ == 0 || reported_) return;
        RemoteActivityHub::get_instance().tool_finished(
            activity_id_,
            R"({"error":"Tool execution terminated unexpectedly."})",
            true);
    }

    void finish(std::string_view result, bool failed) {
        if (activity_id_ == 0 || reported_) return;
        reported_ = true;
        RemoteActivityHub::get_instance().tool_finished(activity_id_, result, failed);
    }

    void fail(std::string_view message) {
        core::utils::JsonWriter writer(message.size() + 32);
        {
            auto error = writer.object();
            writer.kv_str("error", message);
        }
        finish(std::move(writer).take(), true);
    }

private:
    std::uint64_t activity_id_ = 0;
    bool reported_ = false;
};

[[nodiscard]] RpcExpected<std::string> build_tool_call_result(
    const ParsedToolCallRequest& request,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    auto& sm = core::tools::ToolManager::get_instance();
    auto tool_def = sm.get_tool_definition(request.name);
    if (!tool_def.has_value()) {
        return std::unexpected(
            RpcError{-32602, std::format("Unknown tool: {}", request.name)});
    }

    RemoteToolCallReporter remote_reporter(
        context, request.name, request.arguments_json);
    if (auto validation_error = validate_tool_arguments(*tool_def, request.arguments_json)) {
        remote_reporter.fail(*validation_error);
        return std::unexpected(RpcError{-32602, *validation_error});
    }

    auto execution_context = context;
    const bool modern = mode == McpProtocolMode::stateless;
    if (modern) {
        static std::atomic<uint64_t> next_request_scope{1};
        execution_context.session_id = std::format(
            "modern-request-{}",
            next_request_scope.fetch_add(1, std::memory_order_relaxed));
    }
    const std::string tool_result = sm.execute_tool(
        request.name,
        request.arguments_json,
        execution_context);
    if (modern) {
        sm.clear_session_state(execution_context.session_id);
    }
    const ToolCallResultClassification classification =
        classify_tool_call_payload(tool_result);
    remote_reporter.finish(tool_result, classification.is_error);
    return build_call_tool_result_from_payload(tool_result, classification);
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

[[nodiscard]] std::string build_task_state_json(
    const core::mcp::McpTaskManager::TaskState& task,
    const std::optional<core::mcp::McpTaskManager::ResultPayload>& payload = std::nullopt,
    std::optional<std::string_view> result_type = std::nullopt) {
    JsonWriter writer(512 + (payload.has_value() ? payload->result_json.size() + payload->error_message.size() : 0));
    {
        auto root = writer.object();
        if (result_type.has_value()) {
            writer.kv_str("resultType", *result_type).comma();
        }
        writer.kv_str("taskId", task.task_id).comma();
        writer.kv_str("status", task.status).comma();
        writer.kv_str("statusMessage", task.status_message).comma();
        writer.kv_str("createdAt", task.created_at).comma();
        writer.kv_str("lastUpdatedAt", task.last_updated_at);
        if (task.ttl_ms.has_value()) writer.comma().kv_num("ttlMs", *task.ttl_ms);
        else writer.comma().key("ttlMs").null_val();
        if (task.poll_interval_ms.has_value()) {
            writer.comma().kv_num("pollIntervalMs", *task.poll_interval_ms);
        }
        if (payload.has_value()) {
            if (task.status == "completed" && !payload->has_error) {
                writer.comma().kv_raw("result", payload->result_json);
            } else if (task.status == "failed") {
                writer.comma().key("error");
                {
                    auto error = writer.object();
                    writer.kv_num("code", payload->error_code).comma()
                        .kv_str("message", payload->error_message);
                }
            }
        }
    }
    return std::move(writer).take();
}

/// Long-running delegations are the ones worth wrapping in an MCP task: `start`
/// and `resume` hand work to a background worker, while `status`, `list`, and
/// `cancel` answer immediately and should stay plain tool calls.
[[nodiscard]] bool is_long_running_delegate_task_call(const ParsedToolCallRequest& request) {
    if (request.name != core::tools::TaskTool::kToolName) return false;

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(request.arguments_json).get(doc) != simdjson::SUCCESS) return false;

    std::string_view action;
    if (doc["action"].get(action) != simdjson::SUCCESS) return false;

    return action == "start" || action == "resume";
}

[[nodiscard]] std::string build_create_task_result(
    const core::mcp::McpTaskManager::CreateResult& create_result)
{
    return build_task_state_json(create_result.task, std::nullopt, "task");
}

[[nodiscard]] RpcExpected<std::string> parse_task_id_param(
    simdjson::ondemand::object& root)
{
    simdjson::ondemand::object params;
    if (auto params_result = parse_params_object(root, params); !params_result) {
        return std::unexpected(params_result.error());
    }

    std::string_view task_id;
    if (params["taskId"].get_string().get(task_id) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'taskId'"});
    }
    return std::string(task_id);
}

/// Per-root mention indexes shared by every completion request.
///
/// The dispatcher is a stateless free-function router, so the cache is owned by
/// this accessor rather than threaded through every handler signature. Its TTL
/// matches the one advertised for workspace-derived lists.
[[nodiscard]] core::workspace::MentionIndexCache& mention_index_cache() {
    static core::workspace::MentionIndexCache cache{std::chrono::milliseconds(kDerivedListTtlMs)};
    return cache;
}

/// Partial `path` value a client wants completed for the workspace file template.
[[nodiscard]] RpcExpected<std::string> parse_completion_query(
    simdjson::ondemand::object& root)
{
    simdjson::ondemand::object params;
    if (auto params_result = parse_params_object(root, params); !params_result) {
        return std::unexpected(params_result.error());
    }

    simdjson::ondemand::object ref;
    if (params["ref"].get_object().get(ref) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'ref'"});
    }

    std::string_view ref_type;
    if (ref["type"].get_string().get(ref_type) != simdjson::SUCCESS
        || ref_type != "ref/resource") {
        return std::unexpected(RpcError{-32602, "Invalid params: only 'ref/resource' is supported"});
    }

    std::string_view ref_uri;
    if (ref["uri"].get_string().get(ref_uri) != simdjson::SUCCESS
        || ref_uri != kFileResourceTemplate) {
        return std::unexpected(RpcError{-32602, "Invalid params: unknown resource template"});
    }

    simdjson::ondemand::object argument;
    if (params["argument"].get_object().get(argument) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'argument'"});
    }

    std::string_view argument_name;
    if (argument["name"].get_string().get(argument_name) != simdjson::SUCCESS
        || argument_name != kFileResourceTemplateArgument) {
        return std::unexpected(RpcError{-32602, "Invalid params: unknown template argument"});
    }

    std::string_view argument_value;
    if (argument["value"].get_string().get(argument_value) != simdjson::SUCCESS) {
        return std::unexpected(RpcError{-32602, "Invalid params: missing 'argument.value'"});
    }
    return std::string(argument_value);
}

/// Completion values are substituted into `file:///{+path}`, so each one is an
/// absolute path with its leading separator removed.
[[nodiscard]] std::string to_template_value(const std::filesystem::path& root,
                                            std::string_view relative_path) {
    std::string value = root.generic_string();
    if (!value.empty() && value.back() == '/') value.pop_back();
    value += '/';
    value += relative_path;
    if (!value.empty() && value.front() == '/') value.erase(value.begin());
    return value;
}

[[nodiscard]] std::string build_completion_result(
    std::string_view query,
    const core::context::SessionContext& context)
{
    const auto& workspace = context.workspace_view();
    std::vector<std::string> values;
    values.reserve(kMaxCompletionValues);

    // A completion result is a set of candidates. The primary root is commonly
    // repeated in the additional roots, and roots may nest, so both the roots
    // and the values they produce are de-duplicated.
    std::unordered_set<std::string> seen_roots;
    std::unordered_set<std::string> seen_values;

    auto append_root = [&](const std::filesystem::path& root) {
        if (root.empty() || values.size() >= kMaxCompletionValues) return;
        if (!seen_roots.insert(root.generic_string()).second) return;

        const auto index = mention_index_cache().index_for(root);
        for (const auto& suggestion :
             core::workspace::search_mention_index(*index, query, kMaxCompletionValues - values.size())) {
            auto value = to_template_value(root, suggestion.display_path);
            if (!seen_values.insert(value).second) continue;
            values.push_back(std::move(value));
        }
    };

    append_root(workspace.primary());
    for (const auto& additional : workspace.additional()) {
        append_root(additional);
    }

    JsonWriter rw(2048);
    {
        auto _res = rw.object();
        rw.key("completion");
        auto _completion = rw.object();
        rw.key("values");
        {
            auto _arr = rw.array();
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i != 0) rw.comma();
                rw.str(values[i]);
            }
        }
        // `total` is omitted: the index is truncated per root while ranking, so
        // an exact match count is not available without a second pass.
        rw.comma().kv_bool("hasMore", values.size() >= kMaxCompletionValues);
    }
    return std::move(rw).take();
}

[[nodiscard]] bool request_supports_tasks(simdjson::ondemand::object& root) {
    simdjson::ondemand::object params;
    if (root["params"].get_object().get(params) != simdjson::SUCCESS) return false;
    simdjson::ondemand::object meta;
    if (params["_meta"].get_object().get(meta) != simdjson::SUCCESS) return false;
    simdjson::ondemand::object capabilities;
    if (meta["io.modelcontextprotocol/clientCapabilities"]
            .get_object().get(capabilities) != simdjson::SUCCESS) return false;
    return has_tasks_extension_capability(capabilities);
}

using MethodHandler = std::string (*)(
    const RequestId&,
    simdjson::ondemand::object&,
    const core::context::SessionContext&,
    McpProtocolMode);

[[nodiscard]] std::string handle_initialize(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    return make_response(id, build_initialize_result(root, context), mode);
}

[[nodiscard]] std::string handle_server_discover(
    const RequestId& id,
    simdjson::ondemand::object&,
    const core::context::SessionContext&,
    McpProtocolMode mode)
{
    return make_response(id, build_discover_result(), mode);
}

[[nodiscard]] std::string handle_resources_templates_list(
    const RequestId& id,
    simdjson::ondemand::object&,
    const core::context::SessionContext&,
    McpProtocolMode mode)
{
    return make_response(id, build_resources_templates_list_result(), mode);
}

[[nodiscard]] std::string handle_prompts_list(
    const RequestId& id,
    simdjson::ondemand::object&,
    const core::context::SessionContext&,
    McpProtocolMode mode)
{
    return make_response(id, build_prompts_list_result(discover_prompt_templates()), mode);
}

[[nodiscard]] std::string handle_prompts_get(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext&,
    McpProtocolMode mode)
{
    auto request = parse_prompt_get_request(root);
    if (!request) return make_rpc_error(id, request.error());

    auto result = build_prompt_get_result(*request);
    if (!result) return make_rpc_error(id, result.error());

    return make_response(id, *result, mode);
}

[[nodiscard]] std::string handle_resources_list(
    const RequestId& id,
    simdjson::ondemand::object&,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    return make_response(id, build_resources_list_result(context), mode);
}

[[nodiscard]] std::string handle_resources_read(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    auto path = parse_resources_read_path(root);
    if (!path) return make_rpc_error(id, path.error());

    auto result = build_resources_read_result(*path, context);
    if (!result) return make_rpc_error(id, result.error());

    return make_response(id, *result, mode);
}

[[nodiscard]] std::string handle_completion_complete(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    auto query = parse_completion_query(root);
    if (!query) return make_rpc_error(id, query.error());

    return make_response(id, build_completion_result(*query, context), mode);
}

[[nodiscard]] std::string handle_tools_list(
    const RequestId& id,
    simdjson::ondemand::object&,
    const core::context::SessionContext&,
    McpProtocolMode mode)
{
    return make_response(id, tools_list_result(), mode);
}

[[nodiscard]] std::string handle_tools_call(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    auto request = parse_tool_call_request(root);
    if (!request) return make_rpc_error(id, request.error());

    auto& sm = core::tools::ToolManager::get_instance();
    auto tool_def = sm.get_tool_definition(request->name);
    if (!tool_def.has_value()) {
        return make_error(id, -32602, std::format("Unknown tool: {}", request->name));
    }

    // Task augmentation is negotiated independently on each stateless request.
    if (request->client_supports_tasks
        && mode == McpProtocolMode::stateless
        && is_long_running_delegate_task_call(*request)) {
        if (auto validation_error = validate_tool_arguments(*tool_def, request->arguments_json)) {
            RemoteToolCallReporter remote_reporter(
                context, request->name, request->arguments_json);
            remote_reporter.fail(*validation_error);
            return make_error(id, -32602, *validation_error);
        }

        const auto create_result = McpTaskManager::get_instance().create_task_tool_call(
            request->name,
            request->arguments_json,
            context,
            std::nullopt);
        return make_response(id, build_create_task_result(create_result), mode);
    }

    auto result = build_tool_call_result(*request, context, mode);
    if (!result) return make_rpc_error(id, result.error());

    return make_response(id, *result, mode);
}

[[nodiscard]] std::string handle_tasks_get(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    if (mode != McpProtocolMode::stateless) {
        return make_error(id, -32601, "Task methods are not available for the negotiated MCP protocol.");
    }
    if (!request_supports_tasks(root)) {
        return make_missing_task_capability_error(id);
    }

    auto task_id = parse_task_id_param(root);
    if (!task_id) return make_rpc_error(id, task_id.error());

    const auto snapshot = McpTaskManager::get_instance().get(*task_id, context.session_id);
    if (!snapshot.has_value()) {
        return make_error(id, -32602, std::format("Task not found: {}", *task_id));
    }
    return make_response(
        id,
        build_task_state_json(snapshot->task, snapshot->result),
        mode);
}

[[nodiscard]] std::string handle_tasks_cancel(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    if (mode != McpProtocolMode::stateless) {
        return make_error(id, -32601, "Task methods are not available for the negotiated MCP protocol.");
    }
    if (!request_supports_tasks(root)) {
        return make_missing_task_capability_error(id);
    }

    auto task_id = parse_task_id_param(root);
    if (!task_id) return make_rpc_error(id, task_id.error());

    const auto task = McpTaskManager::get_instance().cancel(*task_id, context.session_id);
    if (!task) {
        switch (task.error()) {
        case McpTaskManager::CancelError::not_found:
            return make_error(id, -32602, std::format("Task not found: {}", *task_id));
        case McpTaskManager::CancelError::already_terminal:
            return make_response(id, "{}", mode);
        }
    }
    return make_response(id, "{}", mode);
}

[[nodiscard]] std::string handle_tasks_update(
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    if (mode != McpProtocolMode::stateless) {
        return make_error(id, -32601, "Task methods are not available for the negotiated MCP protocol.");
    }
    if (!request_supports_tasks(root)) {
        return make_missing_task_capability_error(id);
    }

    auto task_id = parse_task_id_param(root);
    if (!task_id) return make_rpc_error(id, task_id.error());

    const auto snapshot = McpTaskManager::get_instance().get(*task_id, context.session_id);
    if (!snapshot.has_value()) {
        return make_error(id, -32602, std::format("Task not found: {}", *task_id));
    }
    return make_response(id, "{}", mode);
}

[[nodiscard]] std::string handle_ping(
    const RequestId& id,
    simdjson::ondemand::object&,
    const core::context::SessionContext&,
    McpProtocolMode mode)
{
    return make_response(id, "{}", mode);
}

struct MethodRoute {
    std::string_view name;
    MethodHandler handler;
};

constexpr std::array<MethodRoute, 14> kMethodRoutes{{
    {"server/discover", &handle_server_discover},
    {"initialize", &handle_initialize},
    {"prompts/list", &handle_prompts_list},
    {"prompts/get", &handle_prompts_get},
    {"resources/templates/list", &handle_resources_templates_list},
    {"resources/list", &handle_resources_list},
    {"resources/read", &handle_resources_read},
    {"completion/complete", &handle_completion_complete},
    {"tools/list", &handle_tools_list},
    {"tools/call", &handle_tools_call},
    {"tasks/get", &handle_tasks_get},
    {"tasks/update", &handle_tasks_update},
    {"tasks/cancel", &handle_tasks_cancel},
    {"ping", &handle_ping},
}};

[[nodiscard]] std::string route_method(
    std::string_view method,
    const RequestId& id,
    simdjson::ondemand::object& root,
    const core::context::SessionContext& context,
    McpProtocolMode mode)
{
    const auto it = std::ranges::find(kMethodRoutes, method, &MethodRoute::name);
    if (it == kMethodRoutes.end()) {
        return make_error(id, -32601, "Method not found");
    }
    return it->handler(id, root, context, mode);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

std::string McpDispatcher::dispatch(const std::string& json_request,
                                   const core::context::SessionContext& context,
                                   McpProtocolMode mode) {
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

    return route_method(method, id, root, context, mode);
}

} // namespace core::mcp
