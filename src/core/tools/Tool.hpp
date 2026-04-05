#pragma once

#include "../context/SessionContext.hpp"

#include <string>
#include <vector>

namespace core::tools {

/**
 * @brief Describes a single parameter accepted by a tool.
 *
 * Maps directly to one entry in the JSON Schema @c properties object returned
 * by @c tools/list.  The @c type field must be a JSON Schema primitive type
 * token: @c "string", @c "boolean", @c "integer", @c "number", @c "object",
 * or @c "array".
 */
struct ToolParameter {
    std::string name;         ///< Parameter key — must be a valid JSON object key
    std::string type;         ///< JSON Schema type token (e.g. @c "string")
    std::string description;  ///< Human-readable description shown to the LLM / UI
    bool        required = false; ///< If true, included in the JSON Schema @c required array
    /// Optional full JSON Schema object for this parameter. If set, serializers
    /// should emit it verbatim and ignore the convenience fields below.
    std::string schema = {};
    /// Raw JSON Schema object for array element type (only meaningful when @c type == "array").
    /// Emitted verbatim as @c "items": <items_schema> in the serialized schema.
    /// Leave empty for untyped arrays or non-array parameters.
    std::string items_schema = {};
};

/**
 * @brief Behavioral hints advertised to MCP clients via the @c annotations field.
 *
 * These flags correspond to the optional @c annotations object in the MCP
 * 2025-11-25 specification (section §Tools).  Clients such as Lampo can use
 * them to render appropriate UI affordances — e.g. a destructive badge, a
 * read-safe icon, or an idempotent retry button — without executing the tool
 * first.
 *
 * All hints default to @c false.  Only set a hint to @c true when you are
 * confident the property holds for @em every invocation of the tool.
 */
struct ToolAnnotations {
    /// Tool never writes to disk, never executes processes, and never modifies
    /// any external state.  Safe to call without user confirmation.
    bool read_only_hint = false;

    /// Tool may irreversibly delete or overwrite data.  Clients should surface
    /// a confirmation prompt or highlight the action in their UI.
    bool destructive_hint = false;

    /// Calling the tool multiple times with identical arguments produces the
    /// same observable result.  Safe for clients to retry on transient failure.
    bool idempotent_hint = false;

    /// Tool reaches outside the local filesystem — e.g. network calls, spawns
    /// arbitrary processes, or interacts with external services.
    bool open_world_hint = false;
};

/**
 * @brief Complete descriptor for one tool, returned by @c Tool::get_definition().
 *
 * The dispatcher serialises this struct into the JSON Schema object emitted by
 * @c tools/list.  Fields map to the MCP 2025-11-25 tool object as follows:
 *
 * | C++ field    | MCP field      | Required |
 * |--------------|----------------|----------|
 * | name         | name           | yes      |
 * | title        | title          | no       |
 * | description  | description    | no       |
 * | parameters   | inputSchema    | yes      |
 * | input_schema | inputSchema    | no       |
 * | output_schema| outputSchema   | no       |
 * | annotations  | annotations    | no       |
 *
 * @note If @c input_schema is empty, serializers construct @c inputSchema from
 *       @c parameters. Tools with complex nested arguments may provide a raw
 *       @c input_schema instead. @c output_schema describes the successful
 *       structure returned in @c structuredContent.
 */
struct ToolDefinition {
    std::string name = {};         ///< Programmatic identifier used in @c tools/call (e.g. @c "read_file")
    std::string title = {};        ///< Human-readable display name shown in Lampo's tool list (e.g. @c "Read File")
    std::string description = {};  ///< Prose description shown to the LLM to guide tool selection
    std::vector<ToolParameter> parameters = {};  ///< Ordered list of accepted parameters
    std::string input_schema = {}; ///< Optional raw JSON Schema object overriding parameter-derived inputSchema
    std::string output_schema = {}; ///< Optional raw JSON Schema object for successful structuredContent
    ToolAnnotations annotations = {};            ///< Behavioral hints for MCP clients (all default to @c false)
};

/**
 * @brief Abstract base class for all filo MCP tools.
 *
 * Every concrete tool must implement two pure-virtual methods:
 *
 *  - @c get_definition() — returns a @c ToolDefinition that the MCP dispatcher
 *    serialises into the @c tools/list response.
 *
 *  - @c execute() — performs the actual work for the supplied
 *    @c SessionContext and returns a JSON string result that is embedded in
 *    the @c tools/call response.
 *
 * ### Tool result contract
 *
 * @c execute() must return a valid JSON object string (@c '{' … @c '}').
 * The top-level key signals success or failure:
 *
 * @code{.json}
 * // Success — first key is NOT "error"
 * {"output": "hello\n", "exit_code": 0}
 *
 * // Failure — first key MUST be "error" (dispatcher relies on this prefix)
 * {"error": "file not found: /tmp/x.txt"}
 * @endcode
 *
 * The dispatcher uses a cheap prefix check to set @c isError in the MCP
 * envelope.  Therefore: @em every error result must begin with @c {"error"
 * (no leading space).  Do not re-order a failed result's keys.
 */
class Tool {
public:
    virtual ~Tool() = default;

    /**
     * @brief Returns the tool's schema and metadata.
     *
     * Called once at startup by @c McpDispatcher::register_tools() to populate
     * the @c ToolManager.  The result is cached; this method is not called
     * again during normal operation.
     *
     * @return A @c ToolDefinition describing the tool's name, description,
     *         parameters, and behavioral annotations.
     */
    virtual ToolDefinition get_definition() const = 0;

    /**
     * @brief Executes the tool with the supplied arguments and session context.
     *
     * @param json_args  A JSON object string containing the arguments passed
     *                   by the LLM/client in the @c tools/call request.
     *                   May be @c "{}" if the client omitted @c arguments.
     * @param context    Per-call execution context. The caller is responsible
     *                   for choosing the authoritative workspace and session
     *                   identity up front and passing it explicitly.
     *
     * @return A JSON object string.  On success the first key must @em not be
     *         @c "error".  On failure the first key must be @c "error".
     *         See the class-level documentation for the full contract.
     */
    virtual std::string execute(const std::string& json_args, const core::context::SessionContext& context) = 0;
};

} // namespace core::tools
