#pragma once

#include <string>

namespace core::mcp {

/**
 * @brief Routes a single MCP 2025-11-25 JSON-RPC message to the correct handler
 *        and returns the JSON-RPC response string.
 *
 * @c McpDispatcher is the single source of truth for the MCP protocol logic.
 * Both the stdio transport (@c exec::mcp::run_server in Server.cpp) and the
 * HTTP Streamable-HTTP transport (@c exec::daemon::run_server in Daemon.cpp)
 * delegate all protocol work to this class so there is exactly one implementation.
 *
 * ### Supported methods (MCP 2025-11-25)
 *
 * | Method                        | Role        | Notes                                              |
 * |-------------------------------|-------------|----------------------------------------------------|
 * | @c initialize                 | Request     | Negotiates protocol version; returns capabilities  |
 * | @c tools/list                 | Request     | Returns all registered tools (cached, no cursor)   |
 * | @c tools/call                 | Request     | Validates args, executes tool, wraps result        |
 * | @c ping                       | Request     | Returns @c {}                                      |
 * | @c notifications/initialized  | Notification| Silently ignored (no response per spec)            |
 * | @c notifications/cancelled    | Notification| Silently ignored (sync execution, nothing to cancel)|
 * | Any other notification        | Notification| Silently ignored                                   |
 *
 * ### tools/call result envelope
 *
 * @code{.json}
 * {
 *   "content": [{"type": "text", "text": "<escaped tool JSON>"}],
 *   "isError": false,
 *   "structuredContent": { <raw tool JSON — present on successful results> }
 * }
 * @endcode
 *
 * ### Thread safety
 * @c dispatch() is thread-safe.  Each call creates all temporaries on the
 * stack (including a per-call simdjson ondemand parser).  The @c ToolManager
 * map is populated once in the constructor and is read-only thereafter, so
 * multiple HTTP handler threads may call @c dispatch() concurrently without
 * synchronisation.
 *
 * ### Error codes
 *
 * | Code    | Meaning                                       |
 * |---------|-----------------------------------------------|
 * | -32700  | Parse error — body is not valid JSON          |
 * | -32600  | Invalid Request — wrong jsonrpc version, etc. |
 * | -32601  | Method not found                              |
 * | -32602  | Invalid params — bad tool name / args         |
 */
class McpDispatcher {
public:
    /**
     * @brief Returns the process-wide singleton.
     *
     * Thread-safe on first call (C++11 magic-static guarantee).  Registers
     * all built-in tools via @c ToolManager on first construction.
     */
    static McpDispatcher& get_instance();

    /**
     * @brief Dispatches one JSON-RPC request line or HTTP body.
     *
     * @param json_request  A complete JSON-RPC 2.0 message as a UTF-8 string.
     *                      Must be a single JSON object (not a batch array —
     *                      MCP does not use JSON-RPC batch).
     *
     * @return The JSON-RPC response string, or an @em empty string for
     *         notifications (messages without an @c id field).  Callers must
     *         @em not send an empty response to the client — silently drop it.
     */
    std::string dispatch(const std::string& json_request);

private:
    McpDispatcher();

    /// Registers all built-in tool instances into the @c ToolManager.
    void register_tools();
};

} // namespace core::mcp
