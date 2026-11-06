#pragma once

#include "../tools/Tool.hpp"
#include "../tools/ToolManager.hpp"
#include "../config/ConfigManager.hpp"
#include "McpClientSession.hpp"
#include "McpServerStatus.hpp"
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace core::llm {
class LLMProvider;
}

namespace core::mcp {

// ---------------------------------------------------------------------------
// McpConnectionManager — connects the configured MCP servers and registers
// their tools in a ToolManager.
//
// MCP servers are accessories: no handshake, OAuth refresh, or teardown ever
// runs on the caller's thread. Every server connects on its own background
// thread and its tools appear in the ToolManager the moment its handshake
// completes, so a slow or dead server delays neither the caller nor the other
// servers. Consumers that read the ToolManager per request (the agent, the
// MCP dispatcher, subagents) pick late tools up on their next request.
//
// Each connect starts a new *round*. A newer round, or shutdown_all(),
// supersedes the previous one: a superseded handshake that completes later is
// shut down on its own thread and never registers anything.
//
// Guarantees: every server of a round settles exactly once (connected or
// failed), whatever throws, so wait_until_settled() always returns; on a
// tool-name clash between servers the one later in the configuration wins,
// whichever finishes first; an MCP tool never replaces a built-in or skill
// tool, so a late handshake cannot change what an existing tool name does.
//
// Singleton.
// ---------------------------------------------------------------------------
class McpConnectionManager {
public:
    using ChangeListener = std::function<void()>;

    static constexpr std::chrono::milliseconds kDefaultShutdownGrace{1000};

    static McpConnectionManager& get_instance() noexcept {
        static McpConnectionManager instance;
        return instance;
    }

    ~McpConnectionManager();
    McpConnectionManager(const McpConnectionManager&) = delete;
    McpConnectionManager& operator=(const McpConnectionManager&) = delete;

    // Start connecting every server in config.mcp_servers and return at once.
    // Replaces the previous set: its tools are unregistered before this
    // returns and its sessions are shut down in the background. Every
    // configured server is published as connecting before this returns, so
    // configured_count() and server_statuses() are complete immediately.
    void connect_all_async(const core::config::AppConfig& config,
                           core::tools::ToolManager& tool_manager,
                           std::shared_ptr<core::llm::LLMProvider> sampling_provider = nullptr,
                           std::string sampling_model = {});

    // connect_all_async() followed by wait_until_settled(), for callers that
    // need the result before they continue (tests, one-shot tools). Never call
    // it on a UI thread.
    void connect_all(const core::config::AppConfig& config,
                     core::tools::ToolManager& tool_manager,
                     std::shared_ptr<core::llm::LLMProvider> sampling_provider = nullptr,
                     std::string sampling_model = {});

    // Wait until every server of the latest round has connected or failed.
    // Returns false when @p timeout elapsed first. The default waits as long
    // as the servers' own request timeouts allow.
    bool wait_until_settled(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) const;

    // Called on a background thread after any server's status changed. Keep
    // it cheap and non-blocking, and do not call set_change_listener() from
    // it. Replacing or clearing the listener waits for a running call to
    // return, so whatever the old one captured may be destroyed afterwards.
    void set_change_listener(ChangeListener listener);

    // Update the sampling backend used by connected and connecting sessions.
    void update_sampling_backend(std::shared_ptr<core::llm::LLMProvider> sampling_provider,
                                 std::string sampling_model = {});

    // Disconnect every server and unregister its tools. Session teardown runs
    // in the background; this waits for it, and for superseded handshakes, at
    // most @p grace, so a dead server can never hold up the process exit.
    void shutdown_all(std::chrono::milliseconds grace = kDefaultShutdownGrace) noexcept;

    // Servers whose handshake has completed in the latest round.
    [[nodiscard]] int connected_count() const noexcept;

    // Servers in the latest round, whatever their state.
    [[nodiscard]] std::size_t configured_count() const noexcept;

    // Registered MCP tool names as "server/tool" (for diagnostics / /help).
    [[nodiscard]] std::vector<std::string> registered_tool_names() const;

    // One entry per configured server, in configuration order, including
    // servers that are still connecting or failed.
    [[nodiscard]] std::vector<McpServerStatus> server_statuses() const;

private:
    McpConnectionManager();

    // Shared with the background threads, which outlive any one call and may
    // outlive this singleton during process teardown.
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace core::mcp
