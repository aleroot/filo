#include "McpConnectionManager.hpp"
#include "McpDynamicTool.hpp"
#include "McpDynamicFeatureTools.hpp"
#include "McpOAuth.hpp"
#include "McpSamplingBridge.hpp"
#include "../config/ConfigManager.hpp"
#include "../config/ConfigSource.hpp"
#include "../logging/Logger.hpp"
#include "../llm/LLMProvider.hpp"
#include "../utils/DetachedWorkTracker.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <format>
#include <mutex>
#include <ranges>
#include <string_view>
#include <thread>
#include <unordered_map>

namespace core::mcp {

namespace {

using Clock = std::chrono::steady_clock;
using SessionPtr = std::shared_ptr<IMcpClientSession>;
using ToolPtr = std::shared_ptr<core::tools::Tool>;
using ToolList = std::vector<ToolPtr>;

/// Every tool an MCP server contributes describes itself with this prefix
/// (McpDynamicTool, McpDynamicFeatureTools); it tells them apart from the
/// built-in and skill tools sharing the ToolManager.
constexpr std::string_view kMcpToolDescriptionPrefix = "[MCP:";

[[nodiscard]] bool is_mcp_tool(const core::tools::ToolManager& tool_manager, const std::string& name) {
    const auto definition = tool_manager.get_tool_definition(name);
    return definition && definition->description.starts_with(kMcpToolDescriptionPrefix);
}

/// Everything one connection attempt produced. A session is present when the
/// handshake completed; an error, when the attempt failed at any point after
/// which the session (if any) must be shut down rather than adopted.
struct Attempt {
    SessionPtr session;
    std::vector<McpToolDef> tool_defs;
    ToolList tools;  // feature tools included
    std::shared_ptr<McpServerActivity> activity;
    std::string error;
    std::chrono::milliseconds duration{0};

    [[nodiscard]] bool succeeded() const noexcept { return session && error.empty(); }
};

/// A family of helper tools a server gets when it supports one capability.
/// Every member is built from the same (session, server name) pair.
template <class... FeatureTools>
struct FeatureToolSet {
    static constexpr std::size_t size = sizeof...(FeatureTools);

    static void append_to(ToolList& tools, const SessionPtr& session, const std::string& server_name) {
        (tools.push_back(std::make_shared<FeatureTools>(session, server_name)), ...);
    }
};

using ResourceFeatureTools =
    FeatureToolSet<McpListResourcesTool, McpListResourceTemplatesTool, McpReadResourceTool>;
using PromptFeatureTools = FeatureToolSet<McpListPromptsTool, McpGetPromptTool>;

/// The tools a connected server contributes: one per advertised tool, plus
/// the resource and prompt helpers when it supports those features.
ToolList make_server_tools(const SessionPtr& session,
                           const std::vector<McpToolDef>& tool_defs,
                           const std::string& server_name,
                           const std::shared_ptr<McpServerActivity>& activity) {
    ToolList tools;
    tools.reserve(tool_defs.size() + ResourceFeatureTools::size + PromptFeatureTools::size);
    tools.append_range(tool_defs | std::views::transform([&](const McpToolDef& tool_def) -> ToolPtr {
        return std::make_shared<McpDynamicTool>(session, tool_def, server_name, activity);
    }));
    if (session->supports_resources()) ResourceFeatureTools::append_to(tools, session, server_name);
    if (session->supports_prompts()) PromptFeatureTools::append_to(tools, session, server_name);
    return tools;
}

/// The OAuth token operation for an HTTP server whose configuration came from
/// a known source, including the forced refresh after a server's 401.
/// Empty for every other server.
McpHttpAuth::TokenLoader make_token_loader(const core::config::McpServerConfig& config) {
    const auto source = config.config_source;
    if (!source || config.transport != "http") return {};
    return [source, name = config.name](bool force_refresh) {
        const auto directory = source->credential_directory(mcp_oauth_provider_id(name)).string();
        return force_refresh ? force_refresh_mcp_oauth_token(name, directory)
                             : load_valid_mcp_oauth_token(name, directory);
    };
}

/// Compose auth and transport for one server and run its handshake. Blocks
/// for as long as the server takes, so it only ever runs on a background
/// thread. Never throws.
Attempt attempt_connection(const core::config::McpServerConfig& config,
                           const McpSamplingCallback& sampling_callback,
                           const std::string& config_dir) noexcept {
    const auto started = Clock::now();
    Attempt attempt;
    try {
        auto http_auth = make_mcp_http_auth(config, config_dir, make_token_loader(config));
        SessionPtr session = make_mcp_session(config, sampling_callback, config_dir, std::move(http_auth));
        attempt.tool_defs = session->initialize();
        attempt.session = std::move(session);
        attempt.activity = std::make_shared<McpServerActivity>();
        attempt.tools = make_server_tools(attempt.session, attempt.tool_defs, config.name, attempt.activity);
    } catch (const std::exception& e) {
        attempt.error = e.what();
    } catch (...) {
        attempt.error = "unknown error";
    }
    if (attempt.error.empty() && !attempt.session) attempt.error = "the server produced no session";
    attempt.duration = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started);
    return attempt;
}

McpServerStatus connecting_status(const core::config::McpServerConfig& config) {
    return McpServerStatus{
        .name = config.name,
        .transport = config.transport,
        .endpoint = describe_mcp_endpoint(config),
        .auth = describe_mcp_auth(config),
    };
}

void mark_connected(McpServerStatus& status, const Attempt& attempt) {
    const auto& session = *attempt.session;
    status.state = McpServerState::connected;
    status.server_info = session.server_info();
    status.tools = attempt.tool_defs | std::views::transform(&McpToolDef::name)
        | std::ranges::to<std::vector<std::string>>();
    status.resources = session.supports_resources();
    status.prompts = session.supports_prompts();
    status.sampling = session.supports_sampling();
    status.connected_at = std::chrono::system_clock::now();
}

void mark_failed(McpServerStatus& status, std::string error, bool needs_login) noexcept {
    status.state = McpServerState::failed;
    status.error = std::move(error);
    status.needs_login = needs_login;
}

/// Remove @p names from @p tool_manager, sparing any that something other
/// than an MCP server has registered under the same name since.
void unregister_mcp_tools(core::tools::ToolManager& tool_manager, const std::vector<std::string>& names) {
    const auto owned = names
        | std::views::filter([&](const std::string& name) { return is_mcp_tool(tool_manager, name); })
        | std::ranges::to<std::vector<std::string>>();
    if (!owned.empty()) tool_manager.unregister_tools(owned);
}

} // namespace

// ---------------------------------------------------------------------------
// State — shared by the manager and every background thread it starts, so a
// thread that outlives a call (or the singleton itself) never dangles.
//
// Invariants, all guarded by `mutex`:
//   * every server of the current round settles exactly once: `pending`
//     drops by one per server, whatever happened, so waiters always wake;
//   * `tool_owner` maps each registered MCP tool name to the configuration
//     index of the server that registered it; `servers[i].tool_names` is the
//     inverse. A name absent from `tool_owner` is never touched;
//   * an MCP tool never replaces a non-MCP tool (built-in, skill).
//
// Lock order: listener_mutex is never taken while mutex is held, and nothing
// that can call back into the UI (logging, the listener) runs under mutex.
// The ToolManager's own lock is a leaf and may be taken under mutex.
// ---------------------------------------------------------------------------
struct McpConnectionManager::State : std::enable_shared_from_this<State> {
    struct ServerEntry {
        std::string name;
        std::size_t index = 0;  // position in the round's configuration
        SessionPtr session;
        std::vector<std::string> tool_names;  // names this server owns
    };

    struct StatusEntry {
        McpServerStatus status;
        std::shared_ptr<McpServerActivity> activity;
    };

    enum class Outcome { superseded, adopted, failed };

    mutable std::mutex mutex;
    mutable std::condition_variable changed;
    std::uint64_t round = 0;       // bumped by every connect and shutdown
    std::size_t pending = 0;       // servers of `round` still connecting
    std::size_t background = 0;    // live background threads, any round
    core::tools::ToolManager* tool_manager = nullptr;
    std::vector<ServerEntry> servers;   // configuration order
    std::vector<StatusEntry> statuses;  // configuration order
    std::unordered_map<std::string, std::size_t> tool_owner;
    std::shared_ptr<McpSamplingBridge> sampling_bridge;

    std::mutex listener_mutex;
    ChangeListener listener;

    // -- background threads -------------------------------------------------

    void leave_background() noexcept {
        {
            std::lock_guard lock(mutex);
            --background;
        }
        changed.notify_all();
    }

    /// Start @p work on a detached, counted thread. False when no thread
    /// could be started; @p work has then not run.
    bool spawn(std::function<void()> work) noexcept {
        {
            std::lock_guard lock(mutex);
            ++background;
        }
        try {
            std::thread([self = shared_from_this(), work = std::move(work)] {
                {
                    const core::utils::DetachedWorkTracker::Guard guard{
                        core::utils::DetachedWorkTracker::instance()};
                    try { work(); } catch (...) {}
                }
                self->leave_background();
            }).detach();
            return true;
        } catch (...) {
            leave_background();
            return false;
        }
    }

    /// Shut @p sessions down in parallel, off the caller's thread.
    void teardown(std::vector<SessionPtr> sessions) noexcept {
        for (auto& session : sessions) {
            if (!session) continue;
            if (spawn([session] { session->shutdown(); })) continue;
            try { session->shutdown(); } catch (...) {}  // no thread to spare
        }
    }

    // -- round bookkeeping (all require mutex) --------------------------------

    [[nodiscard]] bool is_current_locked(std::uint64_t connect_round, std::size_t index) const noexcept {
        return connect_round == round && tool_manager && index < statuses.size();
    }

    /// Start a new round: retire the previous one's servers and hand back
    /// their sessions for teardown.
    [[nodiscard]] std::vector<SessionPtr> begin_round_locked(std::vector<StatusEntry> round_statuses,
                                                             core::tools::ToolManager* manager) {
        ++round;
        auto retired = retire_locked();
        tool_manager = manager;
        statuses = std::move(round_statuses);
        pending = manager ? statuses.size() : 0;
        return retired;
    }

    /// Detach every server: unregister its tools and hand back its sessions.
    [[nodiscard]] std::vector<SessionPtr> retire_locked() {
        std::vector<SessionPtr> sessions;
        sessions.reserve(servers.size());
        std::vector<std::string> names;
        for (auto& server : servers) {
            names.append_range(server.tool_names);
            sessions.push_back(std::move(server.session));
        }
        servers.clear();
        tool_owner.clear();
        if (tool_manager) unregister_mcp_tools(*tool_manager, names);
        return sessions;
    }

    /// Register a connected server's tools and record it. On a clash with
    /// another server the one later in the configuration wins, whichever
    /// finished first; a clash with a non-MCP tool is lost and reported in
    /// @p shadowed.
    void adopt_locked(std::size_t index, const std::string& name, const Attempt& attempt,
                      std::vector<std::string>& shadowed) {
        ServerEntry entry{.name = name, .index = index, .session = attempt.session};
        for (const auto& tool : attempt.tools) {
            auto tool_name = tool->get_definition().name;
            if (const auto owner = tool_owner.find(tool_name); owner != tool_owner.end()) {
                if (owner->second > index) continue;
                if (owner->second != index) disown_locked(tool_name, owner->second);
            } else if (tool_manager->has_tool(tool_name)) {
                shadowed.push_back(std::move(tool_name));
                continue;
            }
            tool_manager->register_tool(tool);
            tool_owner.insert_or_assign(tool_name, index);
            if (!std::ranges::contains(entry.tool_names, tool_name)) {
                entry.tool_names.push_back(std::move(tool_name));
            }
        }
        servers.insert(std::ranges::upper_bound(servers, index, {}, &ServerEntry::index),
                       std::move(entry));
    }

    void disown_locked(const std::string& tool_name, std::size_t owner_index) {
        const auto owner = std::ranges::find(servers, owner_index, &ServerEntry::index);
        if (owner != servers.end()) std::erase(owner->tool_names, tool_name);
    }

    /// Undo a partial adopt_locked(): unregister whatever @p index registered.
    void abandon_locked(std::size_t index) noexcept {
        try {
            std::vector<std::string> names;
            std::erase_if(tool_owner, [&](const auto& owned) {
                if (owned.second != index) return false;
                names.push_back(owned.first);
                return true;
            });
            std::erase_if(servers, [index](const ServerEntry& server) { return server.index == index; });
            if (tool_manager) unregister_mcp_tools(*tool_manager, names);
        } catch (...) {}
    }

    /// Publish @p attempt as server @p index's result. A registration that
    /// throws is rolled back and published as a failure.
    Outcome publish_locked(std::size_t index, const core::config::McpServerConfig& config,
                           Attempt& attempt, std::vector<std::string>& shadowed) {
        auto& entry = statuses[index];
        entry.status.connect_duration = attempt.duration;
        if (attempt.succeeded()) {
            try {
                adopt_locked(index, config.name, attempt, shadowed);
                mark_connected(entry.status, attempt);
                entry.activity = attempt.activity;
                return Outcome::adopted;
            } catch (const std::exception& e) {
                abandon_locked(index);
                attempt.error = std::format("could not register its tools: {}", e.what());
            } catch (...) {
                abandon_locked(index);
                attempt.error = "could not register its tools";
            }
        }
        mark_failed(entry.status, attempt.error, mcp_error_needs_login(config, attempt.error));
        return Outcome::failed;
    }

    // -- results --------------------------------------------------------------

    /// Settle server @p index of @p connect_round with @p attempt. A session
    /// that is not adopted (superseded round, failed registration) is shut
    /// down here, so call it on a background thread whenever it has one.
    void record(std::uint64_t connect_round, std::size_t index,
                const core::config::McpServerConfig& config, Attempt attempt) noexcept {
        auto outcome = Outcome::superseded;
        std::vector<std::string> shadowed;
        {
            std::lock_guard lock(mutex);
            if (is_current_locked(connect_round, index)) {
                // Settle first: nothing below may leave a waiter hanging.
                if (pending > 0) --pending;
                try {
                    outcome = publish_locked(index, config, attempt, shadowed);
                } catch (...) {
                    outcome = Outcome::failed;
                }
            }
        }
        changed.notify_all();

        if (attempt.session && outcome != Outcome::adopted) {
            try { attempt.session->shutdown(); } catch (...) {}
        }
        try {
            report(outcome, config, attempt, shadowed);
        } catch (...) {}
        if (outcome != Outcome::superseded) notify_listener();
    }

    static void report(Outcome outcome, const core::config::McpServerConfig& config,
                       const Attempt& attempt, const std::vector<std::string>& shadowed) {
        switch (outcome) {
        case Outcome::superseded:
            core::logging::debug("[MCP] Dropped the superseded connection to '{}'.", config.name);
            return;
        case Outcome::adopted: {
            const auto& session = *attempt.session;
            const auto on_off = [](bool on) { return on ? "on" : "off"; };
            core::logging::info(
                "[MCP] Connected to '{}' in {} ms - {} tool(s), {} feature tool(s), "
                "resources={}, prompts={}, sampling={}.",
                config.name, attempt.duration.count(), attempt.tool_defs.size(),
                attempt.tools.size() - attempt.tool_defs.size(),
                on_off(session.supports_resources()), on_off(session.supports_prompts()),
                on_off(session.supports_sampling()));
            for (const auto& name : shadowed) {
                core::logging::warn("[MCP] '{}' tool '{}' is not available: a non-MCP tool already has that name.",
                                    config.name, name);
            }
            return;
        }
        case Outcome::failed:
            core::logging::warn("[MCP] Could not connect to '{}': {}", config.name, attempt.error);
            if (mcp_error_needs_login(config, attempt.error)) {
                core::logging::warn("[MCP] '{}' appears to require OAuth. Run: /mcp login {}",
                                    config.name, config.name);
            }
            return;
        }
    }

    void set_listener(ChangeListener replacement) noexcept {
        std::lock_guard lock(listener_mutex);
        listener = std::move(replacement);
    }

    void notify_listener() noexcept {
        std::lock_guard lock(listener_mutex);
        if (!listener) return;
        try { listener(); } catch (...) {}
    }
};

McpConnectionManager::McpConnectionManager() : state_(std::make_shared<State>()) {}

McpConnectionManager::~McpConnectionManager() {
    // Process teardown: the ToolManager and the UI may already be gone, so
    // only make sure no background thread will ever touch them again.
    try {
        state_->set_listener(nullptr);
        std::lock_guard lock(state_->mutex);
        ++state_->round;
        state_->pending = 0;
        state_->tool_manager = nullptr;
    } catch (...) {}
}

void McpConnectionManager::connect_all_async(const core::config::AppConfig& config,
                                             core::tools::ToolManager& tool_manager,
                                             std::shared_ptr<core::llm::LLMProvider> sampling_provider,
                                             std::string sampling_model) {
    const std::string config_dir = core::config::ConfigManager::get_instance().get_config_dir();
    const auto& servers = config.mcp_servers;
    auto statuses = servers
        | std::views::transform([](const auto& server) {
              return State::StatusEntry{.status = connecting_status(server), .activity = nullptr};
          })
        | std::ranges::to<std::vector<State::StatusEntry>>();

    std::uint64_t connect_round = 0;
    std::vector<SessionPtr> retired;
    McpSamplingCallback sampling_callback;
    {
        std::lock_guard lock(state_->mutex);
        retired = state_->begin_round_locked(std::move(statuses), &tool_manager);
        connect_round = state_->round;

        auto& bridge = state_->sampling_bridge;
        if (!sampling_provider) {
            bridge.reset();
        } else if (!bridge) {
            bridge = std::make_shared<McpSamplingBridge>(std::move(sampling_provider),
                                                         std::move(sampling_model));
        } else {
            bridge->set_backend(std::move(sampling_provider), std::move(sampling_model));
        }
        if (bridge) {
            sampling_callback = [bridge](std::string_view server_name, std::string_view params_json) {
                return bridge->create_message(server_name, params_json);
            };
        }
    }
    state_->changed.notify_all();
    state_->teardown(std::move(retired));

    for (std::size_t index = 0; index < servers.size(); ++index) {
        const auto& server = servers[index];
        const bool started = state_->spawn(
            [state = state_, connect_round, index, server, sampling_callback, config_dir] {
                state->record(connect_round, index, server,
                              attempt_connection(server, sampling_callback, config_dir));
            });
        if (!started) {
            state_->record(connect_round, index, server,
                           Attempt{.error = "could not start a background connection thread"});
        }
    }
}

void McpConnectionManager::connect_all(const core::config::AppConfig& config,
                                       core::tools::ToolManager& tool_manager,
                                       std::shared_ptr<core::llm::LLMProvider> sampling_provider,
                                       std::string sampling_model) {
    connect_all_async(config, tool_manager, std::move(sampling_provider), std::move(sampling_model));
    wait_until_settled();
}

bool McpConnectionManager::wait_until_settled(std::chrono::milliseconds timeout) const {
    std::unique_lock lock(state_->mutex);
    const auto settled = [&] { return state_->pending == 0; };
    if (timeout == std::chrono::milliseconds::max()) {
        state_->changed.wait(lock, settled);
        return true;
    }
    return state_->changed.wait_for(lock, timeout, settled);
}

void McpConnectionManager::set_change_listener(ChangeListener listener) {
    state_->set_listener(std::move(listener));
}

void McpConnectionManager::update_sampling_backend(
    std::shared_ptr<core::llm::LLMProvider> sampling_provider,
    std::string sampling_model)
{
    std::lock_guard lock(state_->mutex);
    if (!state_->sampling_bridge) return;
    state_->sampling_bridge->set_backend(std::move(sampling_provider), std::move(sampling_model));
}

void McpConnectionManager::shutdown_all(std::chrono::milliseconds grace) noexcept {
    try {
        std::vector<SessionPtr> retired;
        {
            std::lock_guard lock(state_->mutex);
            retired = state_->begin_round_locked({}, nullptr);
            state_->sampling_bridge.reset();
        }
        state_->changed.notify_all();
        state_->teardown(std::move(retired));

        std::unique_lock lock(state_->mutex);
        if (!state_->changed.wait_for(lock, grace, [&] { return state_->background == 0; })) {
            const auto left = state_->background;
            lock.unlock();
            core::logging::warn(
                "[MCP] {} server connection(s) still closing after {} ms; leaving them to finish "
                "in the background.",
                left, grace.count());
        }
    } catch (...) {}
}

int McpConnectionManager::connected_count() const noexcept {
    std::lock_guard lock(state_->mutex);
    return static_cast<int>(state_->servers.size());
}

std::size_t McpConnectionManager::configured_count() const noexcept {
    std::lock_guard lock(state_->mutex);
    return state_->statuses.size();
}

std::vector<std::string> McpConnectionManager::registered_tool_names() const {
    std::lock_guard lock(state_->mutex);
    std::vector<std::string> names;
    for (const auto& server : state_->servers) {
        names.append_range(server.tool_names | std::views::transform([&](const std::string& tool) {
            return std::format("{}/{}", server.name, tool);
        }));
    }
    return names;
}

std::vector<McpServerStatus> McpConnectionManager::server_statuses() const {
    std::lock_guard lock(state_->mutex);
    return state_->statuses | std::views::transform([](const State::StatusEntry& entry) {
               auto status = entry.status;
               if (entry.activity) status.activity = entry.activity->snapshot();
               return status;
           })
        | std::ranges::to<std::vector<McpServerStatus>>();
}

} // namespace core::mcp
