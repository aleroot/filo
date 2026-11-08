#include "AcpAgentSession.hpp"

#include "exec/ProviderRuntime.hpp"

#include "core/agent/AgentMode.hpp"
#include "core/agent/PermissionGate.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/changes/MutationScope.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/context/SessionContext.hpp"
#include "core/llm/ModelMetadata.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderCredentialStatus.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/llm/ReasoningEffortResolution.hpp"
#include "core/logging/Logger.hpp"
#include "core/mcp/McpConnectionManager.hpp"
#include "core/memory/MemorySystem.hpp"
#include "core/permissions/PermissionSystem.hpp"
#include "core/session/SessionHandoff.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/tools/ActivateSkillTool.hpp"
#include "core/tools/BuiltinToolRegistry.hpp"
#include "core/tools/SkillRegistry.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/workspace/Workspace.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <map>
#include <charconv>
#include <chrono>
#include <future>
#include <unordered_set>
#include <utility>

namespace exec::acp {
namespace {

constexpr std::array<std::string_view, 8> kEffortValues{
    "auto", "off", "low", "medium", "high", "xhigh", "max", "ultra",
};

[[nodiscard]] std::string effort_option(std::string_view stored) {
    if (stored.empty()) return "auto";
    if (stored == "none") return "off";
    return std::string(stored);
}

[[nodiscard]] bool known_mode(std::string_view mode) {
    const auto requested = core::utils::str::to_lower_ascii_copy(mode);
    if (requested == "plan") return true;
    for (const auto& descriptor : core::agent::kAgentModes) {
        if (core::utils::str::to_lower_ascii_copy(descriptor.name) == requested) return true;
    }
    return false;
}

/// The id an editor sees for an agent mode. The read-only mode is "plan":
/// Xcode's plan control looks for a mode with exactly that id.
[[nodiscard]] std::string mode_id(std::string_view agent_mode) {
    const auto parsed = core::agent::agent_mode_from_string(agent_mode);
    if (parsed == core::agent::AgentMode::Research) return "plan";
    return core::utils::str::to_lower_ascii_copy(core::agent::to_string(parsed));
}

[[nodiscard]] std::string mode_name(std::string_view id) {
    if (id.empty()) return {};
    std::string name(id);
    name.front() = static_cast<char>(std::toupper(static_cast<unsigned char>(name.front())));
    return name;
}

/// Xcode's "Implement this Plan" control leaves plan mode by sending a prompt
/// that carries this reminder.
constexpr std::string_view kXcodeExitedPlanMode = "The user exited plan mode";

/// Routes ask_user_question calls to the editor session that made them. The
/// tool is registered once per process; each session registers itself here.
class QuestionRouter {
public:
    static QuestionRouter& instance() {
        static QuestionRouter router;
        return router;
    }

    void add(const std::string& session_id, AgentSession* session) {
        std::lock_guard lock(mu_);
        sessions_[session_id] = session;
    }

    void remove(const std::string& session_id, AgentSession* session) {
        std::lock_guard lock(mu_);
        if (const auto it = sessions_.find(session_id); it != sessions_.end() && it->second == session) {
            sessions_.erase(it);
        }
    }

    void route(core::tools::QuestionRequest request) {
        // Not held while asking: the answer arrives on the read loop, which
        // must stay free to open other sessions meanwhile. A question is asked
        // inside its session's prompt, and the server keeps a session alive
        // until its prompt returns, so the pointer outlives the call.
        AgentSession* session = nullptr;
        {
            std::lock_guard lock(mu_);
            if (const auto it = sessions_.find(request.session_id); it != sessions_.end()) {
                session = it->second;
            }
        }
        if (!session) {
            request.promise->set_value(std::nullopt);
            return;
        }
        session->answer_question(std::move(request));
    }

private:
    std::mutex mu_;
    std::map<std::string, AgentSession*> sessions_;
};

[[nodiscard]] std::filesystem::path media_directory(const std::string& session_id) {
    return std::filesystem::temp_directory_path() / "filo-acp" / session_id;
}

[[nodiscard]] std::string mcp_fingerprint(const core::config::McpServerConfig& server) {
    std::string out = server.name + "\n" + server.transport + "\n" + server.command
        + "\n" + server.url + "\n";
    for (const auto& arg : server.args) out += arg + "\n";
    out += "|\n";
    for (const auto& variable : server.env) out += variable + "\n";
    out += "|\n";
    for (const auto& [name, value] : server.headers) out += name + "=" + value + "\n";
    return out;
}

void connect_mcp_servers(const std::vector<core::config::McpServerConfig>& extra,
                         const std::shared_ptr<core::llm::LLMProvider>& provider,
                         const std::string& model) {
    // One process, many editor threads. Later sessions add servers; they must
    // not drop an earlier thread's servers or reconnect an unchanged set.
    static std::mutex mutex;
    static std::vector<core::config::McpServerConfig> client_servers;
    static std::string connected;
    core::config::AppConfig config;
    std::string fingerprint;
    {
        std::lock_guard lock(mutex);
        for (const auto& server : extra) {
            const auto existing = std::ranges::find_if(
                client_servers,
                [&](const auto& configured) { return configured.name == server.name; });
            if (existing != client_servers.end()) *existing = server;
            else client_servers.push_back(server);
        }
        config = core::config::ConfigManager::get_instance().get_config();
        for (const auto& server : client_servers) {
            const auto existing = std::ranges::find_if(
                config.mcp_servers,
                [&](const auto& configured) { return configured.name == server.name; });
            if (existing != config.mcp_servers.end()) *existing = server;
            else config.mcp_servers.push_back(server);
        }
        for (const auto& server : config.mcp_servers) fingerprint += mcp_fingerprint(server);
        if (fingerprint == connected) return;
        connected = fingerprint;
    }
    if (config.mcp_servers.empty()) return;
    core::logging::info("ACP connecting {} MCP server(s)", config.mcp_servers.size());
    auto& manager = core::mcp::McpConnectionManager::get_instance();
    // Not awaited here: the session answers at once, and the first prompt
    // waits for the handshakes it needs (see AgentSession::prompt).
    manager.connect_all_async(
        config,
        core::tools::ToolManager::get_instance(),
        provider,
        model);
}

/// How long a prompt waits for MCP servers still connecting.
constexpr auto kMcpSettleWait = std::chrono::seconds(10);

/// The models a configured provider offers, from the registry family it maps
/// to: a configured name such as "claude" is listed under "anthropic".
[[nodiscard]] std::vector<core::llm::ModelInfo> provider_models(const std::string& provider_name) {
    const auto config = core::config::ConfigManager::get_instance().get_config();
    auto api_type = core::config::ApiType::Unknown;
    if (const auto it = config.providers.find(provider_name); it != config.providers.end()) {
        api_type = it->second.api_type;
    }
    auto models = core::llm::ModelRegistry::instance().get_by_provider(
        core::llm::model_registry_provider_key(provider_name, api_type));
    std::ranges::sort(models, {}, &core::llm::ModelInfo::canonical_id);
    return models;
}

/// Most entries the model picker offers per provider other than the current.
constexpr std::size_t kMaxOtherProviderModels = 32;

/// What every editor session in this process shares: the registered tools,
/// the memory system, and the process-wide workspace that skill and MCP root
/// discovery read. The first session also sets the process directory and the
/// project configuration; later ones only add their roots to skill discovery.
class SharedHost {
public:
    [[nodiscard]] core::workspace::WorkspaceSnapshot workspace_for(
        const std::filesystem::path& primary,
        const std::vector<std::filesystem::path>& additional,
        const ClientCapabilities& client) {
        std::lock_guard lock(mu_);
        auto& workspace = core::workspace::Workspace::get_instance();
        if (!memory_) {
            std::error_code ec;
            std::filesystem::current_path(primary, ec);
            const auto base = workspace.snapshot();
            workspace.initialize(primary, {}, base.enforce, base.scratch);
            core::tools::SkillRegistry::grant_package_read_roots(workspace);
            base_ = workspace.snapshot();
            memory_ = core::memory::make_memory_system(core::memory::MemoryConfig{
                .tool_recovery = core::config::ConfigManager::get_instance().get_config().tool_recovery,
            });
        }
        auto snapshot = base_;
        snapshot.primary = primary;
        for (const auto& extra : additional) {
            if (!extra.empty()) snapshot.additional.push_back(extra);
        }
        bool new_roots = !registered_;
        for (auto& root : core::workspace::ordered_roots(snapshot)) {
            if (std::ranges::find(skill_roots_, root) == skill_roots_.end()) {
                skill_roots_.push_back(std::move(root));
                new_roots = true;
            }
        }
        if (!registered_) {
            register_tools(client);
        } else if (new_roots
                   && !core::tools::SkillRegistry::discover_instruction_skills(skill_roots_).empty()) {
            // Replacing every built-in would reset tools that keep state for
            // sessions already running; only the skill list grows.
            core::tools::ToolManager::get_instance().register_tool(
                std::make_shared<core::tools::ActivateSkillTool>(skill_roots_));
        }
        return snapshot;
    }

    [[nodiscard]] std::shared_ptr<core::memory::MemorySystem> memory() {
        std::lock_guard lock(mu_);
        return memory_;
    }

private:
    void register_tools(const ClientCapabilities& client) {
        auto registration = core::tools::agent_builtin_tool_options();
        registration.memory_store = memory_->semantic();
        // Questions become an editor form. Without form support the tool is
        // left out rather than leaving the model waiting on a dialog.
        registration.include_ask_user_question = client.elicitation_form;
        registration.ask_user_question_callback = route_question;
        registration.workspace_roots = skill_roots_;
        core::tools::register_builtin_tools(core::tools::ToolManager::get_instance(), std::move(registration));
        registered_ = true;
    }

    std::mutex mu_;
    bool registered_ = false;
    core::workspace::WorkspaceSnapshot base_;
    std::shared_ptr<core::memory::MemorySystem> memory_;
    std::vector<std::filesystem::path> skill_roots_;
};

[[nodiscard]] SharedHost& shared_host() {
    static SharedHost host;
    return host;
}

[[nodiscard]] ProviderRuntime process_runtime(const HostOptions& host) {
    static std::mutex mutex;
    static ProviderRuntime runtime;
    static bool ready = false;
    std::lock_guard lock(mutex);
    if (!ready) {
        runtime = load_provider_runtime(host.startup_model);
        ready = runtime.error.empty();
    }
    return runtime;
}

} // namespace

void route_question(core::tools::QuestionRequest request) {
    QuestionRouter::instance().route(std::move(request));
}

AgentSession::AgentSession(std::shared_ptr<core::agent::Agent> agent,
                           std::string id,
                           std::filesystem::path cwd,
                           std::string provider_name,
                           Options options,
                           std::vector<core::llm::Message> replay,
                           std::filesystem::path sessions_dir)
    : id_(std::move(id))
    , cwd_(std::move(cwd))
    , sessions_dir_(sessions_dir.empty()
                        ? core::session::SessionStore::default_sessions_dir()
                        : std::move(sessions_dir))
    , provider_name_(std::move(provider_name))
    , created_at_(options.created_at.empty() ? core::session::SessionStore::now_iso8601()
                                             : options.created_at)
    , options_(std::move(options))
    , replay_(std::move(replay))
    , agent_(std::move(agent)) {
    titled_ = std::ranges::any_of(replay_, [](const core::llm::Message& message) {
        return message.role == "user" && !message.synthetic;
    });
    if (const auto parsed = core::agent::agent_mode_from_string(agent_->get_mode());
        parsed != core::agent::AgentMode::Research) {
        mode_before_plan_ = agent_->get_mode();
    }
    install_permissions();
    agent_->set_session_id(id_);
    core::budget::BudgetTracker::get_instance().reset_session(id_);
    for (const auto& message : replay_) {
        if (message.role == "system") continue;
        agent_->append_history_message(message);
    }
    save();
    QuestionRouter::instance().add(id_, this);
}

AgentSession::~AgentSession() {
    QuestionRouter::instance().remove(id_, this);
}

std::string AgentSession::id() const { return id_; }

std::string AgentSession::provider_name() const {
    std::lock_guard lock(state_mu_);
    return provider_name_;
}

std::vector<ModeChoice> AgentSession::modes() const {
    std::vector<ModeChoice> choices;
    choices.reserve(core::agent::kAgentModes.size());
    for (const auto& descriptor : core::agent::kAgentModes) {
        const auto id = mode_id(descriptor.name);
        choices.push_back(ModeChoice{
            .id = id,
            .name = mode_name(id),
            .description = std::string(descriptor.description),
        });
    }
    return choices;
}

std::vector<CommandSpec> AgentSession::commands() {
    return {
        {"help", "Show the commands this editor session accepts"},
        {"mode", "Set the agent mode (auto, build, debug, plan, execute)"},
        {"model", "Set the model, as a model id or provider/model"},
        {"effort", "Set reasoning effort (auto, off, low, medium, high, xhigh, max, ultra)"},
        {"compact", "Summarise the conversation so far"},
    };
}

std::vector<ConfigSelect> AgentSession::config_selects() const {
    std::vector<ConfigSelect> selects;
    ConfigSelect mode{
        .id = "mode",
        .name = "Mode",
        .description = "How the agent approaches the work",
        .category = "mode",
        .current = mode_id(agent_->get_mode()),
    };
    for (const auto& choice : modes()) {
        mode.options.push_back(ConfigChoice{choice.id, choice.name, choice.description});
    }
    selects.push_back(std::move(mode));

    ConfigSelect model{
        .id = "model",
        .name = "Model",
        .category = "model",
        .current = agent_->get_active_model_name(),
    };
    const auto current_provider = provider_name();
    auto known = provider_models(current_provider);
    if (known.size() > 64) known.resize(64);
    bool saw_current = model.current.empty();
    for (const auto& info : known) {
        if (info.canonical_id.empty()) continue;
        if (info.canonical_id == model.current) saw_current = true;
        model.options.push_back(ConfigChoice{
            info.canonical_id,
            info.display_name.empty() ? info.canonical_id : info.display_name,
            {},
        });
    }
    if (!saw_current) {
        model.options.insert(model.options.begin(), ConfigChoice{model.current, model.current, {}});
    }
    // Other signed-in providers, as provider/model so a pick switches both.
    auto& providers = core::llm::ProviderManager::get_instance();
    for (const auto& name : providers.names()) {
        if (name == current_provider) continue;
        std::shared_ptr<core::llm::LLMProvider> provider;
        try {
            provider = providers.get_provider(name);
        } catch (const std::exception&) {
            continue;
        }
        // Only services that can answer now; the picker is not a setup screen.
        if (core::llm::provider_credential_state(provider) != core::llm::ProviderCredentialState::Ready) {
            continue;
        }
        std::size_t added = 0;
        for (const auto& info : provider_models(name)) {
            if (info.canonical_id.empty()) continue;
            if (++added > kMaxOtherProviderModels) break;
            const auto shown = info.display_name.empty() ? info.canonical_id : info.display_name;
            model.options.push_back(ConfigChoice{name + "/" + info.canonical_id, shown + " · " + name, {}});
        }
    }
    selects.push_back(std::move(model));

    ConfigSelect effort{
        .id = "effort",
        .name = "Effort",
        .description = "Reasoning effort for this session",
        .category = "thought_level",
        .current = effort_option(agent_->get_effort_level()),
    };
    for (const auto value : kEffortValues) {
        effort.options.push_back(ConfigChoice{std::string(value), std::string(value), {}});
    }
    selects.push_back(std::move(effort));
    return selects;
}

core::utils::JsonValue AgentSession::opened_result() const {
    auto result = core::utils::JsonValue::object();
    result.set("sessionId", id_);
    const auto selects = config_selects();
    result.set("modes", mode_state(mode_id(agent_->get_mode()), modes()));
    result.set("configOptions", config_options(selects));
    for (const auto& select : selects) {
        if (select.id == "model") result.set("models", model_state(select));
    }
    return result;
}

core::utils::JsonValue AgentSession::config_state() const {
    return config_options(config_selects());
}

void AgentSession::announce(ClientChannel& client) {
    client.notify("session/update", commands_update(id_, commands()));
}

void AgentSession::emit(const core::utils::JsonValue& params) {
    ClientChannel* channel = nullptr;
    {
        std::lock_guard lock(state_mu_);
        channel = channel_;
    }
    if (channel) channel->notify("session/update", params);
}

void AgentSession::emit_state(ClientChannel& client) {
    auto mode = core::utils::JsonValue::object();
    mode.set("sessionId", id_);
    auto mode_update = core::utils::JsonValue::object();
    mode_update.set("sessionUpdate", "current_mode_update");
    mode_update.set("currentModeId", mode_id(agent_->get_mode()));
    mode.set("update", std::move(mode_update));
    client.notify("session/update", mode);

    auto config = core::utils::JsonValue::object();
    config.set("sessionId", id_);
    auto config_update = core::utils::JsonValue::object();
    config_update.set("sessionUpdate", "config_option_update");
    config_update.set("configOptions", config_options(config_selects()));
    config.set("update", std::move(config_update));
    client.notify("session/update", config);
}

void AgentSession::install_permissions() {
    if (options_.yolo) {
        agent_->set_permission_profile(core::agent::PermissionProfile::Autonomous);
        return;
    }
    agent_->set_permission_profile(core::agent::PermissionProfile::Interactive);
    std::unordered_set<std::string> trusted;
    for (const auto& name : options_.trusted_tools) {
        trusted.emplace(core::tools::names::canonical_alias(name));
    }
    agent_->set_permission_fn(
        [this, trusted = std::move(trusted)](std::string_view tool, std::string_view args) {
            if (cancel_requested_.load(std::memory_order_acquire)) return false;
            const auto canonical = std::string(core::tools::names::canonical_alias(tool));
            if (trusted.contains(canonical)) return true;
            {
                std::lock_guard lock(state_mu_);
                for (const auto& rule : allow_rules_) {
                    if (core::permissions::session_allow_rule_matches(rule, tool, args)) {
                        return true;
                    }
                }
            }

            std::string call_id;
            ClientChannel* channel = nullptr;
            {
                std::lock_guard lock(state_mu_);
                call_id = permission_call_id_;
                channel = channel_;
            }
            if (!channel) return false;
            if (call_id.empty()) call_id = "call";
            const auto rule = core::permissions::make_session_allow_rule(tool, args);
            const auto always_name = rule.empty()
                ? std::string{}
                : "Always allow " + core::permissions::describe_session_allow_rule(rule);
            const auto params = permission_params(
                id_,
                call_id,
                tool_title(tool, args),
                rule.empty() ? std::string_view{} : std::string_view{"allow-always"},
                always_name);
            core::utils::JsonValue result;
            try {
                result = channel->request("session/request_permission", params);
            } catch (const std::exception&) {
                return false;
            }
            const auto decision = interpret_permission(
                result, rule.empty() ? std::string_view{} : std::string_view{"allow-always"});
            if (decision.cancelled) {
                cancel_requested_.store(true, std::memory_order_release);
                agent_->request_stop();
                return false;
            }
            if (decision.remember && !rule.empty()) {
                std::lock_guard lock(state_mu_);
                allow_rules_.push_back(rule);
            }
            return decision.allowed;
        });
}

void AgentSession::capture_before(LiveCall& call) const {
    const auto kind = tool_kind(call.name);
    if (kind != "edit" && kind != "delete") return;
    // The same paths the tool itself will be allowed to touch.
    const auto scope = core::changes::mutation_scope(
        call.name, call.args, agent_->session_context_snapshot(), core::tools::ToolAnnotations{});
    for (const auto& path : scope.paths) {
        if (std::ranges::any_of(call.before, [&](const auto& seen) { return seen.first == path; })) {
            continue;
        }
        call.before.emplace_back(path, read_file_side(path));
    }
}

void AgentSession::publish_tool(const LiveCall& call,
                                std::string status,
                                std::string output,
                                bool initial) {
    std::vector<core::utils::JsonValue> diffs;
    if (!initial && status == "completed") {
        for (const auto& [path, before] : call.before) {
            if (auto diff = file_diff(path, before, read_file_side(path))) {
                diffs.push_back(std::move(*diff));
            }
        }
    }
    ToolCallView view{
        .tool_call_id = call.id,
        .name = call.name,
        .title = tool_title(call.name, call.args),
        .kind = tool_kind(call.name),
        .status = std::move(status),
        .raw_input = initial ? call.args : std::string{},
        .raw_output = output,
        // A diff already shows what an edit did; its JSON receipt adds nothing.
        .content_text = diffs.empty() && !output.empty() ? tool_display_text(call.name, output)
                                                         : std::string{},
        .diffs = std::move(diffs),
        .locations = tool_locations(call.args, cwd_),
        .include_raw_output = !output.empty() && output.size() <= kMaxToolTextChars,
    };
    emit(tool_update(id_, view, initial));
}

void AgentSession::publish_plan() {
    std::vector<PlanItem> entries;
    for (const auto& todo : agent_->get_todos()) {
        PlanItem item;
        item.content = todo.text;
        item.status = std::string(core::session::to_string(todo.status));
        item.priority = todo.status == core::session::TodoStatus::InProgress ? "high" : "medium";
        entries.push_back(std::move(item));
    }
    if (!entries.empty()) emit(plan_update(id_, entries));
}

void AgentSession::publish_usage() {
    // `used` is what the next request carries, not the tokens spent so far.
    const auto context = agent_->context_window_snapshot();
    if (context.max_context_tokens <= 0) return;
    const auto cost = core::budget::BudgetTracker::get_instance().session_cost_usd(id_);
    emit(usage_update(
        id_,
        static_cast<long long>(context.estimated_context_tokens),
        context.max_context_tokens,
        cost > 0 ? std::optional<double>(cost) : std::nullopt));
}

void AgentSession::publish_info(std::string_view prompt_text) {
    std::optional<std::string> title;
    if (!titled_) {
        auto text = session_title(prompt_text);
        if (!text.empty()) {
            title = std::move(text);
            titled_ = true;
        }
    }
    emit(session_info_update(id_, std::move(title), core::session::SessionStore::now_iso8601()));
}

void AgentSession::emit_status(std::string_view text) {
    const auto trimmed = core::utils::str::trim_ascii_copy(text);
    if (trimmed.empty()) return;
    ClientChannel* channel = nullptr;
    {
        std::lock_guard lock(state_mu_);
        channel = channel_;
    }
    if (!channel) return;
    // Filo frames status lines as "[...]"; a notice title is plain text.
    std::string_view body = trimmed;
    if (body.size() >= 2 && body.front() == '[' && body.back() == ']') {
        body = body.substr(1, body.size() - 2);
    }
    if (channel->capabilities().notices) {
        const auto newline = body.find('\n');
        const auto title = core::utils::str::trim_ascii_copy(body.substr(0, newline));
        const auto detail = newline == std::string_view::npos
            ? std::string{}
            : core::utils::str::trim_ascii_copy(body.substr(newline + 1));
        channel->notify("session/update", notice_update(id_, "info", title, detail));
    } else {
        channel->notify("session/update", text_update(id_, "agent_thought_chunk", std::string(trimmed) + "\n"));
    }
}

void AgentSession::save() const {
    if (discarded_.load(std::memory_order_acquire)) return;
    core::session::SessionData data;
    data.session_id = id_;
    data.created_at = created_at_;
    data.last_active_at = core::session::SessionStore::now_iso8601();
    data.working_dir = cwd_.string();
    data.provider = provider_name();
    data.model = agent_->get_active_model_name();
    data.mode = agent_->get_mode();
    for (auto message : agent_->get_history()) {
        if (message.role == "system") continue;
        if (message.role == "user" && !message.synthetic) ++data.stats.turn_count;
        data.messages.push_back(std::move(message));
    }
    data.todos = agent_->get_todos();
    const auto totals = core::budget::BudgetTracker::get_instance().session_total(id_);
    data.stats.prompt_tokens = totals.prompt_tokens;
    data.stats.completion_tokens = totals.completion_tokens;
    data.stats.cost_usd = core::budget::BudgetTracker::get_instance().session_cost_usd(id_);
    data.handoff_summary = core::session::build_handoff_summary(data);

    std::string error;
    core::session::SessionStore store(sessions_dir_);
    if (!store.save(data, &error)) {
        core::logging::warn("ACP session save failed: {}", error);
    }
}

void AgentSession::replay(ClientChannel& client) {
    for (const auto& message : replay_) {
        if (message.synthetic || message.role == "system") continue;
        if (message.role == "user") {
            const auto text = core::llm::message_text_for_display(message);
            if (!text.empty()) {
                client.notify("session/update",
                              text_update(id_, "user_message_chunk", text));
            }
            continue;
        }
        if (message.role == "assistant") {
            if (!message.reasoning_content.empty()) {
                client.notify("session/update",
                              text_update(id_, "agent_thought_chunk", message.reasoning_content));
            }
            if (!message.content.empty()) {
                client.notify("session/update",
                              text_update(id_, "agent_message_chunk", message.content));
            }
            for (const auto& call : message.tool_calls) {
                LiveCall live{call.id, call.function.name, call.function.arguments, true};
                ToolCallView view{
                    .tool_call_id = live.id,
                    .name = live.name,
                    .title = tool_title(live.name, live.args),
                    .kind = tool_kind(live.name),
                    .status = "completed",
                    .raw_input = live.args,
                    .locations = tool_locations(live.args, cwd_),
                };
                client.notify("session/update", tool_update(id_, view, true));
            }
            continue;
        }
        if (message.role == "tool" && !message.tool_call_id.empty()) {
            LiveCall live{message.tool_call_id, message.name, {}, true};
            ToolCallView view{
                .tool_call_id = live.id,
                .status = tool_result_failed(message.content) ? "failed" : "completed",
                .raw_output = message.content,
                .content_text = message.content,
                .include_raw_output = message.content.size() <= kMaxToolTextChars,
            };
            client.notify("session/update", tool_update(id_, view, false));
        }
    }
}

std::string AgentSession::run_slash(std::string_view name,
                                    std::string_view argument,
                                    ClientChannel& client) {
    const auto say = [&](std::string text) {
        client.notify("session/update", text_update(id_, "agent_message_chunk", text));
    };
    if (name == "help") {
        say("Commands: /help, /mode <mode>, /model <model>, /effort <level>, /compact.\n"
            "Mode, model and effort are also in the editor's session controls.\n"
            "File changes ask for permission unless this process was started with --yolo.");
        return "end_turn";
    }
    if (name == "mode") {
        if (argument.empty()) {
            say("Mode is " + mode_id(agent_->get_mode()) + ".");
            return "end_turn";
        }
        if (auto error = set_mode(argument, client)) say(*error);
        else say("Mode is " + mode_id(agent_->get_mode()) + ".");
        return "end_turn";
    }
    if (name == "model") {
        if (argument.empty()) {
            say("Model is " + agent_->get_active_model_name() + ".");
            return "end_turn";
        }
        if (auto error = set_config("model", core::utils::JsonValue{std::string(argument)}, client)) {
            say(*error);
        } else {
            say("Model is " + agent_->get_active_model_name() + ".");
        }
        return "end_turn";
    }
    if (name == "effort") {
        if (argument.empty()) {
            say("Effort is " + effort_option(agent_->get_effort_level()) + ".");
            return "end_turn";
        }
        if (auto error = set_config("effort", core::utils::JsonValue{std::string(argument)}, client)) {
            say(*error);
        } else {
            say("Effort is " + effort_option(agent_->get_effort_level()) + ".");
        }
        return "end_turn";
    }
    if (name == "compact") {
        std::promise<bool> done;
        agent_->compact_history_then(
            [&](const std::string& status) {
                if (!status.empty()) say(status);
            },
            [&](bool compacted) { done.set_value(compacted); });
        const bool compacted = done.get_future().get();
        if (compacted) save();
        else say("Could not compact the conversation.");
        return "end_turn";
    }
    say("Unknown command /" + std::string(name) + ". Try /help.");
    return "end_turn";
}

std::string AgentSession::prompt(const core::utils::JsonValue& blocks, ClientChannel& client) {
    // A cancel that arrived before the turn was entered still wins. Clearing
    // the flag with a plain store would drop it.
    if (cancel_requested_.exchange(false, std::memory_order_acq_rel)) return "cancelled";
    {
        std::lock_guard lock(state_mu_);
        channel_ = &client;
        calls_.clear();
        permission_call_id_.clear();
    }
    struct ClearChannel {
        AgentSession* self;
        ~ClearChannel() {
            std::lock_guard lock(self->state_mu_);
            self->channel_ = nullptr;
        }
    } clear{this};

    std::string error;
    auto parsed = parse_prompt(blocks, media_directory(id_), &error);
    if (!parsed) throw std::runtime_error(error.empty() ? "invalid prompt" : error);
    if (!parsed->grant_paths.empty()) agent_->grant_workspace_paths(parsed->grant_paths);

    // Only Filo's own commands are taken out of the conversation. Anything
    // else that starts with a slash, such as a pasted absolute path, is a
    // prompt for the model.
    const auto trimmed = core::utils::str::trim_ascii_copy(parsed->text);
    if (parsed->parts.empty() && trimmed.starts_with('/') && trimmed.find('\n') == std::string::npos) {
        const auto gap = trimmed.find(' ');
        const auto name = core::utils::str::to_lower_ascii_copy(
            gap == std::string::npos ? trimmed.substr(1) : trimmed.substr(1, gap - 1));
        const auto argument = gap == std::string::npos
            ? std::string_view{}
            : std::string_view(trimmed).substr(gap + 1);
        if (std::ranges::any_of(commands(), [&](const CommandSpec& command) { return command.name == name; })) {
            return run_slash(name, core::utils::str::trim_ascii_copy(argument), client);
        }
    }

    // Xcode's "Implement this Plan" control asks to leave plan mode with a
    // reminder in the prompt rather than a mode change Filo would understand.
    if (core::agent::is_read_only_mode(core::agent::agent_mode_from_string(agent_->get_mode()))
        && trimmed.find(kXcodeExitedPlanMode) != std::string::npos) {
        std::string previous;
        {
            std::lock_guard lock(state_mu_);
            previous = mode_before_plan_;
        }
        static_cast<void>(set_mode(previous, client));
    }

    // Servers the editor passed when it opened the session are still
    // connecting on the first prompt; their tools belong in this turn.
    {
        auto& mcp = core::mcp::McpConnectionManager::get_instance();
        const auto deadline = std::chrono::steady_clock::now() + kMcpSettleWait;
        bool settled = false;
        while (!(settled = mcp.wait_until_settled(std::chrono::milliseconds(200)))
               && std::chrono::steady_clock::now() < deadline
               && !cancel_requested_.load(std::memory_order_acquire)) {
        }
        if (cancel_requested_.exchange(false, std::memory_order_acq_rel)) return "cancelled";
        if (!settled) {
            emit_status("[Some MCP servers are still connecting; their tools may be missing from this turn.]");
        }
    }

    core::llm::Message message{
        .role = "user",
        .content = parsed->text,
        .content_parts = std::move(parsed->parts),
    };
    std::atomic<bool> exhausted{false};
    std::promise<void> finished;
    std::atomic<bool> finished_once{false};
    auto finish = [&] {
        if (!finished_once.exchange(true, std::memory_order_acq_rel)) finished.set_value();
    };

    core::agent::Agent::TurnCallbacks callbacks;
    callbacks.on_reasoning = [this](const std::string& text) {
        if (!text.empty()) emit(text_update(id_, "agent_thought_chunk", text));
    };
    callbacks.on_status_log = [this](const std::string& text) {
        if (text.empty()) return;
        core::logging::info("[acp] {}", text);
        emit_status(text);
    };
    auth_required_.store(false, std::memory_order_release);
    const bool terminal_auth = client.capabilities().terminal_auth;
    callbacks.on_authentication_required = [this, terminal_auth](
                                               const core::llm::AuthenticationRecoveryRequest& request) {
        if (terminal_auth) {
            // The prompt fails with auth_required and the editor offers its
            // terminal sign-in, then reconnects.
            auth_required_.store(true, std::memory_order_release);
            return;
        }
        // Signing in needs a browser or a terminal this editor cannot open.
        const auto provider = request.provider_id.empty() ? std::string("the provider") : request.provider_id;
        std::string text = "\n\nSign-in for " + provider + " is no longer valid";
        if (!request.reason.empty()) text += " (" + request.reason + ")";
        text += ". Run `filo` in a terminal, sign in again with /login, then resend this message.\n";
        emit(text_update(id_, "agent_message_chunk", text));
    };
    callbacks.on_step_budget_event = [&](core::agent::Agent::StepBudgetEvent event) {
        if (event == core::agent::Agent::StepBudgetEvent::Exhausted) {
            exhausted.store(true, std::memory_order_release);
        }
    };
    callbacks.cancellation_requested = [this] {
        return cancel_requested_.load(std::memory_order_acquire);
    };
    callbacks.on_tool_start = [this](const core::llm::ToolCall& call) {
        LiveCall live{call.id, call.function.name, call.function.arguments, false};
        // Before permission is asked, so before the tool can run.
        capture_before(live);
        {
            std::lock_guard lock(state_mu_);
            permission_call_id_ = live.id;
            calls_.push_back(live);
        }
        publish_tool(live, "pending", {}, true);
    };
    callbacks.on_tool_finish = [this](const core::llm::ToolCall& call, const core::llm::Message& result) {
        LiveCall live{call.id, call.function.name, call.function.arguments, true};
        {
            std::lock_guard lock(state_mu_);
            for (auto& existing : calls_) {
                if (existing.id != call.id) continue;
                existing.finished = true;
                live.args = existing.args;
                live.before = std::move(existing.before);
                break;
            }
        }
        const auto failed = tool_result_failed(result.content);
        publish_tool(live, failed ? "failed" : "completed", result.content, false);
        if (core::tools::names::canonical_alias(call.function.name) == core::tools::names::kWriteTodos) {
            publish_plan();
        }
    };

    agent_->send_message(
        std::move(message),
        [this](const std::string& text) {
            if (!text.empty()) emit(text_update(id_, "agent_message_chunk", text));
        },
        [this](const std::string& name, const std::string& detail) {
            LiveCall live;
            bool found = false;
            std::string status;
            {
                std::lock_guard lock(state_mu_);
                for (auto& call : calls_) {
                    if (call.finished || call.name != name) continue;
                    found = true;
                    live = call;
                    if (!detail.empty() && detail.front() == '[') {
                        call.finished = true;
                        live.finished = true;
                        status = "failed";
                    } else {
                        status = "in_progress";
                    }
                    break;
                }
            }
            if (!found) return;
            publish_tool(live, status, status == "failed" ? detail : std::string{}, false);
        },
        finish,
        std::move(callbacks));
    finished.get_future().wait();

    publish_usage();
    publish_info(trimmed);
    save();
    if (auth_required_.exchange(false, std::memory_order_acq_rel)) {
        throw AuthRequired("Sign-in for " + provider_name() + " expired. Sign in again to continue.");
    }
    if (cancel_requested_.load(std::memory_order_acquire) || agent_->is_stop_requested()) {
        return "cancelled";
    }
    if (exhausted.load(std::memory_order_acquire)) return "max_turn_requests";
    return "end_turn";
}

void AgentSession::cancel() {
    cancel_requested_.store(true, std::memory_order_release);
    agent_->request_stop();
}

void AgentSession::discard() {
    discarded_.store(true, std::memory_order_release);
}

void AgentSession::answer_question(core::tools::QuestionRequest request) {
    ClientChannel* channel = nullptr;
    std::string call_id;
    {
        std::lock_guard lock(state_mu_);
        channel = channel_;
        for (auto it = calls_.rbegin(); it != calls_.rend(); ++it) {
            if (!it->finished
                && core::tools::names::canonical_alias(it->name) == core::tools::names::kAskUserQuestion) {
                call_id = it->id;
                break;
            }
        }
    }
    std::optional<std::vector<std::pair<std::string, std::string>>> answers;
    if (channel && channel->capabilities().elicitation_form
        && !cancel_requested_.load(std::memory_order_acquire)) {
        try {
            const auto result = channel->request(
                "elicitation/create", question_form(id_, call_id, request.questions));
            answers = question_answers(result, request.questions);
        } catch (const std::exception& ex) {
            core::logging::warn("ACP question failed: {}", ex.what());
        }
    }
    request.promise->set_value(std::move(answers));
}

std::optional<std::string> AgentSession::set_mode(std::string_view mode_id, ClientChannel& client) {
    if (!known_mode(mode_id)) return "Unknown mode '" + std::string(mode_id) + "'.";
    const auto current = core::agent::agent_mode_from_string(agent_->get_mode());
    if (!core::agent::is_read_only_mode(current)) {
        std::lock_guard lock(state_mu_);
        mode_before_plan_ = agent_->get_mode();
    }
    agent_->set_mode(std::string(mode_id));
    emit_state(client);
    save();
    return std::nullopt;
}

std::optional<std::string> AgentSession::set_config(std::string_view config_id,
                                                    const core::utils::JsonValue& value,
                                                    ClientChannel& client) {
    if (!value.is_string()) return "Configuration values must be strings.";
    const auto selected = std::string(value.as_string());
    if (config_id == "mode") return set_mode(selected, client);
    if (config_id == "effort") {
        const auto parsed = core::llm::parse_effort_selection(selected);
        if (!parsed) return "Unknown effort '" + selected + "'.";
        agent_->set_effort_level(*parsed);
        emit_state(client);
        save();
        return std::nullopt;
    }
    if (config_id == "model") {
        // "provider/model" only when the prefix names a provider: model ids
        // such as OpenRouter's "anthropic/claude-…" contain a slash too.
        const auto slash = selected.find('/');
        auto& providers = core::llm::ProviderManager::get_instance();
        if (slash != std::string::npos && slash > 0 && slash + 1 < selected.size()
            && providers.has_provider(selected.substr(0, slash))) {
            const auto provider_name = selected.substr(0, slash);
            const auto model_name = selected.substr(slash + 1);
            try {
                auto provider = providers.get_provider(provider_name);
                agent_->set_provider(std::move(provider));
                agent_->set_active_provider_name(provider_name);
                {
                    std::lock_guard lock(state_mu_);
                    provider_name_ = provider_name;
                }
                agent_->set_active_model(model_name);
            } catch (const std::exception& ex) {
                return ex.what();
            }
        } else if (selected.empty()) {
            return "Model name is empty.";
        } else {
            agent_->set_active_model(selected);
        }
        emit_state(client);
        save();
        return std::nullopt;
    }
    return "Unknown configuration option '" + std::string(config_id) + "'.";
}

std::expected<std::unique_ptr<Session>, OpenError>
open_session(const OpenRequest& request, const HostOptions& host) {
    std::error_code ec;
    if (request.cwd.empty() || !request.cwd.is_absolute()) {
        return std::unexpected(OpenError::invalid_params("cwd must be an absolute path"));
    }
    if (!std::filesystem::is_directory(request.cwd, ec) || ec) {
        return std::unexpected(OpenError::invalid_params("cwd is not a directory"));
    }
    for (const auto& extra : request.additional_directories) {
        if (!extra.empty() && !extra.is_absolute()) {
            return std::unexpected(
                OpenError::invalid_params("additionalDirectories entries must be absolute paths"));
        }
    }
    auto cwd = std::filesystem::weakly_canonical(request.cwd, ec);
    if (ec) cwd = request.cwd;

    std::string id;
    std::vector<core::llm::Message> replay;
    std::optional<core::session::SessionData> resumed;
    if (request.resume_id.has_value()) {
        // Resolve the record before touching the process workspace, so a
        // missing session cannot retarget the working directory.
        core::session::SessionStore store(core::session::SessionStore::default_sessions_dir());
        resumed = store.load_by_id(*request.resume_id);
        if (!resumed) return std::unexpected(OpenError::not_found("session not found"));
        id = resumed->session_id;
        replay = resumed->messages;
    } else {
        id = core::session::SessionStore::generate_id();
    }

    const auto runtime = process_runtime(host);
    if (!runtime.error.empty() || !runtime.provider) {
        return std::unexpected(OpenError::internal(
            runtime.error.empty() ? "No provider is available." : runtime.error));
    }

    // Tools resolve paths and run commands in each session's own workspace,
    // so one process can serve several projects, as TUI threads do.
    const auto snapshot = shared_host().workspace_for(cwd, request.additional_directories, request.client);
    auto memory = shared_host().memory();

    auto context = core::context::make_session_context(
        std::move(snapshot),
        core::context::SessionTransport::acp,
        id);
    context.steering_policy = host.steering;
    auto agent = std::make_shared<core::agent::Agent>(
        runtime.provider,
        core::tools::ToolManager::get_instance(),
        std::move(context),
        core::agent::ToolResultStore::default_root(),
        std::shared_ptr<core::power::SleepInhibitor>{},
        core::session::SessionStatsRegistry::shared_instance(),
        &core::budget::BudgetTracker::get_instance(),
        std::move(memory));
    // A resumed conversation continues on the provider and model it used.
    // When that provider is gone, both fall back together: a model name is
    // only meaningful to the provider that serves it.
    std::shared_ptr<core::llm::LLMProvider> provider = runtime.provider;
    std::string provider_name = runtime.provider_name;
    std::string model_name = runtime.model_name;
    if (resumed && !resumed->provider.empty() && !resumed->model.empty()) {
        auto& providers = core::llm::ProviderManager::get_instance();
        if (resumed->provider == runtime.provider_name) {
            model_name = resumed->model;
        } else if (providers.has_provider(resumed->provider)) {
            provider = providers.get_provider(resumed->provider);
            provider_name = resumed->provider;
            model_name = resumed->model;
        } else {
            core::logging::warn("ACP session {} used provider '{}', which is not configured; using '{}'",
                                id, resumed->provider, runtime.provider_name);
        }
    } else if (resumed && !resumed->model.empty()) {
        model_name = resumed->model;
    }
    if (provider != runtime.provider) agent->set_provider(provider);
    agent->set_active_provider_name(provider_name);
    agent->set_active_model(model_name);
    agent->set_mode(resumed && !resumed->mode.empty() ? resumed->mode : runtime.default_mode);
    if (host.startup_effort.has_value()) agent->set_effort_level(*host.startup_effort);
    if (!request.additional_directories.empty()) {
        agent->grant_workspace_paths(request.additional_directories);
    }

    try {
        connect_mcp_servers(request.mcp_servers, runtime.provider, agent->get_active_model_name());
    } catch (const std::exception& ex) {
        core::logging::warn("ACP MCP connect failed: {}", ex.what());
    }

    AgentSession::Options options{
        .yolo = host.yolo,
        .trusted_tools = host.trusted_tools,
        .created_at = resumed ? resumed->created_at : std::string{},
    };
    return std::make_unique<AgentSession>(
        std::move(agent),
        std::move(id),
        std::move(cwd),
        provider_name,
        std::move(options),
        std::move(replay));
}

namespace {

class StoreCatalog final : public SessionCatalog {
public:
    explicit StoreCatalog(std::filesystem::path dir) : dir_(std::move(dir)) {}

    [[nodiscard]] std::expected<SessionPage, std::string> list(
        const std::optional<std::filesystem::path>& cwd,
        std::string_view cursor) const override {
        // The cursor is the offset into the filtered, newest-first list.
        std::size_t offset = 0;
        if (!cursor.empty()) {
            const auto parsed = std::from_chars(cursor.data(), cursor.data() + cursor.size(), offset);
            if (parsed.ec != std::errc{} || parsed.ptr != cursor.data() + cursor.size()) {
                return std::unexpected(std::string("Invalid cursor"));
            }
        }
        SessionPage page;
        std::size_t seen = 0;
        for (const auto& info : core::session::SessionStore(dir_).list()) {
            // A conversation, not a session opened and never prompted.
            if ((info.turn_count <= 0 && info.preview.empty()) || info.working_dir.empty()) continue;
            if (cwd && !core::session::SessionStore::working_dirs_match(info.working_dir, cwd->string())) {
                continue;
            }
            if (seen++ < offset) continue;
            if (page.sessions.size() == kPageSize) {
                page.next_cursor = std::to_string(offset + kPageSize);
                break;
            }
            page.sessions.push_back(ListedSession{
                .session_id = info.session_id,
                .cwd = info.working_dir,
                .title = session_title(info.name.empty() ? info.preview : info.name),
                .updated_at = info.last_active_at.empty() ? info.created_at : info.last_active_at,
            });
        }
        return page;
    }

    void remove(std::string_view session_id) override {
        std::string error;
        if (!core::session::SessionStore(dir_).remove(session_id, &error)) {
            core::logging::warn("ACP session delete failed: {}", error);
        }
    }

private:
    static constexpr std::size_t kPageSize = 50;
    std::filesystem::path dir_;
};

} // namespace

std::unique_ptr<SessionCatalog> make_store_catalog(std::filesystem::path sessions_dir) {
    return std::make_unique<StoreCatalog>(std::move(sessions_dir));
}

} // namespace exec::acp
