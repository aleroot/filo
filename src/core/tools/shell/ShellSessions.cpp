#include "ShellSessions.hpp"

#include "CommandProcess.hpp"
#include "ShellExecutorFactory.hpp"
#include "ShellUtils.hpp"
#include "../../utils/ProcessUtils.hpp"

#include <algorithm>
#include <format>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <signal.h>

namespace core::tools::shell {
namespace fs = std::filesystem;

struct ShellSessions::CommandEntry {
    std::shared_ptr<CommandProcess> process;
    std::string owner; // Protected by the registry mutex.
    std::string text;
    std::string working_dir;
    std::chrono::steady_clock::time_point last_used{std::chrono::steady_clock::now()};
    std::mutex step; // One call at a time reads a session.
};

struct ShellSessions::State {
    struct Session {
        explicit Session(std::unique_ptr<IShellExecutor> process) : executor(std::move(process)) {}
        std::unique_ptr<IShellExecutor> executor;
        std::string session_id; // Protected by the registry mutex, including while running.
        std::mutex mutex;
        std::chrono::steady_clock::time_point last_used{std::chrono::steady_clock::now()};
    };
    using Command = CommandEntry;
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions;
    std::unordered_map<const Session*, ActiveCommand> active;
    std::shared_ptr<Session> anonymous;
    std::unordered_map<int, std::shared_ptr<Command>> commands;
    std::unordered_map<const Command*, ActiveCommand> command_steps;
    std::mt19937 process_ids{std::random_device{}()};

    /// codex-rs process_manager.rs allocate_process_id: random, 1000-99999.
    [[nodiscard]] int allocate_process_id() {
        std::uniform_int_distribution<int> range(1'000, 99'999);
        int id = range(process_ids);
        while (commands.contains(id)) id = range(process_ids);
        return id;
    }

    /// Makes room for one more session: an ended one first, else the least
    /// recently used one no call is reading. The caller destroys it unlocked.
    [[nodiscard]] std::shared_ptr<Command> evict_for_new_command() {
        if (commands.size() < kMaxCommandSessions) return {};
        auto victim = commands.end();
        for (auto it = commands.begin(); it != commands.end(); ++it) {
            if (it->second.use_count() != 1) continue;
            if (it->second->process->finished()) { victim = it; break; }
            if (victim == commands.end() || it->second->last_used < victim->second->last_used)
                victim = it;
        }
        if (victim == commands.end()) return {};
        auto removed = std::move(victim->second);
        commands.erase(victim);
        return removed;
    }

    /// Removes every session @p owner holds, killing them now; they are
    /// destroyed when the last call reading one lets go.
    [[nodiscard]] std::vector<std::shared_ptr<Command>> take_commands(std::string_view owner) {
        std::vector<std::shared_ptr<Command>> removed;
        for (auto it = commands.begin(); it != commands.end();) {
            if (it->second->owner == owner) {
                it->second->process->kill();
                it->second->process->interrupt();
                removed.push_back(std::move(it->second));
                it = commands.erase(it);
            } else {
                ++it;
            }
        }
        return removed;
    }

    void mark_used(const std::shared_ptr<Session>& session) {
        std::lock_guard lock(mutex);
        session->last_used = std::chrono::steady_clock::now();
    }
};

ShellSessions::ShellSessions() : ShellSessions(make_shell_executor()) {}
ShellSessions::ShellSessions(std::unique_ptr<IShellExecutor> anonymous_executor)
    : state_(std::make_unique<State>()) {
    state_->anonymous = std::make_shared<State::Session>(
        anonymous_executor ? std::move(anonymous_executor) : make_shell_executor());
}
ShellSessions::~ShellSessions() = default;

IShellExecutor::Result ShellSessions::run(ActiveCommand command, const fs::path& workspace,
    const core::landrun::LandrunPolicy& policy, std::chrono::milliseconds timeout,
    const std::function<void(int)>& before_dispatch, const std::function<bool()>& cancelled) {
    std::shared_ptr<State::Session> session, evicted;
    {
        std::lock_guard lock(state_->mutex);
        if (command.session_id.empty()) {
            session = state_->anonymous;
        } else {
            auto& slot = state_->sessions[command.session_id];
            if (!slot) {
                slot = std::make_shared<State::Session>(make_shell_executor());
                slot->session_id = command.session_id;
            }
            session = slot;
            constexpr std::size_t max_sessions = 64;
            if (state_->sessions.size() > max_sessions) {
                auto oldest = state_->sessions.end();
                for (auto it = state_->sessions.begin(); it != state_->sessions.end(); ++it) {
                    if (it->second.use_count() != 1) continue;
                    if (oldest == state_->sessions.end() || it->second->last_used < oldest->second->last_used)
                        oldest = it;
                }
                if (oldest != state_->sessions.end()) {
                    evicted = std::move(oldest->second);
                    state_->sessions.erase(oldest);
                }
            }
        }
        session->last_used = std::chrono::steady_clock::now();

    }
    // Eviction must never hold up unrelated commands behind the registry lock.
    evicted.reset();
    std::lock_guard execution_lock(session->mutex);
    struct Activity {
        State& state;
        const State::Session* owner;
        ~Activity() {
            std::lock_guard lock(state.mutex);
            state.active.erase(owner);
        }
    } activity{*state_, session.get()};
    if (!command.session_id.empty()) {
        std::lock_guard lock(state_->mutex);
        command.session_id = session->session_id;
        command.started_at = std::chrono::steady_clock::now();
        state_->active[session.get()] = command;
    }
    session->executor->configure_landrun(policy);
    if (!session->executor->is_alive() && (!command.session_id.empty() || before_dispatch)) {
        (void)session->executor->run(std::format("cd '{}'", detail::shell_single_quote(workspace.string())),
                                    {}, std::chrono::seconds(5));
    }
    if (before_dispatch) before_dispatch(session->executor->process_group());
    auto result = session->executor->run(command.command, command.working_dir, timeout, cancelled);
    state_->mark_used(session);
    return result;
}

ShellSessions::CommandStep ShellSessions::start_command(ActiveCommand command,
    const fs::path& workspace, const core::landrun::LandrunPolicy& policy,
    std::chrono::milliseconds yield, bool tty, const std::function<void(int)>& before_dispatch) {
    const auto started = std::chrono::steady_clock::now();
    std::string error;
    std::shared_ptr<CommandProcess> process = CommandProcess::spawn(
        command.command,
        command.working_dir.empty() ? workspace : fs::path(command.working_dir),
        policy, tty, error);
    if (!process) return {.error = std::move(error)};
    if (before_dispatch) before_dispatch(process->process_group());

    auto entry = std::make_shared<CommandEntry>();
    entry->process = std::move(process);
    entry->owner = command.session_id;
    entry->text = command.command;
    entry->working_dir = command.working_dir;
    std::shared_ptr<CommandEntry> evicted;
    int id = 0;
    {
        std::lock_guard lock(state_->mutex);
        evicted = state_->evict_for_new_command();
        id = state_->allocate_process_id();
        state_->commands.emplace(id, entry);
    }
    evicted.reset();
    auto step = step_command(id, entry, {}, yield, std::move(command.tool_call_id));
    step.wall_time = std::chrono::steady_clock::now() - started;
    return step;
}

ShellSessions::CommandStep ShellSessions::continue_command(std::string_view session_id,
    int process_id, std::string_view input, std::chrono::milliseconds yield,
    std::string tool_call_id) {
    const auto started = std::chrono::steady_clock::now();
    std::shared_ptr<CommandEntry> entry;
    {
        std::lock_guard lock(state_->mutex);
        if (const auto it = state_->commands.find(process_id);
            it != state_->commands.end() && it->second->owner == session_id) {
            entry = it->second;
        }
    }
    // codex-rs UnifiedExecError::UnknownProcessId.
    if (!entry) return {.error = std::format("Unknown process id {}", process_id)};
    auto step = step_command(process_id, entry, input, yield, std::move(tool_call_id));
    step.wall_time = std::chrono::steady_clock::now() - started;
    return step;
}

ShellSessions::CommandStep ShellSessions::step_command(int id,
    const std::shared_ptr<CommandEntry>& entry, std::string_view input,
    std::chrono::milliseconds yield, std::string tool_call_id) {
    std::lock_guard step_lock(entry->step);
    {
        std::lock_guard lock(state_->mutex);
        state_->command_steps[entry.get()] = ActiveCommand{
            .session_id = entry->owner,
            .command = entry->text,
            .working_dir = entry->working_dir,
            .tool_call_id = std::move(tool_call_id),
            .started_at = std::chrono::steady_clock::now(),
        };
    }
    struct Activity {
        State& state;
        const CommandEntry* entry;
        ~Activity() {
            std::lock_guard lock(state.mutex);
            state.command_steps.erase(entry);
        }
    } activity{*state_, entry.get()};

    CommandStep step;
    // A process that ended still has its output read; one that runs but
    // stopped reading its stdin is reported.
    if (!input.empty() && !entry->process->write(input) && !entry->process->finished()) {
        return {.error = std::format("Session {} did not accept the input.", id)};
    }
    auto taken = entry->process->wait_and_take(std::chrono::steady_clock::now() + yield);
    step.output = std::move(taken.output);
    std::lock_guard lock(state_->mutex);
    entry->last_used = std::chrono::steady_clock::now();
    if (taken.exit_code) {
        step.exit_code = taken.exit_code;
        if (const auto it = state_->commands.find(id);
            it != state_->commands.end() && it->second == entry) {
            state_->commands.erase(it);
        }
    } else if (const auto it = state_->commands.find(id);
               it != state_->commands.end() && it->second == entry) {
        step.process_id = id;
    } else {
        // Cleared or evicted while this call waited: it was killed.
        step.exit_code = -1;
    }
    return step;
}

void ShellSessions::clear(std::string_view session_id) {
    if (session_id.empty()) return;
    std::shared_ptr<State::Session> removed;
    std::vector<std::shared_ptr<CommandEntry>> commands;
    {
        std::lock_guard lock(state_->mutex);
        if (auto it = state_->sessions.find(std::string(session_id)); it != state_->sessions.end()) {
            removed = std::move(it->second);
            state_->sessions.erase(it);
        }
        if (removed) state_->active.erase(removed.get());
        commands = state_->take_commands(session_id);
    }
}

void ShellSessions::move(std::string_view from, std::string_view to) {
    if (from == to || from.empty() || to.empty()) return;
    const std::string source(from), destination(to);
    std::string identity(to), active_identity(to);
    std::lock_guard lock(state_->mutex);
    for (auto& [_, command] : state_->commands) {
        if (command->owner == source) command->owner = destination;
    }
    for (auto& [_, step] : state_->command_steps) {
        if (step.session_id == source) step.session_id = destination;
    }
    const auto it = state_->sessions.find(source);
    if (it == state_->sessions.end()) return;
    if (state_->sessions.contains(destination))
        throw std::logic_error("Cannot replace another session's shell state.");
    const auto session = it->second;
    // Insert before erasing: allocation failure leaves the original owner intact.
    state_->sessions.emplace(destination, session);
    state_->sessions.erase(source);
    session->session_id.swap(identity);
    if (auto active = state_->active.find(session.get()); active != state_->active.end())
        active->second.session_id.swap(active_identity);
}

bool ShellSessions::interrupt(std::string_view session_id) {
    std::shared_ptr<State::Session> session;
    bool stopped_command = false;
    {
        std::lock_guard lock(state_->mutex);
        if (session_id.empty()) session = state_->anonymous;
        else if (const auto it = state_->sessions.find(std::string(session_id)); it != state_->sessions.end())
            session = it->second;
        // A command session a call is reading is the running tool: stop it.
        // One left running in the background between calls is not.
        for (const auto& [_, command] : state_->commands) {
            if (command->owner != session_id || !state_->command_steps.contains(command.get()))
                continue;
            command->process->kill();
            command->process->interrupt();
            stopped_command = true;
        }
    }
    const bool stopped_shell = session && session->executor->interrupt();
    return stopped_shell || stopped_command;
}

std::vector<ShellSessions::ActiveCommand> ShellSessions::active_commands() const {
    std::lock_guard lock(state_->mutex);
    std::vector<ActiveCommand> commands;
    commands.reserve(state_->active.size() + state_->command_steps.size());
    for (const auto& [_, command] : state_->active) commands.push_back(command);
    for (const auto& [_, step] : state_->command_steps) commands.push_back(step);
    std::ranges::sort(commands, {}, &ActiveCommand::started_at);
    return commands;
}

bool ShellSessions::quiesce_session(std::string_view session_id) {
    std::shared_ptr<State::Session> session;
    std::vector<std::shared_ptr<CommandEntry>> commands;
    {
        std::lock_guard lock(state_->mutex);
        if (session_id.empty()) session = state_->anonymous;
        else if (const auto it = state_->sessions.find(std::string(session_id)); it != state_->sessions.end())
            session = it->second;
        commands = state_->take_commands(session_id);
    }
    // Every command session this conversation started is stopped, and its
    // process group must be gone before anything is restored under it.
    std::vector<int> groups;
    for (const auto& command : commands) groups.push_back(command->process->process_group());
    commands.clear();
    bool commands_quiet = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    for (const int group : groups) {
        // A group that still has members keeps its id, so signalling it can
        // only reach what the command left running in the background.
        while (core::utils::process::group_may_exist(group)) {
            ::killpg(group, SIGKILL);
            if (std::chrono::steady_clock::now() >= deadline) {
                commands_quiet = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    if (!session) return commands_quiet;
    std::unique_lock lock(session->mutex, std::try_to_lock);
    return commands_quiet && lock.owns_lock() && session->executor->quiesce();
}

std::optional<std::chrono::steady_clock::time_point> ShellSessions::reap_idle(
    std::span<const std::string> session_ids, std::chrono::steady_clock::duration idle_timeout) {
    std::vector<std::shared_ptr<State::Session>> reaped;
    std::optional<std::chrono::steady_clock::time_point> next_deadline;
    {
        std::lock_guard lock(state_->mutex);
        const auto now = std::chrono::steady_clock::now();
        for (const auto& id : session_ids) {
            const auto it = state_->sessions.find(id);
            if (it == state_->sessions.end()) continue;
            const auto deadline = it->second->last_used + idle_timeout;
            if (deadline <= now && it->second.use_count() == 1) {
                reaped.push_back(std::move(it->second));
                state_->sessions.erase(it);
            } else if (!next_deadline || deadline < *next_deadline) next_deadline = deadline;
        }
    }
    return next_deadline;
}

} // namespace core::tools::shell
