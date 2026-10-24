#include "ShellSessions.hpp"

#include "ShellExecutorFactory.hpp"
#include "ShellUtils.hpp"

#include <algorithm>
#include <format>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace core::tools::shell {
namespace fs = std::filesystem;

struct ShellSessions::State {
    struct Session {
        explicit Session(std::unique_ptr<IShellExecutor> process) : executor(std::move(process)) {}
        std::unique_ptr<IShellExecutor> executor;
        std::string session_id; // Protected by the registry mutex, including while running.
        std::mutex mutex;
        std::chrono::steady_clock::time_point last_used{std::chrono::steady_clock::now()};
    };
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions;
    std::unordered_map<const Session*, ActiveCommand> active;
    std::shared_ptr<Session> anonymous;

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
    const std::function<void(int)>& before_dispatch) {
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
    auto result = session->executor->run(command.command, command.working_dir, timeout);
    state_->mark_used(session);
    return result;
}

void ShellSessions::clear(std::string_view session_id) {
    if (session_id.empty()) return;
    std::shared_ptr<State::Session> removed;
    {
        std::lock_guard lock(state_->mutex);
        if (auto it = state_->sessions.find(std::string(session_id)); it != state_->sessions.end()) {
            removed = std::move(it->second);
            state_->sessions.erase(it);
        }
        if (removed) state_->active.erase(removed.get());
    }
}

void ShellSessions::move(std::string_view from, std::string_view to) {
    if (from == to || from.empty() || to.empty()) return;
    const std::string source(from), destination(to);
    std::string identity(to), active_identity(to);
    std::lock_guard lock(state_->mutex);
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
    {
        std::lock_guard lock(state_->mutex);
        if (session_id.empty()) session = state_->anonymous;
        else if (const auto it = state_->sessions.find(std::string(session_id)); it != state_->sessions.end())
            session = it->second;
    }
    return session && session->executor->interrupt();
}

std::vector<ShellSessions::ActiveCommand> ShellSessions::active_commands() const {
    std::lock_guard lock(state_->mutex);
    std::vector<ActiveCommand> commands;
    commands.reserve(state_->active.size());
    for (const auto& [_, command] : state_->active) commands.push_back(command);
    std::ranges::sort(commands, {}, &ActiveCommand::started_at);
    return commands;
}

bool ShellSessions::quiesce_session(std::string_view session_id) {
    std::shared_ptr<State::Session> session;
    {
        std::lock_guard lock(state_->mutex);
        if (session_id.empty()) session = state_->anonymous;
        else if (const auto it = state_->sessions.find(std::string(session_id)); it != state_->sessions.end())
            session = it->second;
    }
    if (!session) return true;
    std::unique_lock lock(session->mutex, std::try_to_lock);
    return lock.owns_lock() && session->executor->quiesce();
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
