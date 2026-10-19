#include "ShellSessions.hpp"

#include "ShellExecutorFactory.hpp"
#include "ShellUtils.hpp"
#include "../../utils/PathUtils.hpp"

#include <algorithm>
#include <format>
#include <mutex>
#include <unordered_map>

namespace core::tools::shell {
namespace fs = std::filesystem;

struct ShellSessions::State {
    struct Session {
        explicit Session(std::unique_ptr<IShellExecutor> process) : executor(std::move(process)) {}
        std::unique_ptr<IShellExecutor> executor;
        std::mutex mutex;
        std::chrono::steady_clock::time_point last_used{std::chrono::steady_clock::now()};
        std::vector<fs::path> roots; // Guarded by State::mutex, like last_used.
    };
    mutable std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions;
    struct Activity {
        ActiveCommand command;
        const Session* owner;
    };
    std::unordered_map<std::string, Activity> active;
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
            if (!slot) slot = std::make_shared<State::Session>(make_shell_executor());
            session = slot;
            constexpr std::size_t max_sessions = 64;
            if (state_->sessions.size() > max_sessions) {
                auto oldest = std::ranges::min_element(state_->sessions, {},
                    [](const auto& entry) { return entry.second->last_used; });
                evicted = std::move(oldest->second);
                state_->sessions.erase(oldest);
            }
        }
        session->last_used = std::chrono::steady_clock::now();
        for (const fs::path& path : {workspace, fs::path(command.working_dir)}) {
            if (path.empty()) continue;
            const auto root = core::utils::path::directory_for(path);
            if (!std::ranges::contains(session->roots, root)) session->roots.push_back(root);
        }
    }
    // Eviction must never hold up unrelated commands behind the registry lock.
    evicted.reset();
    std::lock_guard execution_lock(session->mutex);
    struct Activity {
        State& state;
        const std::string& id;
        const State::Session* owner;
        ~Activity() {
            if (id.empty()) return;
            std::lock_guard lock(state.mutex);
            const auto it = state.active.find(id);
            if (it != state.active.end() && it->second.owner == owner) state.active.erase(it);
        }
    } activity{*state_, command.session_id, session.get()};
    if (!command.session_id.empty()) {
        std::lock_guard lock(state_->mutex);
        command.started_at = std::chrono::steady_clock::now();
        state_->active[command.session_id] = {command, session.get()};
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
        state_->active.erase(std::string(session_id));
    }
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
    for (const auto& [_, activity] : state_->active) commands.push_back(activity.command);
    std::ranges::sort(commands, {}, &ActiveCommand::started_at);
    return commands;
}

bool ShellSessions::quiesce_workspace(const fs::path& root) {
    const auto key = core::utils::path::directory_for(root);
    std::vector<std::shared_ptr<State::Session>> sessions;
    {
        std::lock_guard lock(state_->mutex);
        const auto collect = [&](const auto& session) {
            if (std::ranges::any_of(session->roots, [&](const auto& path) {
                    return core::utils::path::overlaps(path, key);
                })) sessions.push_back(session);
        };
        collect(state_->anonymous);
        for (const auto& [_, session] : state_->sessions) collect(session);
    }
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(sessions.size());
    for (const auto& session : sessions) {
        locks.emplace_back(session->mutex, std::try_to_lock);
        if (!locks.back().owns_lock()) return false;
    }
    return std::ranges::all_of(sessions, [](const auto& session) { return session->executor->quiesce(); });
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
