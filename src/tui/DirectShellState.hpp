#pragma once

#include "core/tools/shell/ShellSessions.hpp"
#include "core/changes/MutationObservations.hpp"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>

namespace tui {

// The UI owns this service; persistent processes belong to sessions, just as
// they do for agent shell tools. Only shutdown accounting is shared by tabs.
class DirectShellState final {
public:
    explicit DirectShellState(std::shared_ptr<core::changes::MutationObservations> observations = {})
        : observations_(observations ? std::move(observations) : std::make_shared<core::changes::MutationObservations>()) {}

    void begin_worker() {
        std::lock_guard lock(mutex_);
        ++workers_;
    }

    void finish_worker() {
        {
            std::lock_guard lock(mutex_);
            if (workers_ > 0) --workers_;
        }
        idle_.notify_all();
    }

    [[nodiscard]] std::optional<core::tools::shell::ShellSessions::ActiveCommand>
    active_info(std::string_view session_id) const {
        const auto commands = shells_.active_commands();
        const auto found = std::ranges::find(commands, session_id,
            &core::tools::shell::ShellSessions::ActiveCommand::session_id);
        if (found == commands.end()) return std::nullopt;
        return *found;
    }

    [[nodiscard]] bool has_active_for(std::string_view session_id) const {
        return active_info(session_id).has_value();
    }

    bool interrupt_active_for(std::string_view session_id) {
        return shells_.interrupt(session_id);
    }

    void clear_session_state(std::string_view session_id) {
        shells_.clear(session_id);
    }

    void move_session_state(std::string_view from, std::string_view to) {
        shells_.move(from, to);
    }

    void interrupt_active() {
        for (const auto& command : shells_.active_commands()) {
            (void)shells_.interrupt(command.session_id);
        }
    }

    void wait_until_idle() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [this] { return workers_ == 0; });
    }

    bool quiesce_session(std::string_view session_id) {
        return shells_.quiesce_session(session_id);
    }

    core::tools::shell::IShellExecutor::Result run(
        std::string_view command, std::string_view working_dir, std::string session_id,
        std::chrono::milliseconds timeout = core::tools::shell::IShellExecutor::kDefaultTimeout) {
        if (session_id.empty()) return {"Direct shells require a session id.", -1};
        try {
            const std::vector<std::filesystem::path> paths{std::filesystem::path(working_dir)};
            const auto observation = observations_->observe({}, paths);
            return shells_.run({
                .session_id = std::move(session_id),
                .command = std::string(command),
                .working_dir = std::string(working_dir),
            }, std::filesystem::path(working_dir), {}, timeout);
        } catch (const std::exception& error) {
            return {error.what(), -1};
        }
    }

private:
    core::tools::shell::ShellSessions shells_;
    std::shared_ptr<core::changes::MutationObservations> observations_;
    std::mutex mutex_;
    std::condition_variable idle_;
    std::size_t workers_ = 0;
};

} // namespace tui
