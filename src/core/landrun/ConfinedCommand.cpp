#include "ConfinedCommand.hpp"
#include "LandrunEnvironment.hpp"
#include "LandrunLaunch.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace core::landrun {

ConfinedCommandResult run_confined_command(
    const LandrunPolicy& policy, std::string_view command,
    const std::function<void(std::string_view)>& consume,
    std::chrono::milliseconds timeout) {
    if (!policy.confines() || policy.allow_network || !consume || timeout.count() <= 0)
        return {.error = "a network-denying confinement policy and a valid consumer are required"};

    int pipefd[2] = {-1, -1};
    pid_t child = -1;
    bool reaped = false;
    auto close_pipe = [&] {
        for (int& fd : pipefd) if (fd >= 0) { ::close(fd); fd = -1; }
    };
    auto terminate = [&] {
        if (child > 0) {
            // Kill the spawned process group as well as its leader. Descendants
            // that deliberately detach may remain sandboxed but are not
            // guaranteed to be terminated by this operation.
            (void)::kill(-child, SIGKILL);
            if (!reaped) {
                (void)::kill(child, SIGKILL);
                while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
                reaped = true;
            }
        }
        close_pipe();
    };
    try {
        auto launch = prepare_landrun_launch(policy, "/bin/sh", {"sh", "-c", std::string(command)});
        auto env_storage = build_landrun_environment(policy, environ);
        std::vector<char*> argv;
        for (auto& value : launch.arguments) argv.push_back(value.data());
        argv.push_back(nullptr);
        std::vector<char*> env;
        for (auto& value : env_storage) env.push_back(value.data());
        env.push_back(nullptr);
        if (::pipe(pipefd) != 0) return {.error = "could not create output pipe"};
        if (pipefd[0] <= STDERR_FILENO || pipefd[1] <= STDERR_FILENO) {
            close_pipe();
            return {.error = "standard descriptors must be open"};
        }

        posix_spawn_file_actions_t actions;
        if (::posix_spawn_file_actions_init(&actions) != 0) {
            close_pipe();
            return {.error = "could not initialize spawn actions"};
        }
        const bool actions_ok =
            ::posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0) == 0
            && ::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO) == 0
            && ::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO) == 0
            && ::posix_spawn_file_actions_addclose(&actions, pipefd[0]) == 0
            && ::posix_spawn_file_actions_addclose(&actions, pipefd[1]) == 0;
        posix_spawnattr_t attr;
        const bool attr_initialized = ::posix_spawnattr_init(&attr) == 0;
        const bool attr_ok = attr_initialized
            && ::posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP) == 0
            && ::posix_spawnattr_setpgroup(&attr, 0) == 0;
        int spawn_error = EINVAL;
        if (actions_ok && attr_ok)
            spawn_error = ::posix_spawn(&child, launch.executable.c_str(), &actions, &attr,
                                        argv.data(), env.data());
        if (attr_initialized) ::posix_spawnattr_destroy(&attr);
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(pipefd[1]);
        pipefd[1] = -1;
        if (spawn_error != 0) {
            close_pipe();
            return {.error = "could not start confined command"};
        }

        const auto deadline = std::chrono::steady_clock::now() + timeout;
        bool eof = false;
        int status = 0;
        std::array<char, 8192> buffer{};
        while (!eof || !reaped) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                terminate();
                return {.timed_out = true, .error = "confined command timed out"};
            }
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            const int delay = static_cast<int>(std::min(remaining, std::chrono::milliseconds(100)).count());
            pollfd descriptor{.fd = eof ? -1 : pipefd[0], .events = POLLIN | POLLHUP, .revents = 0};
            const int ready = ::poll(&descriptor, 1, delay);
            if (ready < 0 && errno != EINTR) {
                terminate();
                return {.error = "failed reading confined command output"};
            }
            if (ready > 0 && (descriptor.revents & (POLLIN | POLLHUP))) {
                const auto count = ::read(pipefd[0], buffer.data(), buffer.size());
                if (count > 0) consume(std::string_view(buffer.data(), static_cast<std::size_t>(count)));
                else if (count == 0) eof = true;
                else if (errno != EINTR) {
                    terminate();
                    return {.error = "failed reading confined command output"};
                }
            }
            if (descriptor.revents & (POLLERR | POLLNVAL)) {
                terminate();
                return {.error = "confined output pipe failed"};
            }
            // Keep the leader's PID reserved until the pipe closes, so a
            // timeout cannot signal an unrelated reused process group.
            if (eof && !reaped) {
                const auto waited = ::waitpid(child, &status, WNOHANG);
                if (waited == child) reaped = true;
                else if (waited < 0 && errno != EINTR) {
                    terminate();
                    return {.error = "could not collect confined command"};
                }
            }
        }
        close_pipe();
        return {.exit_code = WIFEXITED(status) ? WEXITSTATUS(status)
                                                : WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1};
    } catch (...) {
        terminate();
        return {.error = "confined command setup or output handling failed"};
    }
}

} // namespace core::landrun
