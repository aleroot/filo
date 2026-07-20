#include "LandrunReadiness.hpp"

#include "LandrunEnvironment.hpp"
#include "LandrunLaunch.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace core::landrun {

LandrunResult verify_landrun_readiness(const LandrunPolicy& policy) {
    if (!policy.enabled()) return {.success = true};

    try {
        const std::filesystem::path target =
            ::access("/usr/bin/true", X_OK) == 0 ? "/usr/bin/true" : "/bin/true";
        auto launch = prepare_landrun_launch(policy, target, {"true"});
        std::vector<char*> arguments;
        arguments.reserve(launch.arguments.size() + 1);
        for (auto& argument : launch.arguments) arguments.push_back(argument.data());
        arguments.push_back(nullptr);

        auto environment_storage = build_landrun_environment(policy, environ);
        std::vector<char*> environment;
        environment.reserve(environment_storage.size() + 1);
        for (auto& entry : environment_storage) environment.push_back(entry.data());
        environment.push_back(nullptr);

        int error_pipe[2];
        if (::pipe(error_pipe) != 0) {
            return {.success = false,
                    .detail = std::format("readiness pipe failed: {}",
                                          std::strerror(errno))};
        }

        posix_spawn_file_actions_t actions;
        const int actions_error = ::posix_spawn_file_actions_init(&actions);
        if (actions_error != 0) {
            ::close(error_pipe[0]);
            ::close(error_pipe[1]);
            return {.success = false,
                    .detail = std::format("readiness spawn setup failed: {}",
                                          std::strerror(actions_error))};
        }
        const bool actions_ready =
            ::posix_spawn_file_actions_adddup2(
                &actions, error_pipe[1], STDERR_FILENO) == 0
            && ::posix_spawn_file_actions_addclose(&actions, error_pipe[0]) == 0
            && ::posix_spawn_file_actions_addclose(&actions, error_pipe[1]) == 0;
        if (!actions_ready) {
            ::posix_spawn_file_actions_destroy(&actions);
            ::close(error_pipe[0]);
            ::close(error_pipe[1]);
            return {.success = false,
                    .detail = "readiness spawn actions could not be configured"};
        }

        pid_t child = -1;
        const int spawn_error = ::posix_spawn(
            &child,
            launch.executable.c_str(),
            &actions,
            nullptr,
            arguments.data(),
            environment.data());
        ::posix_spawn_file_actions_destroy(&actions);
        ::close(error_pipe[1]);
        if (spawn_error != 0) {
            ::close(error_pipe[0]);
            return {.success = false,
                    .detail = std::format("readiness helper failed to spawn: {}",
                                          std::strerror(spawn_error))};
        }
        const auto terminate_child = [&] {
            ::kill(child, SIGKILL);
            while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
        };

        std::string diagnostics;
        char buffer[1024];
        constexpr auto kReadinessTimeout = std::chrono::seconds{5};
        const auto deadline = std::chrono::steady_clock::now() + kReadinessTimeout;
        bool timed_out = false;
        while (true) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            if (remaining <= std::chrono::milliseconds::zero()) {
                timed_out = true;
                break;
            }
            pollfd descriptor{
                .fd = error_pipe[0],
                .events = POLLIN | POLLHUP,
                .revents = 0,
            };
            const int ready = ::poll(
                &descriptor, 1, static_cast<int>(remaining.count()));
            if (ready == 0) {
                timed_out = true;
                break;
            }
            if (ready < 0) {
                if (errno == EINTR) continue;
                const int poll_error = errno;
                terminate_child();
                ::close(error_pipe[0]);
                return {.success = false,
                        .detail = std::format("readiness diagnostics failed: {}",
                                              std::strerror(poll_error))};
            }
            const ssize_t count = ::read(error_pipe[0], buffer, sizeof(buffer));
            if (count > 0) {
                constexpr std::size_t kMaximumDiagnostics = 8192;
                const auto remaining = kMaximumDiagnostics - std::min(
                    diagnostics.size(), kMaximumDiagnostics);
                diagnostics.append(buffer, static_cast<std::size_t>(count)
                    > remaining ? remaining : static_cast<std::size_t>(count));
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) {
                const int read_error = errno;
                terminate_child();
                ::close(error_pipe[0]);
                return {.success = false,
                        .detail = std::format("readiness diagnostics failed: {}",
                                              std::strerror(read_error))};
            }
            break;
        }
        ::close(error_pipe[0]);

        if (timed_out) {
            terminate_child();
            return {.success = false,
                    .detail = "readiness helper timed out after 5 seconds"};
        }

        int status = 0;
        pid_t waited = -1;
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child) {
            return {.success = false,
                    .detail = std::format("readiness helper wait failed: {}",
                                          std::strerror(errno))};
        }
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            return {.success = true};
        }
        while (!diagnostics.empty()
               && (diagnostics.back() == '\n' || diagnostics.back() == '\r')) {
            diagnostics.pop_back();
        }
        const std::string outcome = WIFEXITED(status)
            ? std::format("exit {}", WEXITSTATUS(status))
            : WIFSIGNALED(status)
                ? std::format("signal {}", WTERMSIG(status))
                : "unknown status";
        return {.success = false,
                .detail = diagnostics.empty()
                    ? std::format("readiness helper failed with {}", outcome)
                    : std::format("readiness helper failed with {}: {}",
                                  outcome, diagnostics)};
    } catch (const std::exception& error) {
        return {.success = false,
                .detail = std::format("readiness check failed: {}", error.what())};
    }
}

} // namespace core::landrun
