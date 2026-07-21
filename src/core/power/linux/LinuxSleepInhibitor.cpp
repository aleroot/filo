#include "LinuxSleepInhibitor.hpp"

#include "core/logging/Logger.hpp"

#if defined(__linux__)
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <expected>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>

extern char** environ;
#endif

#include <memory>
#include <string>

namespace core::power {

#if defined(__linux__)
namespace {

using namespace std::chrono_literals;

[[nodiscard]] bool set_close_on_exec(int fd) noexcept {
    const int flags = ::fcntl(fd, F_GETFD);
    return flags >= 0 && ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

class Pipe final {
public:
    [[nodiscard]] static std::expected<Pipe, int> create() noexcept {
        int pipe_fds[2] = {-1, -1};
        if (::pipe(pipe_fds) != 0) {
            return std::unexpected(errno);
        }
        if (!set_close_on_exec(pipe_fds[0]) || !set_close_on_exec(pipe_fds[1])) {
            const int error = errno;
            ::close(pipe_fds[0]);
            ::close(pipe_fds[1]);
            return std::unexpected(error);
        }
        return Pipe(pipe_fds[0], pipe_fds[1]);
    }

    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    Pipe(Pipe&& other) noexcept
        : read_fd_(std::exchange(other.read_fd_, -1))
        , write_fd_(std::exchange(other.write_fd_, -1)) {}

    Pipe& operator=(Pipe&& other) noexcept {
        if (this != &other) {
            close_read();
            close_write();
            read_fd_ = std::exchange(other.read_fd_, -1);
            write_fd_ = std::exchange(other.write_fd_, -1);
        }
        return *this;
    }

    ~Pipe() {
        close_read();
        close_write();
    }

    [[nodiscard]] int read_fd() const noexcept { return read_fd_; }
    [[nodiscard]] int write_fd() const noexcept { return write_fd_; }

    void close_read() noexcept { close_fd(read_fd_); }
    void close_write() noexcept { close_fd(write_fd_); }

    [[nodiscard]] int release_write() noexcept {
        return std::exchange(write_fd_, -1);
    }

private:
    Pipe(int read_fd, int write_fd) noexcept
        : read_fd_(read_fd), write_fd_(write_fd) {}

    static void close_fd(int& fd) noexcept {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    int read_fd_ = -1;
    int write_fd_ = -1;
};

class SpawnFileActions final {
public:
    SpawnFileActions() noexcept
        : error_(::posix_spawn_file_actions_init(&actions_)) {}

    SpawnFileActions(const SpawnFileActions&) = delete;
    SpawnFileActions& operator=(const SpawnFileActions&) = delete;

    ~SpawnFileActions() {
        if (error_ == 0) {
            ::posix_spawn_file_actions_destroy(&actions_);
        }
    }

    [[nodiscard]] int error() const noexcept { return error_; }

    [[nodiscard]] int duplicate(int fd, int target_fd) noexcept {
        return ::posix_spawn_file_actions_adddup2(&actions_, fd, target_fd);
    }

    [[nodiscard]] int close(int fd) noexcept {
        return ::posix_spawn_file_actions_addclose(&actions_, fd);
    }

    [[nodiscard]] const posix_spawn_file_actions_t* native_handle() const noexcept {
        return &actions_;
    }

private:
    posix_spawn_file_actions_t actions_{};
    int error_ = 0;
};

[[nodiscard]] bool wait_for_child(
    pid_t process_id,
    std::chrono::milliseconds timeout) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    int status = 0;

    do {
        const pid_t result = ::waitpid(process_id, &status, WNOHANG);
        if (result == process_id) {
            return true;
        }
        if (result < 0 && errno != EINTR) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(10ms);
    } while (true);
}

void terminate_and_reap(pid_t process_id) noexcept {
    if (process_id <= 0 || wait_for_child(process_id, 200ms)) {
        return;
    }
    (void)::kill(process_id, SIGTERM);
    if (wait_for_child(process_id, 200ms)) {
        return;
    }
    (void)::kill(process_id, SIGKILL);
    (void)wait_for_child(process_id, 100ms);
}

class LinuxSleepInhibitionLease final : public SleepInhibitionLease {
public:
    LinuxSleepInhibitionLease(pid_t process_id, int input_fd) noexcept
        : process_id_(process_id), input_fd_(input_fd) {}

    ~LinuxSleepInhibitionLease() override {
        if (input_fd_ >= 0) {
            ::close(input_fd_);
        }
        if (process_id_ <= 0) {
            return;
        }
        terminate_and_reap(process_id_);
    }

    [[nodiscard]] bool is_active() const noexcept override {
        return process_id_ > 0 && input_fd_ >= 0;
    }

private:
    pid_t process_id_ = -1;
    int input_fd_ = -1;
};

[[nodiscard]] bool wait_for_readiness(int ready_fd) noexcept {
    pollfd descriptor{
        .fd = ready_fd,
        .events = POLLIN,
        .revents = 0,
    };

    int poll_result = 0;
    do {
        poll_result = ::poll(&descriptor, 1, 1500);
    } while (poll_result < 0 && errno == EINTR);
    if (poll_result <= 0 || (descriptor.revents & (POLLIN | POLLHUP)) == 0) {
        return false;
    }

    char ready = '\0';
    ssize_t bytes_read = 0;
    do {
        bytes_read = ::read(ready_fd, &ready, 1);
    } while (bytes_read < 0 && errno == EINTR);
    return bytes_read == 1 && ready == 'R';
}

} // namespace
#endif

std::shared_ptr<SleepInhibitionLease>
LinuxSleepInhibitor::inhibit(std::string_view reason) {
#if !defined(__linux__)
    (void)reason;
    return {};
#else
    // systemd-inhibit talks to logind. Avoid launching a helper when systemd
    // is not the active system manager (for example in minimal CI containers).
    if (::access("/run/systemd/system", F_OK) != 0) {
        return {};
    }

    auto input_pipe_result = Pipe::create();
    if (!input_pipe_result.has_value()) {
        core::logging::warn(
            "Could not create the Linux sleep-inhibition input pipe: {}",
            std::strerror(input_pipe_result.error()));
        return {};
    }
    auto ready_pipe_result = Pipe::create();
    if (!ready_pipe_result.has_value()) {
        core::logging::warn(
            "Could not create the Linux sleep-inhibition readiness pipe: {}",
            std::strerror(ready_pipe_result.error()));
        return {};
    }
    Pipe input_pipe = std::move(*input_pipe_result);
    Pipe ready_pipe = std::move(*ready_pipe_result);

    SpawnFileActions actions;
    if (actions.error() != 0) {
        core::logging::warn(
            "Could not prepare the Linux sleep inhibitor: {}",
            std::strerror(actions.error()));
        return {};
    }

    int action_error = actions.duplicate(input_pipe.read_fd(), STDIN_FILENO);
    if (action_error == 0) {
        action_error = actions.duplicate(ready_pipe.write_fd(), STDOUT_FILENO);
    }
    for (const int fd : {
             input_pipe.read_fd(),
             input_pipe.write_fd(),
             ready_pipe.read_fd(),
             ready_pipe.write_fd()}) {
        if (action_error == 0 && fd != STDIN_FILENO && fd != STDOUT_FILENO) {
            action_error = actions.close(fd);
        }
    }

    std::string why_argument = "--why=" + (
        reason.empty() ? std::string("Filo is working") : std::string(reason));
    char executable[] = "systemd-inhibit";
    char what_argument[] = "--what=sleep";
    char who_argument[] = "--who=filo";
    char mode_argument[] = "--mode=block";
    char shell[] = "/bin/sh";
    char shell_option[] = "-c";
    char shell_command[] = "printf R; exec /bin/cat";
    char* arguments[] = {
        executable,
        what_argument,
        who_argument,
        why_argument.data(),
        mode_argument,
        shell,
        shell_option,
        shell_command,
        nullptr,
    };

    pid_t process_id = -1;
    int spawn_error = action_error;
    if (spawn_error == 0) {
        spawn_error = ::posix_spawnp(
            &process_id,
            executable,
            actions.native_handle(),
            nullptr,
            arguments,
            environ);
    }
    input_pipe.close_read();
    ready_pipe.close_write();

    if (spawn_error != 0) {
        core::logging::warn(
            "Could not start the Linux systemd sleep inhibitor: {}",
            std::strerror(spawn_error));
        return {};
    }

    const bool ready = wait_for_readiness(ready_pipe.read_fd());
    ready_pipe.close_read();
    if (!ready) {
        input_pipe.close_write();
        terminate_and_reap(process_id);
        core::logging::warn(
            "Linux systemd sleep inhibitor did not acquire its lock in time");
        return {};
    }

    return std::make_shared<LinuxSleepInhibitionLease>(
        process_id,
        input_pipe.release_write());
#endif
}

} // namespace core::power
