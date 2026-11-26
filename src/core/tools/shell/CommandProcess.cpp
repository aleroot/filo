#include "CommandProcess.hpp"

#include "ShellUtils.hpp"
#include "../../landrun/LandrunEnvironment.hpp"
#include "../../landrun/LandrunLaunch.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <format>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace core::tools::shell {
namespace {

// After the process exits, output a backgrounded child still writes is
// collected this long (codex-rs unified_exec TRAILING_OUTPUT_GRACE), then the
// session ends even if that child keeps the pipe open.
constexpr auto kTrailingOutputGrace = std::chrono::milliseconds{100};
constexpr int kPollIntervalMs = 50;
constexpr std::size_t kReadChunkBytes = 64 * 1024;
constexpr auto kWriteTimeout = std::chrono::seconds{2};

[[nodiscard]] int exit_code_from_status(int status) noexcept {
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

void close_fd(int& fd) noexcept {
    if (fd >= 0) ::close(fd);
    fd = -1;
}

[[nodiscard]] bool set_cloexec(int fd) noexcept {
    const int flags = ::fcntl(fd, F_GETFD);
    return flags >= 0 && ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

} // namespace

std::unique_ptr<CommandProcess> CommandProcess::spawn(std::string_view command,
                                                      const std::filesystem::path& working_dir,
                                                      const core::landrun::LandrunPolicy& policy,
                                                      std::string& error) {
    std::string script;
    if (!working_dir.empty()) {
        script = std::format("cd -- '{}' || exit 1\n",
                             detail::shell_single_quote(working_dir.string()));
    }
    script += command;

    const bool has_bash = ::access("/bin/bash", X_OK) == 0;
    std::vector<std::string> arguments;
    if (has_bash) arguments = {"bash", "--norc", "--noprofile", "-c", std::move(script)};
    else arguments = {"sh", "-c", std::move(script)};

    core::landrun::LandrunLaunch launch;
    try {
        launch = core::landrun::prepare_landrun_launch(
            policy, has_bash ? "/bin/bash" : "/bin/sh", std::move(arguments));
    } catch (const std::exception& ex) {
        error = std::format("the sandbox could not prepare the command: {}", ex.what());
        return nullptr;
    }
    std::vector<char*> argv;
    argv.reserve(launch.arguments.size() + 1);
    for (auto& argument : launch.arguments) argv.push_back(argument.data());
    argv.push_back(nullptr);
    auto environment = core::landrun::build_landrun_environment(policy, environ);
    std::vector<char*> envp;
    envp.reserve(environment.size() + 1);
    for (auto& entry : environment) envp.push_back(entry.data());
    envp.push_back(nullptr);

    std::array<int, 2> in{-1, -1};
    std::array<int, 2> out{-1, -1};
    const auto close_all = [&] {
        for (int* fd : {&in[0], &in[1], &out[0], &out[1]}) close_fd(*fd);
    };
    // Every end is close-on-exec, so a command another thread starts never
    // inherits this one's pipes; dup2 clears the flag on the child's stdio.
    if (::pipe(in.data()) != 0 || ::pipe(out.data()) != 0
        || !set_cloexec(in[0]) || !set_cloexec(in[1])
        || !set_cloexec(out[0]) || !set_cloexec(out[1])) {
        error = std::format("could not create pipes: {}", std::strerror(errno));
        close_all();
        return nullptr;
    }

    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        close_all();
        error = "could not prepare the process";
        return nullptr;
    }
    if (::posix_spawnattr_init(&attributes) != 0) {
        ::posix_spawn_file_actions_destroy(&actions);
        close_all();
        error = "could not prepare the process";
        return nullptr;
    }
    sigset_t default_signals;
    short flags = POSIX_SPAWN_SETSIGDEF;
#ifdef POSIX_SPAWN_SETSID
    // A new session: no controlling terminal to block on, and the process
    // leads the group kill() signals.
    flags |= POSIX_SPAWN_SETSID;
#else
    flags |= POSIX_SPAWN_SETPGROUP;
#endif
    const bool ready =
        ::posix_spawn_file_actions_adddup2(&actions, in[0], STDIN_FILENO) == 0
        && ::posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO) == 0
        && ::posix_spawn_file_actions_adddup2(&actions, out[1], STDERR_FILENO) == 0
        && sigemptyset(&default_signals) == 0
        && sigaddset(&default_signals, SIGINT) == 0
        && sigaddset(&default_signals, SIGPIPE) == 0
        && ::posix_spawnattr_setsigdefault(&attributes, &default_signals) == 0
#ifndef POSIX_SPAWN_SETSID
        && ::posix_spawnattr_setpgroup(&attributes, 0) == 0
#endif
        && ::posix_spawnattr_setflags(&attributes, flags) == 0;
    pid_t pid = -1;
    const int spawn_error = ready
        ? ::posix_spawn(&pid, launch.executable.c_str(), &actions, &attributes,
                        argv.data(), envp.data())
        : EINVAL;
    ::posix_spawn_file_actions_destroy(&actions);
    ::posix_spawnattr_destroy(&attributes);
    if (spawn_error != 0) {
        close_all();
        error = std::format("could not start the command: {}", std::strerror(spawn_error));
        return nullptr;
    }
#ifndef POSIX_SPAWN_SETSID
    ::setpgid(pid, pid);
#endif

    close_fd(in[0]);
    close_fd(out[1]);
    if (const int flags = ::fcntl(in[1], F_GETFL); flags >= 0) {
        ::fcntl(in[1], F_SETFL, flags | O_NONBLOCK);
    }
    std::unique_ptr<CommandProcess> process(new CommandProcess());
    process->pid_ = pid;
    process->pgid_ = pid;
    process->stdin_fd_ = in[1];
    process->stdout_fd_ = out[0];
    process->reader_ = std::jthread([raw = process.get()](std::stop_token stop) {
        raw->read_loop(std::move(stop));
    });
    return process;
}

CommandProcess::~CommandProcess() {
    kill();
    reader_.request_stop();
    if (reader_.joinable()) reader_.join();
    // The reader stops without reaping when it is told to: the process was
    // just killed, so this wait is short.
    if (!exit_code_.has_value() && pid_ > 0) {
        int status = 0;
        while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
    }
    close_fd(stdin_fd_);
    close_fd(stdout_fd_);
}

void CommandProcess::read_loop(std::stop_token stop) {
    std::array<char, kReadChunkBytes> buffer{};
    bool eof = false;
    bool exited = false;
    std::optional<std::chrono::steady_clock::time_point> exited_at;
    while (!stop.stop_requested()) {
        if (!eof) {
            pollfd fd{.fd = stdout_fd_, .events = POLLIN, .revents = 0};
            const int ready = ::poll(&fd, 1, kPollIntervalMs);
            if (ready > 0) {
                const ssize_t n = ::read(stdout_fd_, buffer.data(), buffer.size());
                if (n > 0) append(std::string_view(buffer.data(), static_cast<std::size_t>(n)));
                else if (n == 0 || (errno != EINTR && errno != EAGAIN)) eof = true;
            } else if (ready < 0 && errno != EINTR) {
                eof = true;
            }
        }
        if (!exited) {
            // Notice the exit without reaping, then reap under the lock kill()
            // takes, so kill() never signals a group id already released.
            // Once the pipe has closed only the exit is left: block for it.
            siginfo_t info{};
            const int waited = ::waitid(P_PID, static_cast<id_t>(pid_), &info,
                                        WEXITED | WNOWAIT | (eof ? 0 : WNOHANG));
            const bool ended = (waited == 0 && info.si_pid == pid_)
                || (waited < 0 && errno != EINTR);
            if (ended) {
                exited = true;
                exited_at = std::chrono::steady_clock::now();
                std::lock_guard lock(mutex_);
                int status = 0;
                exit_code_ = ::waitpid(pid_, &status, 0) == pid_ ? exit_code_from_status(status)
                                                                : -1;
            }
        }
        if (exited && (eof || std::chrono::steady_clock::now() - *exited_at >= kTrailingOutputGrace)) {
            break;
        }
    }
    {
        std::lock_guard lock(mutex_);
        finished_ = true;
    }
    changed_.notify_all();
}

void CommandProcess::append(std::string_view bytes) {
    {
        std::lock_guard lock(mutex_);
        unread_.append(bytes);
        if (unread_.size() > kMaxUnreadBytes) {
            // Keep the head and the newest output; drop what lies between.
            const std::size_t head = kMaxUnreadBytes / 2;
            const std::size_t excess = unread_.size() - kMaxUnreadBytes;
            unread_.erase(head, excess);
            omitted_ += excess;
        }
    }
    changed_.notify_all();
}

std::string CommandProcess::take_unread() {
    std::string output;
    if (omitted_ > 0) {
        const std::size_t head = std::min(unread_.size(), kMaxUnreadBytes / 2);
        output = unread_.substr(0, head);
        output += std::format("…{} chars truncated…", omitted_);
        output.append(unread_, head);
    } else {
        output = std::move(unread_);
    }
    unread_.clear();
    omitted_ = 0;
    return output;
}

bool CommandProcess::write(std::string_view bytes) {
    std::lock_guard lock(write_mutex_);
    if (stdin_fd_ < 0) return false;
    // stdin is non-blocking: a process that stops reading cannot hold the
    // caller past this deadline once its pipe is full.
    const auto deadline = std::chrono::steady_clock::now() + kWriteTimeout;
    while (!bytes.empty()) {
        const ssize_t written = ::write(stdin_fd_, bytes.data(), bytes.size());
        if (written >= 0) {
            bytes.remove_prefix(static_cast<std::size_t>(written));
            continue;
        }
        if (errno == EINTR) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) return false;
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) return false;
        pollfd fd{.fd = stdin_fd_, .events = POLLOUT, .revents = 0};
        if (::poll(&fd, 1, static_cast<int>(left.count())) < 0 && errno != EINTR) return false;
    }
    return true;
}

CommandProcess::Taken CommandProcess::wait_and_take(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    interrupted_ = false;
    changed_.wait_until(lock, deadline, [this] { return finished_ || interrupted_; });
    Taken taken{.output = take_unread()};
    if (finished_) taken.exit_code = exit_code_.value_or(-1);
    return taken;
}

void CommandProcess::interrupt() {
    {
        std::lock_guard lock(mutex_);
        interrupted_ = true;
    }
    changed_.notify_all();
}

void CommandProcess::kill() noexcept {
    // Only while the process is unreaped: its group id cannot have been
    // handed to another process yet.
    std::lock_guard lock(mutex_);
    if (pgid_ > 0 && !exit_code_.has_value()) ::killpg(pgid_, SIGKILL);
}

bool CommandProcess::finished() const {
    std::lock_guard lock(mutex_);
    return finished_;
}

} // namespace core::tools::shell
