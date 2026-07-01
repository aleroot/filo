#pragma once

#include <algorithm>
#include <atomic>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <format>
#include <poll.h>
#include <random>
#include <signal.h>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace core::tools::detail {

enum class WriteAllResult {
    success,
    failed_no_progress,
    failed_partial_progress,
};

template <typename WriteFn>
[[nodiscard]] inline WriteAllResult write_all_classified(std::string_view data, WriteFn&& write_fn) {
    const char* ptr = data.data();
    std::size_t remaining = data.size();
    bool wrote_any = false;
    while (remaining > 0) {
        const ssize_t written = write_fn(ptr, remaining);
        if (written < 0) {
            if (errno == EINTR) continue;
            return wrote_any
                ? WriteAllResult::failed_partial_progress
                : WriteAllResult::failed_no_progress;
        }
        if (written == 0) {
            return wrote_any
                ? WriteAllResult::failed_partial_progress
                : WriteAllResult::failed_no_progress;
        }
        wrote_any = true;
        ptr += written;
        remaining -= static_cast<std::size_t>(written);
    }
    return WriteAllResult::success;
}

[[nodiscard]] constexpr bool should_retry_stdin_write(WriteAllResult result) noexcept {
    return result == WriteAllResult::failed_no_progress;
}

/**
 * @brief Persistent bash subprocess with pipe-based I/O.
 *
 * Maintains a single long-lived bash process.  State (environment variables,
 * working directory, shell functions, aliases) persists across run() calls.
 *
 * ### Completion detection
 * After each user command a @c printf sentinel is injected into bash's stdin.
 * bash prints the sentinel line containing the exit code once it reaches that
 * instruction.  An EXIT trap is installed at session start so that commands
 * like @c exit @c N also produce the sentinel before the process terminates,
 * allowing correct exit-code reporting even when the session dies.
 *
 * ### Timeout handling
 * When a command exceeds its timeout, the entire bash process group is
 * killed with SIGKILL (compiler, spawned sub-processes included).  The session
 * is torn down so the next run() call gets a clean, fresh bash process.
 * The partial output collected up to that point is returned with exit_code=-1.
 *
 * ### Session restart
 * If bash exits for any reason (user called @c exit, OOM-kill, timeout kill,
 * etc.) the next call to run() transparently restarts the session.
 *
 * ### Process group
 * The child bash is placed in its own process group (setpgid) so that
 * killpg() on timeout terminates bash and every sub-process it spawned
 * (compilers, linkers, sub-makes, etc.) atomically.
 *
 * ### Thread safety
 * NOT thread-safe.  Callers must serialize access with a mutex.
 */
class ShellSession {
public:
    // 10-minute default — covers compilation, npm install, docker build, etc.
    // Callers can override per-command via the timeout argument to run().
    static constexpr std::chrono::milliseconds kDefaultTimeout{600'000};
    static constexpr std::size_t               kMaxOutput{4 * 1024 * 1024};

    struct Result {
        std::string output;
        int         exit_code{-1};
    };

    ShellSession()  = default;
    ~ShellSession() { stop(); }

    ShellSession(const ShellSession&)            = delete;
    ShellSession& operator=(const ShellSession&) = delete;

    /**
     * @brief Execute a shell command and return its merged stdout+stderr.
     *
     * Starts (or restarts) the bash session as needed.  If the command
     * exceeds @p timeout the bash process group is killed and the session is
     * reset; the next call gets a fresh session.
     *
     * @param command  Arbitrary bash command string.
     * @param timeout  Maximum time to wait for the command to complete.
     *                 Defaults to kDefaultTimeout (10 minutes).
     */
    Result run(std::string_view command,
               std::chrono::milliseconds timeout = kDefaultTimeout) {
        int stale_exit_code = -1;
        [[maybe_unused]] const bool reaped_stale =
            reap_if_exited(stale_exit_code, true);
        interrupted_.store(false, std::memory_order_release);
        if (!is_alive()) start();
        if (!is_alive()) {
            return {"[ShellSession] Failed to start bash process.\n", -1};
        }

        // Inject the user command followed by our sentinel.
        // printf is used (not echo) to avoid locale / -e flag variations.
        std::string full{command};
        full += "\nprintf '\\n";
        full += sentinel_;
        full += ":%d\\n' $?\n";

        // A stale zombie process can pass kill(pid, 0) briefly while stdin is
        // already closed. In that case, restart once and retry transparently.
        const auto send_command = [&]() {
            const WriteAllResult first_attempt = write_all(stdin_fd_, full);
            if (first_attempt == WriteAllResult::success) return true;
            if (!should_retry_stdin_write(first_attempt)) return false;

            stop();
            start();
            if (!is_alive()) return false;
            return write_all(stdin_fd_, full) == WriteAllResult::success;
        };

        if (!send_command()) {
            stop();
            return {"[ShellSession] Write to bash stdin failed.\n", -1};
        }

        return read_until_done(timeout);
    }

    /// Returns true if the bash process is still running.
    [[nodiscard]] bool is_alive() const {
        const pid_t pid = pid_.load(std::memory_order_acquire);
        return pid > 0 && ::kill(pid, 0) == 0;
    }

    /// Kill the current session and start a fresh one.
    void reset() {
        stop();
        start();
    }

    bool interrupt() {
        interrupted_.store(true, std::memory_order_release);
        const pid_t pgid = pgid_.load(std::memory_order_acquire);
        if (pgid <= 0) return false;
        return ::killpg(pgid, SIGKILL) == 0 || errno == ESRCH;
    }

private:
    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    static WriteAllResult write_all(int fd, std::string_view data) {
        return write_all_classified(data, [fd](const char* ptr, std::size_t remaining) {
            return ::write(fd, ptr, remaining);
        });
    }

    static std::string make_sentinel() {
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_int_distribution<uint64_t> dist;
        return std::format("FILO_END_{:016x}{:016x}", dist(rng), dist(rng));
    }

    void start() {
        sentinel_ = make_sentinel();

        int stdin_pipe[2];   // parent writes commands here
        int stdout_pipe[2];  // parent reads output from here

        if (::pipe(stdin_pipe) != 0) return;
        if (::pipe(stdout_pipe) != 0) {
            ::close(stdin_pipe[0]);
            ::close(stdin_pipe[1]);
            return;
        }

        const pid_t child_pid = ::fork();
        if (child_pid < 0) {
            ::close(stdin_pipe[0]);  ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]); ::close(stdout_pipe[1]);
            pid_.store(-1, std::memory_order_release);
            return;
        }

        if (child_pid == 0) {
            // ---- child ----
            // Place child in its own process group so killpg() on timeout
            // terminates bash AND all sub-processes it spawned.
            ::setpgid(0, 0);

            ::dup2(stdin_pipe[0],  STDIN_FILENO);
            ::dup2(stdout_pipe[1], STDOUT_FILENO);
            ::dup2(stdout_pipe[1], STDERR_FILENO);
            ::close(stdin_pipe[0]);  ::close(stdin_pipe[1]);
            ::close(stdout_pipe[0]); ::close(stdout_pipe[1]);

            ::setenv("HISTFILE", "/dev/null", 1);
            ::setenv("HISTSIZE", "0", 1);
            ::signal(SIGINT, SIG_DFL);

            const bool has_bash = ::access("/bin/bash", X_OK) == 0;
            if (has_bash) {
                ::execl("/bin/bash", "bash", "--norc", "--noprofile",
                        nullptr);
            }
            ::execl("/bin/sh", "sh", nullptr);
            ::_exit(127);
        }

        // ---- parent ----
        // Mirror setpgid so there is no race: if the parent calls killpg()
        // before the child's setpgid() completes, it still finds the group.
        ::setpgid(child_pid, child_pid);
        pid_.store(child_pid, std::memory_order_release);
        pgid_.store(child_pid, std::memory_order_release);

        ::close(stdin_pipe[0]);
        ::close(stdout_pipe[1]);
        stdin_fd_  = stdin_pipe[1];
        stdout_fd_ = stdout_pipe[0];

        // Install EXIT trap so that `exit N` still prints the sentinel before
        // the process terminates, giving us the correct exit code.
        std::string trap_cmd = "trap 'printf \"\\n" + sentinel_ +
                               ":%d\\n\" \"$?\"' EXIT\n";
        if (write_all(stdin_fd_, trap_cmd) != WriteAllResult::success) {
            stop();
            return;
        }
    }

    void close_pipes() noexcept {
        if (stdin_fd_ >= 0)  { ::close(stdin_fd_);  stdin_fd_  = -1; }
        if (stdout_fd_ >= 0) { ::close(stdout_fd_); stdout_fd_ = -1; }
    }

    void clear_process_state() noexcept {
        pid_.store(-1, std::memory_order_release);
        pgid_.store(-1, std::memory_order_release);
    }

    [[nodiscard]] bool wait_for_child_exit(pid_t pid,
                                           std::chrono::milliseconds wait) noexcept {
        if (pid <= 0) return true;
        const auto deadline = std::chrono::steady_clock::now() + wait;
        int status = 0;
        while (true) {
            pid_t ret = -1;
            do {
                ret = ::waitpid(pid, &status, WNOHANG);
            } while (ret < 0 && errno == EINTR);

            if (ret == pid || (ret < 0 && errno == ECHILD)) return true;
            if (ret < 0) return false;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            ::usleep(1'000);
        }
    }

    // Graceful shutdown (SIGTERM -> wait -> SIGKILL if still alive).
    // Kills the entire process group so no orphaned sub-processes are left.
    void stop() {
        close_pipes();
        const pid_t pid = pid_.load(std::memory_order_acquire);
        const pid_t pgid = pgid_.load(std::memory_order_acquire);
        if (pid > 0) {
            if (pgid > 0) ::killpg(pgid, SIGTERM);
            if (!wait_for_child_exit(pid, std::chrono::milliseconds{100}) && pgid > 0) {
                ::killpg(pgid, SIGKILL);
                [[maybe_unused]] const bool reaped =
                    wait_for_child_exit(pid, std::chrono::milliseconds{250});
            }
            clear_process_state();
        }
    }

    [[nodiscard]] static int exit_code_from_status(int status) noexcept {
        if (WIFEXITED(status)) return WEXITSTATUS(status);
        if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
        return -1;
    }

    [[nodiscard]] bool reap_if_exited(int& exit_code, bool kill_process_group) {
        const pid_t pid = pid_.load(std::memory_order_acquire);
        if (pid <= 0) return false;

        int status = 0;
        pid_t ret = -1;
        do {
            ret = ::waitpid(pid, &status, WNOHANG);
        } while (ret < 0 && errno == EINTR);

        if (ret == 0) return false;
        if (ret < 0 && errno != ECHILD) return false;

        exit_code = ret > 0 ? exit_code_from_status(status) : -1;
        const pid_t pgid = pgid_.load(std::memory_order_acquire);
        if (kill_process_group && pgid > 0) {
            ::killpg(pgid, SIGKILL);
        }
        close_pipes();
        pid_.store(-1, std::memory_order_release);
        pgid_.store(-1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool reap_if_exited_for(int& exit_code,
                                          bool kill_process_group,
                                          std::chrono::milliseconds wait) {
        const auto deadline = std::chrono::steady_clock::now() + wait;
        while (true) {
            if (reap_if_exited(exit_code, kill_process_group)) return true;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            ::usleep(1'000);
        }
    }

    // Immediate hard-kill used after a timeout.
    // No grace period - the command already exceeded its allowed time.
    void kill_session() {
        close_pipes();
        const pid_t pid = pid_.load(std::memory_order_acquire);
        const pid_t pgid = pgid_.load(std::memory_order_acquire);
        if (pid > 0) {
            if (pgid > 0) ::killpg(pgid, SIGKILL);
            [[maybe_unused]] const bool reaped =
                wait_for_child_exit(pid, std::chrono::milliseconds{250});
            clear_process_state();
        }
    }

    [[nodiscard]] bool consume_interrupt_requested() noexcept {
        return interrupted_.exchange(false, std::memory_order_acq_rel);
    }

    [[nodiscard]] Result interrupted_result(std::string raw) noexcept {
        raw += "\n[INTERRUPTED: command cancelled by user]\n";
        return {std::move(raw), -1};
    }

    // Drain stdout until the sentinel appears.
    // Called after output truncation to keep the session synchronised so the
    // next command can be run cleanly.
    [[nodiscard]] bool drain_sentinel_until(
        std::chrono::steady_clock::time_point deadline) {
        if (stdout_fd_ < 0) return false;

        const std::string prefix = "\n" + sentinel_ + ":";
        // Rolling tail — only needs to be large enough to detect the sentinel
        // across two consecutive read() chunks.
        std::string tail;
        tail.reserve(prefix.size() * 2 + 32);

        while (std::chrono::steady_clock::now() < deadline) {
            const auto remaining_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now()).count();
            if (remaining_ms <= 0) break;

            const int poll_ms = static_cast<int>(std::min<std::int64_t>(remaining_ms, 250));
            struct pollfd pfd{stdout_fd_, POLLIN | POLLHUP, 0};
            const int ret = ::poll(&pfd, 1, poll_ms);
            if (ret < 0 && errno == EINTR) continue;
            if (ret < 0) break;
            if (ret == 0) {
                int ignored_exit_code = -1;
                if (reap_if_exited(ignored_exit_code, true)) break;
                continue;
            }

            std::array<char, 4096> buf{};
            const ssize_t n = ::read(stdout_fd_, buf.data(), buf.size());
            if (n <= 0) {
                int ignored_exit_code = -1;
                [[maybe_unused]] const bool reaped = reap_if_exited_for(
                    ignored_exit_code,
                    true,
                    std::chrono::milliseconds{50});
                break;  // bash closed its end (exited)
            }

            tail.append(buf.data(), static_cast<std::size_t>(n));
            // Keep a rolling window just large enough to detect the sentinel
            // even when it spans two consecutive reads.
            const std::size_t keep = prefix.size() + 32;
            if (tail.size() > keep * 2)
                tail = tail.substr(tail.size() - keep);

            if (tail.find(prefix) != std::string::npos) return true;
        }
        return false;
    }

    Result read_until_done(std::chrono::milliseconds timeout) {
        std::string raw;
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        const std::string prefix = "\n" + sentinel_ + ":";

        while (true) {
            // Check for sentinel in accumulated output.
            const auto pos = raw.find(prefix);
            if (pos != std::string::npos) {
                const std::size_t code_start = pos + prefix.size();
                std::size_t code_end = code_start;
                while (code_end < raw.size() && raw[code_end] != '\n')
                    ++code_end;

                int exit_code = 0;
                try {
                    exit_code = std::stoi(
                        raw.substr(code_start, code_end - code_start));
                } catch (...) {}

                std::string output = raw.substr(0, pos);

                // If the command exited the persistent shell, tear the session
                // down now. Descendants can keep stdout open after bash exits,
                // so EOF is not a reliable liveness signal.
                int ignored_exit_code = -1;
                [[maybe_unused]] const bool reaped =
                    reap_if_exited_for(
                        ignored_exit_code,
                        true,
                        std::chrono::milliseconds{10});

                return {std::move(output), exit_code};
            }

            // Timeout check.
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                // Hard-kill the bash process group — the running command
                // (e.g. a compiler) may still be alive and must not corrupt
                // subsequent commands with its leftover output.
                // The next run() call will transparently start a fresh session.
                kill_session();
                raw += "\n[TIMEOUT: command exceeded time limit]\n";
                return {std::move(raw), -1};
            }

            const auto remaining_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - now).count();

            const int poll_ms = static_cast<int>(std::min<std::int64_t>(remaining_ms, 250));
            struct pollfd pfd{stdout_fd_, POLLIN | POLLHUP, 0};
            const int ret = ::poll(&pfd, 1, poll_ms);
            if (ret < 0 && errno == EINTR) continue;
            if (ret < 0) {
                kill_session();
                return {std::move(raw), -1};
            }
            if (ret == 0) {
                int shell_exit_code = -1;
                if (reap_if_exited(shell_exit_code, true)) {
                    if (consume_interrupt_requested()) {
                        return interrupted_result(std::move(raw));
                    }
                    return {std::move(raw), shell_exit_code};
                }
                continue;
            }

            if ((pfd.revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
                int shell_exit_code = -1;
                if (reap_if_exited(shell_exit_code, true)) {
                    if (consume_interrupt_requested()) {
                        return interrupted_result(std::move(raw));
                    }
                    return {std::move(raw), shell_exit_code};
                }
                continue;
            }

            std::array<char, 4096> buf{};
            const ssize_t n = ::read(stdout_fd_, buf.data(), buf.size());
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                int shell_exit_code = -1;
                if (reap_if_exited_for(
                        shell_exit_code,
                        true,
                        std::chrono::milliseconds{50})) {
                    if (consume_interrupt_requested()) {
                        return interrupted_result(std::move(raw));
                    }
                    return {std::move(raw), shell_exit_code};
                }
                // stdout closed before the sentinel while the shell still
                // appears alive; reset instead of blocking in waitpid().
                kill_session();
                return {std::move(raw), -1};
            }

            const auto chunk = static_cast<std::size_t>(n);
            if (raw.size() + chunk > kMaxOutput) {
                raw.append(buf.data(), kMaxOutput - raw.size());
                raw += "\n... [OUTPUT TRUNCATED AT 4MB] ...";
                // Drain the remaining output up to the sentinel so the session
                // stays synchronised for the next command. If the command never
                // reaches the sentinel before the original deadline, reset the
                // shell so future calls cannot receive stale output.
                if (!drain_sentinel_until(deadline)) {
                    kill_session();
                    raw += "\n[TIMEOUT: command exceeded time limit while draining "
                           "truncated output]\n";
                }
                // Exit code is unknown after truncation — report -1 so callers
                // do not mistakenly treat a failed command as successful.
                return {std::move(raw), -1};
            }
            raw.append(buf.data(), chunk);
        }
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    int                stdin_fd_{-1};
    int                stdout_fd_{-1};
    std::atomic<pid_t> pid_{-1};
    std::atomic<pid_t> pgid_{-1};   // process group id (== pid_ while alive)
    std::atomic<bool>  interrupted_{false};
    std::string        sentinel_;
};

} // namespace core::tools::detail
