#pragma once

#include "../../landrun/LandrunPolicy.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace core::tools::shell {

// One command running in its own process, the way codex-rs unified exec runs
// `exec_command`: a fresh shell per command, stdin left open for later
// writes, and output collected in the background so a caller can return
// while the command keeps running and read what it wrote since.
//
// The process leads its own process group (a new session), so kill() and
// the destructor reach everything it started.
class CommandProcess {
public:
    /// Unread output kept between two reads, as codex-rs keeps
    /// UNIFIED_EXEC_OUTPUT_MAX_BYTES: past it the middle is dropped.
    static constexpr std::size_t kMaxUnreadBytes = 1024 * 1024;

    /// What one wait returned: the output since the previous take, and the
    /// exit code once the process has ended.
    struct Taken {
        std::string output;
        std::optional<int> exit_code;
    };

    /// Starts `bash -c` running @p command in @p working_dir under @p policy.
    /// Nothing, with @p error set, when the process could not start.
    [[nodiscard]] static std::unique_ptr<CommandProcess> spawn(
        std::string_view command,
        const std::filesystem::path& working_dir,
        const core::landrun::LandrunPolicy& policy,
        std::string& error);

    ~CommandProcess();
    CommandProcess(const CommandProcess&) = delete;
    CommandProcess& operator=(const CommandProcess&) = delete;

    [[nodiscard]] int process_group() const noexcept { return pgid_; }

    /// Writes @p bytes to the process's stdin. False once it is closed.
    [[nodiscard]] bool write(std::string_view bytes);

    /// Blocks until the process has ended, @p deadline passes, or interrupt()
    /// is called, then hands over what it wrote since the previous take.
    [[nodiscard]] Taken wait_and_take(std::chrono::steady_clock::time_point deadline);

    /// Ends a wait in progress early.
    void interrupt();

    /// Kills the process group. Output collected so far stays readable.
    void kill() noexcept;

    [[nodiscard]] bool finished() const;

private:
    CommandProcess() = default;
    void read_loop();
    void append(std::string_view bytes);
    [[nodiscard]] std::string take_unread();

    int pid_ = -1;
    int pgid_ = -1;
    int stdin_fd_ = -1;
    int stdout_fd_ = -1;
    std::thread reader_;
    std::atomic<bool> stop_reader_{false};

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::string unread_;
    std::size_t omitted_ = 0;
    std::optional<int> exit_code_;
    bool finished_ = false;
    bool interrupted_ = false;
    std::mutex write_mutex_;
};

} // namespace core::tools::shell
