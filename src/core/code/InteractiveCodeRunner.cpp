#include "InteractiveCodeRunner.hpp"

#include "core/landrun/LandrunEnvironment.hpp"
#include "core/landrun/LandrunLaunch.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <format>
#include <system_error>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace core::code {
namespace {

constexpr std::size_t kCaptureLimit = 2 * 1024 * 1024;
constexpr std::size_t kCaptureHead = 3 * kCaptureLimit / 4;
constexpr std::size_t kCaptureTail = kCaptureLimit - kCaptureHead;

class BoundedCapture {
public:
    void append(std::string_view chunk) {
        total_ += chunk.size();
        if (!truncated_ && data_.size() + chunk.size() <= kCaptureLimit) {
            data_.append(chunk);
            return;
        }
        if (!truncated_) {
            truncated_ = true;
            std::string overflow = data_ + std::string(chunk);
            data_.assign(overflow.substr(0, std::min(kCaptureHead, overflow.size())));
            tail_.assign(overflow.size() > kCaptureTail
                ? overflow.substr(overflow.size() - kCaptureTail)
                : overflow);
            return;
        }
        tail_.append(chunk);
        if (tail_.size() > kCaptureTail) {
            tail_.erase(0, tail_.size() - kCaptureTail);
        }
    }

    [[nodiscard]] std::string finish() const {
        if (!truncated_) return data_;
        return data_ + std::format(
            "\n\n[... Filo omitted {} bytes from the middle ...]\n\n",
            total_ > data_.size() + tail_.size()
                ? total_ - data_.size() - tail_.size()
                : 0) + tail_;
    }

    [[nodiscard]] std::size_t total() const noexcept { return total_; }
    [[nodiscard]] bool truncated() const noexcept { return truncated_; }

private:
    std::string data_;
    std::string tail_;
    std::size_t total_ = 0;
    bool truncated_ = false;
};

#if !defined(_WIN32)
class FileDescriptor {
public:
    explicit FileDescriptor(int value = -1) : value_(value) {}
    ~FileDescriptor() { reset(); }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept
        : value_(std::exchange(other.value_, -1)) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) reset(std::exchange(other.value_, -1));
        return *this;
    }
    [[nodiscard]] int get() const noexcept { return value_; }
    void reset(int value = -1) noexcept {
        if (value_ >= 0) ::close(value_);
        value_ = value;
    }
private:
    int value_;
};

class TemporaryScript {
public:
    static std::expected<TemporaryScript, std::string> create(
        const ExecutionPlan& plan,
        const std::filesystem::path& directory) {
        if (directory.empty()) {
            return std::unexpected("Code-run temporary directory is unavailable.");
        }
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        if (ec) return std::unexpected("Could not create code-run directory: " + ec.message());

        std::string pattern = (directory / ("block-XXXXXX." + plan.script_extension)).string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        const int suffix_length = static_cast<int>(plan.script_extension.size() + 1);
        FileDescriptor descriptor(::mkstemps(writable.data(), suffix_length));
        if (descriptor.get() < 0) {
            return std::unexpected(std::format(
                "Could not create temporary script: {}", std::strerror(errno)));
        }
        TemporaryScript script(std::filesystem::path(writable.data()));

        std::string_view remaining(plan.prepared_source);
        while (!remaining.empty()) {
            const auto written = ::write(descriptor.get(), remaining.data(), remaining.size());
            if (written < 0) {
                if (errno == EINTR) continue;
                return std::unexpected(std::format(
                    "Could not write temporary script: {}", std::strerror(errno)));
            }
            remaining.remove_prefix(static_cast<std::size_t>(written));
        }
        if (::fchmod(descriptor.get(), S_IRUSR | S_IWUSR) != 0) {
            return std::unexpected(std::format(
                "Could not protect temporary script: {}", std::strerror(errno)));
        }
        return std::move(script);
    }

    TemporaryScript(TemporaryScript&& other) noexcept
        : path_(std::exchange(other.path_, {})) {}
    TemporaryScript& operator=(TemporaryScript&& other) noexcept {
        if (this != &other) {
            remove();
            path_ = std::exchange(other.path_, {});
        }
        return *this;
    }
    ~TemporaryScript() { remove(); }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    explicit TemporaryScript(std::filesystem::path path) : path_(std::move(path)) {}
    void remove() noexcept {
        if (path_.empty()) return;
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
        path_.clear();
    }
    std::filesystem::path path_;
};

class ScopedIgnoredSignal {
public:
    explicit ScopedIgnoredSignal(int signal) : signal_(signal) {
        struct sigaction action{};
        action.sa_handler = SIG_IGN;
        sigemptyset(&action.sa_mask);
        active_ = ::sigaction(signal_, &action, &previous_) == 0;
    }
    ~ScopedIgnoredSignal() {
        if (active_) (void)::sigaction(signal_, &previous_, nullptr);
    }
private:
    int signal_;
    struct sigaction previous_{};
    bool active_ = false;
};

void reset_child_signal(int signal) {
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    (void)::sigaction(signal, &action, nullptr);
}

void close_inherited_descriptors(long descriptor_limit) noexcept {
#if defined(__linux__) && defined(__NR_close_range)
    if (::syscall(
            __NR_close_range,
            static_cast<unsigned int>(STDERR_FILENO + 1),
            ~0U,
            0U) == 0) {
        return;
    }
#endif
    const long limit = descriptor_limit > STDERR_FILENO
        ? descriptor_limit
        : 65536;
    for (long descriptor = STDERR_FILENO + 1; descriptor < limit; ++descriptor) {
        while (::close(static_cast<int>(descriptor)) != 0 && errno == EINTR) {}
    }
}

void write_live_output(std::string_view chunk) {
    while (!chunk.empty()) {
        const auto written = ::write(STDOUT_FILENO, chunk.data(), chunk.size());
        if (written < 0) {
            if (errno == EINTR) continue;
            return;
        }
        chunk.remove_prefix(static_cast<std::size_t>(written));
    }
}

bool create_cloexec_pipe(std::array<int, 2>& descriptors) noexcept {
#if defined(__linux__) && defined(__NR_pipe2) && defined(O_CLOEXEC)
    if (::syscall(__NR_pipe2, descriptors.data(), O_CLOEXEC) == 0) return true;
    if (errno != ENOSYS && errno != EINVAL) return false;
#endif
    if (::pipe(descriptors.data()) != 0) return false;
    for (const int descriptor : descriptors) {
        const int flags = ::fcntl(descriptor, F_GETFD);
        if (flags < 0 || ::fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0) {
            ::close(descriptors[0]);
            ::close(descriptors[1]);
            descriptors = {-1, -1};
            return false;
        }
    }
    return true;
}

class PosixInteractiveCodeRunner final : public IInteractiveCodeRunner {
public:
    std::expected<CodeRunResult, std::string> run(
        const ExecutionPlan& plan,
        const std::filesystem::path& working_directory,
        const std::filesystem::path& temporary_directory,
        const core::landrun::LandrunPolicy& policy) const override {
        auto script_result = TemporaryScript::create(plan, temporary_directory);
        if (!script_result) return std::unexpected(script_result.error());
        auto script = std::move(*script_result);

        std::vector<std::string> target_arguments{plan.interpreter};
        target_arguments.insert(target_arguments.end(),
                                plan.interpreter_arguments.begin(),
                                plan.interpreter_arguments.end());
        target_arguments.push_back(script.path().string());

        core::landrun::LandrunLaunch launch;
        try {
            launch = core::landrun::prepare_landrun_launch(
                policy, plan.interpreter_path, std::move(target_arguments));
        } catch (const std::exception& error) {
            return std::unexpected(error.what());
        }

        std::vector<char*> argv;
        argv.reserve(launch.arguments.size() + 1);
        for (auto& argument : launch.arguments) argv.push_back(argument.data());
        argv.push_back(nullptr);

        auto environment_storage = core::landrun::build_landrun_environment(policy, environ);
        std::vector<char*> environment;
        environment.reserve(environment_storage.size() + 1);
        for (auto& entry : environment_storage) environment.push_back(entry.data());
        environment.push_back(nullptr);

        std::array<int, 2> output_pipe{};
        if (!create_cloexec_pipe(output_pipe)) {
            return std::unexpected(std::format("Could not create output pipe: {}",
                                               std::strerror(errno)));
        }
        FileDescriptor read_end(output_pipe[0]);
        FileDescriptor write_end(output_pipe[1]);

        const auto started = std::chrono::steady_clock::now();
        const long descriptor_limit = ::sysconf(_SC_OPEN_MAX);
        const pid_t child = ::fork();
        if (child < 0) {
            return std::unexpected(std::format("Could not start code block: {}",
                                               std::strerror(errno)));
        }
        if (child == 0) {
            (void)::dup2(write_end.get(), STDOUT_FILENO);
            (void)::dup2(write_end.get(), STDERR_FILENO);
            read_end.reset();
            write_end.reset();
            close_inherited_descriptors(descriptor_limit);
            reset_child_signal(SIGINT);
            reset_child_signal(SIGQUIT);
            reset_child_signal(SIGTERM);
            if (!working_directory.empty()
                && ::chdir(working_directory.c_str()) != 0) {
                const std::string message = std::format(
                    "filo: cannot enter {}: {}\n",
                    working_directory.string(), std::strerror(errno));
                (void)::write(STDERR_FILENO, message.data(), message.size());
                ::_exit(126);
            }
            ::execve(launch.executable.c_str(), argv.data(), environment.data());
            const std::string message = std::format(
                "filo: cannot execute {}: {}\n", launch.executable, std::strerror(errno));
            (void)::write(STDERR_FILENO, message.data(), message.size());
            ::_exit(126);
        }

        write_end.reset();
        ScopedIgnoredSignal ignore_interrupt(SIGINT);
        ScopedIgnoredSignal ignore_quit(SIGQUIT);
        BoundedCapture capture;
        std::array<char, 16 * 1024> buffer{};
        while (true) {
            const auto count = ::read(read_end.get(), buffer.data(), buffer.size());
            if (count > 0) {
                const std::string_view chunk(buffer.data(), static_cast<std::size_t>(count));
                write_live_output(chunk);
                capture.append(chunk);
                continue;
            }
            if (count < 0 && errno == EINTR) continue;
            break;
        }
        read_end.reset();

        int status = 0;
        while (::waitpid(child, &status, 0) < 0) {
            if (errno != EINTR) {
                return std::unexpected(std::format("Could not wait for code block: {}",
                                                   std::strerror(errno)));
            }
        }

        CodeRunResult result{
            .output = sanitize_terminal_output(capture.finish()),
            .output_bytes = capture.total(),
            .output_truncated = capture.truncated(),
            .duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started),
        };
        if (WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
        if (WIFSIGNALED(status)) result.terminating_signal = WTERMSIG(status);
        return result;
    }
};
#else
class UnsupportedInteractiveCodeRunner final : public IInteractiveCodeRunner {
public:
    std::expected<CodeRunResult, std::string> run(
        const ExecutionPlan&,
        const std::filesystem::path&,
        const std::filesystem::path&,
        const core::landrun::LandrunPolicy&) const override {
        return std::unexpected("Interactive code-block execution is not available on Windows yet.");
    }
};
#endif

} // namespace

std::unique_ptr<IInteractiveCodeRunner> make_interactive_code_runner() {
#if defined(_WIN32)
    return std::make_unique<UnsupportedInteractiveCodeRunner>();
#else
    return std::make_unique<PosixInteractiveCodeRunner>();
#endif
}

} // namespace core::code
