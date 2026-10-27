#include "SystemKeyring.hpp"

#ifndef _WIN32
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <array>
#include <vector>

#ifndef _WIN32
extern char** environ;
#endif

namespace core::auth {

namespace {

/// A keyring item is a token document, not a file: anything larger is not one.
constexpr std::size_t kMaxSecretBytes = 1U << 20;

/// How a store's command-line helper is invoked and how it reports a missing item.
struct LookupCommand {
    std::vector<std::string> argv;
    /// Exit status meaning "no such item"; any other failure that wrote to
    /// stderr means the store itself could not be consulted.
    int not_found_status = -1;
};

using LookupResult = std::expected<std::optional<std::string>, std::string>;

#ifndef _WIN32

[[nodiscard]] std::string first_line(std::string text) {
    if (const auto end = text.find_first_of("\r\n"); end != std::string::npos) text.resize(end);
    return text;
}

class FileActions {
public:
    FileActions() : error_(::posix_spawn_file_actions_init(&actions_)) {}
    ~FileActions() {
        if (error_ == 0) ::posix_spawn_file_actions_destroy(&actions_);
    }
    FileActions(const FileActions&) = delete;
    FileActions& operator=(const FileActions&) = delete;

    [[nodiscard]] bool ok() const noexcept { return error_ == 0; }
    [[nodiscard]] posix_spawn_file_actions_t* get() noexcept { return &actions_; }

private:
    posix_spawn_file_actions_t actions_{};
    int error_;
};

class Pipe {
public:
    Pipe() {
        if (::pipe(fds_.data()) != 0) fds_ = {-1, -1};
    }
    ~Pipe() {
        close_read();
        close_write();
    }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;

    [[nodiscard]] bool ok() const noexcept { return fds_[0] >= 0; }
    [[nodiscard]] int read_end() const noexcept { return fds_[0]; }
    [[nodiscard]] int write_end() const noexcept { return fds_[1]; }
    void close_read() noexcept { close_fd(fds_[0]); }
    void close_write() noexcept { close_fd(fds_[1]); }

private:
    static void close_fd(int& fd) noexcept {
        if (fd >= 0) ::close(fd);
        fd = -1;
    }
    std::array<int, 2> fds_{-1, -1};
};

/// Runs @p command with stdin closed and returns its exit status, stdout and stderr.
struct CommandOutput {
    int status = -1;
    std::string out;
    std::string err;
};

[[nodiscard]] std::expected<CommandOutput, std::string> run(const LookupCommand& command) {
    const std::string& tool = command.argv.front();
    Pipe out;
    Pipe err;
    FileActions actions;
    if (!out.ok() || !err.ok() || !actions.ok()) {
        return std::unexpected("could not start " + tool + ": " + std::strerror(errno));
    }
    const bool prepared =
        ::posix_spawn_file_actions_addopen(actions.get(), STDIN_FILENO, "/dev/null", O_RDONLY, 0) == 0
        && ::posix_spawn_file_actions_adddup2(actions.get(), out.write_end(), STDOUT_FILENO) == 0
        && ::posix_spawn_file_actions_adddup2(actions.get(), err.write_end(), STDERR_FILENO) == 0
        && ::posix_spawn_file_actions_addclose(actions.get(), out.read_end()) == 0
        && ::posix_spawn_file_actions_addclose(actions.get(), err.read_end()) == 0;
    if (!prepared) return std::unexpected("could not start " + tool);

    std::vector<char*> argv;
    argv.reserve(command.argv.size() + 1);
    for (const auto& arg : command.argv) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int spawn_error =
        ::posix_spawnp(&pid, argv.front(), actions.get(), nullptr, argv.data(), environ);
    out.close_write();
    err.close_write();
    if (spawn_error == ENOENT) return std::unexpected(tool + " is not installed");
    if (spawn_error != 0) {
        return std::unexpected("could not start " + tool + ": " + std::strerror(spawn_error));
    }

    // Drain both pipes together so a chatty stderr cannot stall the child.
    CommandOutput result;
    std::array<pollfd, 2> fds{{{out.read_end(), POLLIN, 0}, {err.read_end(), POLLIN, 0}}};
    std::array<std::string*, 2> sinks{&result.out, &result.err};
    bool oversized = false;
    while (fds[0].fd >= 0 || fds[1].fd >= 0) {
        if (::poll(fds.data(), fds.size(), -1) < 0) {
            if (errno == EINTR) continue;
            break;
        }
        for (std::size_t i = 0; i < fds.size(); ++i) {
            if (fds[i].fd < 0 || fds[i].revents == 0) continue;
            char buffer[4096];
            const ssize_t n = ::read(fds[i].fd, buffer, sizeof(buffer));
            if (n > 0) {
                if (sinks[i]->size() + static_cast<std::size_t>(n) > kMaxSecretBytes) {
                    oversized = true;
                } else {
                    sinks[i]->append(buffer, static_cast<std::size_t>(n));
                }
            } else if (n == 0 || errno != EINTR) {
                fds[i].fd = -1;
            }
        }
    }

    int wait_status = 0;
    while (::waitpid(pid, &wait_status, 0) < 0) {
        if (errno != EINTR) return std::unexpected(tool + " could not be waited for");
    }
    if (oversized) return std::unexpected(tool + " returned more than a keyring item can hold");
    result.status = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : -1;
    return result;
}

[[nodiscard]] LookupResult lookup(const LookupCommand& command) {
    auto output = run(command);
    if (!output.has_value()) return std::unexpected(std::move(output.error()));
    if (output->status == 0) {
        std::string secret = std::move(output->out);
        // The helpers terminate the secret with a newline that is not part of it.
        while (secret.ends_with('\n') || secret.ends_with('\r')) secret.pop_back();
        return secret;
    }
    if (output->status == command.not_found_status || output->err.empty()) return std::nullopt;
    return std::unexpected(command.argv.front() + " failed: " + first_line(std::move(output->err)));
}

#endif // !_WIN32

class PlatformKeyring final : public IKeyring {
public:
    [[nodiscard]] std::string_view description() const noexcept override {
#if defined(__APPLE__)
        return "the macOS login keychain";
#elif defined(_WIN32)
        return {};
#else
        return "the Secret Service keyring";
#endif
    }

    [[nodiscard]] LookupResult read(const KeyringEntry& entry) const override {
#if defined(__APPLE__)
        // `security` exits 44 (errSecItemNotFound) when no item matches.
        return lookup({.argv = {"/usr/bin/security", "find-generic-password", "-s", entry.service,
                                "-a", entry.account, "-w"},
                       .not_found_status = 44});
#elif defined(_WIN32)
        static_cast<void>(entry);
        return std::unexpected(std::string("Filo cannot read the Windows credential store"));
#else
        // go-keyring and Rust's keyring crate both tag items with these attributes.
        return lookup({.argv = {"secret-tool", "lookup", "service", entry.service, "username",
                                entry.account}});
#endif
    }
};

} // namespace

std::shared_ptr<const IKeyring> system_keyring() {
    static const auto keyring = std::make_shared<const PlatformKeyring>();
    return keyring;
}

std::string describe_keyring_entry(const KeyringEntry& entry) {
    return "service=" + entry.service + ", account=" + entry.account;
}

} // namespace core::auth
