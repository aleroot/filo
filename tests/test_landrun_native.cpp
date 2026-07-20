#include "core/landrun/LandrunDriverFactory.hpp"
#include "core/landrun/LandrunHelper.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"
#include "core/landrun/LandrunReadiness.hpp"
#include "core/landrun/LandrunRuntime.hpp"
#include "core/landrun/LandrunSettings.hpp"
#include "core/tools/shell/PosixShellExecutor.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <unistd.h>

extern char** environ;

namespace {

std::string shell_quote(std::string_view value) {
    std::string quoted{"'"};
    for (const char c : value) {
        if (c == '\'') quoted += "'\\''";
        else quoted += c;
    }
    quoted += '\'';
    return quoted;
}

std::filesystem::path find_repository_root() {
    std::error_code ec;
    auto path = std::filesystem::current_path(ec);
    while (!ec && !path.empty()) {
        if (std::filesystem::exists(path / ".git", ec) && !ec) return path;
        const auto parent = path.parent_path();
        if (parent == path) break;
        path = parent;
    }
    return {};
}

core::landrun::LandrunPolicy make_policy(const std::filesystem::path& allowed,
                                         bool shared_temp = false) {
    core::landrun::LandrunPolicy policy{
        .mode = core::landrun::LandrunMode::workspace_write,
        .allow_network = false,
    };
    core::landrun::add_readable_root(policy, allowed);
    core::landrun::add_writable_root(policy, allowed);
    if (const auto runtime_root = core::landrun::LandrunSettings::instance().runtime_root();
        !runtime_root.empty()) {
        core::landrun::add_readable_root(policy, runtime_root);
        core::landrun::add_writable_root(policy, runtime_root);
    }
    if (shared_temp) {
        core::landrun::add_readable_root(policy, "/tmp");
        core::landrun::add_writable_root(policy, "/tmp");
        const auto host_tmpdir = core::landrun::LandrunSettings::instance().host_tmpdir();
        core::landrun::add_readable_root(policy, host_tmpdir);
        core::landrun::add_writable_root(policy, host_tmpdir);
    }
#if defined(__APPLE__)
    for (const char* root : {"/System", "/usr", "/bin", "/sbin", "/Library", "/dev",
                             "/Applications/Xcode.app", "/opt/homebrew", "/opt/local",
                             "/private/etc"}) {
        std::error_code ec;
        if (std::filesystem::exists(root, ec)) core::landrun::add_readable_root(policy, root);
    }
    core::landrun::add_protected_read_path(policy, allowed / ".env");
#else
    for (const char* root : {"/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev"}) {
        std::error_code ec;
        if (std::filesystem::exists(root, ec)) core::landrun::add_readable_root(policy, root);
    }
#endif
    return policy;
}

int run_restricted_child(const std::filesystem::path& allowed,
                         const std::filesystem::path& denied_file,
                         const std::filesystem::path& denied_rename)
{
    auto policy = make_policy(allowed);
    const auto driver = core::landrun::make_landrun_driver();
    if (const auto result = driver->apply(policy); !result.success) {
        std::cerr << result.detail << '\n';
        return 10;
    }

    const auto allowed_file = allowed / "created-by-landrun";
    int fd = ::open(allowed_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        std::cerr << "allowed workspace write failed: " << std::strerror(errno) << '\n';
        return 11;
    }
    constexpr char value[] = "ok";
    if (::write(fd, value, sizeof(value) - 1) != sizeof(value) - 1) return 12;
    ::close(fd);

    errno = 0;
    fd = ::open(denied_file.c_str(), O_WRONLY | O_TRUNC);
    if (fd >= 0) {
        ::close(fd);
        return 13;
    }
    if (errno != EPERM && errno != EACCES) return 14;

    errno = 0;
    if (::rename(allowed_file.c_str(), denied_rename.c_str()) == 0) return 15;
    if (errno != EPERM && errno != EACCES && errno != EXDEV) return 16;

    errno = 0;
    fd = ::open(denied_file.c_str(), O_RDONLY);
#if defined(__APPLE__)
    if (fd < 0) {
        std::cerr << "Codex-compatible host read failed: " << std::strerror(errno) << '\n';
        return 17;
    }
    ::close(fd);
#else
    if (fd >= 0) {
        ::close(fd);
        return 17;
    }
    if (errno != EPERM && errno != EACCES) return 18;
#endif

#if defined(__APPLE__)
    errno = 0;
    fd = ::open((allowed / ".env").c_str(), O_RDONLY);
    if (fd >= 0) {
        ::close(fd);
        return 19;
    }
    if (errno != EPERM && errno != EACCES) return 20;
#endif

    errno = 0;
    const int network_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (network_fd >= 0) {
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(9);
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        const int connected = ::connect(
            network_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address));
        const int network_errno = errno;
        ::close(network_fd);
        if (connected == 0 || (network_errno != EPERM && network_errno != EACCES)) return 21;
    } else if (errno != EPERM && errno != EACCES) {
        return 22;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (core::landrun::is_landrun_helper_invocation(argc, argv)) {
        return core::landrun::run_landrun_helper(argc - 2, argv + 2);
    }
    if (argc == 5 && std::string_view(argv[1]) == "--restricted-child") {
        return run_restricted_child(argv[2], argv[3], argv[4]);
    }
    if (std::getenv("FILO_LANDRUN") != nullptr) {
        std::cerr << "SKIP: native Seatbelt cannot be stacked inside an existing Filo sandbox\n";
        return 77;
    }

    const auto driver = core::landrun::make_landrun_driver();
    const auto probe = driver->probe();
    if (!probe.available) {
        std::cerr << "SKIP: " << probe.detail << '\n';
        return 77;
    }

    const auto base = std::filesystem::temp_directory_path()
        / ("filo-native-" + std::to_string(::getpid()));
    const auto allowed = base / "allowed";
    const auto denied_file = base / "denied-existing";
    const auto denied_rename = base / "denied-renamed";
    std::error_code ec;
    std::filesystem::create_directories(allowed, ec);
    if (ec) return 20;
    int denied_fd = ::open(denied_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (denied_fd < 0) return 21;
    ::close(denied_fd);
    int secret_fd = ::open((allowed / ".env").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (secret_fd < 0) return 22;
    constexpr char secret[] = "TOKEN=must-not-leak";
    (void)::write(secret_fd, secret, sizeof(secret) - 1);
    ::close(secret_fd);

    std::string executable = std::filesystem::weakly_canonical(argv[0], ec).string();
    if (ec || executable.empty()) return 22;
    std::string allowed_text = allowed.string();
    std::string denied_text = denied_file.string();
    std::string rename_text = denied_rename.string();
    char* child_argv[] = {
        executable.data(),
        const_cast<char*>("--restricted-child"),
        allowed_text.data(),
        denied_text.data(),
        rename_text.data(),
        nullptr,
    };
    pid_t child = -1;
    const int spawn_result = ::posix_spawn(
        &child, executable.c_str(), nullptr, nullptr, child_argv, environ);
    if (spawn_result != 0) return 23;
    int status = 0;
    if (::waitpid(child, &status, 0) != child) return 24;

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::filesystem::remove_all(base, ec);
        return WIFEXITED(status) ? WEXITSTATUS(status) : 25;
    }

    core::landrun::LandrunRuntime runtime;
    core::tools::shell::PosixShellExecutor shell;
    const auto repository_root = find_repository_root();
    if (repository_root.empty()) {
        std::cerr << "could not locate the Git repository root\n";
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    const auto shell_denied_file = repository_root
        / (".filo-native-denied-" + std::to_string(::getpid()));
    int shell_denied_fd = ::open(
        shell_denied_file.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (shell_denied_fd < 0) {
        std::cerr << "could not create the shell write-denial fixture: "
                  << std::strerror(errno) << '\n';
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    (void)::write(shell_denied_fd, "unchanged", 9);
    ::close(shell_denied_fd);
    const core::workspace::SessionWorkspace workspace{
        core::workspace::WorkspaceSnapshot{.primary = allowed, .enforce = true}};
    auto shell_policy = core::landrun::LandrunPolicyCompiler::compile(
        workspace, core::landrun::LandrunMode::workspace_write);
    core::landrun::add_readable_root(shell_policy, repository_root);
    if (const auto readiness = core::landrun::verify_landrun_readiness(shell_policy);
        !readiness.success) {
        std::cerr << readiness.detail << '\n';
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    shell.configure_landrun(std::move(shell_policy));
    const auto global_tmp_file = std::filesystem::path("/tmp")
        / ("filo-native-shared-tmp-" + std::to_string(::getpid()));
    const auto effective_tmpdir =
        core::landrun::LandrunSettings::instance().effective_tmpdir();
    const int inherited_fd = ::open("/dev/null", O_RDONLY);
    if (inherited_fd < 0) {
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    if (::fcntl(inherited_fd, F_SETFD, 0) != 0) {
        ::close(inherited_fd);
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    ::setenv("OPENAI_API_KEY", "must-not-leak", 1);
    const std::string command =
        "if { true <&" + std::to_string(inherited_fd)
        + "; } 2>/dev/null; then exit 32; fi; printf shell-ok > '"
        + (allowed / "from-shell").string()
        + "'; if printf denied > '" + shell_denied_file.string()
        + "'; then exit 31; fi; printf shared-temp > '" + global_tmp_file.string()
        + "'; test -z \"${OPENAI_API_KEY+x}\"; test \"$(cat '"
        + (allowed / "from-shell").string()
        + "')\" = shell-ok; test \"$HOME\" = '"
        + (runtime.root() / "home").string()
        + "'; test \"$TMPDIR\" = '" + effective_tmpdir.string()
        + "'; printf temp-ok > \"$TMPDIR/probe\"; git --version >/dev/null; clang --version >/dev/null; "
          "python3 -c 'print(1)' >/dev/null; git -C "
        + shell_quote(repository_root.string())
        + " --no-pager diff --stat HEAD >/dev/null";
    const auto shell_result = shell.run(command, {}, std::chrono::seconds{10});
    ::unsetenv("OPENAI_API_KEY");
    ::close(inherited_fd);
    if (shell_result.exit_code != 0) {
        std::cerr << "sandboxed shell failed with exit code "
                  << shell_result.exit_code << ": " << shell_result.output << '\n';
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    std::filesystem::remove(shell_denied_file, ec);
    if (!std::filesystem::exists(global_tmp_file, ec)) {
        std::filesystem::remove_all(base, ec);
        return 27;
    }
    std::filesystem::remove(global_tmp_file, ec);
    if (!std::filesystem::exists(effective_tmpdir / "probe", ec)) {
        std::filesystem::remove_all(base, ec);
        return 28;
    }
    std::filesystem::remove(effective_tmpdir / "probe", ec);

    std::filesystem::remove_all(base, ec);
    return 0;
}
