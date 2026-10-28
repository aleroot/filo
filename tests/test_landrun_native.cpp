#include "core/landrun/LandrunDriverFactory.hpp"
#include "core/landrun/ConfinedCommand.hpp"
#include "core/landrun/LandrunHelper.hpp"
#include "core/landrun/LandrunPolicyCompiler.hpp"
#include "core/landrun/LandrunReadiness.hpp"
#include "core/landrun/LandrunRuntime.hpp"
#include "core/landrun/LandrunSettings.hpp"
#include "core/tools/shell/PosixShellExecutor.hpp"
#include "core/scm/GitProvider.hpp"
#include "core/review/Git.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <poll.h>
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

std::filesystem::path find_source_root() {
    std::error_code ec;
    auto path = std::filesystem::current_path(ec);
    while (!ec && !path.empty()) {
        const bool has_build_definition =
            std::filesystem::is_regular_file(path / "CMakeLists.txt", ec);
        ec.clear();
        const bool has_landrun_sources =
            std::filesystem::is_directory(path / "src/core/landrun", ec);
        if (!ec && has_build_definition && has_landrun_sources) return path;
        ec.clear();
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
        // Native sandboxes are intentionally irreversible and macOS refuses to
        // stack Seatbelt profiles in some already-confined CI/agent hosts. The
        // backend exists there, so probe() succeeds, but this integration test
        // cannot exercise it. Preserve failures for malformed policies and all
        // other initialization errors; only the host-level EPERM is a skip.
        if (result.detail.contains(std::strerror(EPERM))) {
            std::cerr << "SKIP: " << result.detail << '\n';
            return 77;
        }
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

/**
 * Protection-only enforcement: no confinement mode, one subtracted path.
 *
 * This is the shape a withheld steering file needs, and the two halves are
 * asserted separately because failing either is a different bug -- denying too
 * little leaks the file, denying too much breaks the user's shell.
 */
int run_protection_child(const std::filesystem::path& allowed,
                         const std::filesystem::path& denied_file)
{
    core::landrun::LandrunPolicy policy;
    core::landrun::add_protected_read_path(policy, denied_file);
    if (policy.confines() || !policy.protects_paths() || !policy.enabled()) {
        return 30;
    }

    const auto driver = core::landrun::make_landrun_driver();
    if (!driver->supports_protected_paths()) {
        // Allow-list-only backend (Landlock). It must *refuse* the subtraction
        // instead of quietly granting it: silently ignoring a deny is precisely
        // how withheld steering would leak. apply() cannot confine anything
        // here, because the refusal happens before any ruleset is built.
        return driver->apply(policy).success ? 36 : 0;
    }

    if (const auto result = driver->apply(policy); !result.success) {
        if (result.detail.contains(std::strerror(EPERM))) {
            std::cerr << "SKIP: " << result.detail << '\n';
            return 77;
        }
        std::cerr << result.detail << '\n';
        return 31;
    }

    // The subtraction itself.
    errno = 0;
    int fd = ::open(denied_file.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::close(fd);
        return 32;
    }
    if (errno != EPERM && errno != EACCES) return 33;

    // ...and nothing else: a sibling still reads, and the host is still
    // writable, because the user never asked for confinement.
    fd = ::open((allowed / "protection-sibling").c_str(), O_RDONLY);
    if (fd < 0) return 34;
    ::close(fd);

    fd = ::open((allowed / "protection-write").c_str(),
                O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return 35;
    ::close(fd);
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
    if (argc == 4 && std::string_view(argv[1]) == "--protection-child") {
        return run_protection_child(argv[2], argv[3]);
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

    // Protection-only enforcement runs in its own child: an applied policy is
    // irreversible, so it cannot share a process with the confinement checks.
    {
        const auto sibling = allowed / "protection-sibling";
        const int sibling_fd =
            ::open(sibling.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (sibling_fd < 0) {
            std::filesystem::remove_all(base, ec);
            return 29;
        }
        constexpr char sibling_content[] = "readable";
        (void)::write(sibling_fd, sibling_content, sizeof(sibling_content) - 1);
        ::close(sibling_fd);

        char* protection_argv[] = {
            executable.data(),
            const_cast<char*>("--protection-child"),
            allowed_text.data(),
            denied_text.data(),
            nullptr,
        };
        pid_t protection_child = -1;
        const int protection_spawn = ::posix_spawn(
            &protection_child, executable.c_str(), nullptr, nullptr,
            protection_argv, environ);
        int protection_status = 0;
        if (protection_spawn != 0
            || ::waitpid(protection_child, &protection_status, 0) != protection_child) {
            std::filesystem::remove_all(base, ec);
            return 29;
        }
        if (!WIFEXITED(protection_status) || WEXITSTATUS(protection_status) != 0) {
            const int code =
                WIFEXITED(protection_status) ? WEXITSTATUS(protection_status) : 29;
            if (code != 77) {
                std::filesystem::remove_all(base, ec);
                return code;
            }
            // The host refused to stack another profile. Report it, but keep the
            // remaining coverage rather than discarding the whole run.
            std::cerr << "SKIP: protection-only enforcement is unavailable here\n";
        }
    }

    core::landrun::LandrunRuntime runtime;
    core::tools::shell::PosixShellExecutor shell;
    const auto source_root = find_source_root();
    if (source_root.empty()) {
        std::cerr << "could not locate the Filo source root\n";
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    // Compare real Git results with and without confinement. This also checks
    // that the helper launched from GitProvider can dispatch from this test
    // executable and that binary status records retain their fingerprint.
    if (std::filesystem::exists(source_root / ".git", ec) && !ec) {
        const auto baseline = core::scm::GitProvider(source_root).repository_snapshot();
        if (!baseline) {
            std::cerr << "could not obtain baseline Git snapshot\n";
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        const pid_t git_child = ::fork();
        if (git_child == 0) {
            core::landrun::LandrunSettings::instance().configure_startup({
                .mode = core::landrun::LandrunMode::workspace_write});
            const auto confined = core::scm::GitProvider(source_root).repository_snapshot();
            const auto revision = core::review::GitClient(source_root).rev_parse("HEAD");
            ::_exit(confined && revision && confined->revision == baseline->revision
                    && *revision == baseline->revision
                    && confined->status_fingerprint == baseline->status_fingerprint ? 0 : 1);
        }
        int git_status = 0;
        if (git_child < 0 || ::waitpid(git_child, &git_status, 0) != git_child
            || !WIFEXITED(git_status) || WEXITSTATUS(git_status) != 0) {
            std::cerr << "confined Git snapshot did not match baseline\n";
            std::filesystem::remove_all(base, ec);
            return 26;
        }
    }
    {
        // Git status may execute a repository-configured fsmonitor helper. A
        // real listener proves that the helper runs but cannot connect when
        // GitProvider is launched through Landrun.
        const auto repository = base / "git-network-fixture";
        std::filesystem::create_directories(repository, ec);
        if (ec) return 26;
        const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in endpoint{};
        endpoint.sin_family = AF_INET;
        endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t endpoint_size = sizeof(endpoint);
        if (listener < 0 || ::bind(listener, reinterpret_cast<sockaddr*>(&endpoint), sizeof(endpoint)) != 0
            || ::listen(listener, 4) != 0
            || ::getsockname(listener, reinterpret_cast<sockaddr*>(&endpoint), &endpoint_size) != 0) {
            if (listener >= 0) ::close(listener);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        const auto marker = repository / "hook-ran";
        const auto hook = repository / "fsmonitor-hook";
        {
            std::ofstream tracked(repository / "tracked");
            tracked << "test\n";
            std::ofstream script(hook);
            script << "#!/bin/sh\nprintf invoked > " << shell_quote(marker.string())
                   << "\ncurl -s --max-time 1 http://127.0.0.1:"
                   << ntohs(endpoint.sin_port)
                   << "/ >/dev/null 2>&1\nprintf 'token\\000'\n";
        }
        std::filesystem::permissions(hook, std::filesystem::perms::owner_all, ec);
        const std::string git = "git -C " + shell_quote(repository.string());
        if (ec || ::system((git + " init -q").c_str()) != 0
            || ::system((git + " -c core.fsmonitor=false add tracked").c_str()) != 0
            || ::system((git + " config core.fsmonitor " + shell_quote(hook.string())).c_str()) != 0) {
            ::close(listener);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        (void)core::scm::GitProvider(repository).get_status_summary();
        pollfd incoming{.fd = listener, .events = POLLIN, .revents = 0};
        const bool baseline_connected = ::poll(&incoming, 1, 0) == 1;
        while (::poll(&incoming, 1, 0) == 1) {
            const int connection = ::accept(listener, nullptr, nullptr);
            if (connection < 0) break;
            ::close(connection);
        }
        const bool baseline_ran = std::filesystem::exists(marker, ec);
        std::filesystem::remove(marker, ec);
        const pid_t git_child = ::fork();
        if (git_child == 0) {
            core::landrun::LandrunSettings::instance().configure_startup({
                .mode = core::landrun::LandrunMode::workspace_write});
            const auto status = core::scm::GitProvider(repository).get_status_summary();
            ::_exit(!status.empty() && std::filesystem::exists(marker) ? 0 : 1);
        }
        int git_status = 0;
        const bool child_ok = git_child > 0 && ::waitpid(git_child, &git_status, 0) == git_child
            && WIFEXITED(git_status) && WEXITSTATUS(git_status) == 0;
        incoming.revents = 0;
        const bool sandbox_connected = ::poll(&incoming, 1, 0) != 0;
        ::close(listener);
        if (!baseline_connected || !baseline_ran || !child_ok || sandbox_connected) {
            std::cerr << "Git fsmonitor network confinement failed\n";
            std::filesystem::remove_all(base, ec);
            return 26;
        }
    }
    const auto shell_denied_file = source_root
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
    core::landrun::add_readable_root(shell_policy, source_root);
    if (const auto readiness = core::landrun::verify_landrun_readiness(shell_policy);
        !readiness.success) {
        std::cerr << readiness.detail << '\n';
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    {
        std::string bytes;
        const auto result = core::landrun::run_confined_command(
            shell_policy, "printf 'a\\000b'; exit 7",
            [&](std::string_view chunk) { bytes.append(chunk); });
        if (result.exit_code != 7 || bytes != std::string("a\0b", 3)) {
            std::cerr << "confined command did not preserve output or exit status: "
                      << result.error << '\n';
            std::filesystem::remove(shell_denied_file, ec);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        const auto timeout = core::landrun::run_confined_command(
            shell_policy, "sleep 5", [](std::string_view) {}, std::chrono::milliseconds(100));
        if (!timeout.timed_out) {
            std::cerr << "confined command timeout was not enforced\n";
            std::filesystem::remove(shell_denied_file, ec);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in endpoint{};
        endpoint.sin_family = AF_INET;
        endpoint.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        endpoint.sin_port = 0;
        socklen_t endpoint_size = sizeof(endpoint);
        if (listener < 0 || ::bind(listener, reinterpret_cast<sockaddr*>(&endpoint), sizeof(endpoint)) != 0
            || ::listen(listener, 2) != 0
            || ::getsockname(listener, reinterpret_cast<sockaddr*>(&endpoint), &endpoint_size) != 0) {
            if (listener >= 0) ::close(listener);
            std::cerr << "could not start local network boundary fixture\n";
            std::filesystem::remove(shell_denied_file, ec);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        const int baseline = ::socket(AF_INET, SOCK_STREAM, 0);
        if (baseline < 0 || ::connect(baseline, reinterpret_cast<sockaddr*>(&endpoint), endpoint_size) != 0) {
            if (baseline >= 0) ::close(baseline);
            ::close(listener);
            std::filesystem::remove(shell_denied_file, ec);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        const int accepted = ::accept(listener, nullptr, nullptr);
        ::close(baseline);
        if (accepted < 0) {
            ::close(listener);
            std::filesystem::remove(shell_denied_file, ec);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
        ::close(accepted);
        const auto python_ok = core::landrun::run_confined_command(
            shell_policy, "python3 -c 'pass'", [](std::string_view) {});
        const std::string probe = "python3 -c 'import socket; socket.create_connection((\"127.0.0.1\", "
            + std::to_string(ntohs(endpoint.sin_port)) + "), timeout=2)'";
        const auto denied = core::landrun::run_confined_command(
            shell_policy, probe, [](std::string_view) {});
        pollfd incoming{.fd = listener, .events = POLLIN, .revents = 0};
        const int pending = ::poll(&incoming, 1, 0);
        ::close(listener);
        if (python_ok.exit_code != 0 || denied.exit_code == 0 || denied.timed_out || pending != 0) {
            std::cerr << "confined command network boundary failed\n";
            std::filesystem::remove(shell_denied_file, ec);
            std::filesystem::remove_all(base, ec);
            return 26;
        }
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
    std::string command =
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
        + "'; printf temp-ok > \"$TMPDIR/probe\"; git --version >/dev/null; c++ --version >/dev/null; "
          "python3 -c 'print(1)' >/dev/null; test -r "
        + shell_quote((source_root / "CMakeLists.txt").string());
    if (std::filesystem::exists(source_root / ".git", ec) && !ec) {
        command += "; git -C " + shell_quote(source_root.string())
            + " --no-pager diff --stat HEAD >/dev/null";
    }
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

    core::tools::shell::PosixShellExecutor read_only_shell;
    auto read_only_policy = core::landrun::LandrunPolicyCompiler::compile(
        workspace, core::landrun::LandrunMode::read_only);
    core::landrun::add_readable_root(read_only_policy, source_root);
    if (const auto readiness = core::landrun::verify_landrun_readiness(read_only_policy);
        !readiness.success) {
        std::cerr << readiness.detail << '\n';
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    read_only_shell.configure_landrun(std::move(read_only_policy));
    const auto read_only_denied_file = allowed / "read-only-write-must-fail";
    const auto read_only_temp_file =
        core::landrun::LandrunSettings::instance().effective_tmpdir(
            core::landrun::LandrunMode::read_only)
        / "read-only-temp-probe";
    const std::string read_only_command =
        "if printf denied > " + shell_quote(read_only_denied_file.string())
        + "; then exit 41; fi; printf temp-ok > "
        + shell_quote(read_only_temp_file.string())
        + "; test -r " + shell_quote((allowed / "from-shell").string());
    const auto read_only_result = read_only_shell.run(
        read_only_command, {}, std::chrono::seconds{10});
    if (read_only_result.exit_code != 0) {
        std::cerr << "read-only sandbox failed with exit code "
                  << read_only_result.exit_code << ": "
                  << read_only_result.output << '\n';
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    if (std::filesystem::exists(read_only_denied_file, ec)
        || !std::filesystem::exists(read_only_temp_file, ec)) {
        std::filesystem::remove(shell_denied_file, ec);
        std::filesystem::remove_all(base, ec);
        return 26;
    }
    std::filesystem::remove(read_only_temp_file, ec);

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
