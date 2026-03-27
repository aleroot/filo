#pragma once

#include <chrono>
#include <string>
#include <string_view>

namespace core::tools::shell {

// ---------------------------------------------------------------------------
// IShellExecutor — platform-agnostic shell execution interface.
//
// Current implementation:
//   PosixShellExecutor  — persistent bash session via fork+pipe (Linux/macOS/WSL)
//
// Future implementations can be added here without touching ShellTool:
//   WindowsShellExecutor — CreateProcess + pwsh/cmd (native Win32, not yet needed)
//
// Thread safety: NOT thread-safe. Callers must serialise access with a mutex.
// ---------------------------------------------------------------------------
class IShellExecutor {
public:
    // 10-minute default — covers compilation, npm install, docker build, etc.
    static constexpr std::chrono::milliseconds kDefaultTimeout{600'000};

    struct Result {
        std::string output;    // merged stdout + stderr
        int         exit_code{-1};
    };

    virtual ~IShellExecutor() = default;

    // Execute a shell command and return its merged output + exit code.
    //
    // working_dir — when non-empty, the command runs with this directory as
    //   its current working directory without permanently changing the
    //   session's own cwd (implemented as a subshell on POSIX).
    virtual Result run(std::string_view command,
                       std::string_view working_dir = {},
                       std::chrono::milliseconds timeout = kDefaultTimeout) = 0;

    // True if the underlying shell process is still alive.
    virtual bool is_alive() const noexcept = 0;

    // Terminate and restart the shell session.
    virtual void reset() = 0;

    // Human-readable name of the shell in use (e.g. "bash").
    virtual std::string_view shell_name() const noexcept = 0;
};

} // namespace core::tools::shell
