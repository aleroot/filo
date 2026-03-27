#pragma once

#ifndef _WIN32

#include "IShellExecutor.hpp"
#include "ShellSession.hpp"
#include "../shell/ShellUtils.hpp"

#include <chrono>
#include <string>
#include <string_view>

namespace core::tools::shell {

// ---------------------------------------------------------------------------
// PosixShellExecutor — persistent bash/sh session for Linux, macOS, and WSL.
//
// Wraps ShellSession (fork+pipe IPC) and adds working-directory subshell
// support.  Session state (cwd, exported env vars, shell functions, activated
// virtualenvs, etc.) persists across run() calls.
// ---------------------------------------------------------------------------
class PosixShellExecutor final : public IShellExecutor {
public:
    PosixShellExecutor()  = default;
    ~PosixShellExecutor() = default;

    Result run(std::string_view command,
               std::string_view working_dir = {},
               std::chrono::milliseconds timeout = kDefaultTimeout) override
    {
        // When working_dir is given, run inside a subshell so the session's
        // own cwd is left unchanged:  (cd '/path/to/dir' && <command>)
        std::string effective;
        if (working_dir.empty()) {
            effective = std::string(command);
        } else {
            effective  = "(cd '";
            effective += detail::shell_single_quote(working_dir);
            effective += "' && ";
            effective += command;
            effective += ')';
        }

        detail::ShellSession::Result r = session_.run(effective, timeout);
        return {std::move(r.output), r.exit_code};
    }

    bool is_alive() const noexcept override { return session_.is_alive(); }
    void reset()   override { session_.reset(); }
    std::string_view shell_name() const noexcept override { return "bash"; }

private:
    detail::ShellSession session_;
};

} // namespace core::tools::shell

#endif // !_WIN32
