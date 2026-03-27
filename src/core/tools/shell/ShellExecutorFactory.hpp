#pragma once

#include "IShellExecutor.hpp"

// Select the right backend at compile time.
// To add a native Windows backend in the future, implement WindowsShellExecutor
// (see IShellExecutor.hpp for the contract) and add the #ifdef _WIN32 branch here.
#ifndef _WIN32
#  include "PosixShellExecutor.hpp"
#endif

#include <memory>

namespace core::tools::shell {

// ---------------------------------------------------------------------------
// make_shell_executor() — compile-time platform selection.
//
// Currently returns PosixShellExecutor on all platforms (Linux, macOS, WSL).
// A future native Windows port would add a WindowsShellExecutor branch here.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::unique_ptr<IShellExecutor> make_shell_executor() {
#ifndef _WIN32
    return std::make_unique<PosixShellExecutor>();
#else
    // Native Windows is not yet implemented.
    // Under WSL, Filo runs as a Linux process and uses PosixShellExecutor.
    // For a future native Win32 port, implement WindowsShellExecutor and
    // return it here.
    static_assert(false,
        "Native Windows shell execution is not yet implemented. "
        "Run Filo under WSL to use it on Windows.");
#endif
}

} // namespace core::tools::shell
