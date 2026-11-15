#pragma once

// ---------------------------------------------------------------------------
// ClientPlatform — what this process runs on, as the Codex CLI reports it in
// its User-Agent: the operating system, its version and the architecture.
//
// The values are read once and kept: a User-Agent must not change between
// two requests of the same process.
// ---------------------------------------------------------------------------

#include <string>
#include <string_view>

namespace core::utils {

/// The operating system and the machine it runs on.
struct ClientPlatform {
    std::string os;           ///< "macOS", "Ubuntu", "Linux".
    std::string version;      ///< "15.6.1", "24.04".
    std::string architecture; ///< "aarch64", "x86_64".
};

/// The platform of this process, read once.
[[nodiscard]] const ClientPlatform& client_platform();

/// The terminal this process runs in, named the way the Codex CLI names it:
/// the value of TERM_PROGRAM (with its version when there is one), else the
/// value of TERM. Empty when the environment names no terminal.
[[nodiscard]] std::string_view terminal_token();

} // namespace core::utils
