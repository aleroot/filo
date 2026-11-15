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
/// the program the terminal sets (with its version), else the TERM
/// capability string, else the name the terminal is known by.
///
/// The environment is read on every call: the User-Agent that carries the
/// token is built once, so the token itself never changes between two
/// requests of the same process.
[[nodiscard]] std::string terminal_token();

/// Filo's own User-Agent: the product and its version, then the platform the
/// client runs on — the shape the Codex CLI uses, without the terminal token
/// (a provider that only wants to know which client it is serving has no
/// use for it).
[[nodiscard]] const std::string& client_user_agent();

} // namespace core::utils
