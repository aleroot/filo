#pragma once

#include <string>
#include <string_view>

namespace core::auth::codex_cli {

// Identity the ChatGPT subscription backend expects from the official Codex
// CLI. The backend gates models on the reported version ("requires a newer
// version of Codex"), so both the `version` header and the catalog query use
// the released CLI version rather than Filo's own.
//
// Audited against openai/codex (codex-rs) at 97239fd and the published
// @openai/codex 0.160.0 package. The originator and User-Agent shape come from
// login/src/auth/default_client.rs; the version is stamped at release time and
// is not in the source tree, so it is taken from the npm release.
inline constexpr std::string_view kOriginator = "codex_cli_rs";
inline constexpr std::string_view kClientVersion = "0.160.0";

[[nodiscard]] const std::string& originator();
[[nodiscard]] const std::string& client_version();
[[nodiscard]] const std::string& user_agent();

} // namespace core::auth::codex_cli
