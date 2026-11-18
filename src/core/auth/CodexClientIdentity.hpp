#pragma once

#include <string>
#include <string_view>

namespace core::auth::codex_cli {

// Identity the ChatGPT subscription backend expects from the official Codex
// CLI. The backend gates models on the reported version ("requires a newer
// version of Codex"), so both the `version` header and the catalog query use
// the released CLI version rather than Filo's own.
//
// Audited against openai/codex (codex-rs) tag rust-v0.170.0. That tag's
// models-manager/models.json is byte-identical to main and is the first
// stable catalog that lists gpt-6.2-sol and gpt-6.2-astra (minimal client
// 0.166.0). The originator and User-Agent shape are unchanged from 0.169.0
// (login/src/auth/default_client.rs). The version is stamped at release time
// and is not in the source tree, so it is taken from the npm package
// @openai/codex 0.170.0.
inline constexpr std::string_view kOriginator = "codex_cli_rs";
inline constexpr std::string_view kClientVersion = "0.170.0";

[[nodiscard]] const std::string& originator();
[[nodiscard]] const std::string& client_version();
[[nodiscard]] const std::string& user_agent();

} // namespace core::auth::codex_cli
