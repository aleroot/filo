#pragma once

#include "ClaudeBilling.hpp"

#include <cpr/cprtypes.h>

#include <string>
#include <string_view>
#include <vector>

/// The client identity Claude Code 2.1.302 presents on `POST /v1/messages`,
/// apart from its system prompt (Filo keeps its own): HTTP headers, the
/// `anthropic-beta` list, per-conversation lineage and request-body gzip.
/// Reference: cortexkit/commons 3ce6295 `claude-oauth-wire` vectors and
/// cortexkit/anthropic-auth, which reconstruct an interactive 2.1.302 capture.
namespace core::llm::claude {

inline constexpr std::string_view kUserAgent = "claude-cli/2.1.302 (external, cli)";
inline constexpr std::string_view kStainlessPackageVersion = "0.128.0";
inline constexpr std::string_view kStainlessRuntimeVersion = "v26.3.0";
inline constexpr std::string_view kOAuthBeta = "oauth-2025-04-20";
inline constexpr std::string_view kStructuredOutputsBeta = "structured-outputs-2025-12-15";
/// Smallest body, in UTF-16 code units, that Claude Code sends gzipped.
inline constexpr std::size_t kGzipMinBodyChars = 4096;

/// `x-stainless-os` / `x-stainless-arch` for the host Filo runs on.
[[nodiscard]] std::string_view stainless_os() noexcept;
[[nodiscard]] std::string_view stainless_arch() noexcept;

/// Headers that do not depend on the request: User-Agent, x-app,
/// anthropic-dangerous-direct-browser-access and the static x-stainless-*.
void apply_client_headers(cpr::Header& headers);

/// Claude Code's `anthropic-beta` list for a conversation turn on @p model.
/// @p oauth keeps the OAuth beta; @p structured_output adds the
/// structured-outputs beta before cache-diagnosis.
[[nodiscard]] std::vector<std::string_view> main_request_betas(
    std::string_view model, bool oauth, bool structured_output);

/// What one request carries about the conversation it belongs to.
struct RequestLineage {
    std::string session_id;          ///< x-claude-code-session-id, metadata session_id.
    BillingLineage billing;          ///< cc_prev_req / cc_prompt_id / indexes.
    std::string previous_message_id; ///< diagnostics.previous_message_id; empty = null.
};

/// Lineage for the next request of the Filo conversation @p conversation.
/// The prompt id is a fresh UUID whenever @p prompt_index changes and is kept
/// for every request that serves the same prompt (tool-result follow-ups).
[[nodiscard]] RequestLineage begin_request(std::string_view conversation, int prompt_index);
/// Remember a successful response's `request-id` / message id for the next request.
void record_request_id(std::string_view conversation, std::string_view request_id);
void record_message_id(std::string_view conversation, std::string_view message_id);

/// 64 hex digits, fixed for the process.
[[nodiscard]] std::string_view device_id();
/// `metadata.user_id`: a JSON string of device, account and session ids.
[[nodiscard]] std::string metadata_user_id(std::string_view account_uuid,
                                           std::string_view session_id);

/// Whether Claude Code would gzip @p body for @p url: api.anthropic.com only,
/// at least kGzipMinBodyChars, no proxy, no client certificate,
/// CLAUDE_CODE_GZIP_REQUEST_BODIES not off, and no earlier rejection.
[[nodiscard]] bool should_gzip_body(std::string_view url, std::string_view body);
/// The wire bytes for a signed body: `cch` reset to 00000, a newline and up to
/// 256 random spaces/tabs appended, one gzip member (level 6, OS byte 255).
[[nodiscard]] std::string gzip_body(std::string_view signed_body);
/// Claude Code's test for a rejected gzipped body: 400/403/415 that did not
/// come from Anthropic, or a 400 saying the body is not valid JSON.
[[nodiscard]] bool is_gzip_rejection(int status, const cpr::Header& headers,
                                     std::string_view body);
/// Stop gzipping for the rest of the process.
void disable_body_gzip() noexcept;

} // namespace core::llm::claude
