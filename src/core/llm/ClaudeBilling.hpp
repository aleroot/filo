#pragma once

#include "Models.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

/// Claude Code's billing attribution block, as Claude Code 2.1.302 writes it:
/// `x-anthropic-billing-header: cc_version=<version>.<suffix>; cc_entrypoint=cli; cch=<token>;`
/// is the first system block, the suffix is derived from the first user
/// prompt, and `cch` signs the final request body.
namespace core::llm::claude {

inline constexpr std::string_view kClaudeCodeVersion = "2.1.302";
inline constexpr std::string_view kBillingEntrypoint = "cli";
inline constexpr std::string_view kBillingMarker = "x-anthropic-billing-header:";

/// The visible prompt of the first user message, chosen as Claude Code does:
/// a `<command-name>` text block from that tag on, else the last non-empty
/// text block, else the string content. Empty when the first user turn holds
/// only tool results.
[[nodiscard]] std::string_view first_user_text(const ChatRequest& request);

/// Three lowercase hex digits: SHA-256 over a salt, the UTF-16 code units at
/// positions 4, 7 and 20 of @p first_user_text ('0' when absent) and @p version.
[[nodiscard]] std::string version_suffix(std::string_view first_user_text,
                                         std::string_view version = kClaudeCodeVersion);

/// Request lineage Claude Code appends after `cch`. Every field is optional.
struct BillingLineage {
    std::string previous_request_id; ///< `cc_prev_req`, only when a valid `req_` id.
    std::string prompt_id;           ///< `cc_prompt_id`, a UUID per user prompt.
    int prompt_index = 0;            ///< `cc_prompt_index`/`cc_turn_index` when > 0.
};

/// True for an Anthropic `request-id` (`req_` + 8..128 of [A-Za-z0-9_-]).
[[nodiscard]] bool is_valid_request_id(std::string_view id) noexcept;

/// User prompts in @p request: user messages, which in Filo never hold only
/// tool results (those are `tool` messages).
[[nodiscard]] int count_user_prompts(const ChatRequest& request) noexcept;

/// The system block text with an unsigned `cch=00000;` placeholder, followed by
/// `cc_prev_req`, then `cc_prompt_id` and, with a prompt index,
/// `cc_turn_origin=human; cc_prompt_index=N; cc_turn_index=N;`.
[[nodiscard]] std::string billing_block(std::string_view first_user_text,
                                        const BillingLineage& lineage = {});

/// XXH64 of @p data.
[[nodiscard]] std::uint64_t xxh64(std::string_view data, std::uint64_t seed) noexcept;

/// The 5-hex-digit `cch` of an unsigned body: XXH64 over the body as
/// JSON.stringify(JSON.parse(body)) emits it, with top-level `model` set to ""
/// and `max_tokens` removed. nullopt when @p unsigned_body is not a JSON object.
[[nodiscard]] std::optional<std::string> body_signature(std::string_view unsigned_body);

/// Rewrite an unsigned body the way Claude Code sends it, then sign it: the
/// top-level fields in Claude Code's order (model, messages, system, tools,
/// tool_choice, metadata, max_tokens, temperature, thinking,
/// context_management, output_config, diagnostics, stream, speed, then any
/// other), serialized as JSON.stringify would, with the billing block's
/// `cch=00000;` replaced by the body signature. Bodies without the
/// placeholder are left unchanged.
void sign_body(std::string& body);

} // namespace core::llm::claude
