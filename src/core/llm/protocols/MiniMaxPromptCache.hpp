#pragma once

/**
 * @file MiniMaxPromptCache.hpp
 * @brief MiniMax-protocol-local prompt-cache breakpoints.
 *
 * Owned by MiniMaxProtocol. MiniMax's Anthropic-compatible endpoint caches
 * the prefix before an explicit block-level `cache_control` marker. MiniMax
 * Code (`anthropic.ts`) places one ephemeral marker on the last system block,
 * one on the last tool and one on the last block of a trailing user message,
 * and sends no top-level marker. The shared Messages encoder's cache option
 * places a top-level automatic marker and a system marker at filo's stable
 * prompt boundary instead, so MiniMax leaves it off and adds MiniMax Code's
 * markers to the encoded request here.
 */

#include <string>

namespace core::llm::protocols {

/// Adds MiniMax Code's three breakpoints to a serialized Messages request in
/// place. A payload this cannot parse is left unchanged, uncached.
void add_minimax_cache_breakpoints(std::string& payload);

} // namespace core::llm::protocols
