#pragma once

/**
 * @file MiniMaxImage.hpp
 * @brief MiniMax-protocol-local image policy.
 *
 * Owned by MiniMaxProtocol. Other providers must not include this header or
 * share these limits.
 *
 * MiniMax's Anthropic-compatible endpoint accepts JPEG, PNG, GIF and WebP
 * images of at most 10 MiB each. MiniMax Code (`native-image-mime.ts`,
 * `user-media.ts`) sends only those formats, demotes anything else to text,
 * and drops an unusable attachment instead of failing the turn. This module
 * applies the same rules from the bytes themselves, so `media_type` always
 * matches the payload. Filo carries no image codec, so it does not re-encode
 * images the way MiniMax Code does to save tokens.
 */

#include "core/llm/Models.hpp"

#include <cstddef>
#include <string_view>

namespace core::llm::protocols {

/// Per-image ceiling documented for MiniMax's Messages endpoint.
inline constexpr std::size_t kMiniMaxImageMaxBytes = 10 * 1024 * 1024;

/// MIME of a format MiniMax accepts, detected from its leading bytes; empty
/// for anything else.
[[nodiscard]] std::string_view minimax_image_mime(std::string_view bytes) noexcept;

/// Labels each inline or local image with its detected MIME and replaces
/// unsupported, oversized or unreadable ones with the shared
/// "[Attached image unavailable: …]" text. Remote URLs are left to MiniMax.
void prepare_minimax_images(ChatRequest& request);

} // namespace core::llm::protocols
