#pragma once

/**
 * @file MiniMaxImage.hpp
 * @brief MiniMax-protocol-local image policy.
 *
 * Owned by MiniMaxProtocol. Other providers must not include this header or
 * share these limits.
 *
 * MiniMax's Anthropic-compatible endpoint accepts JPEG, PNG, GIF and WebP
 * images of at most 10 MiB each. This module prepares a request's images the
 * way MiniMax Code does before they reach the wire:
 *  - the media type is detected from the bytes (`native-image-mime.ts`), and
 *    an unusable image becomes text instead of failing the turn
 *    (`user-media.ts`);
 *  - a user message inlines at most four images (`max_attachments_count`,
 *    `minimax-model-catalog.ts`), and a request keeps only the newest twenty
 *    images whose base64 fits the body limit (`request-image-limit.ts`,
 *    `selectMultimodalAttachments`). Older ones become a note naming the file;
 *  - each user message that attached local images starts with a
 *    `<system-reminder>` listing their paths and whether they were sent inline
 *    (`local-turn-input-preparation.ts`), so the model can tie each image to a
 *    file and pass that path to tools.
 * Filo carries no image codec, so it does not re-encode images the way
 * MiniMax Code does to save tokens.
 */

#include "core/llm/Models.hpp"

#include <cstddef>
#include <string_view>

namespace core::llm::protocols {

/// Per-image ceiling documented for MiniMax's Messages endpoint.
inline constexpr std::size_t kMiniMaxImageMaxBytes = 10 * 1024 * 1024;
/// Images one user message may inline (MiniMax Code's `max_attachments_count`).
inline constexpr std::size_t kMiniMaxImagesPerMessage = 4;
/// Images one request may carry, newest first (MiniMax Code's
/// `DEFAULT_MAX_IMAGES_PER_REQUEST`).
inline constexpr std::size_t kMiniMaxImagesPerRequest = 20;

/// MIME of a format MiniMax accepts, detected from its leading bytes; empty
/// for anything else.
[[nodiscard]] std::string_view minimax_image_mime(std::string_view bytes) noexcept;

/// Applies the policy above to the request copy about to be sent. Remote URLs
/// are left to MiniMax and do not count toward the limits.
void prepare_minimax_images(ChatRequest& request);

} // namespace core::llm::protocols
