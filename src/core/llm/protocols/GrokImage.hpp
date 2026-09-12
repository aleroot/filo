#pragma once

/**
 * @file GrokImage.hpp
 * @brief Grok-protocol-local image policy matching xAI Grok Build / api.x.ai.
 *
 * Owned by GrokProtocol / GrokResponsesProtocol. Other providers must not
 * include this header or share these limits.
 *
 * grok-build (`xai-grok-image` + `image_normalize`) validates and, when
 * needed, transcodes attachments before they hit inference. The engines only
 * sample JPEG/PNG/WebP; GIF/BMP/TIFF/ICO must be PNG on the wire. The backend
 * also rejects images with either side under 8 px or fewer than 512 total
 * pixels, and truncated JPEG/PNG/WebP payloads.
 *
 * Filo does not transcode (no image codec). Unsupported or unsafe inputs are
 * replaced with an explanatory text placeholder so a 400 cannot poison the
 * rest of the conversation — the same recovery idea as grok-build's
 * `IMAGE_STRIP_PLACEHOLDER`.
 */

#include "core/llm/Models.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace core::llm::protocols {

enum class GrokImageFormat {
    Png,
    Jpeg,
    WebP,
    Gif,
    Bmp,
    Tiff,
    Ico,
    Unknown,
};

[[nodiscard]] GrokImageFormat sniff_grok_image_format(std::string_view bytes) noexcept;

/// MIME for engine-native formats; empty for formats that must not go on the wire.
[[nodiscard]] std::string_view grok_image_mime(GrokImageFormat format) noexcept;

[[nodiscard]] bool grok_image_needs_png_transcode(GrokImageFormat format) noexcept;

[[nodiscard]] bool grok_image_structurally_complete(std::string_view bytes);

[[nodiscard]] std::optional<std::pair<std::uint32_t, std::uint32_t>>
grok_image_dimensions(std::string_view bytes);

/// Rewrite image content parts so Grok only receives engine-native, complete
/// images that clear the backend's dimension floor.
void prepare_grok_images(ChatRequest& request);

} // namespace core::llm::protocols
