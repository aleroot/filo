#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace core::utils::image {

/// Image formats recognized from their leading bytes. Only PNG, JPEG, GIF
/// and WebP are accepted by every vision API Filo talks to; the others are
/// recognized so callers can say what a file is instead of "binary".
enum class ImageFormat { png, jpeg, gif, webp, bmp, tiff, heif, avif, psd };

struct ImageHeader {
    ImageFormat format = ImageFormat::png;
    /// 0 when the header does not state it within the bytes inspected (a
    /// truncated prefix, or a JPEG whose metadata segments outgrow it).
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

/// Identify an image from its leading bytes and read its dimensions from the
/// header. Codec-free: nothing is decoded. A complete signature identifies
/// the format (the WHATWG MIME Sniffing image patterns for the four vision
/// formats); dimensions are filled in only when the prefix contains them.
[[nodiscard]] std::optional<ImageHeader> sniff(std::string_view bytes) noexcept;
[[nodiscard]] std::string_view format_name(ImageFormat format) noexcept;
/// MIME type for a format vision APIs accept (PNG, JPEG, GIF, WebP), or empty.
[[nodiscard]] std::string_view vision_mime_type(ImageFormat format) noexcept;

} // namespace core::utils::image
