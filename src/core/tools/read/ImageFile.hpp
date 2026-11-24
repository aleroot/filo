#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::tools::read {

/// Largest image sent to a model. Anthropic rejects an image whose base64
/// form exceeds 5 MiB, the tightest limit among the supported APIs, so the
/// raw bytes stop at 3/4 of that.
inline constexpr std::uintmax_t kMaxImageBytes = 5 * 1024 * 1024 / 4 * 3;
/// Anthropic rejects either side above 8000 px; the others downscale.
inline constexpr std::uint32_t kMaxImageDimension = 8000;

/// Whether @p path carries an image file extension (case-insensitive).
[[nodiscard]] bool has_image_extension(const std::filesystem::path& path);

/// `read` result for the image at @p resolved, or nullopt when the file is
/// not an image (or cannot be opened, so the text path reports why). The
/// result names the file in `kind`/`path`/`mime_type`; the agent attaches the
/// pixels to the tool result as an image input. @p requested is the path as
/// the caller wrote it, used in messages.
[[nodiscard]] std::optional<std::string> read_image_file(std::string_view requested,
                                                         const std::filesystem::path& resolved);

/// The image a `read_image_file` result describes, with its bytes.
struct DescribedImage {
    std::string path;
    std::string mime_type;
    std::string bytes;
};

/// The image @p result describes, read again so its bytes can be kept with
/// the result, or nullopt when @p result describes no image. When the file
/// no longer holds the image described (rewritten or emptied since), @p result
/// is replaced by an error asking to read it again, and nullopt is returned.
[[nodiscard]] std::optional<DescribedImage> load_described_image(std::string& result);

} // namespace core::tools::read
