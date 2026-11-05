#include "ImageFile.hpp"
#include "ReadTypes.hpp"
#include "../shell/ShellUtils.hpp"
#include "../../net/NetworkTraffic.hpp"  // format_bytes: unit-agnostic size text.
#include "../../utils/AsciiUtils.hpp"
#include "../../utils/ImageHeader.hpp"
#include "../../utils/JsonWriter.hpp"
#include <algorithm>
#include <array>
#include <format>
#include <fstream>
#include <system_error>

namespace core::tools::read {
namespace {

namespace image = core::utils::image;
using core::net::format_bytes;

// A JPEG states its size in the SOF segment, after any EXIF, ICC and
// Photoshop segments. 256 KiB covers those in practice; past it the image is
// still sent, only without dimensions.
constexpr std::size_t kHeaderScanBytes = 256 * 1024;

// One command the model can run to turn the file into something it can view.
[[nodiscard]] std::string conversion_hint(std::string_view path, bool resize) {
    const auto source = std::format("'{}'", detail::shell_single_quote(path));
#if defined(__APPLE__)
    return resize
        ? std::format("sips -Z 2000 -s format png {} --out /tmp/filo-image.png", source)
        : std::format("sips -s format png {} --out /tmp/filo-image.png", source);
#else
    return resize
        ? std::format("magick {} -resize '2000x2000>' /tmp/filo-image.png", source)
        : std::format("magick {} /tmp/filo-image.png", source);
#endif
}

} // namespace

bool has_image_extension(const std::filesystem::path& path) {
    const auto extension = path.extension().string();
    constexpr std::array<std::string_view, 14> kExtensions = {
        ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".gif", ".webp",
        ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif", ".psd"};
    return std::ranges::any_of(kExtensions, [&](std::string_view candidate) {
        return core::utils::ascii::iequals(extension, candidate);
    });
}

std::optional<std::string> read_image_file(std::string_view requested,
                                           const std::filesystem::path& resolved) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(resolved, ec);
    if (ec) return std::nullopt;
    std::ifstream input(resolved, std::ios::binary);
    if (!input) return std::nullopt;
    const auto prefix = read_prefix(input, static_cast<std::size_t>(std::min<std::uintmax_t>(size, kHeaderScanBytes)));
    const auto header = image::sniff(prefix);
    if (!header) return std::nullopt;

    const auto name = image::format_name(header->format);
    const auto mime = image::vision_mime_type(header->format);
    if (mime.empty()) {
        return format_read_error(requested, resolved.string(), std::format(
            "{} images are not accepted by model APIs (PNG, JPEG, GIF and WebP are). "
            "Convert it with run_terminal_command, e.g. `{}`, then read the PNG.",
            name, conversion_hint(requested, false)));
    }
    if (size > kMaxImageBytes || header->width > kMaxImageDimension || header->height > kMaxImageDimension) {
        return format_read_error(requested, resolved.string(), std::format(
            "{} {}x{}, {} exceeds the image input limit ({} px per side, {}). "
            "Downscale it with run_terminal_command, e.g. `{}`, then read the PNG.",
            name, header->width, header->height, format_bytes(size),
            kMaxImageDimension, format_bytes(kMaxImageBytes), conversion_hint(requested, true)));
    }

    const auto dimensions = header->width && header->height
        ? std::format("{}x{}", header->width, header->height)
        : std::string("unknown size");
    core::utils::JsonWriter writer;
    {
        auto object = writer.object();
        writer.kv_str("kind", "image").comma()
            .kv_str("path", resolved.string()).comma()
            .kv_str("mime_type", mime).comma()
            .kv_num("width", header->width).comma()
            .kv_num("height", header->height).comma()
            .kv_num("bytes", size).comma()
            .kv_str("content", std::format(
                "[Image {}: {} {}, {}. The image is attached to this result as image input; "
                "if you cannot see it, the active model does not accept images.]",
                requested, name, dimensions, format_bytes(size)));
    }
    return std::move(writer).take();
}

} // namespace core::tools::read
