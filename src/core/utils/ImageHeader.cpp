#include "ImageHeader.hpp"

namespace core::utils::image {
namespace {

[[nodiscard]] std::uint32_t byte_at(std::string_view bytes, std::size_t at) noexcept {
    return static_cast<unsigned char>(bytes[at]);
}
[[nodiscard]] std::uint32_t be16(std::string_view b, std::size_t at) noexcept {
    return byte_at(b, at) << 8 | byte_at(b, at + 1);
}
[[nodiscard]] std::uint32_t be32(std::string_view b, std::size_t at) noexcept {
    return be16(b, at) << 16 | be16(b, at + 2);
}
[[nodiscard]] std::uint32_t le16(std::string_view b, std::size_t at) noexcept {
    return byte_at(b, at) | byte_at(b, at + 1) << 8;
}
[[nodiscard]] std::uint32_t le24(std::string_view b, std::size_t at) noexcept {
    return le16(b, at) | byte_at(b, at + 2) << 16;
}
[[nodiscard]] std::uint32_t le32(std::string_view b, std::size_t at) noexcept {
    return le16(b, at) | le16(b, at + 2) << 16;
}

[[nodiscard]] ImageHeader png_header(std::string_view b) noexcept {
    ImageHeader header{.format = ImageFormat::png};
    if (b.size() >= 24 && b.substr(12, 4) == "IHDR") {
        header.width = be32(b, 16);
        header.height = be32(b, 20);
    }
    return header;
}

[[nodiscard]] ImageHeader gif_header(std::string_view b) noexcept {
    ImageHeader header{.format = ImageFormat::gif};
    if (b.size() >= 10) {
        header.width = le16(b, 6);
        header.height = le16(b, 8);
    }
    return header;
}

[[nodiscard]] ImageHeader jpeg_header(std::string_view b) noexcept {
    ImageHeader header{.format = ImageFormat::jpeg};
    std::size_t i = 2;
    while (i + 4 <= b.size()) {
        if (byte_at(b, i) != 0xFF) return header;
        const auto marker = byte_at(b, i + 1);
        if (marker == 0xFF) { ++i; continue; }  // fill byte
        // Standalone markers carry no length.
        if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) { i += 2; continue; }
        if (marker == 0xD9 || marker == 0xDA) return header;  // EOI / scan data
        const auto length = be16(b, i + 2);
        const bool frame = marker >= 0xC0 && marker <= 0xCF
            && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if (frame && i + 9 <= b.size()) {
            header.height = be16(b, i + 5);
            header.width = be16(b, i + 7);
            return header;
        }
        if (length < 2) return header;
        i += 2 + length;
    }
    return header;
}

[[nodiscard]] ImageHeader webp_header(std::string_view b) noexcept {
    ImageHeader header{.format = ImageFormat::webp};
    if (b.size() < 30) return header;
    const auto chunk = b.substr(12, 4);
    if (chunk == "VP8 " && byte_at(b, 23) == 0x9D && byte_at(b, 24) == 0x01 && byte_at(b, 25) == 0x2A) {
        header.width = le16(b, 26) & 0x3FFF;
        header.height = le16(b, 28) & 0x3FFF;
    } else if (chunk == "VP8L" && byte_at(b, 20) == 0x2F) {
        const auto bits = le32(b, 21);
        header.width = (bits & 0x3FFF) + 1;
        header.height = (bits >> 14 & 0x3FFF) + 1;
    } else if (chunk == "VP8X") {
        header.width = le24(b, 24) + 1;
        header.height = le24(b, 27) + 1;
    }
    return header;
}

[[nodiscard]] std::optional<ImageFormat> iso_media_format(std::string_view b) noexcept {
    if (b.size() < 12 || b.substr(4, 4) != "ftyp") return std::nullopt;
    const auto brand = b.substr(8, 4);
    if (brand == "avif" || brand == "avis") return ImageFormat::avif;
    for (const std::string_view heif : {"heic", "heix", "hevc", "hevx", "heim", "heis", "mif1", "msf1"})
        if (brand == heif) return ImageFormat::heif;
    return std::nullopt;
}

// "BM" alone is too weak a magic; also require a known DIB header size.
[[nodiscard]] std::optional<ImageHeader> bmp_header(std::string_view b) noexcept {
    if (b.size() < 26 || !b.starts_with("BM")) return std::nullopt;
    const auto dib = le32(b, 14);
    if (dib != 12 && dib != 40 && dib != 52 && dib != 56 && dib != 64 && dib != 108 && dib != 124)
        return std::nullopt;
    const bool core = dib == 12;
    // Negative height marks a top-down bitmap.
    const auto height = static_cast<std::int32_t>(core ? le16(b, 20) : le32(b, 22));
    return ImageHeader{
        .format = ImageFormat::bmp,
        .width = core ? le16(b, 18) : le32(b, 18),
        .height = static_cast<std::uint32_t>(height < 0 ? -static_cast<std::int64_t>(height) : height),
    };
}

} // namespace

std::optional<ImageHeader> sniff(std::string_view b) noexcept {
    if (b.starts_with("\x89PNG\r\n\x1a\n")) return png_header(b);
    if (b.starts_with("\xFF\xD8\xFF")) return jpeg_header(b);
    if (b.starts_with("GIF87a") || b.starts_with("GIF89a")) return gif_header(b);
    if (b.size() >= 12 && b.starts_with("RIFF") && b.substr(8, 4) == "WEBP") return webp_header(b);
    if (const auto format = iso_media_format(b)) return ImageHeader{.format = *format};
    if (b.starts_with(std::string_view("II*\0", 4)) || b.starts_with(std::string_view("MM\0*", 4)))
        return ImageHeader{.format = ImageFormat::tiff};
    if (b.size() >= 26 && b.starts_with("8BPS"))
        return ImageHeader{.format = ImageFormat::psd, .width = be32(b, 18), .height = be32(b, 14)};
    return bmp_header(b);
}

std::string_view format_name(ImageFormat format) noexcept {
    switch (format) {
        case ImageFormat::png: return "PNG";
        case ImageFormat::jpeg: return "JPEG";
        case ImageFormat::gif: return "GIF";
        case ImageFormat::webp: return "WebP";
        case ImageFormat::bmp: return "BMP";
        case ImageFormat::tiff: return "TIFF";
        case ImageFormat::heif: return "HEIF";
        case ImageFormat::avif: return "AVIF";
        case ImageFormat::psd: return "PSD";
    }
    return "image";
}

std::string_view vision_mime_type(ImageFormat format) noexcept {
    switch (format) {
        case ImageFormat::png: return "image/png";
        case ImageFormat::jpeg: return "image/jpeg";
        case ImageFormat::gif: return "image/gif";
        case ImageFormat::webp: return "image/webp";
        case ImageFormat::bmp:
        case ImageFormat::tiff:
        case ImageFormat::heif:
        case ImageFormat::avif:
        case ImageFormat::psd: break;
    }
    return {};
}

} // namespace core::utils::image
