#include "GrokImage.hpp"

#include "core/utils/Base64.hpp"

#include <filesystem>
#include <fstream>

namespace core::llm::protocols {
namespace {

// Backend MIN_IMAGE_PIXELS / MIN_VISION_SIDE_PX from grok-build image_normalize.
constexpr std::uint32_t kMinSidePx = 8;
constexpr std::uint64_t kMinTotalPx = 512;
// Public xAI image-understanding limit (docs: 20 MiB).
constexpr std::uintmax_t kMaxImageBytes = 20ull * 1024ull * 1024ull;

[[nodiscard]] std::uint32_t read_be16(std::string_view bytes, std::size_t at) noexcept {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at])) << 8)
        | static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 1]));
}

[[nodiscard]] std::uint32_t read_be32(std::string_view bytes, std::size_t at) noexcept {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at])) << 24)
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 1])) << 16)
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 2])) << 8)
        | static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 3]));
}

[[nodiscard]] std::uint32_t read_le16(std::string_view bytes, std::size_t at) noexcept {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at]))
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 1])) << 8);
}

[[nodiscard]] std::uint32_t read_le24(std::string_view bytes, std::size_t at) noexcept {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at]))
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 1])) << 8)
        | (static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[at + 2])) << 16);
}

[[nodiscard]] std::uint32_t crc32_ieee(std::string_view data) noexcept {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (unsigned char byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

[[nodiscard]] bool starts_with(std::string_view bytes, std::string_view prefix) noexcept {
    return bytes.size() >= prefix.size() && bytes.substr(0, prefix.size()) == prefix;
}

[[nodiscard]] std::optional<std::size_t> skip_jpeg_segment(
    std::string_view bytes,
    std::size_t at) noexcept {
    if (at + 2 > bytes.size()) return std::nullopt;
    const std::size_t len = read_be16(bytes, at);
    if (len < 2) return std::nullopt;
    const std::size_t end = at + len;
    if (end > bytes.size()) return std::nullopt;
    return end;
}

} // namespace

GrokImageFormat sniff_grok_image_format(std::string_view bytes) noexcept {
    if (bytes.size() >= 8
        && starts_with(bytes, std::string_view{"\x89PNG\r\n\x1a\n", 8})) {
        return GrokImageFormat::Png;
    }
    if (bytes.size() >= 3
        && static_cast<unsigned char>(bytes[0]) == 0xFF
        && static_cast<unsigned char>(bytes[1]) == 0xD8
        && static_cast<unsigned char>(bytes[2]) == 0xFF) {
        return GrokImageFormat::Jpeg;
    }
    if (bytes.size() >= 12
        && starts_with(bytes, "RIFF")
        && bytes.substr(8, 4) == "WEBP") {
        return GrokImageFormat::WebP;
    }
    if (starts_with(bytes, "GIF87a") || starts_with(bytes, "GIF89a")) {
        return GrokImageFormat::Gif;
    }
    if (starts_with(bytes, "BM")) {
        return GrokImageFormat::Bmp;
    }
    if (bytes.size() >= 4
        && (starts_with(bytes, std::string_view{"II*\0", 4})
            || starts_with(bytes, std::string_view{"MM\0*", 4}))) {
        return GrokImageFormat::Tiff;
    }
    if (bytes.size() >= 4
        && static_cast<unsigned char>(bytes[0]) == 0
        && static_cast<unsigned char>(bytes[1]) == 0
        && (static_cast<unsigned char>(bytes[2]) == 1
            || static_cast<unsigned char>(bytes[2]) == 2)
        && static_cast<unsigned char>(bytes[3]) == 0) {
        return GrokImageFormat::Ico;
    }
    return GrokImageFormat::Unknown;
}

std::string_view grok_image_mime(GrokImageFormat format) noexcept {
    switch (format) {
        case GrokImageFormat::Png:  return "image/png";
        case GrokImageFormat::Jpeg: return "image/jpeg";
        case GrokImageFormat::WebP: return "image/webp";
        case GrokImageFormat::Gif:  return "image/gif";
        case GrokImageFormat::Bmp:  return "image/bmp";
        case GrokImageFormat::Tiff: return "image/tiff";
        case GrokImageFormat::Ico:  return "image/x-icon";
        case GrokImageFormat::Unknown: break;
    }
    return {};
}

bool grok_image_needs_png_transcode(GrokImageFormat format) noexcept {
    switch (format) {
        case GrokImageFormat::Gif:
        case GrokImageFormat::Bmp:
        case GrokImageFormat::Tiff:
        case GrokImageFormat::Ico:
            return true;
        case GrokImageFormat::Png:
        case GrokImageFormat::Jpeg:
        case GrokImageFormat::WebP:
        case GrokImageFormat::Unknown:
            break;
    }
    return false;
}

bool grok_image_structurally_complete(std::string_view bytes) {
    switch (sniff_grok_image_format(bytes)) {
        case GrokImageFormat::Jpeg: {
            // Port of grok-build jpeg_reaches_eoi: walk markers, honor
            // byte-stuffing, require a top-level EOI.
            const std::size_t n = bytes.size();
            if (n < 2
                || static_cast<unsigned char>(bytes[0]) != 0xFF
                || static_cast<unsigned char>(bytes[1]) != 0xD8) {
                return false;
            }
            std::size_t i = 2;
            while (true) {
                while (i < n && static_cast<unsigned char>(bytes[i]) != 0xFF) {
                    ++i;
                }
                while (i < n && static_cast<unsigned char>(bytes[i]) == 0xFF) {
                    ++i;
                }
                if (i >= n) return false;
                const unsigned char marker = static_cast<unsigned char>(bytes[i]);
                ++i;
                if (marker == 0x00) continue;
                if (marker == 0xD9) return true; // EOI
                if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) continue;
                if (marker == 0xDA) {
                    const auto next = skip_jpeg_segment(bytes, i);
                    if (!next) return false;
                    i = *next;
                    while (true) {
                        while (i < n && static_cast<unsigned char>(bytes[i]) != 0xFF) {
                            ++i;
                        }
                        if (i + 1 >= n) return false;
                        const unsigned char next_marker =
                            static_cast<unsigned char>(bytes[i + 1]);
                        if (next_marker == 0x00) {
                            i += 2;
                            continue;
                        }
                        if (next_marker == 0xFF) {
                            ++i;
                            continue;
                        }
                        if (next_marker >= 0xD0 && next_marker <= 0xD7) {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    continue;
                }
                const auto next = skip_jpeg_segment(bytes, i);
                if (!next) return false;
                i = *next;
            }
        }
        case GrokImageFormat::Png: {
            constexpr std::string_view kSig{"\x89PNG\r\n\x1a\n", 8};
            if (!starts_with(bytes, kSig)) return false;
            std::size_t i = kSig.size();
            while (i + 8 <= bytes.size()) {
                const std::size_t len = read_be32(bytes, i);
                const bool is_iend = bytes.substr(i + 4, 4) == "IEND";
                const std::size_t data_start = i + 8;
                if (len > bytes.size() - data_start) return false;
                const std::size_t data_end = data_start + len;
                if (data_end + 4 > bytes.size()) return false;
                const std::uint32_t expected = read_be32(bytes, data_end);
                const std::uint32_t actual = crc32_ieee(bytes.substr(i + 4, 4 + len));
                if (actual != expected) return false;
                if (is_iend) return true;
                i = data_end + 4;
            }
            return false;
        }
        case GrokImageFormat::WebP: {
            if (bytes.size() < 12) return false;
            if (!starts_with(bytes, "RIFF") || bytes.substr(8, 4) != "WEBP") {
                return false;
            }
            const std::size_t riff_size =
                static_cast<std::size_t>(read_le16(bytes, 4))
                | (static_cast<std::size_t>(read_le16(bytes, 6)) << 16);
            return riff_size + 8 <= bytes.size();
        }
        case GrokImageFormat::Gif:
        case GrokImageFormat::Bmp:
        case GrokImageFormat::Tiff:
        case GrokImageFormat::Ico:
        case GrokImageFormat::Unknown:
            break;
    }
    return false;
}

std::optional<std::pair<std::uint32_t, std::uint32_t>>
grok_image_dimensions(std::string_view bytes) {
    switch (sniff_grok_image_format(bytes)) {
        case GrokImageFormat::Png: {
            if (bytes.size() < 24) return std::nullopt;
            if (bytes.substr(12, 4) != "IHDR") return std::nullopt;
            const std::uint32_t width = read_be32(bytes, 16);
            const std::uint32_t height = read_be32(bytes, 20);
            if (width == 0 || height == 0) return std::nullopt;
            return std::pair{width, height};
        }
        case GrokImageFormat::Jpeg: {
            const std::size_t n = bytes.size();
            if (n < 4) return std::nullopt;
            std::size_t i = 2;
            while (i + 1 < n) {
                while (i < n && static_cast<unsigned char>(bytes[i]) != 0xFF) ++i;
                while (i < n && static_cast<unsigned char>(bytes[i]) == 0xFF) ++i;
                if (i >= n) return std::nullopt;
                const unsigned char marker = static_cast<unsigned char>(bytes[i]);
                ++i;
                if (marker == 0x00 || marker == 0xD9 || marker == 0x01
                    || (marker >= 0xD0 && marker <= 0xD7)) {
                    continue;
                }
                if (marker == 0xDA) return std::nullopt;
                const bool sof = (marker >= 0xC0 && marker <= 0xC3)
                    || (marker >= 0xC5 && marker <= 0xC7)
                    || (marker >= 0xC9 && marker <= 0xCB)
                    || (marker >= 0xCD && marker <= 0xCF);
                if (sof) {
                    if (i + 7 > n) return std::nullopt;
                    const std::uint32_t height = read_be16(bytes, i + 3);
                    const std::uint32_t width = read_be16(bytes, i + 5);
                    if (width == 0 || height == 0) return std::nullopt;
                    return std::pair{width, height};
                }
                const auto next = skip_jpeg_segment(bytes, i);
                if (!next) return std::nullopt;
                i = *next;
            }
            return std::nullopt;
        }
        case GrokImageFormat::WebP: {
            if (bytes.size() < 16) return std::nullopt;
            const auto chunk = bytes.substr(12, 4);
            if (chunk == "VP8X") {
                if (bytes.size() < 30) return std::nullopt;
                const std::uint32_t width = read_le24(bytes, 24) + 1;
                const std::uint32_t height = read_le24(bytes, 27) + 1;
                return std::pair{width, height};
            }
            if (chunk == "VP8 ") {
                // Lossy bitstream: 3-byte frame tag, then 0x9d 0x01 0x2a, then
                // 14-bit width/height. Chunk header is 8 bytes at offset 12.
                constexpr std::size_t kBitstream = 20;
                if (bytes.size() < kBitstream + 10) return std::nullopt;
                if (static_cast<unsigned char>(bytes[kBitstream + 3]) != 0x9D
                    || static_cast<unsigned char>(bytes[kBitstream + 4]) != 0x01
                    || static_cast<unsigned char>(bytes[kBitstream + 5]) != 0x2A) {
                    return std::nullopt;
                }
                const std::uint32_t packed_w = read_le16(bytes, kBitstream + 6);
                const std::uint32_t packed_h = read_le16(bytes, kBitstream + 8);
                return std::pair{packed_w & 0x3FFFu, packed_h & 0x3FFFu};
            }
            if (chunk == "VP8L") {
                constexpr std::size_t kBitstream = 20;
                if (bytes.size() < kBitstream + 5) return std::nullopt;
                if (static_cast<unsigned char>(bytes[kBitstream]) != 0x2F) {
                    return std::nullopt;
                }
                const std::uint32_t bits =
                    static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[kBitstream + 1]))
                    | (static_cast<std::uint32_t>(
                           static_cast<unsigned char>(bytes[kBitstream + 2]))
                        << 8)
                    | (static_cast<std::uint32_t>(
                           static_cast<unsigned char>(bytes[kBitstream + 3]))
                        << 16)
                    | (static_cast<std::uint32_t>(
                           static_cast<unsigned char>(bytes[kBitstream + 4]))
                        << 24);
                const std::uint32_t width = (bits & 0x3FFFu) + 1;
                const std::uint32_t height = ((bits >> 14) & 0x3FFFu) + 1;
                return std::pair{width, height};
            }
            return std::nullopt;
        }
        case GrokImageFormat::Gif:
        case GrokImageFormat::Bmp:
        case GrokImageFormat::Tiff:
        case GrokImageFormat::Ico:
        case GrokImageFormat::Unknown:
            break;
    }
    return std::nullopt;
}

namespace {

[[nodiscard]] std::string format_name(GrokImageFormat format) {
    switch (format) {
        case GrokImageFormat::Png:  return "PNG";
        case GrokImageFormat::Jpeg: return "JPEG";
        case GrokImageFormat::WebP: return "WebP";
        case GrokImageFormat::Gif:  return "GIF";
        case GrokImageFormat::Bmp:  return "BMP";
        case GrokImageFormat::Tiff: return "TIFF";
        case GrokImageFormat::Ico:  return "ICO";
        case GrokImageFormat::Unknown: break;
    }
    return "unrecognised";
}

[[nodiscard]] std::string placeholder(std::string_view path, std::string_view reason) {
    std::string text = "[Attached image unavailable: ";
    text.append(path);
    if (!reason.empty()) {
        text.append(" — ");
        text.append(reason);
    }
    text.append("]");
    return text;
}

[[nodiscard]] std::optional<std::string> decode_data_url(std::string_view url) {
    constexpr std::string_view kPrefix = "data:";
    if (!url.starts_with(kPrefix)) return std::nullopt;
    const auto comma = url.find(',');
    if (comma == std::string_view::npos) return std::nullopt;
    const auto header = url.substr(kPrefix.size(), comma - kPrefix.size());
    if (!header.ends_with(";base64")) return std::nullopt;
    return core::utils::Base64::decode(url.substr(comma + 1));
}

enum class LoadKind { Ok, Unreadable, TooLarge };

struct LoadedBytes {
    LoadKind kind = LoadKind::Unreadable;
    std::string bytes;
};

[[nodiscard]] LoadedBytes load_image_file(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) return {};
    if (size > kMaxImageBytes) return {.kind = LoadKind::TooLarge};
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    LoadedBytes loaded{.kind = LoadKind::Ok};
    loaded.bytes.resize(static_cast<std::size_t>(size));
    if (size > 0) {
        input.read(loaded.bytes.data(), static_cast<std::streamsize>(size));
        if (!input) return {};
    }
    return loaded;
}

struct InspectResult {
    bool keep = true;
    std::string mime;
    std::string placeholder_text;
};

[[nodiscard]] InspectResult inspect_bytes(std::string_view bytes, std::string_view path) {
    if (bytes.empty()) {
        return {.keep = false, .placeholder_text = placeholder(path, "image is empty")};
    }
    if (bytes.size() > kMaxImageBytes) {
        return {
            .keep = false,
            .placeholder_text = placeholder(
                path,
                "exceeds Grok's 20 MiB image-understanding limit"),
        };
    }

    const auto format = sniff_grok_image_format(bytes);
    if (format == GrokImageFormat::Unknown) {
        return {
            .keep = false,
            .placeholder_text = placeholder(
                path,
                "unrecognised image format (Grok accepts JPEG, PNG, and WebP)"),
        };
    }
    if (grok_image_needs_png_transcode(format)) {
        return {
            .keep = false,
            .placeholder_text = placeholder(
                path,
                "Grok's vision endpoint rejects "
                    + format_name(format)
                    + "; convert it to PNG or JPEG"),
        };
    }
    if (!grok_image_structurally_complete(bytes)) {
        return {
            .keep = false,
            .placeholder_text = placeholder(
                path,
                "image bytes are truncated or corrupt"),
        };
    }

    const auto size = grok_image_dimensions(bytes);
    if (!size) {
        return {
            .keep = false,
            .placeholder_text = placeholder(path, "could not read image dimensions"),
        };
    }
    const auto [width, height] = *size;
    if (width < kMinSidePx || height < kMinSidePx) {
        return {
            .keep = false,
            .placeholder_text = placeholder(
                path,
                std::to_string(width) + "×" + std::to_string(height)
                    + " is below Grok's 8×8 px minimum"),
        };
    }
    const auto pixels = static_cast<std::uint64_t>(width) * height;
    if (pixels < kMinTotalPx) {
        return {
            .keep = false,
            .placeholder_text = placeholder(
                path,
                std::to_string(width) + "×" + std::to_string(height) + " = "
                    + std::to_string(pixels)
                    + " px is below Grok's 512-pixel minimum"),
        };
    }

    return {
        .keep = true,
        .mime = std::string(grok_image_mime(format)),
    };
}

} // namespace

void prepare_grok_images(ChatRequest& request) {
    for (auto& msg : request.messages) {
        for (auto& part : msg.content_parts) {
            if (part.type != ContentPartType::Image) continue;

            InspectResult result{.keep = true};
            const std::string_view label = media_reference(part);

            if (!part.url.empty()) {
                if (part.url.starts_with("http://") || part.url.starts_with("https://")) {
                    continue; // xAI fetches public URLs server-side
                }
                if (auto decoded = decode_data_url(part.url)) {
                    result = inspect_bytes(*decoded, label);
                } else if (is_data_url_for_media(part.type, part.url)) {
                    result = {
                        .keep = false,
                        .placeholder_text = placeholder(label, "malformed image data URI"),
                    };
                } else {
                    continue;
                }
            } else if (!part.path.empty()) {
                const auto loaded = load_image_file(part.path);
                if (loaded.kind == LoadKind::Unreadable) {
                    continue; // shared encoder emits the generic unavailable text
                }
                if (loaded.kind == LoadKind::TooLarge) {
                    result = {
                        .keep = false,
                        .placeholder_text = placeholder(
                            label,
                            "exceeds Grok's 20 MiB image-understanding limit"),
                    };
                } else {
                    result = inspect_bytes(loaded.bytes, label);
                }
            } else {
                continue;
            }

            if (result.keep) {
                if (!result.mime.empty()) {
                    part.mime_type = result.mime;
                }
                continue;
            }
            part = ContentPart::make_text(std::move(result.placeholder_text));
        }
    }
}

} // namespace core::llm::protocols
