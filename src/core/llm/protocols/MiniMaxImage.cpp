#include "MiniMaxImage.hpp"

#include "core/utils/Base64.hpp"

#include <string>

namespace core::llm::protocols {
namespace {

/// Base64 characters covering the 12 bytes the longest signature needs.
constexpr std::size_t kSignatureBase64Chars = 16;

/// Byte count of padded base64, without decoding it.
[[nodiscard]] std::size_t decoded_size(std::string_view base64) noexcept {
    const std::size_t padding = base64.ends_with("==") ? 2 : base64.ends_with('=') ? 1 : 0;
    return base64.size() / 4 * 3 - padding;
}

void prepare_image(ContentPart& part) {
    const auto encoded = encode_image_part(part);
    // Remote URLs are fetched by MiniMax; unreadable sources are reported by
    // the shared Messages encoder.
    if (!encoded || encoded->is_url_reference()) return;
    const auto signature = core::utils::Base64::decode(
        std::string_view(encoded->base64_data).substr(0, kSignatureBase64Chars));
    const auto mime = signature ? minimax_image_mime(*signature) : std::string_view{};
    if (mime.empty() || decoded_size(encoded->base64_data) > kMiniMaxImageMaxBytes) {
        part = ContentPart::make_text(unavailable_image_attachment_text(part.path.empty() ? "<image>" : part.path));
        return;
    }
    if (encoded->mime_type == mime) return;
    part.url = "data:" + std::string(mime) + ";base64," + encoded->base64_data;
    part.mime_type = mime;
}

} // namespace

// The image signatures of the WHATWG MIME Sniffing Standard, section 6.1.
std::string_view minimax_image_mime(std::string_view bytes) noexcept {
    if (bytes.starts_with("\xFF\xD8\xFF")) return "image/jpeg";
    if (bytes.starts_with("\x89PNG\r\n\x1A\n")) return "image/png";
    if (bytes.starts_with("GIF87a") || bytes.starts_with("GIF89a")) return "image/gif";
    if (bytes.size() >= 12 && bytes.starts_with("RIFF") && bytes.substr(8, 4) == "WEBP") return "image/webp";
    return {};
}

void prepare_minimax_images(ChatRequest& request) {
    for (auto& message : request.messages) {
        for (auto& part : message.content_parts) {
            if (part.type == ContentPartType::Image) prepare_image(part);
        }
    }
}

} // namespace core::llm::protocols
