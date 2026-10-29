#include "MiniMaxImage.hpp"

#include "core/llm/MiniMaxModelTraits.hpp"
#include "core/utils/Base64.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace core::llm::protocols {
namespace {

/// Base64 characters covering the 12 bytes the longest signature needs.
constexpr std::size_t kSignatureBase64Chars = 16;

/// Why an image is not sent inline.
enum class Withheld { No, Unsupported, MessageFull, RequestFull, BodyFull };

/// A local or inline image of the request.
struct Attachment {
    std::size_t message = 0;
    std::size_t part = 0;
    std::string path;
    std::string mime;
    std::size_t base64_bytes = 0;
    Withheld withheld = Withheld::Unsupported;
};

[[nodiscard]] std::string reason(Withheld withheld) {
    switch (withheld) {
    case Withheld::Unsupported:
        return std::format("MiniMax accepts only JPEG, PNG, GIF or WebP images up to {} MiB",
                           kMiniMaxImageMaxBytes / (1024 * 1024));
    case Withheld::MessageFull:
        return std::format("MiniMax accepts at most {} images per message", kMiniMaxImagesPerMessage);
    case Withheld::RequestFull:
        return std::format("this request keeps only the {} most recent images", kMiniMaxImagesPerRequest);
    case Withheld::BodyFull: return "the request body limit leaves no room for it";
    case Withheld::No: break;
    }
    return {};
}

/// Byte count of padded base64, without decoding it.
[[nodiscard]] std::size_t decoded_size(std::string_view base64) noexcept {
    const std::size_t padding = base64.ends_with("==") ? 2 : base64.ends_with('=') ? 1 : 0;
    return base64.size() / 4 * 3 - padding;
}

/// Labels the image with the MIME its bytes declare. Remote URLs are fetched
/// by MiniMax itself, so they yield no attachment.
[[nodiscard]] std::optional<Attachment> inspect(ContentPart& part) {
    const auto encoded = encode_image_part(part);
    if (encoded && encoded->is_url_reference()) return std::nullopt;
    Attachment attachment{.path = part.path, .mime = part.mime_type};
    if (!encoded) return attachment;
    attachment.mime = encoded->mime_type;
    const auto signature = core::utils::Base64::decode(
        std::string_view(encoded->base64_data).substr(0, kSignatureBase64Chars));
    const auto mime = signature ? minimax_image_mime(*signature) : std::string_view{};
    if (mime.empty() || decoded_size(encoded->base64_data) > kMiniMaxImageMaxBytes) return attachment;
    if (encoded->mime_type != mime) {
        part.url = "data:" + std::string(mime) + ";base64," + encoded->base64_data;
        part.mime_type = mime;
    }
    attachment.mime = mime;
    attachment.base64_bytes = encoded->base64_data.size();
    attachment.withheld = Withheld::No;
    return attachment;
}

[[nodiscard]] std::string withheld_text(const Attachment& attachment) {
    std::string text = "[Image omitted: " + reason(attachment.withheld);
    if (!attachment.path.empty()) text += ". Original file: " + attachment.path;
    return text + ']';
}

/// MiniMax Code's attachment reminder: the model learns which file each
/// image is, and can hand that path to a tool. Empty without local files.
[[nodiscard]] std::string attachment_reminder(std::span<const Attachment> attachments) {
    using core::utils::str::escape_xml;
    std::string entries;
    for (const auto& attachment : attachments) {
        if (attachment.path.empty()) continue;
        const bool sent = attachment.withheld == Withheld::No;
        entries += std::format(R"(<attachment name="{}" mime="{}" path="{}" kind="image" inline="{}">)",
                               escape_xml(std::filesystem::path(attachment.path).filename().string()),
                               escape_xml(attachment.mime.empty() ? "application/octet-stream" : attachment.mime),
                               escape_xml(attachment.path), sent);
        entries += '\n';
        entries += sent ? "This media attachment is also provided inline to the model."
                        : "This image was not sent inline: " + reason(attachment.withheld)
                              + ". Use the exact local path with an appropriate file/tool workflow when its contents are needed.";
        entries += "\n</attachment>\n";
    }
    if (entries.empty()) return {};
    return "<system-reminder>\nThe user provided local attachments for this turn. "
           "Use the exact file paths below and do not guess filenames.\n" + entries + "</system-reminder>";
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
    // Oldest first: within a message the first images are the ones kept.
    std::vector<Attachment> attachments;
    for (std::size_t m = 0; m < request.messages.size(); ++m) {
        auto& parts = request.messages[m].content_parts;
        std::size_t inlined = 0;
        for (std::size_t p = 0; p < parts.size(); ++p) {
            if (parts[p].type != ContentPartType::Image) continue;
            auto attachment = inspect(parts[p]);
            if (!attachment) continue;
            attachment->message = m;
            attachment->part = p;
            if (attachment->withheld == Withheld::No && ++inlined > kMiniMaxImagesPerMessage) {
                attachment->withheld = Withheld::MessageFull;
            }
            attachments.push_back(std::move(*attachment));
        }
    }

    // Newest first: history images give way to the ones just attached.
    std::size_t kept = 0;
    std::size_t body = 0;
    for (auto& attachment : attachments | std::views::reverse) {
        if (attachment.withheld != Withheld::No) continue;
        if (kept == kMiniMaxImagesPerRequest) {
            attachment.withheld = Withheld::RequestFull;
        } else if (body + attachment.base64_bytes > kMiniMaxRequestBodyLimit) {
            attachment.withheld = Withheld::BodyFull;
        } else {
            ++kept;
            body += attachment.base64_bytes;
        }
    }

    for (const auto& attachment : attachments) {
        if (attachment.withheld == Withheld::No) continue;
        request.messages[attachment.message].content_parts[attachment.part] =
            ContentPart::make_text(withheld_text(attachment));
    }

    // Reminders go in last: inserting them shifts the part indices above.
    for (std::size_t m = 0; m < request.messages.size(); ++m) {
        auto& message = request.messages[m];
        if (message.role != "user") continue;
        const auto reminder = attachment_reminder(std::ranges::equal_range(attachments, m, {}, &Attachment::message));
        if (!reminder.empty()) message.content_parts.insert(message.content_parts.begin(), ContentPart::make_text(reminder));
    }
}

} // namespace core::llm::protocols
