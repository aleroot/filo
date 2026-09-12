#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/GrokImage.hpp"
#include "core/llm/protocols/GrokProtocol.hpp"
#include "core/utils/Base64.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

std::string from_hex(std::string_view hex) {
    std::string out;
    out.reserve(hex.size() / 2);
    auto nibble = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        return 0;
    };
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<char>((nibble(hex[i]) << 4) | nibble(hex[i + 1])));
    }
    return out;
}

// 32×32 RGB PNG (1024 px — above Grok's 512-pixel floor).
const std::string kPng32 = from_hex(
    "89504e470d0a1a0a0000000d4948445200000020000000200802000000fc18eda3"
    "000000274944415478daedcdb109000008c0b0feffb45ee12004b2a7a953090402"
    "814020100804822fc10232c3fc2e70fd233d0000000049454e44ae426082");

// 16×16 RGB PNG (256 px — below the 512-pixel floor).
const std::string kPng16 = from_hex(
    "89504e470d0a1a0a0000000d494844520000001000000010080200000090916836"
    "000000164944415478da63f8cfc040126218d530aa61f86a000090f9ff01f2eee8"
    "570000000049454e44ae426082");

const std::string kGif = from_hex(
    "47494638396101000080000000000000ffffff21f90401000000002c00000000"
    "01000000020144003b");

std::filesystem::path write_temp(std::string_view name, std::string_view bytes) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(out);
    return path;
}

ChatRequest image_request(const std::filesystem::path& path,
                          std::string mime = "image/png") {
    ChatRequest req;
    req.model = "grok-4.6";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_image_attachment(path.string()),
        .content_parts = {
            ContentPart::make_text("What is in this image?"),
            ContentPart::make_image(path.string(), std::move(mime)),
        },
    });
    return req;
}

} // namespace

TEST_CASE("Grok image sniff recognises engine-native and transcode formats",
          "[grok][image]") {
    CHECK(sniff_grok_image_format(kPng32) == GrokImageFormat::Png);
    CHECK(sniff_grok_image_format(kGif) == GrokImageFormat::Gif);
    CHECK(sniff_grok_image_format("not-an-image") == GrokImageFormat::Unknown);
    CHECK(sniff_grok_image_format(std::string{"\xff\xd8\xff\xe0", 4})
          == GrokImageFormat::Jpeg);
    CHECK(grok_image_needs_png_transcode(GrokImageFormat::Gif));
    CHECK(grok_image_needs_png_transcode(GrokImageFormat::Bmp));
    CHECK(grok_image_needs_png_transcode(GrokImageFormat::Ico));
    CHECK_FALSE(grok_image_needs_png_transcode(GrokImageFormat::Png));
    CHECK_FALSE(grok_image_needs_png_transcode(GrokImageFormat::Jpeg));
    CHECK_FALSE(grok_image_needs_png_transcode(GrokImageFormat::WebP));
}

TEST_CASE("Grok image PNG structure and dimensions match grok-build's floor",
          "[grok][image]") {
    REQUIRE(grok_image_structurally_complete(kPng32));
    REQUIRE(grok_image_structurally_complete(kPng16));
    const auto big = grok_image_dimensions(kPng32);
    REQUIRE(big.has_value());
    CHECK(big->first == 32);
    CHECK(big->second == 32);
    const auto small = grok_image_dimensions(kPng16);
    REQUIRE(small.has_value());
    CHECK(small->first == 16);
    CHECK(small->second == 16);

    auto truncated = kPng32.substr(0, kPng32.size() - 8);
    CHECK_FALSE(grok_image_structurally_complete(truncated));
}

TEST_CASE("prepare_grok_images keeps a valid 32x32 PNG and fixes MIME from bytes",
          "[grok][image]") {
    const auto path = write_temp("filo-grok-32.png", kPng32);
    ChatRequest req = image_request(path, "image/gif");
    prepare_grok_images(req);
    REQUIRE(req.messages[0].content_parts.size() == 2);
    CHECK(req.messages[0].content_parts[1].type == ContentPartType::Image);
    CHECK(req.messages[0].content_parts[1].mime_type == "image/png");
}

TEST_CASE("prepare_grok_images strips images below Grok's 512-pixel floor",
          "[grok][image]") {
    const auto path = write_temp("filo-grok-16.png", kPng16);
    ChatRequest req = image_request(path);
    prepare_grok_images(req);
    REQUIRE(req.messages[0].content_parts.size() == 2);
    REQUIRE(req.messages[0].content_parts[1].type == ContentPartType::Text);
    CHECK_THAT(req.messages[0].content_parts[1].text,
               Catch::Matchers::ContainsSubstring("512-pixel minimum"));
}

TEST_CASE("prepare_grok_images strips GIF instead of sending a format Grok rejects",
          "[grok][image]") {
    const auto path = write_temp("filo-grok.gif", kGif);
    ChatRequest req = image_request(path, "image/gif");
    prepare_grok_images(req);
    REQUIRE(req.messages[0].content_parts[1].type == ContentPartType::Text);
    CHECK_THAT(req.messages[0].content_parts[1].text,
               Catch::Matchers::ContainsSubstring("rejects GIF"));
}

TEST_CASE("prepare_grok_images leaves public image URLs for the server to fetch",
          "[grok][image]") {
    ChatRequest req;
    req.model = "grok-4.6";
    req.messages.push_back(Message{
        .role = "user",
        .content_parts = {
            ContentPart::make_image_url("https://example.com/shot.png"),
        },
    });
    prepare_grok_images(req);
    REQUIRE(req.messages[0].content_parts[0].type == ContentPartType::Image);
    CHECK(req.messages[0].content_parts[0].url == "https://example.com/shot.png");
}

TEST_CASE("GrokProtocol serializes a valid PNG as an OpenAI image_url part",
          "[grok][image][serializer]") {
    const auto path = write_temp("filo-grok-wire.png", kPng32);
    const auto payload = GrokProtocol{}.serialize(image_request(path));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"image_url")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("data:image/png;base64,"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("detail":"auto")"));
}

TEST_CASE("GrokProtocol does not send a GIF that would 400 the vision endpoint",
          "[grok][image][serializer]") {
    const auto path = write_temp("filo-grok-wire.gif", kGif);
    const auto payload = GrokProtocol{}.serialize(image_request(path, "image/gif"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("type":"image_url")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("rejects GIF"));
}

TEST_CASE("GrokResponsesProtocol emits Responses input_image after Grok image policy",
          "[grok][image][responses][serializer]") {
    const auto path = write_temp("filo-grok-responses.png", kPng32);
    GrokResponsesProtocol protocol({}, /*enable_hosted_tools=*/false);
    const auto payload = protocol.serialize(image_request(path));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"input_image")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("data:image/png;base64,"));
}

TEST_CASE("prepare_grok_images validates data-URI attachments",
          "[grok][image]") {
    ChatRequest req;
    req.model = "grok-4.6";
    req.messages.push_back(Message{
        .role = "user",
        .content_parts = {
            ContentPart::make_image_url(
                "data:image/png;base64," + core::utils::Base64::encode(kPng16)),
        },
    });
    prepare_grok_images(req);
    REQUIRE(req.messages[0].content_parts[0].type == ContentPartType::Text);
    CHECK_THAT(req.messages[0].content_parts[0].text,
               Catch::Matchers::ContainsSubstring("512-pixel minimum"));
}
