#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/MiniMaxImage.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"
#include "core/utils/Base64.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

using namespace core::llm;
using namespace core::llm::protocols;
using Catch::Matchers::ContainsSubstring;

namespace {

const std::string kPng("\x89PNG\r\n\x1A\n\0\0\0\rIHDR", 16);
const std::string kJpeg("\xFF\xD8\xFF\xE0\0\x10JFIF\0", 11);
const std::string kGif("GIF89a\x01\0\x01\0", 10);
const std::string kWebp("RIFF\x24\0\0\0WEBPVP8 ", 16);
const std::string kBmp("BM\x36\0\0\0\0\0", 8);

std::string data_url(std::string_view mime, std::string_view bytes) {
    return "data:" + std::string(mime) + ";base64," + core::utils::Base64::encode(bytes);
}

ChatRequest request_with(std::vector<ContentPart> parts) {
    ChatRequest request;
    request.model = "MiniMax-M3";
    request.messages.push_back(Message{.role = "user", .content_parts = std::move(parts)});
    return request;
}

std::filesystem::path temp_file(std::string_view name, std::string_view bytes) {
    const auto path = std::filesystem::temp_directory_path()
        / (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::string(name));
    std::ofstream(path, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

} // namespace

TEST_CASE("MiniMax recognises the image formats its endpoint accepts", "[minimax][vision]") {
    CHECK(minimax_image_mime(kJpeg) == "image/jpeg");
    CHECK(minimax_image_mime(kPng) == "image/png");
    CHECK(minimax_image_mime(kGif) == "image/gif");
    CHECK(minimax_image_mime("GIF87a") == "image/gif");
    CHECK(minimax_image_mime(kWebp) == "image/webp");
    CHECK(minimax_image_mime(kBmp).empty());
    CHECK(minimax_image_mime(std::string_view("RIFF\x24\0\0\0WAVE", 12)).empty());
    CHECK(minimax_image_mime("png-bytes").empty());
    CHECK(minimax_image_mime("").empty());
}

TEST_CASE("MiniMax images keep a correct label and are relabelled from their bytes", "[minimax][vision]") {
    const auto png = data_url("image/png", kPng);
    auto request = request_with({ContentPart::make_image_url(png),
                                 ContentPart::make_image_url(data_url("image/png", kJpeg)),
                                 ContentPart::make_image_url(data_url("image/jpg", kJpeg)),
                                 ContentPart::make_image_url("https://example.test/shot.bmp")});
    prepare_minimax_images(request);
    const auto& parts = request.messages.front().content_parts;
    REQUIRE(parts.size() == 4);
    CHECK(parts[0].url == png);
    CHECK(parts[1].url == data_url("image/jpeg", kJpeg));
    CHECK(parts[1].mime_type == "image/jpeg");
    CHECK(parts[2].url == data_url("image/jpeg", kJpeg));
    CHECK(parts[3].url == "https://example.test/shot.bmp");
}

TEST_CASE("MiniMax labels a local image by its content, not its extension", "[minimax][vision]") {
    const auto path = temp_file("photo.png", kJpeg);
    auto request = request_with({ContentPart::make_image(path.string())});
    prepare_minimax_images(request);
    const auto& parts = request.messages.front().content_parts;
    REQUIRE(parts.size() == 2);
    CHECK(parts[1].type == ContentPartType::Image);
    CHECK(parts[1].mime_type == "image/jpeg");
    CHECK(parts[1].url == data_url("image/jpeg", kJpeg));
    std::filesystem::remove(path);
}

TEST_CASE("MiniMax tells the model which file each attached image is", "[minimax][vision]") {
    const auto path = temp_file("shot \"1\" & <2>.png", kPng);
    auto request = request_with({ContentPart::make_text("What color is "), ContentPart::make_image(path.string()),
                                 ContentPart::make_image_url(data_url("image/png", kPng))});
    request.messages.insert(request.messages.begin(), Message{.role = "assistant", .content_parts = {
        ContentPart::make_image(path.string())}});
    prepare_minimax_images(request);

    const auto& parts = request.messages.back().content_parts;
    REQUIRE(parts.size() == 4);
    REQUIRE(parts[0].type == ContentPartType::Text);
    // temp_file prefixes the name with a timestamp; only the chosen suffix needs escaping.
    const std::string_view suffix = "shot \"1\" & <2>.png";
    const auto unescaped = [&](const std::string& value) { return value.substr(0, value.size() - suffix.size()); };
    const std::string escaped_suffix = "shot &quot;1&quot; &amp; &lt;2&gt;.png";
    CHECK(parts[0].text == "<system-reminder>\nThe user provided local attachments for this turn. "
                           "Use the exact file paths below and do not guess filenames.\n"
                           "<attachment name=\"" + unescaped(path.filename().string()) + escaped_suffix
                           + "\" mime=\"image/png\" path=\"" + unescaped(path.string()) + escaped_suffix
                           + "\" kind=\"image\" inline=\"true\">\n"
                           "This media attachment is also provided inline to the model.\n"
                           "</attachment>\n</system-reminder>");
    CHECK(parts[1].text == "What color is ");
    CHECK(parts[2].type == ContentPartType::Image);
    CHECK(parts[3].type == ContentPartType::Image);
    // Only user turns carry the reminder; a pasted image has no file to name.
    CHECK(request.messages.front().content_parts.size() == 1);
    std::filesystem::remove(path);
}

TEST_CASE("MiniMax withholds images its endpoint would reject instead of failing the turn", "[minimax][vision]") {
    const auto bmp = temp_file("diagram.bmp", kBmp);
    auto at_limit = kPng;
    at_limit.resize(kMiniMaxImageMaxBytes, '\0');
    auto over_limit = at_limit + '\0';
    auto request = request_with({ContentPart::make_image(bmp.string()),
                                 ContentPart::make_image_url(data_url("image/png", "png-bytes")),
                                 ContentPart::make_image_url(data_url("image/png", at_limit)),
                                 ContentPart::make_image_url(data_url("image/png", over_limit))});
    prepare_minimax_images(request);
    const auto& parts = request.messages.front().content_parts;
    REQUIRE(parts.size() == 5);
    const std::string unsupported = "[Image omitted: MiniMax accepts only JPEG, PNG, GIF or WebP images up to 10 MiB";
    CHECK_THAT(parts[0].text, ContainsSubstring(
        "<attachment name=\"" + bmp.filename().string() + "\" mime=\"image/bmp\" path=\"" + bmp.string()
        + "\" kind=\"image\" inline=\"false\">\nThis image was not sent inline: MiniMax accepts only"));
    CHECK(parts[1].text == unsupported + ". Original file: " + bmp.string() + "]");
    CHECK(parts[2].text == unsupported + "]");
    CHECK(parts[3].type == ContentPartType::Image);
    CHECK(parts[4].text == unsupported + "]");
    std::filesystem::remove(bmp);
}

TEST_CASE("MiniMax inlines at most four images per message", "[minimax][vision]") {
    const auto png = ContentPart::make_image_url(data_url("image/png", kPng));
    auto request = request_with({png, png, png, png, png, ContentPart::make_image_url("https://example.test/a.png")});
    prepare_minimax_images(request);
    const auto& parts = request.messages.front().content_parts;
    REQUIRE(parts.size() == 6);
    for (std::size_t i = 0; i < kMiniMaxImagesPerMessage; ++i) CHECK(parts[i].type == ContentPartType::Image);
    CHECK(parts[4].text == "[Image omitted: MiniMax accepts at most 4 images per message]");
    CHECK(parts[5].url == "https://example.test/a.png");
}

TEST_CASE("MiniMax keeps only the newest images of a long conversation", "[minimax][vision]") {
    const auto png = ContentPart::make_image_url(data_url("image/png", kPng));
    ChatRequest request;
    request.model = "MiniMax-M3";
    for (int turn = 0; turn < 6; ++turn) {
        request.messages.push_back(Message{.role = "user", .content_parts = {png, png, png, png}});
        request.messages.push_back(Message{.role = "assistant", .content = "Seen."});
    }
    prepare_minimax_images(request);
    std::size_t images = 0;
    for (const auto& message : request.messages) {
        for (const auto& part : message.content_parts) images += part.type == ContentPartType::Image;
    }
    CHECK(images == kMiniMaxImagesPerRequest);
    for (const auto& part : request.messages.front().content_parts) {
        CHECK(part.text == "[Image omitted: this request keeps only the 20 most recent images]");
    }
    CHECK(request.messages[2].content_parts.front().type == ContentPartType::Image);
}

TEST_CASE("MiniMax sends the detected media type on the wire", "[minimax][vision]") {
    MiniMaxProtocol protocol(true);
    auto request = request_with({ContentPart::make_text("What is this?"),
                                 ContentPart::make_image_url(data_url("image/png", kJpeg))});
    protocol.prepare_request(request);
    const auto payload = protocol.serialize(request);
    CHECK_THAT(payload, ContainsSubstring(R"("type":"base64","media_type":"image/jpeg","data":")"
                                          + core::utils::Base64::encode(kJpeg) + "\""));
}
