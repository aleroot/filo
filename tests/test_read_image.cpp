#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/llm/Models.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/MistralProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/tools/ReadTool.hpp"
#include "core/tools/read/ImageFile.hpp"
#include "core/utils/ImageHeader.hpp"
#include "TestSessionContext.hpp"
#include <simdjson.h>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>

using namespace core::tools;
using Catch::Matchers::ContainsSubstring;

namespace {

// Smallest byte strings that carry each format's real header layout.
std::string png(std::uint32_t width, std::uint32_t height) {
    std::string bytes("\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR", 16);
    for (const auto value : {width, height})
        for (int shift = 24; shift >= 0; shift -= 8) bytes.push_back(static_cast<char>(value >> shift & 0xFF));
    bytes += std::string("\x08\x06\0\0\0", 5);
    return bytes;
}
std::string jpeg(std::uint16_t width, std::uint16_t height) {
    // SOI, an APP1 segment the scan must skip, then SOF0.
    std::string bytes("\xFF\xD8\xFF\xE1\x00\x08" "Exif\0\0", 12);
    bytes += std::string("\xFF\xC0\x00\x11\x08", 5);
    bytes.push_back(static_cast<char>(height >> 8)); bytes.push_back(static_cast<char>(height & 0xFF));
    bytes.push_back(static_cast<char>(width >> 8)); bytes.push_back(static_cast<char>(width & 0xFF));
    bytes += std::string("\x03\x01\x22\x00", 4);
    return bytes;
}
std::string webp(std::string_view chunk, std::string_view payload) {
    std::string bytes = "RIFF";
    bytes += std::string(4, '\0');
    bytes += "WEBP";
    bytes += chunk;
    bytes += std::string(4, '\0');
    bytes += payload;
    bytes.resize(std::max<std::size_t>(bytes.size(), 32), '\0');
    return bytes;
}

struct Fixture {
    std::filesystem::path root;
    Fixture() {
        std::string pattern = (std::filesystem::temp_directory_path() / "filo-read-image-XXXXXX").string();
        const char* directory = ::mkdtemp(pattern.data());
        if (!directory) throw std::runtime_error("Cannot create read image test directory.");
        root = std::filesystem::canonical(directory);
    }
    ~Fixture() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    auto context() const {
        return test_support::make_session_context({.primary = root, .enforce = true},
            core::context::SessionTransport::cli, "read-image-test");
    }
    void write(std::string_view name, std::string_view bytes) const {
        std::ofstream file(root / name, std::ios::binary);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
};

simdjson::dom::element parse(simdjson::dom::parser& parser, const std::string& json) {
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    return doc;
}

core::llm::Message tool_message(std::string id, std::string content) {
    return {.role = "tool", .content = std::move(content), .name = "read", .tool_call_id = std::move(id)};
}

} // namespace

TEST_CASE("Image sniffing reads dimensions from headers without decoding", "[read][image]") {
    using core::utils::image::ImageFormat;
    using core::utils::image::sniff;
    const auto p = sniff(png(1280, 720));
    REQUIRE(p);
    CHECK(p->format == ImageFormat::png);
    CHECK((p->width == 1280 && p->height == 720));

    const auto j = sniff(jpeg(640, 480));
    REQUIRE(j);
    CHECK(j->format == ImageFormat::jpeg);
    CHECK((j->width == 640 && j->height == 480));

    const auto g = sniff(std::string("GIF89a\x40\x01\xF0\x00", 10));
    REQUIRE(g);
    CHECK((g->format == ImageFormat::gif && g->width == 320 && g->height == 240));

    // Lossy: 14-bit little-endian sizes after the 9d 01 2a start code.
    const auto lossy = sniff(webp("VP8 ", std::string("\0\0\0\x9D\x01\x2A\x20\x03\x58\x02", 10)));
    REQUIRE(lossy);
    CHECK((lossy->format == ImageFormat::webp && lossy->width == 800 && lossy->height == 600));
    // Lossless: width-1 and height-1 packed into 14-bit fields.
    const std::uint32_t bits = (100 - 1) | (50 - 1) << 14;
    std::string lossless_payload("\x2F", 1);
    for (int shift = 0; shift < 32; shift += 8) lossless_payload.push_back(static_cast<char>(bits >> shift & 0xFF));
    const auto lossless = sniff(webp("VP8L", lossless_payload));
    REQUIRE(lossless);
    CHECK((lossless->width == 100 && lossless->height == 50));
    // Extended: 24-bit canvas width-1 and height-1.
    const auto extended = sniff(webp("VP8X", std::string("\0\0\0\0\x9F\x0F\0\x3F\x02\0", 10)));
    REQUIRE(extended);
    CHECK((extended->width == 4000 && extended->height == 576));

    CHECK(sniff(std::string("II*\0\x08\0\0\0", 8))->format == ImageFormat::tiff);
    CHECK(sniff(std::string("\0\0\0\x18" "ftypheic\0\0\0\0", 16))->format == ImageFormat::heif);
    CHECK(sniff(std::string("\0\0\0\x18" "ftypavif\0\0\0\0", 16))->format == ImageFormat::avif);
    CHECK(core::utils::image::vision_mime_type(ImageFormat::tiff).empty());
    CHECK(core::utils::image::vision_mime_type(ImageFormat::webp) == "image/webp");

    CHECK_FALSE(sniff("BM is how this note starts, and it is still text"));
    CHECK_FALSE(sniff("plain text"));
    CHECK_FALSE(sniff(std::string("\x89PNG", 4)));
    // A complete signature names the format even before the header arrives.
    const auto bare = sniff(std::string("\x89PNG\r\n\x1a\n", 8));
    REQUIRE(bare);
    CHECK((bare->format == ImageFormat::png && bare->width == 0 && bare->height == 0));
}

TEST_CASE("read returns an image as an attachment descriptor, not as text", "[read][image]") {
    Fixture fixture;
    fixture.write("shot.png", png(1280, 720));
    fixture.write("noextension", png(64, 32));
    fixture.write("photo.JPG", jpeg(640, 480));
    ReadTool tool;
    simdjson::dom::parser parser;

    const auto shot = tool.execute(R"({"path":"shot.png"})", fixture.context());
    auto doc = parse(parser, shot);
    CHECK(std::string_view(doc["kind"].get_string().value()) == "image");
    CHECK(std::string_view(doc["mime_type"].get_string().value()) == "image/png");
    CHECK(std::string_view(doc["path"].get_string().value()) == (fixture.root / "shot.png").string());
    CHECK(doc["width"].get_uint64().value() == 1280);
    CHECK(doc["height"].get_uint64().value() == 720);
    CHECK_THAT(std::string(doc["content"].get_string().value()), ContainsSubstring("PNG 1280x720"));

    // Any view of a single image returns the image.
    doc = parse(parser, tool.execute(R"({"path":"shot.png","view":"auto"})", fixture.context()));
    CHECK(std::string_view(doc["kind"].get_string().value()) == "image");
    doc = parse(parser, tool.execute(R"({"path":"photo.JPG"})", fixture.context()));
    CHECK(std::string_view(doc["mime_type"].get_string().value()) == "image/jpeg");

    // Without an image extension the content still identifies it.
    doc = parse(parser, tool.execute(R"({"path":"noextension"})", fixture.context()));
    CHECK(std::string_view(doc["kind"].get_string().value()) == "image");
    CHECK(doc["width"].get_uint64().value() == 64);
}

TEST_CASE("read explains how to make an unsupported or oversized image viewable", "[read][image]") {
    Fixture fixture;
    fixture.write("scan.tiff", std::string("II*\0\x08\0\0\0", 8));
    fixture.write("huge.png", png(12000, 300));
    fixture.write("fake.png", "this is a text file named like an image");
    ReadTool tool;
    simdjson::dom::parser parser;

    auto doc = parse(parser, tool.execute(R"({"path":"scan.tiff"})", fixture.context()));
    const std::string tiff_error(doc["error"].get_string().value());
    CHECK_THAT(tiff_error, ContainsSubstring("TIFF images are not accepted"));
    CHECK_THAT(tiff_error, ContainsSubstring("scan.tiff"));

    doc = parse(parser, tool.execute(R"({"path":"huge.png"})", fixture.context()));
    const std::string huge_error(doc["error"].get_string().value());
    CHECK_THAT(huge_error, ContainsSubstring("12000x300"));
    CHECK_THAT(huge_error, ContainsSubstring("Downscale"));

    // A file that only looks like an image by name is read as text.
    doc = parse(parser, tool.execute(R"({"path":"fake.png"})", fixture.context()));
    CHECK(std::string_view(doc["content"].get_string().value()) == "this is a text file named like an image");
}

TEST_CASE("read names an image inside a batch instead of calling it undecodable", "[read][image]") {
    Fixture fixture;
    fixture.write("shot.png", png(10, 10));
    fixture.write("notes.txt", "notes");
    ReadTool tool;
    simdjson::dom::parser parser;
    auto doc = parse(parser, tool.execute(R"({"path":["notes.txt","shot.png"]})", fixture.context()));
    simdjson::dom::array unread;
    REQUIRE(doc["unread"].get(unread) == simdjson::SUCCESS);
    REQUIRE(unread.size() == 1);
    CHECK_THAT(std::string(unread.at(0)["error"].get_string().value()), ContainsSubstring("read it on its own"));
}

TEST_CASE("Tool-result images move to one follow-up user message where tool messages are text-only",
          "[llm][image][tool-media]") {
    using core::llm::ContentPart;
    using core::llm::ToolResultMediaPlacement;
    core::llm::ChatRequest base;
    base.messages.push_back({.role = "user", .content = "look"});
    base.messages.push_back({.role = "assistant", .tool_calls = {
        {.id = "a", .function = {.name = "read", .arguments = "{}"}},
        {.id = "b", .function = {.name = "read", .arguments = "{}"}}}});
    base.messages.push_back(tool_message("a", R"({"kind":"image"})"));
    base.messages.back().content_parts.push_back(ContentPart::make_image("/tmp/a.png", "image/png"));
    base.messages.push_back(tool_message("b", R"({"content":"text"})"));
    base.messages.back().content_parts.push_back(ContentPart::make_text("text"));
    base.messages.push_back({.role = "assistant", .content = "done"});

    auto native = base;
    core::llm::place_tool_result_media(native, ToolResultMediaPlacement::native);
    CHECK(native.messages.size() == base.messages.size());
    CHECK(native.messages[2].content_parts.size() == 1);

    auto lifted = base;
    core::llm::place_tool_result_media(lifted, ToolResultMediaPlacement::follow_up_user);
    REQUIRE(lifted.messages.size() == base.messages.size() + 1);
    CHECK(lifted.messages[2].role == "tool");
    CHECK(lifted.messages[2].content == R"({"kind":"image"})");
    CHECK(lifted.messages[2].content_parts.empty());
    CHECK(lifted.messages[3].tool_call_id == "b");
    CHECK(lifted.messages[3].content_parts.size() == 1); // a result without media is untouched
    const auto& follow_up = lifted.messages[4];
    CHECK(follow_up.role == "user");
    CHECK(follow_up.synthetic);
    REQUIRE(follow_up.content_parts.size() == 2);
    CHECK_THAT(follow_up.content_parts[0].text, ContainsSubstring("tool call a (read): /tmp/a.png"));
    CHECK(follow_up.content_parts[1].type == core::llm::ContentPartType::Image);
    CHECK(follow_up.content_parts[1].path == "/tmp/a.png");
    CHECK(lifted.messages[5].role == "assistant");

    auto omitted = base;
    core::llm::place_tool_result_media(omitted, ToolResultMediaPlacement::omit);
    CHECK(omitted.messages.size() == base.messages.size());
    CHECK_FALSE(core::llm::request_has_image_input(omitted));
    CHECK(omitted.messages[2].content == R"({"kind":"image"})");

    CHECK(core::llm::protocols::AnthropicProtocol{}.tool_result_media() == ToolResultMediaPlacement::native);
    CHECK(core::llm::protocols::OpenAIProtocol{}.tool_result_media() == ToolResultMediaPlacement::follow_up_user);
    CHECK(core::llm::protocols::MistralProtocol{}.tool_result_media() == ToolResultMediaPlacement::omit);
}

TEST_CASE("Anthropic Messages carries a tool-result image inside the tool_result block",
          "[llm][image][tool-media][anthropic]") {
    Fixture fixture;
    fixture.write("shot.png", png(2, 2));
    core::llm::ChatRequest req;
    req.model = "claude-sonnet-4-5";
    req.messages.push_back({.role = "user", .content = "look"});
    req.messages.push_back({.role = "assistant", .tool_calls = {{.id = "call-1", .function = {.name = "read", .arguments = "{}"}}}});
    req.messages.push_back(tool_message("call-1", R"({"kind":"image"})"));
    req.messages.back().content_parts.push_back(
        core::llm::ContentPart::make_image((fixture.root / "shot.png").string(), "image/png"));

    const auto payload = core::llm::protocols::AnthropicProtocol{}.serialize(req);
    simdjson::dom::parser parser;
    auto doc = parse(parser, payload);
    simdjson::dom::array messages;
    REQUIRE(doc["messages"].get(messages) == simdjson::SUCCESS);
    REQUIRE(messages.size() == 3);
    auto block = messages.at(2)["content"].at(0);
    CHECK(std::string_view(block["type"].get_string().value()) == "tool_result");
    simdjson::dom::array content;
    REQUIRE(block["content"].get(content) == simdjson::SUCCESS);
    REQUIRE(content.size() == 2);
    CHECK(std::string_view(content.at(0)["type"].get_string().value()) == "text");
    CHECK(std::string_view(content.at(1)["type"].get_string().value()) == "image");
    CHECK(std::string_view(content.at(1)["source"]["media_type"].get_string().value()) == "image/png");
    CHECK(std::string_view(content.at(1)["source"]["data"].get_string().value())
          == core::utils::Base64::encode(png(2, 2)));
}

TEST_CASE("Old tool-result images are detached once newer ones fill the request budget",
          "[llm][image][tool-media]") {
    using core::llm::ContentPart;
    Fixture fixture;
    // Three images of 4 MiB: the newest two fit the 10 MiB budget.
    const std::string four_mib(4 * 1024 * 1024, 'x');
    for (const auto* name : {"a.png", "b.png", "c.png"}) fixture.write(name, four_mib);
    core::llm::ChatRequest req;
    req.messages.push_back({.role = "user", .content = "look"});
    for (const auto* id : {"a", "b", "c"}) {
        req.messages.push_back({.role = "assistant", .tool_calls = {{.id = id, .function = {.name = "read", .arguments = "{}"}}}});
        req.messages.push_back(tool_message(id, "described"));
        req.messages.back().content_parts.push_back(
            ContentPart::make_image((fixture.root / std::format("{}.png", id)).string(), "image/png"));
    }

    core::llm::bound_tool_result_media(req);
    CHECK(req.messages[2].content_parts.empty());
    CHECK_THAT(req.messages[2].content, ContainsSubstring("described\n[The image of this result was detached"));
    CHECK_THAT(req.messages[2].content, ContainsSubstring((fixture.root / "a.png").string()));
    CHECK(req.messages[4].content_parts.size() == 1);
    CHECK(req.messages[6].content_parts.size() == 1);

    // Applying it again changes nothing: the detached prefix is stable.
    const auto once = req.messages[2].content;
    core::llm::bound_tool_result_media(req);
    CHECK(req.messages[2].content == once);

    // The count bounds small images too.
    core::llm::ChatRequest many;
    fixture.write("tiny.png", png(2, 2));
    for (std::size_t i = 0; i <= core::llm::kMaxToolResultImages; ++i) {
        many.messages.push_back(tool_message(std::to_string(i), "described"));
        many.messages.back().content_parts.push_back(
            ContentPart::make_image((fixture.root / "tiny.png").string(), "image/png"));
    }
    core::llm::bound_tool_result_media(many);
    CHECK(core::llm::count_image_inputs(many) == core::llm::kMaxToolResultImages);
    CHECK(many.messages.front().content_parts.empty());

    // A format that omits tool media drops it without promising a view.
    auto omitted = many;
    omitted.messages.push_back(tool_message("extra", "described"));
    omitted.messages.back().content_parts.push_back(
        ContentPart::make_image((fixture.root / "tiny.png").string(), "image/png"));
    core::llm::fit_tool_result_media(omitted, core::llm::ToolResultMediaPlacement::omit);
    CHECK_FALSE(core::llm::request_has_image_input(omitted));
    CHECK(omitted.messages.back().content == "described");
}
