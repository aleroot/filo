#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextMentions.hpp"
#include "core/llm/MediaSnapshots.hpp"
#include "core/llm/Models.hpp"
#include "core/session/SessionStore.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using core::llm::ContentPart;
using core::llm::MediaSnapshots;

namespace {

struct Fixture {
    fs::path root;
    Fixture() {
        std::string pattern = (fs::temp_directory_path() / "filo-media-snapshots-XXXXXX").string();
        const char* directory = ::mkdtemp(pattern.data());
        if (!directory) throw std::runtime_error("Cannot create media snapshot test directory.");
        root = fs::canonical(directory);
    }
    ~Fixture() { std::error_code ec; fs::remove_all(root, ec); }
    void write(const fs::path& path, std::string_view bytes) const {
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary | std::ios::trunc) << bytes;
    }
};

} // namespace

TEST_CASE("A pinned video keeps the bytes it was attached with",
          "[media][video][regression]") {
    Fixture fixture;
    const auto video = fixture.root / "work" / "render.mp4";
    fixture.write(video, "first-render");
    const MediaSnapshots snapshots(fixture.root / "media" / "session");

    auto part = ContentPart::make_video(video.string(), "video/mp4");
    snapshots.pin(part);
    REQUIRE_FALSE(part.snapshot.empty());
    CHECK(part.path == video.string());  // Still called what it was attached as.
    CHECK(fs::path(part.snapshot).parent_path() == snapshots.directory());
    CHECK(part.media_file() == part.snapshot);

    SECTION("rewritten, the request still carries the attached bytes") {
        fixture.write(video, "second-render, longer");
        const auto encoded = core::llm::encode_media_part(part);
        REQUIRE(encoded);
        CHECK(encoded->base64_data == core::utils::Base64::encode("first-render"));
        CHECK(encoded->mime_type == "video/mp4");
    }
    SECTION("truncated or deleted, likewise") {
        fs::remove(video);
        const auto encoded = core::llm::encode_media_part(part);
        REQUIRE(encoded);
        CHECK(encoded->base64_data == core::utils::Base64::encode("first-render"));
    }
    SECTION("the same contents are stored once") {
        const auto copy = fixture.root / "elsewhere.mp4";
        fixture.write(copy, "first-render");
        auto again = ContentPart::make_video(copy.string(), "video/mp4");
        snapshots.pin(again);
        CHECK(again.snapshot == part.snapshot);
    }
    SECTION("pinning again changes nothing") {
        fixture.write(video, "second-render");
        const auto before = part.snapshot;
        snapshots.pin(part);
        CHECK(part.snapshot == before);
    }
}

TEST_CASE("Only a readable local file is pinned", "[media][video]") {
    Fixture fixture;
    const MediaSnapshots snapshots(fixture.root / "media" / "session");

    auto missing = ContentPart::make_video((fixture.root / "gone.mp4").string(), "video/mp4");
    snapshots.pin(missing);
    CHECK(missing.snapshot.empty());

    fixture.write(fixture.root / "empty.mp4", "");
    auto empty = ContentPart::make_video((fixture.root / "empty.mp4").string(), "video/mp4");
    snapshots.pin(empty);
    CHECK(empty.snapshot.empty());
    CHECK_FALSE(core::llm::encode_media_part(empty));  // Sent as a note, never as an empty source.

    auto remote = ContentPart::make_video_url("https://example.com/clip.mp4");
    snapshots.pin(remote);
    CHECK(remote.snapshot.empty());

    auto text = ContentPart::make_text("hello");
    snapshots.pin(text);
    CHECK(text.snapshot.empty());
}

TEST_CASE("An @-mentioned video is pinned when the prompt has a snapshot store",
          "[media][video][context]") {
    Fixture fixture;
    const auto video = fixture.root / "clips" / "flow.mp4";
    fixture.write(video, "mp4-bytes");
    const auto prompt = R"(Watch @"clips/flow.mp4")";

    const MediaSnapshots snapshots(fixture.root / "media" / "session");
    const auto pinned = core::context::expand_prompt(prompt, fixture.root, {.media_snapshots = &snapshots});
    REQUIRE(pinned.content_parts.size() == 2);
    const auto& part = pinned.content_parts[1];
    CHECK(part.type == core::llm::ContentPartType::Video);
    CHECK(part.path == video.string());
    CHECK(fs::path(part.snapshot).parent_path() == snapshots.directory());

    const auto unpinned = core::context::expand_prompt(prompt, fixture.root);
    REQUIRE(unpinned.content_parts.size() == 2);
    CHECK(unpinned.content_parts[1].snapshot.empty());
}

TEST_CASE("An @-mentioned image is pinned as a snapshot, not held inline",
          "[media][image][context]") {
    Fixture fixture;
    const auto image = fixture.root / "shots" / "error.png";
    fixture.write(image, "png-bytes");
    const auto prompt = R"(Look at @"shots/error.png")";

    const MediaSnapshots snapshots(fixture.root / "media" / "session");
    const auto pinned = core::context::expand_prompt(prompt, fixture.root, {.media_snapshots = &snapshots});
    REQUIRE(pinned.content_parts.size() == 2);
    const auto& part = pinned.content_parts[1];
    CHECK(part.type == core::llm::ContentPartType::Image);
    CHECK(part.path == image.string());
    CHECK(part.url.empty());  // The history keeps a path, not the image.
    CHECK(fs::path(part.snapshot).parent_path() == snapshots.directory());

    // What is sent is what was attached, however the file changes later.
    fixture.write(image, "overwritten");
    const auto encoded = core::llm::encode_image_part(part);
    REQUIRE(encoded);
    CHECK(encoded->data_url() == "data:image/png;base64,cG5nLWJ5dGVz");

    // Without a snapshot store the bytes are kept inline, as before.
    const auto inline_part = core::context::expand_prompt(prompt, fixture.root).content_parts[1];
    CHECK(inline_part.snapshot.empty());
    CHECK(inline_part.url == "data:image/png;base64,b3ZlcndyaXR0ZW4=");
}

TEST_CASE("A session keeps where its media snapshots are, and removes them with it",
          "[media][session]") {
    Fixture fixture;
    const core::session::SessionStore store(fixture.root / "sessions");
    core::session::SessionData data;
    data.session_id = "abcd1234";
    auto video = ContentPart::make_video("/work/render.mp4", "video/mp4");
    const auto media = core::session::SessionStore::media_dir(store.sessions_dir(), data.session_id);
    video.snapshot = (media / std::string(64, 'a')).string();
    data.messages.push_back({.role = "user", .content = "watch", .content_parts = {video}});
    fixture.write(video.snapshot, "bytes");
    REQUIRE(store.save(data));

    const auto loaded = store.load_by_id(data.session_id);
    REQUIRE(loaded);
    REQUIRE(loaded->messages.size() == 1);
    REQUIRE(loaded->messages[0].content_parts.size() == 1);
    CHECK(loaded->messages[0].content_parts[0].path == "/work/render.mp4");
    CHECK(loaded->messages[0].content_parts[0].snapshot == video.snapshot);

    REQUIRE(store.remove(data.session_id));
    CHECK_FALSE(fs::exists(media));

    // A crafted ID never names a directory outside the store.
    CHECK(core::session::SessionStore::media_dir(store.sessions_dir(), "../escape").empty());
}

TEST_CASE("A branched session keeps its media when the original is deleted",
          "[media][session][regression]") {
    Fixture fixture;
    const core::session::SessionStore store(fixture.root / "sessions");
    const auto video = fixture.root / "work" / "render.mp4";
    fixture.write(video, "attached-render");

    core::session::SessionData original;
    original.session_id = "aaaa0001";
    const MediaSnapshots original_media(
        core::session::SessionStore::media_dir(store.sessions_dir(), original.session_id));
    auto part = ContentPart::make_video(video.string(), "video/mp4");
    original_media.pin(part);
    REQUIRE_FALSE(part.snapshot.empty());
    original.messages.push_back({.role = "user", .content = "watch", .content_parts = {part}});
    REQUIRE(store.save(original));

    auto branch = original;  // What /fork, /branch and rotation do.
    branch.session_id = "bbbb0002";
    REQUIRE(store.save(branch));
    REQUIRE(store.remove(original.session_id));
    fs::remove(video);

    const auto loaded = store.load_by_id(branch.session_id);
    REQUIRE(loaded);
    const auto& kept = loaded->messages.at(0).content_parts.at(0);
    CHECK(fs::path(kept.snapshot).parent_path()
          == core::session::SessionStore::media_dir(store.sessions_dir(), branch.session_id));
    const auto encoded = core::llm::encode_media_part(kept);
    REQUIRE(encoded);
    CHECK(encoded->base64_data == core::utils::Base64::encode("attached-render"));

    // Saving it again stores nothing new.
    REQUIRE(store.save(*loaded));
    CHECK(loaded->messages.at(0).content_parts.at(0).snapshot == kept.snapshot);
}
