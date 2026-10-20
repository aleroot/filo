#include <catch2/catch_test_macros.hpp>

#if defined(__APPLE__)

#include "core/session/SessionStore.hpp"
#include "tui/lampo/LampoCliSession.hpp"
#include "tui/viewer/LampoDiffViewer.hpp"

#import <AppKit/AppKit.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

namespace {

namespace fs = std::filesystem;

using tui::lampo::CliSession;

/// A session id in the shape Lampo accepts: exactly sixteen hexadecimal
/// characters. Random per process, because the suite runs each case in one.
[[nodiscard]] std::string make_session_id() {
    return core::session::SessionStore::generate_id() + core::session::SessionStore::generate_id();
}

[[nodiscard]] std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

[[nodiscard]] NSPasteboard* session_board(const std::string& id) {
    return [NSPasteboard pasteboardWithName:[NSString stringWithUTF8String:
        (std::string(tui::lampo::kComparerProtocol.pasteboard_prefix) + id).c_str()]];
}

[[nodiscard]] NSPasteboardType comparer_type() {
    return [NSString stringWithUTF8String:tui::lampo::kComparerProtocol.pasteboard_type.data()];
}

} // namespace

TEST_CASE("An older Lampo receives the original patch and returns ordinary comments", "[lampo]") {
    @autoreleasepool {
        const auto id = make_session_id();
        constexpr std::string_view patch = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n";
        auto session = CliSession::create(tui::lampo::kComparerProtocol, id, patch);
        REQUIRE(session);
        NSPasteboard* board = session_board(id);
        NSDictionary* request = [board propertyListForType:comparer_type()];
        REQUIRE([request[@"reply"] isEqualToString:@"comments"]);
        CHECK([request[@"accept"] isEqualToString:@"annotations"]);
        CHECK(read_file(session->file()) == patch);

        const std::string comments = "File: x\nKeep the old format.\n";
        { std::ofstream output(session->file()); output << comments; }
        [board clearContents];
        [board setPropertyList:@{@"version": @"1", @"session": [NSString stringWithUTF8String:id.c_str()],
                                 @"state": @"saved"} forType:comparer_type()];
        const auto reply = session->poll();
        REQUIRE(reply);
        CHECK(reply->reply.empty());
        const auto result = tui::viewer::decode_lampo_review_reply(read_file(session->file()), reply->reply);
        CHECK(result.comments() == comments);
        CHECK_FALSE(result.annotations());
    }
}

TEST_CASE("A negotiated Lampo request reuses the granted file and ignores its own announcement", "[lampo]") {
    @autoreleasepool {
        const auto id = make_session_id();
        auto created = CliSession::create(tui::lampo::kComparerProtocol, id, "plain patch");
        REQUIRE(created);
        const auto file = created->file();
        CliSession session = std::move(*created);
        NSPasteboard* board = session_board(id);
        CHECK_FALSE(session.poll());
        [board clearContents];
        // A peer publishes in two calls. An empty board between them is not
        // a consumed answer, even if setting its data keeps the same count.
        CHECK_FALSE(session.poll());
        [board setPropertyList:@{@"version": @"1", @"session": [NSString stringWithUTF8String:id.c_str()],
                                 @"state": @"ready", @"reply": @"annotations"} forType:comparer_type()];
        const auto ready = session.poll();
        REQUIRE(ready);
        CHECK(ready->reply == "annotations");
        constexpr std::string_view document = R"({"version":1,"patch":"plain patch","annotations":[]})";
        REQUIRE(session.update_request(ready->reply, document));
        CHECK(session.file() == file);
        CHECK(read_file(file) == document);
        NSDictionary* request = [board propertyListForType:comparer_type()];
        CHECK([request[@"state"] isEqualToString:@"request"]);
        CHECK([request[@"reply"] isEqualToString:@"annotations"]);
        CHECK(request[@"accept"] == nil);
        CHECK_FALSE(session.poll());
        CHECK_FALSE(session.update_request("annotations", "duplicate"));
        CHECK(read_file(file) == document);

        { std::ofstream output(file); output << R"({"version":1,"annotations":[]})"; }
        [board clearContents];
        [board setPropertyList:@{@"version": @"1", @"session": [NSString stringWithUTF8String:id.c_str()],
                                 @"state": @"saved", @"reply": @"annotations"} forType:comparer_type()];
        const auto saved = session.poll();
        REQUIRE(saved);
        const auto result = tui::viewer::decode_lampo_review_reply(read_file(file), saved->reply);
        REQUIRE(result.annotations());
        CHECK(result.annotations()->empty());
    }
}

TEST_CASE("A Lampo session puts its payload exactly where Lampo looks", "[lampo]") {
    const std::string session_id = make_session_id();
    fs::path directory;
    {
        auto session = CliSession::create(
            tui::lampo::kComparerProtocol,
            session_id,
            "diff --git a/x b/x\n");
        REQUIRE(session.has_value());

        directory = session->file().parent_path();
        CHECK(session->file().filename() == "changes.patch");
        // The shape Lampo's CliSessionDirectory recognises: a prefixed
        // directory holding the one file the protocol reads. Anything else is
        // treated as an ordinary document open and no window appears.
        CHECK(directory.filename() == "lampo-comparer-view-v1-" + session_id);
        CHECK(read_file(session->file()) == "diff --git a/x b/x\n");

        // Owner-only: nobody else on this machine reads what a turn changed.
        const auto file_permissions = fs::status(session->file()).permissions();
        CHECK((file_permissions & fs::perms::owner_read) != fs::perms::none);
        CHECK((file_permissions & fs::perms::group_read) == fs::perms::none);
        CHECK((file_permissions & fs::perms::others_read) == fs::perms::none);
        const auto directory_permissions = fs::status(directory).permissions();
        CHECK((directory_permissions & fs::perms::owner_all) == fs::perms::owner_all);
        CHECK((directory_permissions & fs::perms::group_all) == fs::perms::none);

        // Nothing has answered yet, and no application has been launched.
        CHECK_FALSE(session->poll().has_value());
        CHECK_FALSE(session->app_running());
    }

    // The session removes its own payload, on every exit path.
    CHECK_FALSE(fs::exists(directory));
}

TEST_CASE("A Lampo session can copy a scratch file it did not write", "[lampo]") {
    const std::string session_id = make_session_id();
    const fs::path source = fs::temp_directory_path() / ("filo-lampo-source-" + session_id);
    {
        std::ofstream output(source, std::ios::binary | std::ios::trunc);
        output << "the draft";
    }

    fs::path payload;
    {
        auto session = CliSession::create_from_file(
            tui::lampo::kPromptEditProtocol,
            session_id,
            source);
        REQUIRE(session.has_value());

        payload = session->file();
        CHECK(payload.filename() == "prompt.md");
        CHECK(payload.parent_path().filename() == "lampo-prompter-edit-v1-" + session_id);
        CHECK(read_file(payload) == "the draft");
        // The source is the caller's scratch buffer, not the session's to keep.
        CHECK(fs::exists(source));
    }

    CHECK_FALSE(fs::exists(payload));
    fs::remove(source);
}

TEST_CASE("Moving a Lampo session hands over its payload exactly once", "[lampo]") {
    const std::string session_id = make_session_id();
    fs::path payload;
    {
        auto first = CliSession::create(tui::lampo::kComparerProtocol, session_id, "patch");
        REQUIRE(first.has_value());
        payload = first->file();

        CliSession second = std::move(*first);
        CHECK(second.file() == payload);
        // The moved-from session owns nothing, so it cannot remove what the
        // other one is still using.
        CHECK(first->file().empty());
        CHECK(fs::exists(payload));
    }
    CHECK_FALSE(fs::exists(payload));
}

#endif // defined(__APPLE__)
