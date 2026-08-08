#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/session/ThreadCatalog.hpp"
#include "core/session/SessionStore.hpp"

#include <chrono>
#include <string>

using Catch::Matchers::ContainsSubstring;
using core::session::SessionInfo;
using core::session::ThreadRecencyGroup;

namespace {

SessionInfo make_info(std::string_view id,
                      std::string_view name,
                      std::string_view last_active,
                      std::string_view preview = {}) {
    SessionInfo info;
    info.session_id = std::string{id};
    info.name = std::string{name};
    info.last_active_at = std::string{last_active};
    info.created_at = std::string{last_active};
    info.preview = std::string{preview};
    info.provider = "grok";
    info.model = "grok-4";
    info.working_dir = "/home/user/project";
    info.turn_count = 2;
    return info;
}

} // namespace

TEST_CASE("ThreadCatalog collapses previews and titles", "[session][thread]") {
    CHECK(core::session::collapse_preview("  hello\n\nworld  ", 72) == "hello world");
    CHECK(core::session::collapse_preview("exact", 5) == "exact");
    CHECK(core::session::collapse_preview("long preview", 7) == "long...");
    CHECK(core::session::collapse_preview("anything", 0).empty());

    core::llm::Message user;
    user.role = "user";
    user.content = "Implement OAuth flow";
    core::llm::Message assistant;
    assistant.role = "assistant";
    assistant.content = "Sure";
    const auto preview = core::session::first_user_message_preview({user, assistant});
    CHECK(preview == "Implement OAuth flow");

    auto named = make_info("aaa11111", "auth-work", "2026-03-22T10:00:00Z", "ignored");
    CHECK(core::session::thread_display_title(named) == "auth-work");

    auto previewed = make_info("bbb22222", "", "2026-03-22T10:00:00Z", "Fix the flaky test");
    CHECK(core::session::thread_display_title(previewed) == "Fix the flaky test");

    auto bare = make_info("ccc33333", "", "2026-03-22T10:00:00Z");
    CHECK_THAT(core::session::thread_display_title(bare), ContainsSubstring("ccc33333"));
}

TEST_CASE("ThreadCatalog relative time and recency groups", "[session][thread]") {
    using namespace std::chrono;
    // Fixed "now": 2026-03-22 12:00:00 UTC
    std::tm tm{};
    tm.tm_year = 2026 - 1900;
    tm.tm_mon = 2;
    tm.tm_mday = 22;
    tm.tm_hour = 12;
    tm.tm_min = 0;
    tm.tm_sec = 0;
    const auto now = system_clock::from_time_t(timegm(&tm));

    CHECK(core::session::format_relative_time("2026-03-22T11:55:00Z", now) == "5m ago");
    CHECK(core::session::format_relative_time("2026-03-22T10:00:00Z", now) == "2h ago");
    CHECK(core::session::format_relative_time("2026-03-21T11:00:00Z", now) == "yesterday");

    CHECK(core::session::thread_recency_group("2026-03-22T11:00:00Z", now)
          == ThreadRecencyGroup::Today);
    CHECK(core::session::thread_recency_group("2026-03-21T11:00:00Z", now)
          == ThreadRecencyGroup::Yesterday);
    CHECK(core::session::thread_recency_group("2026-03-18T11:00:00Z", now)
          == ThreadRecencyGroup::LastWeek);
    CHECK(core::session::thread_recency_group("2026-02-01T11:00:00Z", now)
          == ThreadRecencyGroup::Older);
}

TEST_CASE("ThreadCatalog filter and grouping", "[session][thread]") {
    using namespace std::chrono;
    std::tm tm{};
    tm.tm_year = 2026 - 1900;
    tm.tm_mon = 2;
    tm.tm_mday = 22;
    tm.tm_hour = 12;
    const auto now = system_clock::from_time_t(timegm(&tm));

    std::vector<SessionInfo> sessions{
        make_info("aaa11111", "oauth", "2026-03-22T11:00:00Z", "Implement OAuth"),
        make_info("bbb22222", "flaky", "2026-03-21T10:00:00Z", "Fix flaky test"),
        make_info("ccc33333", "", "2026-03-10T10:00:00Z", "Older work"),
    };

    const auto filtered = core::session::filter_threads(sessions, "OAUTH");
    REQUIRE(filtered.size() == 1);
    CHECK(filtered[0].session_id == "aaa11111");

    const auto by_preview = core::session::filter_threads(sessions, "flaky");
    REQUIRE(by_preview.size() == 1);
    CHECK(by_preview[0].session_id == "bbb22222");

    const auto groups = core::session::group_threads_by_recency(sessions, now);
    REQUIRE(groups.size() >= 2);
    CHECK(groups.front().group == ThreadRecencyGroup::Today);
    CHECK(groups.front().indices.size() == 1);
    CHECK(sessions[groups.front().indices[0]].session_id == "aaa11111");
}

TEST_CASE("ThreadCatalog always pins main before more recently active threads",
          "[session][thread][ordering]") {
    std::vector<SessionInfo> threads{
        make_info("worker02", "filo 2", "2026-03-22T12:00:00Z"),
        make_info("main0001", "renamed-main", "2026-03-22T09:00:00Z"),
        make_info("worker01", "filo", "2026-03-22T11:00:00Z"),
    };

    core::session::order_active_threads(threads, "main0001");

    REQUIRE(threads.size() == 3);
    CHECK(threads[0].session_id == "main0001");
    CHECK(threads[1].session_id == "worker02");
    CHECK(threads[2].session_id == "worker01");
}

TEST_CASE("SessionStore list populates thread previews", "[session][thread][store]") {
    namespace fs = std::filesystem;
    const auto dir = fs::temp_directory_path() / "filo_test_thread_preview";
    fs::remove_all(dir);
    fs::create_directories(dir);

    core::session::SessionStore store{dir};
    core::session::SessionData data;
    data.session_id = "preview01";
    data.created_at = "2026-03-22T10:00:00Z";
    data.last_active_at = "2026-03-22T11:00:00Z";
    data.provider = "grok";
    data.model = "grok-4";
    data.messages.push_back({"user", "Ship thread management", "", "", {}});
    data.messages.push_back({"assistant", "On it", "", "", {}});
    data.stats.turn_count = 1;
    REQUIRE(store.save(data));

    const auto infos = store.list();
    REQUIRE(infos.size() == 1);
    CHECK(infos[0].preview == "Ship thread management");
    CHECK(core::session::thread_display_title(infos[0]) == "Ship thread management");

    fs::remove_all(dir);
}
