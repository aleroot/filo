#include <catch2/catch_test_macros.hpp>

#include "core/agent/ToolResultStore.hpp"
#include "core/changes/RecoveryStore.hpp"
#include "core/session/ActiveSessionLease.hpp"
#include "core/session/ExecutionJournal.hpp"
#include "core/session/SessionRetention.hpp"
#include "core/session/SessionStore.hpp"
#include "core/utils/Gzip.hpp"
#include "core/utils/InterprocessFile.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stop_token>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using core::session::RetentionPolicy;
using core::session::SessionStore;
using Clock = std::chrono::system_clock;
using std::chrono::days;

struct RetentionFixture {
    fs::path root;
    fs::path sessions;
    SessionStore store;
    Clock::time_point now = Clock::now();

    RetentionFixture() : root(make_root()), sessions(root / "sessions"), store(sessions) {
        fs::create_directories(sessions);
    }
    ~RetentionFixture() { std::error_code ec; fs::remove_all(root, ec); }

    static fs::path make_root() {
        auto pattern = (fs::temp_directory_path() / "filo-retention-XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()));
        return pattern;
    }

    /// Dates @p path and everything below it @p age ago.
    void age(const fs::path& path, days age) const {
        const auto when = fs::file_time_type::clock::from_sys(now - age);
        if (fs::is_directory(path) && !fs::is_symlink(path)) {
            for (const auto& entry : fs::recursive_directory_iterator(path)) fs::last_write_time(entry.path(), when);
        }
        fs::last_write_time(path, when);
    }

    /// A saved session last used @p idle ago, with a journal, a revert
    /// checkpoint and an offloaded tool result.
    core::session::SessionData session(std::string id, days idle, std::string name = {}) const {
        core::session::SessionData data;
        data.session_id = std::move(id);
        data.name = std::move(name);
        data.created_at = SessionStore::to_iso8601(now - idle - days{1});
        data.last_active_at = SessionStore::to_iso8601(now - idle);
        REQUIRE(store.save(data));
        REQUIRE(journal().append(data.session_id, "agent", "", "clear", "{}"));
        checkpoint(data.session_id);
        REQUIRE(results().store(data.session_id, "call-1", "output"));
        age(store.compute_path(data), idle);
        age(journal().path_for(data.session_id), idle);
        age(recovery(data.session_id), idle);
        age(tool_results(data.session_id), idle);
        return data;
    }

    void checkpoint(std::string_view id) const {
        const auto turn = core::changes::RecoveryStore::new_directory(recovery(id));
        fs::create_directories(turn);
        std::ofstream(turn / "manifest.json") << "{}";
        fs::create_directories(recovery(id) / "objects");
        std::ofstream(recovery(id) / "objects" / std::string(64, 'a')) << "bytes";
    }

    [[nodiscard]] core::session::ExecutionJournal journal() const { return core::session::ExecutionJournal(sessions / "journals"); }
    [[nodiscard]] core::agent::ToolResultStore results() const {
        return core::agent::ToolResultStore(SessionStore::tool_results_dir(sessions));
    }
    [[nodiscard]] fs::path recovery(std::string_view id) const {
        return core::changes::RecoveryStore::session_root_in(sessions, id);
    }
    [[nodiscard]] fs::path tool_results(std::string_view id) const {
        return SessionStore::tool_results_dir(sessions) / id;
    }
    [[nodiscard]] bool stored(const core::session::SessionData& data) const {
        return fs::exists(store.compute_path(data));
    }
    [[nodiscard]] bool anything_of(std::string_view id) const {
        return fs::exists(journal().path_for(id)) || fs::exists(recovery(id)) || fs::exists(tool_results(id));
    }

    core::session::RetentionReport run(RetentionPolicy policy) const {
        return core::session::apply_retention(store, policy, now);
    }
};

} // namespace

TEST_CASE("Retention days parse from settings strings", "[session][retention]") {
    using core::session::parse_retention_days;
    const std::optional<days> fallback = days{7};
    CHECK(parse_retention_days(std::nullopt, fallback) == fallback);
    CHECK(parse_retention_days(std::string{}, fallback) == fallback);
    CHECK(parse_retention_days(std::string("30"), fallback) == days{30});
    CHECK(parse_retention_days(std::string("0"), fallback) == std::nullopt);
    CHECK(parse_retention_days(std::string("-3"), fallback) == fallback);
    CHECK(parse_retention_days(std::string("30d"), fallback) == fallback);
    CHECK(parse_retention_days(std::string(" 30"), fallback) == fallback);
    CHECK(parse_retention_days(std::string("99999999999"), fallback) == fallback);
    CHECK(parse_retention_days(std::string("100000"), fallback) == std::nullopt);  // Forever.
    CHECK(parse_retention_days(std::string("abc"), std::nullopt) == std::nullopt);
}

TEST_CASE("The default policy keeps every conversation", "[session][retention]") {
    RetentionFixture fixture;
    const auto old = fixture.session("aaaa0001", days{4000});
    const auto report = fixture.run(RetentionPolicy{.sessions = std::nullopt, .revert_history = std::nullopt});
    CHECK(fixture.stored(old));
    CHECK(fs::exists(fixture.journal().path_for(old.session_id)));
    CHECK(fs::exists(fixture.recovery(old.session_id)));
    CHECK(fs::exists(fixture.tool_results(old.session_id)));
    CHECK(report.sessions == 0);
    CHECK(report.checkpoints == 0);
}

TEST_CASE("Idle conversations are deleted with everything stored for them", "[session][retention]") {
    RetentionFixture fixture;
    const auto idle = fixture.session("aaaa0001", days{40});
    const auto recent = fixture.session("aaaa0002", days{5});
    const auto named = fixture.session("aaaa0003", days{400}, "keep me");
    const auto open = fixture.session("aaaa0004", days{40});
    auto lease = core::session::ActiveSessionLease::acquire(fixture.store, open);
    REQUIRE(lease);

    const auto report = fixture.run(RetentionPolicy{.sessions = days{30}, .revert_history = std::nullopt});

    CHECK_FALSE(fixture.stored(idle));
    CHECK_FALSE(fixture.anything_of(idle.session_id));
    CHECK(fixture.stored(recent));
    CHECK(fs::exists(fixture.recovery(recent.session_id)));
    CHECK(fixture.stored(named));
    CHECK(fs::exists(fixture.recovery(named.session_id)));
    CHECK(fixture.stored(open));
    CHECK(fs::exists(fixture.recovery(open.session_id)));
    CHECK(report.sessions == 1);
    CHECK(report.errors.size() == 1);  // The open one, which is kept.
}

TEST_CASE("A conversation saved recently is kept whatever its recorded activity", "[session][retention]") {
    RetentionFixture fixture;
    const auto data = fixture.session("aaaa0001", days{40});
    fs::last_write_time(fixture.store.compute_path(data),
                        fs::file_time_type::clock::from_sys(fixture.now));
    fixture.run(RetentionPolicy{.sessions = days{30}, .revert_history = days{30}});
    CHECK(fixture.stored(data));
    CHECK(fs::exists(fixture.recovery(data.session_id)));
}

TEST_CASE("Revert history of idle conversations is deleted, the conversation stays",
          "[session][retention][recovery]") {
    RetentionFixture fixture;
    const auto idle = fixture.session("aaaa0001", days{10});
    const auto recent = fixture.session("aaaa0002", days{2});
    const auto open = fixture.session("aaaa0003", days{10});
    auto lease = core::session::ActiveSessionLease::acquire(fixture.store, open);
    REQUIRE(lease);
    // A turn that ran recently keeps its session's history, however old the
    // rest of it is.
    const auto busy = fixture.session("aaaa0004", days{10});
    fixture.checkpoint(busy.session_id);
    // Recovery of a session that was never saved, long ago.
    fixture.checkpoint("bbbb0001");
    fixture.age(fixture.recovery("bbbb0001"), days{10});

    const auto report = fixture.run(RetentionPolicy{.sessions = std::nullopt, .revert_history = days{7}});

    CHECK(fixture.stored(idle));
    CHECK_FALSE(fs::exists(fixture.recovery(idle.session_id)));
    CHECK(fs::exists(fixture.journal().path_for(idle.session_id)));
    CHECK(fs::exists(fixture.tool_results(idle.session_id)));
    CHECK(fs::exists(fixture.recovery(recent.session_id)));
    CHECK(fs::exists(fixture.recovery(open.session_id)));
    CHECK(fs::exists(fixture.recovery(busy.session_id)));
    CHECK_FALSE(fs::exists(fixture.recovery("bbbb0001")));
    CHECK(report.checkpoints == 2);
}

TEST_CASE("Revert history of a session that is open but not yet saved is kept",
          "[session][retention][recovery]") {
    RetentionFixture fixture;
    core::session::SessionData live;
    live.session_id = "cccc0001";
    live.created_at = SessionStore::to_iso8601(fixture.now - days{20});
    auto lease = core::session::ActiveSessionLease::acquire(fixture.store, live);
    REQUIRE(lease);
    fixture.checkpoint(live.session_id);
    fixture.age(fixture.recovery(live.session_id), days{20});
    fixture.run(RetentionPolicy{.sessions = days{1}, .revert_history = days{1}});
    CHECK(fs::exists(fixture.recovery(live.session_id)));
}

TEST_CASE("Journals and tool results of no conversation follow conversation retention",
          "[session][retention]") {
    RetentionFixture fixture;
    const auto kept = fixture.session("aaaa0001", days{5});
    REQUIRE(fixture.journal().append("dddd0001", "agent", "", "clear", "{}"));
    REQUIRE(fixture.results().store("dddd0001", "call-1", "output"));
    REQUIRE(fixture.journal().append("dddd0002", "agent", "", "clear", "{}"));
    REQUIRE(fixture.results().store("dddd0002", "call-1", "output"));
    fixture.age(fixture.journal().path_for("dddd0001"), days{40});
    fixture.age(fixture.tool_results("dddd0001"), days{40});

    SECTION("Kept while conversations are") {
        fixture.run(RetentionPolicy{.sessions = std::nullopt, .revert_history = days{7}});
        CHECK(fs::exists(fixture.journal().path_for("dddd0001")));
        CHECK(fs::exists(fixture.tool_results("dddd0001")));
    }
    SECTION("Deleted once idle past it") {
        const auto report = fixture.run(RetentionPolicy{.sessions = days{30}, .revert_history = days{7}});
        CHECK_FALSE(fs::exists(fixture.journal().path_for("dddd0001")));
        CHECK_FALSE(fs::exists(fixture.tool_results("dddd0001")));
        CHECK(fs::exists(fixture.journal().path_for("dddd0002")));  // Recent.
        CHECK(fs::exists(fixture.tool_results("dddd0002")));
        CHECK(report.leftovers >= 2);
    }
    CHECK(fixture.stored(kept));
    CHECK(fs::exists(fixture.journal().path_for(kept.session_id)));
    CHECK(fs::exists(fixture.tool_results(kept.session_id)));
}

TEST_CASE("Stale lock files of deleted sessions are removed, live ones are not",
          "[session][retention]") {
    RetentionFixture fixture;
    const auto kept = fixture.session("aaaa0001", days{1});
    const auto kept_path = fixture.store.compute_path(kept);
    REQUIRE(core::utils::InterprocessFileLock::acquire(core::utils::lock_path_for(kept_path)));
    const auto touch = [&](const fs::path& path, days age) {
        std::ofstream(path).flush();
        fixture.age(path, age);
    };
    const auto gone = fixture.sessions / "session-20240101-000000-eeee0001.json";
    touch(core::utils::lock_path_for(gone), days{3});
    touch(core::session::active_session_lease_path(gone), days{3});
    const auto fresh = fixture.sessions / "session-20240101-000000-eeee0002.json";
    touch(core::utils::lock_path_for(fresh), days{0});
    const auto journal_lock = core::utils::lock_path_for(fixture.journal().path_for("eeee0003"));
    touch(journal_lock, days{3});
    const auto held = fixture.sessions / "session-20240101-000000-eeee0004.json";
    touch(core::session::active_session_lease_path(held), days{3});
    auto lease = core::utils::InterprocessFileLock::try_acquire(core::session::active_session_lease_path(held));
    REQUIRE(lease);
    const auto foreign = fixture.sessions / "notes.lock";
    touch(foreign, days{3});

    fixture.run(RetentionPolicy{.sessions = std::nullopt, .revert_history = std::nullopt});

    CHECK_FALSE(fs::exists(core::utils::lock_path_for(gone)));
    CHECK_FALSE(fs::exists(core::session::active_session_lease_path(gone)));
    CHECK_FALSE(fs::exists(journal_lock));
    CHECK(fs::exists(core::utils::lock_path_for(fresh)));
    CHECK(fs::exists(core::session::active_session_lease_path(held)));
    CHECK(fs::exists(core::utils::lock_path_for(kept_path)));
    CHECK(fs::exists(foreign));
    CHECK(fs::exists(core::utils::lock_path_for(fixture.journal().path_for(kept.session_id))));
}

TEST_CASE("A stopped retention pass deletes nothing", "[session][retention]") {
    RetentionFixture fixture;
    const auto idle = fixture.session("aaaa0001", days{400});
    std::stop_source stop;
    stop.request_stop();
    const auto report = core::session::apply_retention(
        fixture.store, RetentionPolicy{.sessions = days{30}, .revert_history = days{7}}, fixture.now,
        stop.get_token());
    CHECK(report.empty());
    CHECK(fixture.stored(idle));
    CHECK(fs::exists(fixture.recovery(idle.session_id)));
}

TEST_CASE("Only one retention pass runs at a time", "[session][retention]") {
    RetentionFixture fixture;
    const auto idle = fixture.session("aaaa0001", days{400});
    auto other = core::utils::InterprocessFileLock::try_acquire(fixture.sessions / ".retention.lock");
    REQUIRE(other);
    const auto report = fixture.run(RetentionPolicy{.sessions = days{30}, .revert_history = days{7}});
    CHECK(report.empty());
    CHECK(fixture.stored(idle));
}

TEST_CASE("Retention compresses conversations saved as plain JSON", "[session][retention][compression]") {
    RetentionFixture fixture;
    const auto data = fixture.session("aaaa0001", days{3});
    const auto path = fixture.store.compute_path(data);
    std::string bytes;
    {
        std::ifstream input(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    const auto json = core::utils::gzip::decompress(bytes);
    REQUIRE(json.has_value());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << *json;
    fixture.age(path, days{3});

    const auto report = fixture.run(RetentionPolicy{.sessions = std::nullopt, .revert_history = std::nullopt});

    CHECK(report.compressed == 1);
    std::ifstream input(path, std::ios::binary);
    const std::string stored{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const auto inflated = core::utils::gzip::decompress(stored);
    REQUIRE(inflated.has_value());
    CHECK(*inflated == *json);
    const auto loaded = fixture.store.load_by_id(data.session_id);
    REQUIRE(loaded.has_value());
    CHECK(loaded->last_active_at == data.last_active_at);
    // Its age is unchanged, so retention still dates it correctly.
    fixture.run(RetentionPolicy{.sessions = days{2}, .revert_history = std::nullopt});
    CHECK_FALSE(fs::exists(path));
}
