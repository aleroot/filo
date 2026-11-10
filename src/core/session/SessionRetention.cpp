#include "SessionRetention.hpp"

#include "ActiveSessionLease.hpp"
#include "ExecutionJournal.hpp"
#include "SessionStore.hpp"
#include "ThreadCatalog.hpp"
#include "../changes/RecoveryStore.hpp"
#include "../utils/InterprocessFile.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <set>
#include <string_view>
#include <system_error>
#include <utility>

namespace core::session {
namespace fs = std::filesystem;
using Clock = std::chrono::system_clock;
using core::utils::InterprocessFileLock;

std::optional<std::chrono::days> parse_retention_days(
    const std::optional<std::string>& value, std::optional<std::chrono::days> fallback) {
    if (!value || value->empty()) return fallback;
    int days = 0;
    const auto* end = value->data() + value->size();
    const auto [ptr, ec] = std::from_chars(value->data(), end, days);
    if (ec != std::errc{} || ptr != end || days < 0) return fallback;
    // A century is forever too, and keeps `now - days` far from overflow.
    if (days == 0 || days > 36'500) return std::nullopt;
    return std::chrono::days{days};
}

namespace {

/// A lock file younger than this may be about to be taken by the writer that
/// created it, so it is never removed.
constexpr std::chrono::hours kLockGrace{24};

// ── Filesystem queries ───────────────────────────────────────────────────────

/// What is at @p path, symlinks not followed; `none` when that cannot be told.
fs::file_type type_of(const fs::path& path) {
    std::error_code ec;
    return fs::symlink_status(path, ec).type();
}

/// Whether anything is at @p path; true when that cannot be told.
bool present(const fs::path& path) { return type_of(path) != fs::file_type::not_found; }

/// What @p directory holds; nullopt when it cannot be listed.
std::optional<std::vector<fs::path>> try_list(const fs::path& directory) {
    std::vector<fs::path> entries;
    std::error_code ec;
    for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec))
        entries.push_back(it->path());
    if (ec) return std::nullopt;
    return entries;
}

/// What @p directory holds; nothing when it cannot be listed.
std::vector<fs::path> listing(const fs::path& directory) {
    return try_list(directory).value_or(std::vector<fs::path>{});
}

/// The directories (not symlinks to them) in @p directory.
std::vector<fs::path> subdirectories_of(const fs::path& directory) {
    auto entries = listing(directory);
    std::erase_if(entries, [](const fs::path& entry) { return type_of(entry) != fs::file_type::directory; });
    return entries;
}

std::optional<Clock::time_point> modified(const fs::path& path) {
    std::error_code ec;
    const auto time = fs::last_write_time(path, ec);
    if (ec) return std::nullopt;
    return std::chrono::time_point_cast<Clock::duration>(fs::file_time_type::clock::to_sys(time));
}

/// Whether none of @p uses was written since @p cutoff. A directory counts as
/// written when it, one of its entries or an entry's manifest is (a turn's
/// manifest is rewritten while the turn runs). False when anything present
/// cannot be dated, so what cannot be aged is kept.
bool idle_since(const std::vector<fs::path>& uses, Clock::time_point cutoff) {
    std::vector<fs::path> dated;
    for (const auto& use : uses) {
        dated.push_back(use);
        if (type_of(use) != fs::file_type::directory) continue;
        const auto entries = try_list(use);
        if (!entries) return false;
        for (const auto& entry : *entries) {
            dated.push_back(entry);
            dated.push_back(entry / "manifest.json");
        }
    }
    bool any = false;
    for (const auto& path : dated) {
        if (!present(path)) continue;
        const auto time = modified(path);
        if (!time || *time >= cutoff) return false;
        any = true;
    }
    return any;
}

bool remove_tree(const fs::path& path, std::string* error) {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec && error) *error = ec.message();
    return !ec;
}

// ── Lock files ───────────────────────────────────────────────────────────────

using LockPathFor = fs::path (*)(const fs::path&);

/// The file @p lock is the @p lock_for lock of; @p lock_for appends a suffix.
std::optional<fs::path> guarded_by(const fs::path& lock, LockPathFor lock_for) {
    const auto suffix = lock_for(fs::path{}).native();
    const auto& name = lock.native();
    if (name.size() <= suffix.size() || !name.ends_with(suffix)) return std::nullopt;
    return fs::path(name.substr(0, name.size() - suffix.size()));
}

std::optional<std::string> session_id_of(const fs::path& file) {
    const auto name = file.filename().string();
    if (const auto id = SessionStore::session_id_from_filename(name)) return std::string(*id);
    return std::nullopt;
}

/// Removes @p lock, guarding @p file, once @p file is gone and no process
/// holds the lock.
bool remove_stale_lock(const fs::path& lock, const fs::path& file, Clock::time_point now) {
    if (present(file)) return false;
    const auto created = modified(lock);
    if (!created || *created > now - kLockGrace) return false;
    const auto held = InterprocessFileLock::try_acquire(lock);
    if (!held || present(file)) return false;
    std::error_code ec;
    return fs::remove(lock, ec);
}

// ── The sessions directory ───────────────────────────────────────────────────

/// What the sessions directory holds, by session ID.
class Catalogue {
public:
    struct Lock {
        fs::path path;
        fs::path guards;  ///< The session file it is the lock or lease of.
    };

    static Catalogue scan(const fs::path& directory) {
        Catalogue catalogue;
        for (const auto& path : listing(directory)) {
            // A lease is a lock file too, so it is told apart first.
            if (const auto file = guarded_by(path, active_session_lease_path)) {
                if (const auto id = session_id_of(*file)) {
                    catalogue.leases_[*id].insert(path);
                    catalogue.locks_.push_back({path, *file});
                }
            } else if (const auto locked = guarded_by(path, core::utils::lock_path_for)) {
                if (session_id_of(*locked)) catalogue.locks_.push_back({path, *locked});
            } else if (const auto id = session_id_of(path)) {
                std::error_code ec;
                if (!fs::is_regular_file(path, ec)) continue;
                catalogue.leases_[*id].insert(active_session_lease_path(path));
                catalogue.sessions_.emplace(*id, path);
            }
        }
        return catalogue;
    }

    /// The saved file of session @p id.
    [[nodiscard]] std::optional<fs::path> session(std::string_view id) const {
        const auto it = sessions_.find(id);
        return it == sessions_.end() ? std::nullopt : std::optional(it->second);
    }

    /// Lock and lease files of the session files they name.
    [[nodiscard]] const std::vector<Lock>& locks() const noexcept { return locks_; }

    /// Every lease of session @p id, held; nullopt while a process holds one.
    [[nodiscard]] std::optional<std::vector<InterprocessFileLock>> claim(std::string_view id) const {
        std::vector<InterprocessFileLock> held;
        const auto it = leases_.find(id);
        if (it == leases_.end()) return held;
        for (const auto& path : it->second) {
            auto lock = InterprocessFileLock::try_acquire(path);
            if (!lock) return std::nullopt;
            held.push_back(std::move(*lock));
        }
        return held;
    }

private:
    std::map<std::string, fs::path, std::less<>> sessions_;
    /// By session: the lease of its saved file and any other lease named for it.
    std::map<std::string, std::set<fs::path>, std::less<>> leases_;
    std::vector<Lock> locks_;
};

// ── The pass ─────────────────────────────────────────────────────────────────

class Sweep {
public:
    Sweep(const SessionStore& store, const RetentionPolicy& policy, Clock::time_point now, std::stop_token stop)
        : store_(store),
          directory_(store.sessions_dir()),
          journals_(ExecutionJournal::root_in(directory_)),
          journal_(journals_),
          sessions_cutoff_(cutoff(policy.sessions, now)),
          revert_history_cutoff_(cutoff(policy.revert_history, now)),
          now_(now),
          stop_(std::move(stop)) {}

    RetentionReport run() && {
        for (const auto step : {&Sweep::sessions, &Sweep::revert_history, &Sweep::leftovers,
                                &Sweep::stale_locks, &Sweep::compress_saved}) {
            if (stop_.stop_requested()) break;
            (this->*step)();
        }
        return std::move(report_);
    }

private:
    static std::optional<Clock::time_point> cutoff(std::optional<std::chrono::days> age, Clock::time_point now) {
        return age ? std::optional(now - *age) : std::nullopt;
    }

    /// Unnamed conversations neither used nor saved since the cutoff.
    void sessions() {
        if (!sessions_cutoff_) return;
        each(store_.list(), [&](const SessionInfo& info) {
            if (!info.name.empty()) return;
            const auto active = parse_iso8601_utc(thread_activity_timestamp(info));
            const auto written = modified(info.path);
            if (!active || !written || std::max(*active, *written) >= *sessions_cutoff_) return;
            std::string error;
            tally(info.session_id, store_.remove(info.session_id, &error), error, report_.sessions);
        });
    }

    /// Revert checkpoints of conversations idle since the cutoff.
    void revert_history() {
        if (!revert_history_cutoff_) return;
        const auto catalogue = Catalogue::scan(directory_);
        each(subdirectories_of(core::changes::RecoveryStore::root_in(directory_)), [&](const fs::path& root) {
            const auto id = root.filename().string();
            if (root != core::changes::RecoveryStore::session_root_in(directory_, id)) return;
            std::vector<fs::path> uses{root};
            if (auto session = catalogue.session(id)) uses.push_back(std::move(*session));
            expire(catalogue, id, uses, *revert_history_cutoff_,
                   [&](std::string* error) { return remove_tree(root, error); }, report_.checkpoints);
        });
    }

    /// Journals and tool results of no saved conversation, idle since the
    /// conversation cutoff.
    void leftovers() {
        if (!sessions_cutoff_) return;
        const auto catalogue = Catalogue::scan(directory_);
        each(listing(journals_), [&](const fs::path& path) {
            const auto id = path.stem().string();
            std::error_code ec;
            if (catalogue.session(id) || journal_.path_for(id) != path || !fs::is_regular_file(path, ec)) return;
            expire(catalogue, id, {path}, *sessions_cutoff_,
                   [&](std::string* error) { return journal_.remove(id, error); }, report_.leftovers);
        });
        each(subdirectories_of(SessionStore::tool_results_dir(directory_)), [&](const fs::path& path) {
            const auto id = path.filename().string();
            if (catalogue.session(id) || SessionStore::tool_results_dir(directory_, id) != path) return;
            expire(catalogue, id, {path}, *sessions_cutoff_,
                   [&](std::string* error) { return remove_tree(path, error); }, report_.leftovers);
        });
    }

    /// Lock files of sessions and journals deleted before their locks were
    /// deleted with them.
    void stale_locks() {
        each(Catalogue::scan(directory_).locks(), [&](const Catalogue::Lock& lock) {
            if (remove_stale_lock(lock.path, lock.guards, now_)) ++report_.leftovers;
        });
        each(listing(journals_), [&](const fs::path& lock) {
            const auto journal = guarded_by(lock, core::utils::lock_path_for);
            if (!journal || journal_.path_for(journal->stem().string()) != *journal) return;
            if (remove_stale_lock(lock, *journal, now_)) ++report_.leftovers;
        });
    }

    /// Conversations saved as plain JSON, before sessions were compressed.
    void compress_saved() {
        each(store_.list(), [&](const SessionInfo& info) {
            std::string error;
            if (store_.compress_saved(info.path, &error)) ++report_.compressed;
            else if (!error.empty()) fail(info.session_id, error);
        });
    }

    /// Visits @p items until a stop is requested.
    template <typename Items, typename Visit>
    void each(const Items& items, Visit visit) const {
        for (const auto& item : items) {
            if (stop_.stop_requested()) return;
            visit(item);
        }
    }

    /// Erases what is stored for session @p id when none of @p uses was
    /// written since @p cutoff, both before and after claiming every lease of
    /// the session: a conversation opened, or used, meanwhile keeps it.
    void expire(const Catalogue& catalogue, std::string_view id, const std::vector<fs::path>& uses,
                Clock::time_point cutoff, const std::function<bool(std::string*)>& erase, std::size_t& count) {
        if (!idle_since(uses, cutoff)) return;
        const auto held = catalogue.claim(id);
        if (!held || !idle_since(uses, cutoff)) return;
        std::string error;
        tally(id, erase(&error), error, count);
    }

    void tally(std::string_view id, bool removed, std::string_view error, std::size_t& count) {
        if (removed) ++count;
        else fail(id, error.empty() ? "not removed" : error);
    }

    void fail(std::string_view id, std::string_view error) { report_.errors.push_back(std::format("{}: {}", id, error)); }

    const SessionStore& store_;
    fs::path directory_;
    fs::path journals_;
    ExecutionJournal journal_;
    std::optional<Clock::time_point> sessions_cutoff_;
    std::optional<Clock::time_point> revert_history_cutoff_;
    Clock::time_point now_;
    std::stop_token stop_;
    RetentionReport report_;
};

} // namespace

RetentionReport apply_retention(const SessionStore& store, const RetentionPolicy& policy,
                                Clock::time_point now, std::stop_token stop) {
    std::error_code ec;
    if (!fs::is_directory(store.sessions_dir(), ec)) return {};
    // One pass at a time: two would race to delete the same trees.
    const auto exclusive = InterprocessFileLock::try_acquire(store.sessions_dir() / ".retention.lock");
    if (!exclusive) return {};
    return Sweep(store, policy, now, std::move(stop)).run();
}

} // namespace core::session
