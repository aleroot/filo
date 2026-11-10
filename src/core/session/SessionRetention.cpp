#include "SessionRetention.hpp"

#include "ActiveSessionLease.hpp"
#include "ExecutionJournal.hpp"
#include "SessionStore.hpp"
#include "ThreadCatalog.hpp"
#include "../changes/RecoveryStore.hpp"
#include "../utils/InterprocessFile.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <filesystem>
#include <format>
#include <functional>
#include <map>
#include <string_view>
#include <system_error>

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

constexpr std::string_view kLeaseSuffix = ".active.lock";
constexpr std::string_view kLockSuffix = ".lock";
/// A lock file younger than this may be about to be taken by the writer that
/// created it, so it is never removed.
constexpr std::chrono::hours kLockGrace{24};

/// A name Filo itself gives a per-session directory or journal.
bool storage_id(std::string_view id) {
    return !id.empty() && id.size() <= 80 && std::ranges::all_of(id, [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_';
    });
}

/// Whether anything is at @p path; true when that cannot be told.
bool present(const fs::path& path) {
    std::error_code ec;
    return fs::symlink_status(path, ec).type() != fs::file_type::not_found;
}

std::optional<Clock::time_point> modified(const fs::path& path) {
    std::error_code ec;
    const auto time = fs::last_write_time(path, ec);
    if (ec) return std::nullopt;
    return std::chrono::time_point_cast<Clock::duration>(fs::file_time_type::clock::to_sys(time));
}

/// The newest change to a file, or to a per-session directory, its entries
/// and their manifests; nullopt when any of them cannot be dated.
std::optional<Clock::time_point> latest_activity(const fs::path& path) {
    auto newest = modified(path);
    std::error_code ec;
    if (!newest || !fs::is_directory(fs::symlink_status(path, ec))) return newest;
    for (fs::directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        for (const auto& entry : {it->path(), it->path() / "manifest.json"}) {
            if (!present(entry)) continue;
            const auto time = modified(entry);
            if (!time) return std::nullopt;
            newest = std::max(*newest, *time);
        }
    }
    if (ec) return std::nullopt;
    return newest;
}

/// What the sessions directory holds, by session ID.
struct Catalogue {
    std::map<std::string, fs::path, std::less<>> sessions;
    std::map<std::string, std::vector<fs::path>, std::less<>> leases;
    std::vector<fs::path> locks;
};

Catalogue scan(const fs::path& directory) {
    Catalogue result;
    std::error_code ec;
    for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (name.ends_with(kLockSuffix)) {
            result.locks.push_back(it->path());
            if (name.ends_with(kLeaseSuffix)) {
                const auto base = std::string_view(name).substr(0, name.size() - kLeaseSuffix.size());
                if (const auto id = SessionStore::session_id_from_filename(base))
                    result.leases[std::string(*id)].push_back(it->path());
            }
            continue;
        }
        std::error_code type_ec;
        if (!it->is_regular_file(type_ec)) continue;
        if (const auto id = SessionStore::session_id_from_filename(name))
            result.sessions.emplace(std::string(*id), it->path());
    }
    return result;
}

/// Every session lease of @p id, held; nullopt when a process holds one.
std::optional<std::vector<InterprocessFileLock>> claim(const Catalogue& catalogue, std::string_view id) {
    std::vector<fs::path> paths;
    if (const auto it = catalogue.leases.find(id); it != catalogue.leases.end()) paths = it->second;
    if (const auto it = catalogue.sessions.find(id); it != catalogue.sessions.end()) {
        auto lease = active_session_lease_path(it->second);
        if (!std::ranges::contains(paths, lease)) paths.push_back(std::move(lease));
    }
    std::vector<InterprocessFileLock> held;
    for (const auto& path : paths) {
        auto lock = InterprocessFileLock::try_acquire(path);
        if (!lock) return std::nullopt;
        held.push_back(std::move(*lock));
    }
    return held;
}

/// Removes a lock file that guards a file which no longer exists, once no
/// process holds it.
bool remove_stale_lock(const fs::path& lock, const fs::path& guarded, Clock::time_point now) {
    if (present(guarded)) return false;
    const auto created = modified(lock);
    if (!created || *created > now - kLockGrace) return false;
    const auto held = InterprocessFileLock::try_acquire(lock);
    if (!held || present(guarded)) return false;
    std::error_code ec;
    return fs::remove(lock, ec);
}

class Pass {
public:
    Pass(const SessionStore& store, Clock::time_point now, std::stop_token stop)
        : store_(store), directory_(store.sessions_dir()), now_(now), stop_(std::move(stop)) {}

    void sessions(std::chrono::days age) {
        const auto cutoff = now_ - age;
        for (const auto& info : store_.list()) {
            if (stop_.stop_requested()) return;
            if (!info.name.empty()) continue;
            const auto active = parse_iso8601_utc(thread_activity_timestamp(info));
            const auto written = modified(info.path);
            if (!active || !written || std::max(*active, *written) >= cutoff) continue;
            std::string error;
            if (store_.remove(info.session_id, &error)) ++report_.sessions;
            else fail(info.session_id, error);
        }
    }

    void revert_history(std::chrono::days age) {
        const auto cutoff = now_ - age;
        const auto catalogue = scan(directory_);
        for_each_directory(directory_ / "recovery", [&](const fs::path& root, std::string_view id) {
            if (root != core::changes::RecoveryStore::session_root_in(directory_, id)) return;
            const auto idle = [&] {
                auto newest = latest_activity(root);
                if (const auto it = catalogue.sessions.find(id); newest && it != catalogue.sessions.end()) {
                    const auto written = modified(it->second);
                    newest = written ? std::optional(std::max(*newest, *written)) : std::nullopt;
                }
                return newest && *newest < cutoff;
            };
            if (!idle()) return;
            const auto held = claim(catalogue, id);
            if (!held || !idle()) return;  // Opened, or used, meanwhile.
            std::error_code ec;
            fs::remove_all(root, ec);
            if (ec) fail(id, ec.message());
            else ++report_.checkpoints;
        });
    }

    /// Journals and tool results whose session is gone (or was never saved).
    void leftovers(std::chrono::days age) {
        const auto cutoff = now_ - age;
        const auto catalogue = scan(directory_);
        const auto orphaned = [&](const fs::path& path, std::string_view id) {
            if (catalogue.sessions.contains(id)) return false;
            const auto newest = latest_activity(path);
            return newest && *newest < cutoff;
        };
        const auto journals = directory_ / "journals";
        const ExecutionJournal journal(journals);
        std::error_code ec;
        for (fs::directory_iterator it(journals, ec), end; !ec && it != end; it.increment(ec)) {
            if (stop_.stop_requested()) return;
            const auto& path = it->path();
            const auto id = path.stem().string();
            std::error_code type_ec;
            if (path.extension() != ".jsonl" || !it->is_regular_file(type_ec) || journal.path_for(id) != path ||
                !orphaned(path, id)) continue;
            const auto held = claim(catalogue, id);
            if (!held || !orphaned(path, id)) continue;
            std::string error;
            if (journal.remove(id, &error)) ++report_.leftovers;
            else fail(id, error);
        }
        for_each_directory(SessionStore::tool_results_dir(directory_), [&](const fs::path& path, std::string_view id) {
            if (!orphaned(path, id)) return;
            const auto held = claim(catalogue, id);
            if (!held || !orphaned(path, id)) return;
            std::error_code remove_ec;
            fs::remove_all(path, remove_ec);
            if (remove_ec) fail(id, remove_ec.message());
            else ++report_.leftovers;
        });
    }

    /// Lock files left by sessions and journals deleted before they were
    /// removed together.
    void stale_locks() {
        for (const auto& lock : scan(directory_).locks) {
            if (stop_.stop_requested()) return;
            const auto name = lock.filename().string();
            const auto suffix = name.ends_with(kLeaseSuffix) ? kLeaseSuffix : kLockSuffix;
            const auto base = std::string_view(name).substr(0, name.size() - suffix.size());
            if (!SessionStore::session_id_from_filename(base)) continue;
            if (remove_stale_lock(lock, directory_ / base, now_)) ++report_.leftovers;
        }
        std::error_code ec;
        for (fs::directory_iterator it(directory_ / "journals", ec), end; !ec && it != end; it.increment(ec)) {
            if (stop_.stop_requested()) return;
            const auto& lock = it->path();
            const auto name = lock.filename().string();
            if (!name.ends_with(".jsonl.lock")) continue;
            if (remove_stale_lock(lock, lock.parent_path() / name.substr(0, name.size() - kLockSuffix.size()), now_))
                ++report_.leftovers;
        }
    }

    [[nodiscard]] bool stopped() const { return stop_.stop_requested(); }
    RetentionReport take() && { return std::move(report_); }

private:
    template <typename Visit>
    void for_each_directory(const fs::path& parent, Visit&& visit) {
        std::error_code ec;
        for (fs::directory_iterator it(parent, ec), end; !ec && it != end; it.increment(ec)) {
            if (stop_.stop_requested()) return;
            std::error_code type_ec;
            if (it->is_symlink(type_ec) || !it->is_directory(type_ec)) continue;
            const auto id = it->path().filename().string();
            if (storage_id(id)) visit(it->path(), id);
        }
    }

    void fail(std::string_view id, std::string_view error) {
        report_.errors.push_back(std::format("{}: {}", id, error.empty() ? "not removed" : error));
    }

    const SessionStore& store_;
    fs::path directory_;
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
    Pass pass(store, now, std::move(stop));
    if (policy.sessions) pass.sessions(*policy.sessions);
    if (policy.revert_history && !pass.stopped()) pass.revert_history(*policy.revert_history);
    if (policy.sessions && !pass.stopped()) pass.leftovers(*policy.sessions);
    if (!pass.stopped()) pass.stale_locks();
    return std::move(pass).take();
}

} // namespace core::session
