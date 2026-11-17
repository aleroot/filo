#include "ExecutionJournal.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/UniqueFd.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <unistd.h>

namespace core::session {

/// The journal file this instance last appended to, still open.
struct ExecutionJournal::Open {
    std::mutex mutex;
    std::filesystem::path path;
    core::utils::UniqueFd fd;
    void close() noexcept {
        fd.reset();
        path.clear();
    }
};

ExecutionJournal::ExecutionJournal(std::filesystem::path root)
    : root_(std::move(root)), open_(std::make_shared<Open>()) {}
std::filesystem::path ExecutionJournal::default_root() {
    return root_in(SessionStore::default_sessions_dir());
}
std::filesystem::path ExecutionJournal::root_in(const std::filesystem::path& sessions_dir) {
    return sessions_dir / "journals";
}
std::filesystem::path ExecutionJournal::path_for(std::string_view id) const {
    if (id.empty() || id.size() > 128 || !std::ranges::all_of(id, [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
            || (c >= '0' && c <= '9') || c == '-' || c == '_';
    })) return {};
    return root_ / (std::string(id) + ".jsonl");
}

bool ExecutionJournal::remove(std::string_view id, std::string* error) const {
    const auto target = path_for(id);
    // No journal can exist under an ID that cannot name one, so there is
    // nothing to remove; failing here would block deleting the session itself.
    if (target.empty()) return true;
    {
        std::lock_guard guard(open_->mutex);
        if (open_->path == target) open_->close();
    }
    std::error_code ec;
    if (!std::filesystem::exists(target, ec)) return !ec;
    const auto lock_path = core::utils::lock_path_for(target);
    auto lock = core::utils::InterprocessFileLock::acquire(lock_path, error);
    if (!lock) return false;
    std::filesystem::remove(target, ec);
    if (ec) {
        if (error) *error = ec.message();
        return false;
    }
    std::filesystem::remove(lock_path, ec);  // Still held, like the session's.
    return true;
}

bool ExecutionJournal::append(std::string_view id, std::string_view actor,
                              std::string_view parent, std::string_view event,
                              std::string_view payload, std::string* error) const {
    const auto fail = [&](std::string message) {
        if (error) *error = std::move(message);
        return false;
    };
    const auto target = path_for(id);
    if (target.empty()) return fail("Invalid execution journal session ID.");
    std::lock_guard guard(open_->mutex);
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    if (ec) return fail(ec.message());
    auto lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(target), error);
    if (!lock) return false;
    struct stat status{};
    if (open_->path == target) {
        // Another writer may have removed or replaced the file since: only a
        // descriptor that is still the file at the path is written through.
        struct stat named{};
        if (::fstat(open_->fd.get(), &status) != 0 || ::lstat(target.c_str(), &named) != 0 ||
            named.st_dev != status.st_dev || named.st_ino != status.st_ino)
            open_->close();
    }
    if (open_->path != target) {
        open_->close();
        const int fd = ::open(target.c_str(), O_CREAT | O_RDWR | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd < 0) return fail(std::strerror(errno));
        open_->fd.reset(fd);
        open_->path = target;
        if (::fchmod(fd, 0600) != 0) {
            open_->close();
            return fail(std::strerror(errno));
        }
    }
    const int fd = open_->fd.get();
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_nlink != 1) {
        open_->close();
        return fail("Execution journal must be a regular file with one link.");
    }

    // Newlines only occur between records: JSON strings escape them. Recover
    // a torn final record under the same lock as the append.
    off_t end = status.st_size;
    char block[4096];
    while (end > 0) {
        const off_t start = std::max<off_t>(0, end - static_cast<off_t>(sizeof(block)));
        const auto count = ::pread(fd, block, static_cast<std::size_t>(end - start), start);
        if (count != end - start) return fail("Cannot inspect execution journal tail.");
        bool found = false;
        for (ssize_t i = count; i > 0; --i) {
            if (block[i - 1] == '\n') { end = start + i; found = true; break; }
        }
        if (found) break;
        end = start;
    }
    if (end != status.st_size && ::ftruncate(fd, end) != 0)
        return fail(std::strerror(errno));
    core::utils::JsonWriter writer;
    {
        auto object = writer.object();
        writer.kv_num("version", kVersion).comma();
        writer.kv_num("offset", end).comma();
        writer.kv_str("session_id", id).comma();
        writer.kv_str("recorded_at", SessionStore::now_iso8601()).comma();
        writer.kv_num("recorded_at_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()).comma();
        writer.kv_str("actor", actor).comma();
        writer.kv_str("parent_tool_call_id", parent).comma();
        writer.kv_str("event", event).comma();
        writer.kv_raw("data", payload);
    }
    std::string line = std::move(writer).take();
    line += '\n';
    std::size_t written = 0;
    while (written < line.size()) {
        const auto count = ::write(fd, line.data() + written, line.size() - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            const int saved_errno = errno;
            (void)::ftruncate(fd, end);
            return fail(std::strerror(saved_errno));
        }
        written += static_cast<std::size_t>(count);
    }
    // No fsync: a record is in the page cache once written, so only a power
    // loss or kernel panic can drop the tail, and the tail repair above copes
    // with that. Syncing every small append rewrote a block per record — the
    // journal wrote about seven times its own size.
    return true;
}
} // namespace core::session
