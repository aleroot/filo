#include "ExecutionJournal.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/JsonWriter.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace core::session {
ExecutionJournal::ExecutionJournal(std::filesystem::path root) : root_(std::move(root)) {}
std::filesystem::path ExecutionJournal::default_root() {
    return SessionStore::default_sessions_dir() / "journals";
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
    std::error_code ec;
    if (!std::filesystem::exists(target, ec)) return !ec;
    auto lock = core::utils::InterprocessFileLock::acquire(core::utils::lock_path_for(target), error);
    if (!lock) return false;
    std::filesystem::remove(target, ec);
    if (ec && error) *error = ec.message();
    return !ec;
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
    std::error_code ec;
    std::filesystem::create_directories(root_, ec);
    if (ec) return fail(ec.message());
    auto lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(target), error);
    if (!lock) return false;
    const int fd = ::open(target.c_str(), O_CREAT | O_RDWR | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return fail(std::strerror(errno));
    struct Close { int fd; ~Close() { ::close(fd); } } close{fd};
    struct stat status{};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_nlink != 1)
        return fail("Execution journal must be a regular file with one link.");
    if (::fchmod(fd, 0600) != 0) return fail(std::strerror(errno));

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
    if (::fsync(fd) != 0) return fail(std::strerror(errno));
    // Persist the directory entry too when creating a journal. Some supported
    // filesystems do not permit directory fsync, so this is best effort.
    if (status.st_size == 0) {
        const int directory = ::open(root_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
        if (directory >= 0) { (void)::fsync(directory); ::close(directory); }
    }
    return true;
}
} // namespace core::session
