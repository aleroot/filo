#include "AmendedFile.hpp"

#include "../utils/Crypto.hpp"
#include "../utils/FileStamp.hpp"
#include "../utils/InterprocessFile.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace core::persistence {
namespace fs = std::filesystem;

namespace {

/// A log opens with this and the hex SHA-256 of the snapshot it amends.
constexpr std::string_view kMagic = "FILOLOG1";
constexpr std::size_t kDigestBytes = 64;
constexpr std::size_t kHeaderBytes = kMagic.size() + kDigestBytes;
/// Each record: its length and CRC-32, both little-endian, then its bytes.
constexpr std::size_t kFrameBytes = 8;

std::string header(std::string_view digest) {
    std::string out(kMagic);
    out += digest;
    return out;
}

std::uint32_t crc_of(std::string_view bytes) noexcept {
    return static_cast<std::uint32_t>(
        ::crc32_z(0, reinterpret_cast<const Bytef*>(bytes.data()), bytes.size()));
}

void put_u32(std::string& out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) out.push_back(static_cast<char>((value >> shift) & 0xff));
}

std::uint32_t get_u32(std::string_view in) noexcept {
    std::uint32_t value = 0;
    for (int i = 3; i >= 0; --i) value = (value << 8) | static_cast<unsigned char>(in[static_cast<std::size_t>(i)]);
    return value;
}

/// The first bytes of the log at @p path, as many as a header holds.
std::string header_on_disk(const fs::path& path) {
    std::array<char, kHeaderBytes> bytes{};
    const core::utils::UniqueFd fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    if (!fd) return {};
    const auto count = ::pread(fd.get(), bytes.data(), bytes.size(), 0);
    return count > 0 ? std::string(bytes.data(), static_cast<std::size_t>(count)) : std::string{};
}

/// Persists the directory entry of a file just created there; best effort,
/// as some filesystems refuse to sync a directory.
void sync_directory(const fs::path& directory) noexcept {
    const core::utils::UniqueFd fd(::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
    if (fd) (void)::fsync(fd.get());
}

} // namespace

AmendedFile::AmendedFile(fs::path snapshot, std::uint64_t min_log_bytes)
    : snapshot_(std::move(snapshot)), log_(log_path(snapshot_)), min_log_bytes_(min_log_bytes) {}

fs::path AmendedFile::log_path(const fs::path& snapshot) {
    auto log = snapshot;
    log += ".log";
    return log;
}

std::expected<std::optional<AmendedFile::Contents>, std::string> AmendedFile::read(
    const fs::path& snapshot, std::size_t max_bytes) {
    auto file = core::utils::read_regular_file(snapshot, max_bytes);
    if (!file) return std::unexpected(file.error());
    if (!file->permissions) return std::optional<Contents>{};
    Contents contents{.snapshot = std::move(file->bytes)};
    auto& position = contents.position;
    position.snapshot_identity = file->stamp.identity;
    position.snapshot_bytes = contents.snapshot.size();
    position.digest = core::utils::crypto::sha256_hex(contents.snapshot);

    auto log = core::utils::read_regular_file(log_path(snapshot), max_bytes);
    if (!log) return std::unexpected(log.error());
    if (!log->permissions || log->bytes.empty()) {
        position.log_bytes = 0;
        return contents;
    }
    const std::string_view bytes = log->bytes;
    if (!bytes.starts_with(header(position.digest))) return contents;  // Amends another snapshot.
    std::size_t offset = kHeaderBytes;
    while (bytes.size() - offset >= kFrameBytes) {
        const auto size = get_u32(bytes.substr(offset));
        const auto crc = get_u32(bytes.substr(offset + 4));
        if (bytes.size() - offset - kFrameBytes < size) break;
        const auto record = bytes.substr(offset + kFrameBytes, size);
        if (crc_of(record) != crc) break;
        contents.records.emplace_back(record);
        offset += kFrameBytes + size;
    }
    // A torn record ends what is read; nothing may be appended after it.
    if (offset == bytes.size()) position.log_bytes = offset;
    return contents;
}

std::expected<void, std::string> AmendedFile::replace(std::string_view bytes,
                                                      std::optional<fs::perms> permissions) {
    log_fd_.reset();
    unsynced_ = entry_unsynced_ = false;
    position_.reset();
    auto digest = core::utils::crypto::sha256_hex(bytes);
    std::error_code ec;
    const bool matching_log = header_on_disk(log_) == header(digest);
    bool live_log = false;
    if (matching_log && core::utils::FileStamp::of(snapshot_).size == bytes.size()) {
        const auto current = core::utils::read_regular_file(snapshot_, bytes.size());
        if (!current) return std::unexpected(current.error());
        live_log = current->permissions.has_value() && current->bytes == bytes;
    }
    const auto remove_log = [&] {
        fs::remove(log_, ec);
        if (!ec) sync_directory(log_.parent_path());
    };
    // A stale log for the incoming bytes must not become live after rename.
    if (matching_log && !live_log) {
        remove_log();
        if (ec) return std::unexpected("Cannot remove " + log_.string() + ": " + ec.message());
    }
    std::string error;
    // If the existing snapshot matches, keep its live amendments until this
    // write succeeds. Until the log is removed the old state is still valid.
    if (!core::utils::atomic_write_file(snapshot_, bytes, &error, permissions)) return std::unexpected(error);
    if (live_log) {
        remove_log();
        if (ec) return std::unexpected("Cannot remove " + log_.string() + ": " + ec.message());
    } else {
        fs::remove(log_, ec);
    }
    position_ = Position{.snapshot_identity = core::utils::FileStamp::of(snapshot_).identity,
                         .snapshot_bytes = bytes.size(),
                         .digest = std::move(digest),
                         .log_bytes = ec ? std::nullopt : std::optional<std::uint64_t>(0)};
    return {};
}

void AmendedFile::adopt(Position position) {
    log_fd_.reset();
    unsynced_ = entry_unsynced_ = false;
    position_ = std::move(position);
}

bool AmendedFile::amend(std::string_view record, Durability durability) {
    if (!position_ || !position_->log_bytes || record.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    auto& position = *position_;
    const auto end = *position.log_bytes;
    // Another writer replaced the snapshot: the log is no longer ours to extend.
    if (core::utils::FileStamp::of(snapshot_).identity != position.snapshot_identity) return false;

    std::string frame = end == 0 ? header(position.digest) : std::string{};
    frame.reserve(frame.size() + kFrameBytes + record.size());
    put_u32(frame, static_cast<std::uint32_t>(record.size()));
    put_u32(frame, crc_of(record));
    frame += record;
    if (end + frame.size() > std::max(position.snapshot_bytes, min_log_bytes_)) return false;

    if (!log_fd_) log_fd_.reset(::open(log_.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (!log_fd_) return false;
    const int fd = log_fd_.get();
    // Only the log as it was left, still at its path, is extended.
    struct stat held{}, named{};
    if (::fstat(fd, &held) != 0 || !S_ISREG(held.st_mode) || static_cast<std::uint64_t>(held.st_size) != end ||
        ::lstat(log_.c_str(), &named) != 0 || named.st_dev != held.st_dev || named.st_ino != held.st_ino) {
        log_fd_.reset();
        return false;
    }
    const auto undo = [&] {
        (void)::ftruncate(fd, static_cast<off_t>(end));
        return false;
    };
    for (std::string_view rest = frame; !rest.empty();) {
        const auto written = ::write(fd, rest.data(), rest.size());
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return undo();
        rest.remove_prefix(static_cast<std::size_t>(written));
    }
    entry_unsynced_ |= end == 0;
    unsynced_ = true;
    if (durability == Durability::Immediate && !sync()) return undo();
    position.log_bytes = end + frame.size();
    return true;
}

bool AmendedFile::sync() {
    if (!unsynced_) return true;
    if (!log_fd_ || ::fsync(log_fd_.get()) != 0) return false;
    // A log created since the last sync needs its directory entry, too.
    if (std::exchange(entry_unsynced_, false)) sync_directory(log_.parent_path());
    unsynced_ = false;
    return true;
}

} // namespace core::persistence
