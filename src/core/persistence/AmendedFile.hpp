#pragma once

#include "../utils/UniqueFd.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::persistence {

/// A file kept as a snapshot plus a log of records that amend it, so that a
/// change costs the bytes of the change instead of a rewrite of the file.
///
/// `replace` writes the snapshot whole (atomically, durably) and starts an
/// empty log; `amend` appends one record to `log_path(snapshot)`. Readers get
/// the snapshot and, in order, the records that amend it. What a record
/// means is the caller's business: this type stores bytes.
///
/// On disk the log opens with the SHA-256 of the snapshot it amends, so a log
/// left behind by a snapshot since replaced — by a crash between writing the
/// snapshot and removing the log, or by another writer — is never applied.
/// Each record carries its length and CRC-32: a record torn by a crash ends
/// the log, and nothing after it is applied.
///
/// Writers must be serialized by the caller (an InterprocessFileLock, as
/// every user holds). One instance is not thread-safe.
class AmendedFile {
public:
    enum class Durability : bool {
        Deferred,   ///< On disk with the next sync, or when the system flushes.
        Immediate,  ///< On disk before amend() returns.
    };

    /// Where a reader left the file. A writer that `adopt`s it can amend the
    /// file as long as nobody changed it since.
    struct Position {
        std::uint64_t snapshot_identity = 0;  ///< FileStamp identity.
        std::uint64_t snapshot_bytes = 0;
        std::string digest;                   ///< Of the snapshot, hex.
        /// End of the last complete record; nullopt when the log on disk is
        /// not this snapshot's or ends in a torn record, so it cannot be
        /// extended.
        std::optional<std::uint64_t> log_bytes;
    };

    struct Contents {
        std::string snapshot;
        std::vector<std::string> records;
        Position position;
    };

    /// The log may grow to the size of its snapshot, and at least to
    /// `min_log_bytes`, before amend() refuses and the caller replaces the
    /// snapshot. Every byte is then written at most about three times however
    /// long the history grows.
    static constexpr std::uint64_t kDefaultMinLogBytes = 64 * 1024;

    explicit AmendedFile(std::filesystem::path snapshot, std::uint64_t min_log_bytes = kDefaultMinLogBytes);

    /// Where the records amending @p snapshot are kept.
    [[nodiscard]] static std::filesystem::path log_path(const std::filesystem::path& snapshot);

    /// The snapshot and the records that amend it; nullopt when there is no
    /// snapshot. Each file is refused beyond @p max_bytes.
    [[nodiscard]] static std::expected<std::optional<Contents>, std::string> read(
        const std::filesystem::path& snapshot, std::size_t max_bytes);

    /// Writes @p bytes as the whole snapshot, with @p permissions when given
    /// (as atomic_write_file), and drops every record amending the old one.
    [[nodiscard]] std::expected<void, std::string> replace(
        std::string_view bytes, std::optional<std::filesystem::perms> permissions = std::nullopt);

    /// Amends the file as `read` left it.
    void adopt(Position position);

    /// Appends @p record. False, leaving the file as it was, when the caller
    /// must `replace` instead: nothing to amend was replaced or adopted, the
    /// snapshot or log changed since, the log has outgrown the snapshot, or
    /// the write failed.
    [[nodiscard]] bool amend(std::string_view record, Durability durability = Durability::Immediate);

    /// Makes every deferred record durable.
    [[nodiscard]] bool sync();

    [[nodiscard]] const std::filesystem::path& snapshot_path() const noexcept { return snapshot_; }

    /// Where the last successful `replace` or `amend` left the file, as
    /// `read` would report it. Only meaningful right after one succeeded.
    [[nodiscard]] const std::optional<Position>& position() const noexcept { return position_; }

private:
    std::filesystem::path snapshot_;
    std::filesystem::path log_;
    std::uint64_t min_log_bytes_;
    std::optional<Position> position_;
    /// The log, held open between records: closing the last writable
    /// descriptor of a file makes macOS write its dirty pages at once, which
    /// would undo Durability::Deferred.
    core::utils::UniqueFd log_fd_;
    bool unsynced_ = false;        ///< Records appended since the last sync.
    bool entry_unsynced_ = false;  ///< The log was created since the last sync.
};

} // namespace core::persistence
