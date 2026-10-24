#pragma once

#include "../utils/FileStamp.hpp"
#include "../utils/PathUtils.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core::changes {

struct RecoveryImage {
    std::string hash; // Empty means absent; empty files have a SHA-256 hash.
    unsigned mode = 0;
    bool operator==(const RecoveryImage&) const = default;
};

struct RecoverySnapshot {
    RecoveryImage image;
    std::string bytes;
    core::utils::FileStamp stamp; // Of the file these bytes were read from.
};

struct RecoveryEntry {
    RecoveryImage before;
    RecoveryImage after;
    bool after_known = false;
    /// FileStamp::settled_identity of the file when it held `before`/`after`;
    /// 0 when unknown. A file still carrying it holds those bytes, so it is
    /// neither read nor hashed again — in this turn or the next.
    std::uint64_t before_stamp = 0;
    std::uint64_t after_stamp = 0;

    /// The image the file was last seen to hold, and its stamp then.
    [[nodiscard]] const RecoveryImage& latest() const noexcept { return after_known ? after : before; }
    [[nodiscard]] std::uint64_t latest_stamp() const noexcept { return after_known ? after_stamp : before_stamp; }
};

enum class RecoveryPhase { Recording, Complete, Interrupted, Reverted };

using RecoveryEntries = std::map<std::filesystem::path, RecoveryEntry, core::utils::path::NativeLess>;

struct RecoveryManifest {
    std::filesystem::path workspace;
    RecoveryPhase phase = RecoveryPhase::Recording;
    bool partial = false;
    bool ambiguous_ownership = false;
    bool inventory_complete = false;
    std::vector<std::filesystem::path> roots;
    std::set<std::filesystem::path, core::utils::path::NativeLess> uncaptured;
    std::vector<int> groups;
    RecoveryEntries entries;
};

// Filesystem persistence only. Contents live in one content-addressed object
// store per session, next to its turn directories, so a file a turn did not
// change costs a manifest line, not a copy: each object is written once, and
// `retain` removes it with the last turn that refers to it. No tool
// execution, writer coordination or diff presentation; loading always
// validates the durable schema and workspace.
class RecoveryStore final {
public:
    static constexpr std::size_t kMaxFileBytes = 8 * 1024 * 1024;
    static constexpr std::size_t kMaxFiles = 4096;
    static constexpr std::size_t kRetainedTurns = 8;

    explicit RecoveryStore(std::filesystem::path directory);
    /// The turn directories under a session root, oldest first: they are
    /// named by creation time.
    [[nodiscard]] static std::expected<std::vector<std::filesystem::path>, std::string> turn_directories(
        const std::filesystem::path& session_root);
    /// Whether this turn's contents live in the session's shared object store.
    /// Turns recorded before it existed keep theirs inside the turn directory.
    [[nodiscard]] bool shares_objects() const;
    [[nodiscard]] static std::filesystem::path session_root(
        const std::filesystem::path& journal, std::string_view session_id);
    [[nodiscard]] static std::filesystem::path session_root_in(
        const std::filesystem::path& sessions_directory, std::string_view session_id);
    [[nodiscard]] static std::filesystem::path new_directory(const std::filesystem::path& root);
    [[nodiscard]] bool is_internal(const std::filesystem::path& path) const;
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
    [[nodiscard]] static std::expected<RecoverySnapshot, std::string> read(const std::filesystem::path& path);
    void save(const RecoveryManifest& manifest) const;
    // An independent marker avoids waiting on a turn's inventory/checkpoint
    // mutex when another writer invalidates ownership of its observations.
    void mark_ambiguous() const;
    [[nodiscard]] std::expected<RecoveryManifest, std::string> load(const std::filesystem::path& workspace) const;
    /// The image of the file at `path`, hashed a chunk at a time: its bytes are
    /// never held whole.
    [[nodiscard]] static std::expected<RecoverySnapshot, std::string> identify(const std::filesystem::path& path);
    /// Copies the file at `path` into the object store while hashing it, a
    /// chunk at a time. Save a Recording manifest first: until a manifest
    /// lists an object, only that phase keeps `retain` from sweeping it.
    /// Absent files succeed with an empty image. Fails, and stores nothing,
    /// when the file is unreadable, larger than `max_bytes` or changes during
    /// the copy; throws when the store cannot be written.
    [[nodiscard]] std::expected<RecoverySnapshot, std::string> store_original(
        const std::filesystem::path& path, std::size_t max_bytes) const;
    /// Makes the objects stored so far durable: one directory sync per batch.
    void sync_objects() const;
    [[nodiscard]] std::expected<std::string, std::string> original(const RecoveryImage& image) const;
    /// Whether this turn's store holds the bytes of `image`. Only
    /// `store_original` writes objects: an image `identify` produced has none,
    /// and neither has an absent file's empty image.
    [[nodiscard]] bool has_object(const RecoveryImage& image) const;
    void acknowledge() const;
    void retain(const std::filesystem::path& workspace) const;
private:
    [[nodiscard]] std::filesystem::path object_path(std::string_view hash) const;
    /// Removes every object no manifest in `referenced` names.
    void sweep_objects(const std::set<std::string_view>& referenced) const;
    std::filesystem::path directory_;
    std::filesystem::path objects_;  ///< Shared by every turn of the session.
    mutable bool prepared_ = false;  ///< The object store exists.
    std::vector<std::filesystem::path> private_roots_;
};

} // namespace core::changes
