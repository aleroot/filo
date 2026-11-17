#pragma once

#include "ObjectStore.hpp"
#include "RecoveryManifest.hpp"
#include "../persistence/AmendedFile.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core::changes {

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
    /// Where the revert history of the sessions saved in @p sessions_directory lives.
    [[nodiscard]] static std::filesystem::path root_in(const std::filesystem::path& sessions_directory);
    [[nodiscard]] static std::filesystem::path session_root_in(
        const std::filesystem::path& sessions_directory, std::string_view session_id);
    [[nodiscard]] static std::filesystem::path new_directory(const std::filesystem::path& root);
    [[nodiscard]] bool is_internal(const std::filesystem::path& path) const;
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
    [[nodiscard]] static std::expected<RecoverySnapshot, std::string> read(const std::filesystem::path& path);
    /// When a save must be on disk: before it returns (`Now`), or with the
    /// next save that is (`Later`). A save of a Recording manifest that only
    /// states what a tool did can wait: recovering a turn still recording
    /// reads every file again, while each later durable save of the same file
    /// makes what came before it durable too.
    enum class Sync : bool { Later, Now };
    /// Commits `manifest`. The first save of a store writes it whole; later
    /// ones record only what changed (see core::persistence::AmendedFile).
    void save(const RecoveryManifest& manifest, Sync sync = Sync::Now) const;
    // An independent marker avoids waiting on a turn's inventory/checkpoint
    // mutex when another writer invalidates ownership of its observations.
    void mark_ambiguous() const;
    [[nodiscard]] std::expected<RecoveryManifest, std::string> load(const std::filesystem::path& workspace) const;
    /// The image of the file at `path`, hashed a chunk at a time: its bytes are
    /// never held whole.
    [[nodiscard]] static std::expected<RecoverySnapshot, std::string> identify(const std::filesystem::path& path);
    /// Puts the contents of the file at `path` in the session's ObjectStore,
    /// which may link them from another session's. Save a Recording manifest
    /// first: until a manifest lists an object, only that phase keeps
    /// `retain` from sweeping it. See ObjectStore::store for failures.
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
    /// Where this turn's objects are read from: the session's store, or for
    /// turns recorded before it existed, their own.
    [[nodiscard]] ObjectStore readable_objects() const;
    std::filesystem::path directory_;
    ObjectStore objects_;  ///< Shared by every turn of the session.
    /// manifest.json, amended by the changes of each later save.
    mutable core::persistence::AmendedFile manifest_file_;
    /// The manifest as last committed; the base of the next change.
    mutable std::optional<RecoveryManifest> committed_;
    std::vector<std::filesystem::path> private_roots_;
};

} // namespace core::changes
