#pragma once

#include "../utils/FileStamp.hpp"

#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
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

/// Whether @p hash names contents: empty (absent), or 64 lowercase hex digits.
[[nodiscard]] bool is_content_hash(std::string_view hash) noexcept;

/// Content-addressed file store: one directory of immutable objects, each
/// named by the SHA-256 of its bytes, so a content is stored once however
/// many files, turns or sessions hold it.
///
/// Storing names the contents first (reading costs the drive nothing) and
/// writes only when no object has them yet, through the first `Placement`
/// that succeeds, cheapest for the drive first: a hard link to a peer
/// store's object, a copy-on-write clone, and last a copy.
class ObjectStore {
public:
    /// One way of putting a file's contents in the store (a Chain of
    /// Responsibility: each declines what it cannot do).
    class Placement {
    public:
        virtual ~Placement() = default;
        /// Puts the contents of @p source, named @p named, at @p object,
        /// using @p scratch for any temporary file. False when this placement
        /// cannot; throws when the store itself cannot be written.
        [[nodiscard]] virtual bool place(const std::filesystem::path& source, const RecoverySnapshot& named,
                                         const std::filesystem::path& object,
                                         const std::filesystem::path& scratch) = 0;
    };

    /// Other stores whose objects may be linked; asked once, when first needed.
    using Peers = std::function<std::vector<std::filesystem::path>()>;

    explicit ObjectStore(std::filesystem::path directory, Peers peers = {});
    ObjectStore(ObjectStore&&) noexcept;
    ObjectStore& operator=(ObjectStore&&) noexcept;
    ~ObjectStore();

    /// The image of the file at @p path, hashed a chunk at a time: its bytes
    /// are never held whole. Absent files succeed with an empty image.
    [[nodiscard]] static std::expected<RecoverySnapshot, std::string> identify(
        const std::filesystem::path& path, std::size_t max_bytes);

    /// Stores the contents of the file at @p path. Absent files succeed with
    /// an empty image and store nothing. Fails, storing nothing, when the
    /// file is unreadable, larger than @p max_bytes or changes while it is
    /// stored; throws when the store cannot be written.
    [[nodiscard]] std::expected<RecoverySnapshot, std::string> store(
        const std::filesystem::path& path, std::size_t max_bytes) const;

    [[nodiscard]] bool contains(std::string_view hash) const;
    /// The bytes of object @p hash, verified against it.
    [[nodiscard]] std::expected<std::string, std::string> read(std::string_view hash, std::size_t max_bytes) const;
    /// Makes the objects stored so far durable: one directory sync per batch.
    void sync() const;
    /// Removes every object @p referenced does not name. Temporary files are
    /// another writer's in-progress work and are left alone.
    void sweep(const std::set<std::string_view>& referenced) const;
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }

private:
    std::filesystem::path directory_;
    mutable bool prepared_ = false;  ///< The directory exists.
    std::vector<std::unique_ptr<Placement>> placements_;
};

} // namespace core::changes
