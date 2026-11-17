#include "ObjectStore.hpp"

#include "../utils/Crypto.hpp"
#include "../utils/FileClone.hpp"
#include "../utils/Hex.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/UniqueFd.hpp"
#include "../utils/Uuid.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <optional>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace core::changes {
namespace fs = std::filesystem;

namespace {

[[noreturn]] void store_failed() {
    throw std::runtime_error("Cannot persist recovery contents: " + std::string(std::strerror(errno)));
}

/// Hashes everything @p reader yields, handing each chunk to @p sink first.
template <typename Sink>
std::expected<std::string, std::string> hash_stream(core::utils::RegularFileReader& reader, Sink&& sink) {
    core::utils::crypto::Sha256 hash;
    for (;;) {
        const auto chunk = reader.next();
        if (!chunk) return std::unexpected(chunk.error());
        if (chunk->empty()) break;
        sink(*chunk);
        hash.update(std::as_bytes(*chunk));
    }
    if (auto done = reader.finish(); !done) return std::unexpected(done.error());
    return core::utils::hex_encode(hash.finish());
}

/// A scratch file that is removed unless kept.
struct Scratch {
    const fs::path& path;
    bool kept = false;
    ~Scratch() {
        if (!kept) ::unlink(path.c_str());
    }
};

/// Makes the finished scratch file private and durable, then the object.
bool commit(const fs::path& scratch, int fd, const fs::path& object) {
    if (::fchmod(fd, 0600) != 0 || ::fsync(fd) != 0) return false;
    return ::rename(scratch.c_str(), object.c_str()) == 0;
}

/// Shares the object a peer store holds for the same contents: objects are
/// immutable and named by their bytes, so a hard link is a copy that costs
/// nothing, and each store's sweep still removes only its own name.
class PeerLink final : public ObjectStore::Placement {
public:
    explicit PeerLink(ObjectStore::Peers peers) : peers_(std::move(peers)) {}

    bool place(const fs::path&, const RecoverySnapshot& named, const fs::path& object, const fs::path&) override {
        if (!stores_) stores_ = peers_ ? peers_() : std::vector<fs::path>{};
        for (const auto& store : *stores_) {
            const auto shared = store / named.image.hash;
            const auto stamp = core::utils::FileStamp::of(shared);
            if (!stamp.regular() || stamp.size != named.stamp.size) continue;
            if (::link(shared.c_str(), object.c_str()) == 0 || errno == EEXIST) return true;
        }
        return false;
    }

private:
    ObjectStore::Peers peers_;
    std::optional<std::vector<fs::path>> stores_;
};

/// A copy-on-write clone shares the source's blocks until either changes.
class CopyOnWriteClone final : public ObjectStore::Placement {
public:
    bool place(const fs::path& source, const RecoverySnapshot& named, const fs::path& object,
               const fs::path& scratch) override {
        if (unsupported_) return false;
        if (const auto cloned = core::utils::clone_file(source, scratch); !cloned) {
            unsupported_ = core::utils::clone_unsupported(cloned.error());
            return false;
        }
        Scratch guard{scratch};
        // The clone is private, so what it holds now it holds for good: it is
        // kept only if it holds the bytes that were named.
        const auto cloned = ObjectStore::identify(scratch, named.stamp.size);
        if (!cloned || cloned->image.hash != named.image.hash) return false;
        const core::utils::UniqueFd fd(::open(scratch.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        if (!fd || !commit(scratch, fd.get(), object)) return false;
        guard.kept = true;
        return true;
    }

private:
    bool unsupported_ = false;  ///< This filesystem never clones: stop trying.
};

/// Writes the bytes out, hashing them on the way: always possible.
class StreamingCopy final : public ObjectStore::Placement {
public:
    bool place(const fs::path& source, const RecoverySnapshot& named, const fs::path& object,
               const fs::path& scratch) override {
        auto reader = core::utils::RegularFileReader::open(source, named.stamp.size);
        if (!reader || !reader->stamp().exists()) return false;
        const core::utils::UniqueFd fd(::open(scratch.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600));
        if (!fd) store_failed();
        Scratch guard{scratch};
        const auto hash = hash_stream(*reader, [&](std::span<const char> chunk) {
            for (auto rest = chunk; !rest.empty();) {
                const auto written = ::write(fd.get(), rest.data(), rest.size());
                if (written < 0 && errno == EINTR) continue;
                if (written <= 0) store_failed();
                rest = rest.subspan(static_cast<std::size_t>(written));
            }
        });
        // Changed since it was named: these are not the bytes asked for.
        if (!hash || *hash != named.image.hash) return false;
        if (!commit(scratch, fd.get(), object)) store_failed();
        guard.kept = true;
        return true;
    }
};

} // namespace

bool is_content_hash(std::string_view hash) noexcept {
    return hash.empty() || (hash.size() == 64 && std::ranges::all_of(hash, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
}

ObjectStore::ObjectStore(fs::path directory, Peers peers) : directory_(std::move(directory)) {
    placements_.push_back(std::make_unique<PeerLink>(std::move(peers)));
    placements_.push_back(std::make_unique<CopyOnWriteClone>());
    placements_.push_back(std::make_unique<StreamingCopy>());
}

ObjectStore::ObjectStore(ObjectStore&&) noexcept = default;
ObjectStore& ObjectStore::operator=(ObjectStore&&) noexcept = default;
ObjectStore::~ObjectStore() = default;

std::expected<RecoverySnapshot, std::string> ObjectStore::identify(const fs::path& path, std::size_t max_bytes) {
    auto reader = core::utils::RegularFileReader::open(path, max_bytes);
    if (!reader) return std::unexpected(reader.error());
    RecoverySnapshot snapshot{.stamp = reader->stamp()};
    if (!snapshot.stamp.exists()) return snapshot;
    auto hash = hash_stream(*reader, [](std::span<const char>) {});
    if (!hash) return std::unexpected(hash.error());
    snapshot.image = {std::move(*hash), snapshot.stamp.mode & 0777};
    return snapshot;
}

std::expected<RecoverySnapshot, std::string> ObjectStore::store(const fs::path& path, std::size_t max_bytes) const {
    auto named = identify(path, max_bytes);
    if (!named || !named->stamp.exists() || contains(named->image.hash)) return named;
    if (!prepared_) {
        fs::create_directories(directory_);
        fs::permissions(directory_, fs::perms::owner_all);
        prepared_ = true;
    }
    const auto object = directory_ / named->image.hash;
    for (const auto& placement : placements_) {
        // Not a hash, so a sweep never takes it for an object.
        const auto scratch = directory_ / std::format("tmp.{}.{}", ::getpid(), core::utils::random_uuid_v4());
        if (placement->place(path, *named, object, scratch)) return named;
    }
    return std::unexpected("The file changed while it was stored.");
}

bool ObjectStore::contains(std::string_view hash) const {
    return !hash.empty() && core::utils::FileStamp::of(directory_ / hash).regular();
}

std::expected<std::string, std::string> ObjectStore::read(std::string_view hash, std::size_t max_bytes) const {
    if (hash.empty()) return std::string{};
    auto file = core::utils::read_regular_file(directory_ / hash, max_bytes);
    if (!file || !file->permissions || core::utils::crypto::sha256_hex(file->bytes) != hash)
        return std::unexpected("Recovery contents failed integrity verification.");
    return std::move(file->bytes);
}

void ObjectStore::sync() const {
    const core::utils::UniqueFd fd(::open(directory_.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd) (void)::fsync(fd.get());
}

void ObjectStore::sweep(const std::set<std::string_view>& referenced) const {
    std::error_code ec;
    for (fs::directory_iterator it(directory_, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        if (name.size() == 64 && is_content_hash(name) && !referenced.contains(name)) {
            std::error_code ignored;
            fs::remove(it->path(), ignored);
        }
    }
}

} // namespace core::changes
