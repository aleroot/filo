#include "RecoveryStore.hpp"

#include "../utils/Crypto.hpp"
#include "../utils/Gzip.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/Uuid.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace core::changes {
namespace fs = std::filesystem;
using core::utils::path::normalized;

namespace {

constexpr std::string_view kObjects = "objects";
constexpr std::string_view kManifest = "manifest.json";
constexpr manifest_json::Limits kLimits{.files = RecoveryStore::kMaxFiles};

/// The object stores of the other sessions under the recovery root that
/// holds @p session, whose objects this session's may link.
std::vector<fs::path> peer_stores(const fs::path& session) {
    std::vector<fs::path> stores;
    const auto root = session.parent_path();
    if (root.filename() != "recovery") return stores;
    std::error_code ec;
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path() != session && fs::is_directory(it->path() / kObjects, ec))
            stores.push_back(it->path() / kObjects);
    }
    return stores;
}

} // namespace

RecoveryStore::RecoveryStore(fs::path directory)
    : directory_(normalized(directory)),
      objects_(directory_.parent_path() / kObjects, [session = directory_.parent_path()] { return peer_stores(session); }),
      manifest_file_(directory_ / kManifest),
      private_roots_{directory_, objects_.directory()} {
    const auto root = directory_.parent_path().parent_path();
    if (root == root_in(root.parent_path())) {
        private_roots_.push_back(root);
        private_roots_.push_back(root.parent_path() / "journals");
    }
}

std::expected<std::vector<fs::path>, std::string> RecoveryStore::turn_directories(const fs::path& session_root) {
    std::vector<fs::path> turns;
    std::error_code ec;
    for (fs::directory_iterator it(session_root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path().filename() != kObjects && it->is_directory(ec) && !it->is_symlink(ec))
            turns.push_back(normalized(it->path()));
    }
    if (ec) return std::unexpected(ec.message());
    std::ranges::sort(turns);
    return turns;
}

bool RecoveryStore::shares_objects() const {
    std::error_code ec;
    return !fs::is_directory(directory_ / kObjects, ec);
}

bool RecoveryStore::is_internal(const fs::path& path) const {
    // Callers pass normalized paths, as every root here is.
    return std::ranges::any_of(private_roots_, [&](const auto& root) {
        return core::utils::path::normalized_within(path, root);
    });
}

fs::path RecoveryStore::session_root(const fs::path& journal, std::string_view id) {
    return journal.empty() ? fs::path{} : session_root_in(journal.parent_path().parent_path(), id);
}

fs::path RecoveryStore::session_root_in(const fs::path& sessions_directory, std::string_view id) {
    if (sessions_directory.empty() || id.empty() || !std::ranges::all_of(id, [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_';
    })) return {};
    return root_in(sessions_directory) / id;
}

fs::path RecoveryStore::root_in(const fs::path& sessions_directory) {
    return sessions_directory / "recovery";
}

fs::path RecoveryStore::new_directory(const fs::path& root) {
    return root / std::format("{}-{}", std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count(), core::utils::random_uuid_v4());
}

std::expected<RecoverySnapshot, std::string> RecoveryStore::read(const fs::path& path) {
    auto file = core::utils::read_regular_file(path, kMaxFileBytes);
    if (!file) return std::unexpected(file.error());
    RecoverySnapshot snapshot;
    if (file->permissions) {
        snapshot.image = {core::utils::crypto::sha256_hex(file->bytes),
                          static_cast<unsigned>(*file->permissions)};
        snapshot.bytes = std::move(file->bytes);
        snapshot.stamp = file->stamp;
    }
    return snapshot;
}

std::expected<RecoverySnapshot, std::string> RecoveryStore::identify(const fs::path& path) {
    return ObjectStore::identify(path, kMaxFileBytes);
}

std::expected<RecoverySnapshot, std::string> RecoveryStore::store_original(
    const fs::path& path, std::size_t max_bytes) const {
    return objects_.store(path, std::min(max_bytes, kMaxFileBytes));
}

void RecoveryStore::sync_objects() const {
    objects_.sync();
}

ObjectStore RecoveryStore::readable_objects() const {
    return ObjectStore(shares_objects() ? objects_.directory() : directory_ / kObjects);
}

std::expected<std::string, std::string> RecoveryStore::original(const RecoveryImage& image) const {
    return readable_objects().read(image.hash, kMaxFileBytes);
}

bool RecoveryStore::has_object(const RecoveryImage& image) const {
    return readable_objects().contains(image.hash);
}

void RecoveryStore::mark_ambiguous() const {
    fs::create_directories(directory_);
    fs::permissions(directory_, fs::perms::owner_all);
    std::string error;
    if (!core::utils::atomic_write_file(directory_ / "ambiguous", "ambiguous\n", &error))
        throw std::runtime_error(error);
}

void RecoveryStore::save(const RecoveryManifest& manifest, Sync sync) const {
    using Durability = core::persistence::AmendedFile::Durability;
    if (fs::create_directories(directory_)) fs::permissions(directory_, fs::perms::owner_all);
    // A manifest names every file a workspace inventory found, so rewriting it
    // for each tool call wrote megabytes a call: only the change is recorded.
    if (committed_ && committed_->workspace == manifest.workspace) {
        auto change = manifest_json::encode_change(*committed_, manifest);
        if (!change && (sync == Sync::Later || manifest_file_.sync())) return;
        // Keep the existing inventory placement: compression here could
        // squeeze a large inventory into the log budget of a tiny base.
        const bool new_inventory = committed_->roots.empty() && !manifest.roots.empty();
        if (!new_inventory && change && change->size() >= 4096) {
            auto compressed = core::utils::gzip::compress(*change, {.level = 1});
            if (compressed.size() < change->size()) *change = std::move(compressed);
        }
        if (change && manifest_file_.amend(*change, sync == Sync::Now ? Durability::Immediate : Durability::Deferred)) {
            committed_ = manifest;
            return;
        }
    }
    committed_.reset();
    // Paths and hashes repeat: compressed, a manifest is about a fifth the size.
    const auto written = manifest_file_.replace(core::utils::gzip::compress(manifest_json::encode(manifest), {.level = 1}));
    if (!written) throw std::runtime_error("Cannot persist recovery manifest: " + written.error());
    committed_ = manifest;
}

std::expected<RecoveryManifest, std::string> RecoveryStore::load(const fs::path& workspace) const {
    auto contents = core::persistence::AmendedFile::read(directory_ / kManifest, kMaxFileBytes);
    if (!contents || !*contents || (*contents)->snapshot.empty())
        return std::unexpected("Recovery manifest is missing or unreadable.");
    auto& snapshot = (*contents)->snapshot;
    if (core::utils::gzip::is_compressed(snapshot)) {
        // Manifests written before compression are plain JSON.
        auto json = core::utils::gzip::decompress(snapshot, 8 * kMaxFileBytes);
        if (!json) return std::unexpected("Invalid recovery manifest.");
        snapshot = std::move(*json);
    }
    auto result = manifest_json::decode(snapshot, workspace, kLimits);
    if (!result) return result;
    for (const auto& change : (*contents)->records) {
        std::optional<std::string> json;
        if (core::utils::gzip::is_compressed(change)) {
            auto decompressed = core::utils::gzip::decompress(change, 8 * kMaxFileBytes);
            if (!decompressed) return std::unexpected("Invalid compressed recovery manifest change.");
            json = std::move(*decompressed);
        }
        if (auto applied = manifest_json::apply_change(json ? *json : change, *result, kLimits); !applied)
            return std::unexpected(applied.error());
    }
    std::error_code marker_error;
    result->ambiguous_ownership |= fs::exists(directory_ / "ambiguous", marker_error) || bool(marker_error);
    return result;
}

void RecoveryStore::acknowledge() const {
    std::error_code ec;
    if (fs::exists(directory_ / "attached", ec)) return;
    std::string error;
    if (!core::utils::atomic_write_file(directory_ / "attached", "", &error))
        throw std::runtime_error(error);
}

void RecoveryStore::retain(const fs::path& workspace) const {
    // A listing that failed proves nothing unreferenced: nothing is swept.
    auto directories = turn_directories(directory_.parent_path());
    if (!directories) return;
    std::vector<std::pair<fs::path, RecoveryManifest>> turns;
    bool all_known = true;
    for (auto& directory : *directories) {
        auto manifest = RecoveryStore(directory).load(workspace);
        // Unreadable, another workspace's, or still recording objects it does
        // not list yet: any of them may need what no manifest names.
        if (!manifest || (directory != directory_ && manifest->phase == RecoveryPhase::Recording)) all_known = false;
        if (!manifest) continue;
        turns.emplace_back(std::move(directory), std::move(*manifest));
    }
    // Newest first; this turn and running ones are kept whatever their age.
    std::ranges::reverse(turns);
    std::size_t completed = 0;
    std::error_code ec;
    std::erase_if(turns, [&](const auto& turn) {
        const auto& [directory, manifest] = turn;
        if (directory == directory_ || manifest.phase == RecoveryPhase::Recording) return false;
        if (++completed < kRetainedTurns) return false;
        fs::remove_all(directory, ec);
        return true;
    });
    if (!all_known) return;
    std::set<std::string_view> referenced;
    for (const auto& entry : turns | std::views::values
                               | std::views::transform(&RecoveryManifest::entries) | std::views::join
                               | std::views::values) {
        referenced.insert(entry.before.hash);
        referenced.insert(entry.after.hash);
    }
    objects_.sweep(referenced);
}

} // namespace core::changes
