#include "RecoveryStore.hpp"

#include "../utils/Crypto.hpp"
#include "../utils/Hex.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/Uuid.hpp"

#include <cctype>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <format>

#include <simdjson.h>
#include <algorithm>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace core::changes {
namespace fs = std::filesystem;
using core::utils::path::normalized;
namespace {
bool valid_hash(std::string_view hash) {
    return hash.empty() || (hash.size() == 64 && std::ranges::all_of(hash, [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
}

std::string_view phase_name(RecoveryPhase phase) {
    switch (phase) {
        case RecoveryPhase::Recording: return "recording";
        case RecoveryPhase::Complete: return "complete";
        case RecoveryPhase::Interrupted: return "interrupted";
        case RecoveryPhase::Reverted: return "reverted";
    }
    std::unreachable();
}
std::optional<RecoveryPhase> parse_phase(std::string_view name) {
    for (auto phase : {RecoveryPhase::Recording, RecoveryPhase::Complete,
                       RecoveryPhase::Interrupted, RecoveryPhase::Reverted}) {
        if (phase_name(phase) == name) return phase;
    }
    return std::nullopt;
}

constexpr std::string_view kObjects = "objects";
} // namespace

RecoveryStore::RecoveryStore(fs::path directory)
    : directory_(normalized(directory)), objects_(directory_.parent_path() / kObjects),
      private_roots_{directory_, objects_} {
    const auto root = directory_.parent_path().parent_path();
    if (root.filename() == "recovery") {
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

fs::path RecoveryStore::object_path(std::string_view hash) const {
    return (shares_objects() ? objects_ : directory_ / kObjects) / hash;
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
    return sessions_directory / "recovery" / id;
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

namespace {

[[noreturn]] void store_failed() {
    throw std::runtime_error("Cannot persist recovery contents: " + std::string(std::strerror(errno)));
}

/// Hashes everything `reader` yields, handing each chunk to `sink` first.
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

} // namespace

std::expected<RecoverySnapshot, std::string> RecoveryStore::identify(const fs::path& path) {
    auto reader = core::utils::RegularFileReader::open(path, kMaxFileBytes);
    if (!reader) return std::unexpected(reader.error());
    RecoverySnapshot snapshot{.stamp = reader->stamp()};
    if (!snapshot.stamp.exists()) return snapshot;
    auto hash = hash_stream(*reader, [](std::span<const char>) {});
    if (!hash) return std::unexpected(hash.error());
    snapshot.image = {std::move(*hash), snapshot.stamp.mode & 0777};
    return snapshot;
}

std::expected<RecoverySnapshot, std::string> RecoveryStore::store_original(
    const fs::path& path, std::size_t max_bytes) const {
    auto source = core::utils::RegularFileReader::open(path, std::min(max_bytes, kMaxFileBytes));
    if (!source) return std::unexpected(source.error());
    RecoverySnapshot snapshot{.stamp = source->stamp()};
    if (!snapshot.stamp.exists()) return snapshot;
    if (!prepared_) {
        fs::create_directories(objects_);
        fs::permissions(objects_, fs::perms::owner_all);
        prepared_ = true;
    }
    // Not a hash, so a sweep never takes it for an object.
    const auto temp = objects_ / std::format("tmp.{}.{}", ::getpid(), core::utils::random_uuid_v4());
    struct Discard {
        const fs::path& temp;
        int fd = -1;
        bool kept = false;
        ~Discard() {
            if (fd >= 0) ::close(fd);
            if (!kept) ::unlink(temp.c_str());
        }
    } discard{temp, ::open(temp.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600)};
    int& fd = discard.fd;
    if (fd < 0) store_failed();
    // One pass: each chunk is written and hashed while it is in hand.
    auto hash = hash_stream(*source, [&](std::span<const char> chunk) {
        for (auto rest = chunk; !rest.empty();) {
            const auto n = ::write(fd, rest.data(), rest.size());
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) store_failed();
            rest = rest.subspan(static_cast<std::size_t>(n));
        }
    });
    if (!hash) return std::unexpected(hash.error());
    const bool durable = ::fsync(fd) == 0;
    if (::close(std::exchange(fd, -1)) != 0 || !durable) store_failed();
    snapshot.image = {std::move(*hash), snapshot.stamp.mode & 0777};
    // Content-addressed: replacing an existing object puts the same bytes back.
    if (::rename(temp.c_str(), (objects_ / snapshot.image.hash).c_str()) != 0) store_failed();
    discard.kept = true;
    return snapshot;
}

void RecoveryStore::sync_objects() const {
    const int fd = ::open(objects_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    (void)::fsync(fd);
    ::close(fd);
}

std::expected<std::string, std::string> RecoveryStore::original(const RecoveryImage& image) const {
    if (image.hash.empty()) return std::string{};
    auto snapshot = read(object_path(image.hash));
    if (!snapshot || snapshot->image.hash != image.hash)
        return std::unexpected("Recovery contents failed integrity verification.");
    return std::move(snapshot->bytes);
}

bool RecoveryStore::has_object(const RecoveryImage& image) const {
    return !image.hash.empty() && core::utils::FileStamp::of(object_path(image.hash)).regular();
}

void RecoveryStore::save(const RecoveryManifest& manifest) const {
    fs::create_directories(directory_);
    fs::permissions(directory_, fs::perms::owner_all);
    core::utils::JsonWriter json;
    {
        auto object = json.object();
        json.kv_num("version", 1).comma().kv_str("workspace", manifest.workspace.string()).comma();
        json.kv_str("phase", phase_name(manifest.phase)).comma().kv_bool("partial", manifest.partial).comma();
        json.kv_bool("inventory_complete", manifest.inventory_complete).comma().key("roots");
        {
            auto array = json.array();
            for (std::size_t i = 0; i < manifest.roots.size(); ++i) {
                if (i) json.comma();
                json.str(manifest.roots[i].string());
            }
        }
        json.comma();
        json.key("uncaptured");
        {
            auto array = json.array();
            bool first = true;
            for (const auto& path : manifest.uncaptured) {
                if (!first) json.comma();
                first = false;
                json.str(path.string());
            }
        }
        json.comma();
        json.key("groups");
        {
            auto array = json.array();
            for (std::size_t i = 0; i < manifest.groups.size(); ++i) {
                if (i) json.comma();
                json.raw(std::to_string(manifest.groups[i]));
            }
        }
        json.comma().key("files");
        auto array = json.array();
        bool first = true;
        for (const auto& [path, entry] : manifest.entries) {
            if (!first) json.comma();
            first = false;
            auto item = json.object();
            json.kv_str("path", path.string()).comma();
            json.kv_str("before", entry.before.hash).comma().kv_num("mode", entry.before.mode).comma();
            json.kv_str("after", entry.after.hash).comma().kv_num("after_mode", entry.after.mode).comma();
            json.kv_bool("after_known", entry.after_known);
            if (entry.before_stamp) json.comma().kv_num("stamp", entry.before_stamp);
            if (entry.after_stamp) json.comma().kv_num("after_stamp", entry.after_stamp);
        }
    }
    std::string error;
    if (!core::utils::atomic_write_file(directory_ / "manifest.json", json.view(), &error))
        throw std::runtime_error("Cannot persist recovery manifest: " + error);
}

std::expected<RecoveryManifest, std::string> RecoveryStore::load(const fs::path& workspace) const {
    const auto contents = core::utils::read_regular_file(directory_ / "manifest.json", kMaxFileBytes);
    if (!contents || contents->bytes.empty()) return std::unexpected("Recovery manifest is missing or unreadable.");
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(contents->bytes).get(doc)) return std::unexpected("Invalid recovery manifest.");
    int64_t version = 0;
    std::string_view root, phase;
    simdjson::dom::array files;
    if (doc["version"].get(version) || version != 1 || doc["workspace"].get(root) ||
        normalized(fs::path(root)) != normalized(workspace) || doc["phase"].get(phase) ||
        doc["files"].get(files)) return std::unexpected("Recovery manifest does not match this workspace.");
    RecoveryManifest result{.workspace = normalized(workspace)};
    const auto parsed_phase = parse_phase(phase);
    if (!parsed_phase) return std::unexpected("Invalid recovery phase.");
    result.phase = *parsed_phase;
    (void)doc["partial"].get(result.partial);
    simdjson::dom::array uncaptured;
    if (!doc["uncaptured"].get(uncaptured)) for (auto item : uncaptured) {
        std::string_view path;
        if (item.get(path) || !fs::path(path).is_absolute() || result.uncaptured.size() >= kMaxFiles)
            return std::unexpected("Invalid uncaptured recovery path.");
        result.uncaptured.emplace(path);
    }
    (void)doc["inventory_complete"].get(result.inventory_complete);
    simdjson::dom::array roots;
    if (!doc["roots"].get(roots)) for (auto root : roots) {
        std::string_view path;
        if (root.get(path) || !fs::path(path).is_absolute() || result.roots.size() >= 64)
            return std::unexpected("Invalid workspace inventory root.");
        result.roots.emplace_back(path);
    }
    for (auto file : files) {
        RecoveryEntry entry;
        std::string_view path, before, after;
        uint64_t mode = 0, after_mode = 0;
        if (file["path"].get(path) || file["before"].get(before) || file["after"].get(after) ||
            file["mode"].get(mode) || file["after_mode"].get(after_mode) ||
            file["after_known"].get(entry.after_known) || !fs::path(path).is_absolute() ||
            !valid_hash(before) || !valid_hash(after) || mode > 0777 || after_mode > 0777 ||
            result.entries.size() >= kMaxFiles)
            return std::unexpected("Invalid recovery file entry.");
        entry.before = {std::string(before), static_cast<unsigned>(mode)};
        entry.after = {std::string(after), static_cast<unsigned>(after_mode)};
        (void)file["stamp"].get(entry.before_stamp);  // Optional: older manifests have none.
        (void)file["after_stamp"].get(entry.after_stamp);
        if (!result.entries.emplace(fs::path(path), std::move(entry)).second)
            return std::unexpected("Duplicate recovery path.");
    }
    simdjson::dom::array groups;
    if (!doc["groups"].get(groups)) for (auto group : groups) {
        int64_t id = 0;
        if (group.get(id) || id <= 1 || id > std::numeric_limits<int>::max() || result.groups.size() >= kMaxFiles)
            return std::unexpected("Invalid recovery process group.");
        result.groups.push_back(static_cast<int>(id));
    }
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
    sweep_objects(referenced);
}

void RecoveryStore::sweep_objects(const std::set<std::string_view>& referenced) const {
    std::error_code ec;
    for (fs::directory_iterator it(objects_, ec), end; !ec && it != end; it.increment(ec)) {
        const auto name = it->path().filename().string();
        // Only finished objects: a temporary file is another writer's business.
        if (name.size() == 64 && valid_hash(name) && !referenced.contains(name)) {
            std::error_code ignored;
            fs::remove(it->path(), ignored);
        }
    }
}

} // namespace core::changes
