#include "RecoveryStore.hpp"

#include "../utils/Crypto.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/Uuid.hpp"

#include <cctype>
#include <chrono>
#include <format>

#include <simdjson.h>
#include <algorithm>
#include <limits>
#include <optional>
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
} // namespace

RecoveryStore::RecoveryStore(fs::path directory)
    : directory_(normalized(directory)), private_roots_{directory_} {
    const auto root = directory_.parent_path().parent_path();
    if (root.filename() == "recovery") {
        private_roots_.push_back(root);
        private_roots_.push_back(root.parent_path() / "journals");
    }
}

bool RecoveryStore::is_internal(const fs::path& path) const {
    return std::ranges::any_of(private_roots_, [&](const auto& root) {
        return core::utils::path::relative_within(path, root).has_value();
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
    }
    return snapshot;
}

void RecoveryStore::put_original(const RecoverySnapshot& snapshot) const {
    if (snapshot.image.hash.empty()) return;
    std::string error;
    if (!core::utils::atomic_write_file(directory_ / "objects" / snapshot.image.hash, snapshot.bytes, &error))
        throw std::runtime_error("Cannot persist recovery contents: " + error);
}

std::expected<std::string, std::string> RecoveryStore::original(const RecoveryImage& image) const {
    if (image.hash.empty()) return std::string{};
    auto snapshot = read(directory_ / "objects" / image.hash);
    if (!snapshot || snapshot->image.hash != image.hash)
        return std::unexpected("Recovery contents failed integrity verification.");
    return std::move(snapshot->bytes);
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
    std::vector<fs::path> completed;
    std::error_code ec;
    for (fs::directory_iterator it(directory_.parent_path(), ec), end; !ec && it != end; it.increment(ec)) {
        if (it->path() == directory_ || !it->is_directory(ec) || it->is_symlink(ec)) continue;
        auto old = RecoveryStore(it->path()).load(workspace);
        if (old && old->phase != RecoveryPhase::Recording) completed.push_back(it->path());
    }
    std::ranges::sort(completed, std::greater{});
    for (std::size_t i = kRetainedTurns - 1; i < completed.size(); ++i) fs::remove_all(completed[i], ec);
}

} // namespace core::changes
