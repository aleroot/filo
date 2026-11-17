#include "RecoveryManifest.hpp"

#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <limits>
#include <ranges>
#include <tuple>
#include <utility>

namespace core::changes::manifest_json {
namespace fs = std::filesystem;
using core::utils::path::normalized;

namespace {

constexpr int kVersion = 1;

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

void write_paths(core::utils::JsonWriter& json, std::string_view key, const auto& paths) {
    json.key(key);
    auto array = json.array();
    bool first = true;
    for (const auto& path : paths) {
        if (!first) json.comma();
        first = false;
        json.str(fs::path(path).string());
    }
}

void write_groups(core::utils::JsonWriter& json, const std::vector<int>& groups) {
    json.key("groups");
    auto array = json.array();
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (i) json.comma();
        json.raw(std::to_string(groups[i]));
    }
}

void write_entry(core::utils::JsonWriter& json, const fs::path& path, const RecoveryEntry& entry) {
    auto item = json.object();
    json.kv_str("path", path.string()).comma();
    json.kv_str("before", entry.before.hash).comma().kv_num("mode", entry.before.mode).comma();
    json.kv_str("after", entry.after.hash).comma().kv_num("after_mode", entry.after.mode).comma();
    json.kv_bool("after_known", entry.after_known);
    if (entry.before_stamp) json.comma().kv_num("stamp", entry.before_stamp);
    if (entry.after_stamp) json.comma().kv_num("after_stamp", entry.after_stamp);
}

void write_state(core::utils::JsonWriter& json, const RecoveryManifest& manifest) {
    json.kv_str("phase", phase_name(manifest.phase)).comma().kv_bool("partial", manifest.partial).comma();
    json.kv_bool("ambiguous_ownership", manifest.ambiguous_ownership).comma();
    json.kv_bool("inventory_complete", manifest.inventory_complete);
}

std::expected<void, std::string> read_paths(simdjson::dom::element doc, std::string_view key,
                                            std::size_t limit, std::string_view invalid, auto&& add) {
    simdjson::dom::array items;
    if (doc[key].get(items)) return {};
    std::size_t count = 0;
    for (auto item : items) {
        std::string_view path;
        if (item.get(path) || !fs::path(path).is_absolute() || ++count > limit)
            return std::unexpected(std::string(invalid));
        add(path);
    }
    return {};
}

/// Reads what `write_state` and the lists wrote. A key that is absent leaves
/// the field as it was: a change names only what changed.
std::expected<void, std::string> read_state(simdjson::dom::element doc, RecoveryManifest& result,
                                            bool amend, Limits limits) {
    std::string_view phase;
    if (!doc["phase"].get(phase)) {
        const auto parsed = parse_phase(phase);
        if (!parsed) return std::unexpected("Invalid recovery phase.");
        result.phase = *parsed;
    }
    std::ignore = doc["partial"].get(result.partial);
    std::ignore = doc["ambiguous_ownership"].get(result.ambiguous_ownership);
    std::ignore = doc["inventory_complete"].get(result.inventory_complete);
    if (simdjson::dom::array present; !doc["uncaptured"].get(present)) result.uncaptured.clear();
    if (auto read = read_paths(doc, "uncaptured", limits.files, "Invalid uncaptured recovery path.",
                               [&](std::string_view path) { result.uncaptured.emplace(path); }); !read)
        return read;
    if (simdjson::dom::array present; !doc["roots"].get(present)) result.roots.clear();
    if (auto read = read_paths(doc, "roots", limits.roots, "Invalid workspace inventory root.",
                               [&](std::string_view path) { result.roots.emplace_back(path); }); !read)
        return read;
    simdjson::dom::array files;
    if (!doc["files"].get(files)) for (auto file : files) {
        RecoveryEntry entry;
        std::string_view path, before, after;
        uint64_t mode = 0, after_mode = 0;
        if (file["path"].get(path) || file["before"].get(before) || file["after"].get(after) ||
            file["mode"].get(mode) || file["after_mode"].get(after_mode) ||
            file["after_known"].get(entry.after_known) || !fs::path(path).is_absolute() ||
            !is_content_hash(before) || !is_content_hash(after) || mode > 0777 || after_mode > 0777)
            return std::unexpected("Invalid recovery file entry.");
        entry.before = {std::string(before), static_cast<unsigned>(mode)};
        entry.after = {std::string(after), static_cast<unsigned>(after_mode)};
        std::ignore = file["stamp"].get(entry.before_stamp);  // Optional: older manifests have none.
        std::ignore = file["after_stamp"].get(entry.after_stamp);
        if (!amend && result.entries.contains(fs::path(path))) return std::unexpected("Duplicate recovery path.");
        result.entries.insert_or_assign(fs::path(path), std::move(entry));
        if (result.entries.size() > limits.files) return std::unexpected("Invalid recovery file entry.");
    }
    if (auto read = read_paths(doc, "removed", limits.files, "Invalid removed recovery path.",
                               [&](std::string_view path) { result.entries.erase(fs::path(path)); }); !read)
        return read;
    simdjson::dom::array groups;
    if (!doc["groups"].get(groups)) {
        result.groups.clear();
        for (auto group : groups) {
            int64_t id = 0;
            if (group.get(id) || id <= 1 || id > std::numeric_limits<int>::max() || result.groups.size() >= limits.files)
                return std::unexpected("Invalid recovery process group.");
            result.groups.push_back(static_cast<int>(id));
        }
    }
    return {};
}

} // namespace

std::string encode(const RecoveryManifest& manifest) {
    core::utils::JsonWriter json;
    {
        auto object = json.object();
        json.kv_num("version", kVersion).comma().kv_str("workspace", manifest.workspace.string()).comma();
        write_state(json, manifest);
        json.comma();
        write_paths(json, "roots", manifest.roots);
        json.comma();
        write_paths(json, "uncaptured", manifest.uncaptured);
        json.comma();
        write_groups(json, manifest.groups);
        json.comma().key("files");
        auto array = json.array();
        bool first = true;
        for (const auto& [path, entry] : manifest.entries) {
            if (!first) json.comma();
            first = false;
            write_entry(json, path, entry);
        }
    }
    return std::move(json).take();
}

std::optional<std::string> encode_change(const RecoveryManifest& from, const RecoveryManifest& to) {
    core::utils::JsonWriter json;
    bool changed = from.phase != to.phase || from.partial != to.partial ||
                   from.ambiguous_ownership != to.ambiguous_ownership ||
                   from.inventory_complete != to.inventory_complete;
    {
        auto object = json.object();
        write_state(json, to);
        if (from.roots != to.roots) {
            json.comma();
            write_paths(json, "roots", to.roots);
            changed = true;
        }
        if (from.uncaptured != to.uncaptured) {
            json.comma();
            write_paths(json, "uncaptured", to.uncaptured);
            changed = true;
        }
        if (from.groups != to.groups) {
            json.comma();
            write_groups(json, to.groups);
            changed = true;
        }
        json.comma().key("files");
        {
            auto array = json.array();
            bool first = true;
            for (const auto& [path, entry] : to.entries) {
                const auto known = from.entries.find(path);
                if (known != from.entries.end() && known->second == entry) continue;
                if (!first) json.comma();
                first = false;
                changed = true;
                write_entry(json, path, entry);
            }
        }
        std::vector<fs::path> removed;
        for (const auto& path : from.entries | std::views::keys)
            if (!to.entries.contains(path)) removed.push_back(path);
        if (!removed.empty()) {
            json.comma();
            write_paths(json, "removed", removed);
            changed = true;
        }
    }
    if (!changed) return std::nullopt;
    return std::move(json).take();
}

std::expected<RecoveryManifest, std::string> decode(std::string_view json, const fs::path& workspace,
                                                    Limits limits) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc)) return std::unexpected("Invalid recovery manifest.");
    int64_t version = 0;
    std::string_view root, phase;
    simdjson::dom::array files;
    if (doc["version"].get(version) || version != kVersion || doc["workspace"].get(root) ||
        normalized(fs::path(root)) != normalized(workspace) || doc["phase"].get(phase) ||
        doc["files"].get(files)) return std::unexpected("Recovery manifest does not match this workspace.");
    RecoveryManifest result{.workspace = normalized(workspace)};
    if (auto read = read_state(doc, result, false, limits); !read) return std::unexpected(read.error());
    return result;
}

std::expected<void, std::string> apply_change(std::string_view json, RecoveryManifest& manifest, Limits limits) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json.data(), json.size()).get(doc)) return std::unexpected("Invalid recovery manifest change.");
    return read_state(doc, manifest, true, limits);
}

} // namespace core::changes::manifest_json
