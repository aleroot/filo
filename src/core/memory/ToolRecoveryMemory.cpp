#include "ToolRecoveryMemory.hpp"
#include "MemoryStore.hpp"

#include "../logging/Logger.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <utility>

namespace core::memory {
namespace {

using Element = simdjson::dom::element;
using Object = simdjson::dom::object;

[[nodiscard]] std::filesystem::path config_directory() {
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path{xdg} / "filo";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path{home} / ".config" / "filo";
    }
    return std::filesystem::temp_directory_path() / "filo";
}

/// Serializes a lesson transaction. Acquiring the lock also serializes threads
/// inside this process, because InterprocessFileLock holds a process-local
/// mutex keyed by the lock path in addition to flock(2).
[[nodiscard]] std::optional<core::utils::InterprocessFileLock> lock_store(
    const std::filesystem::path& path) {
    std::string error;
    auto lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path), &error);
    if (!lock) {
        core::logging::warn(
            "[ToolRecovery] Could not lock {}: {}", path.string(), error);
    }
    return lock;
}

} // namespace

FileToolRecoveryMemory::FileToolRecoveryMemory(std::filesystem::path path)
    : path_(std::move(path)) {}

std::filesystem::path FileToolRecoveryMemory::default_path() {
    return config_directory() / "tool-recovery.json";
}

std::string clamp_hint(std::string text) {
    if (text.size() <= kMaxHintChars) {
        return text;
    }
    // Never split a UTF-8 sequence: parameter names reach us from MCP tool
    // schemas and may contain multi-byte characters.
    std::size_t cut = kMaxHintChars;
    while (cut > 0
           && (static_cast<unsigned char>(text[cut]) & 0xC0U) == 0x80U) {
        --cut;
    }
    text.resize(cut);
    text += "…";
    return text;
}

std::optional<std::string> FileToolRecoveryMemory::recall(
    const RecoveryKey& key) {
    const auto lock = lock_store(path_);
    if (!lock) {
        return std::nullopt;
    }
    const auto loaded = load_unlocked();
    if (loaded.status != LoadStatus::Ok) {
        return std::nullopt;
    }
    const auto it = std::ranges::find_if(loaded.entries, [&](const Entry& entry) {
        return entry.key == key;
    });
    if (it == loaded.entries.end() || it->evidence < kMinEvidence) {
        return std::nullopt;
    }
    return it->hint;
}

std::vector<std::string> FileToolRecoveryMemory::recall_runtime_hints(
    const std::string& tool,
    const std::string& schema_fingerprint) {
    const auto lock = lock_store(path_);
    if (!lock) {
        return {};
    }
    const auto loaded = load_unlocked();
    if (loaded.status != LoadStatus::Ok) {
        return {};
    }

    std::vector<const Entry*> matches;
    for (const auto& entry : loaded.entries) {
        if (entry.key.tool == tool
            && entry.key.schema_fingerprint == schema_fingerprint
            && entry.key.issue_code == kRuntimeFailureCode
            && entry.evidence >= kMinEvidence) {
            matches.push_back(&entry);
        }
    }
    // Strongest evidence first, so the cap keeps the best-supported advice.
    std::ranges::stable_sort(matches, [](const Entry* a, const Entry* b) {
        return a->evidence > b->evidence;
    });
    if (matches.size() > kMaxRuntimeHints) {
        matches.resize(kMaxRuntimeHints);
    }

    std::vector<std::string> hints;
    hints.reserve(matches.size());
    for (const Entry* entry : matches) {
        hints.push_back(entry->hint);
    }
    return hints;
}

void FileToolRecoveryMemory::record(const RecoveryLesson& lesson) {
    if (lesson.key.tool.empty() || lesson.key.issue_code.empty()
        || lesson.hint.empty()) {
        return;
    }

    // Read-modify-write inside one lock: concurrent agents merge their
    // lessons instead of overwriting each other's evidence.
    const auto lock = lock_store(path_);
    if (!lock) {
        return;
    }
    auto loaded = load_unlocked();
    if (loaded.status != LoadStatus::Ok) {
        return;
    }
    auto& entries = loaded.entries;

    const auto it = std::ranges::find_if(entries, [&](const Entry& entry) {
        return entry.key == lesson.key;
    });
    const std::string now = MemoryStore::now_iso8601();
    const std::string hint = clamp_hint(lesson.hint);
    if (it != entries.end()) {
        if (it->hint == hint) {
            it->evidence = std::min(it->evidence + 1, kMaxEvidence);
        } else {
            // Conflicting evidence: the previous conclusion is no longer
            // trustworthy, so the new one restarts as a candidate.
            it->hint = hint;
            it->evidence = 1;
        }
        it->updated_at = now;
    } else {
        entries.push_back(Entry{
            .key = lesson.key,
            .hint = hint,
            .evidence = 1,
            .updated_at = now,
        });
        evict_overflow(entries);
    }
    save_unlocked(entries);
}

std::size_t FileToolRecoveryMemory::entry_count() const {
    const auto lock = lock_store(path_);
    if (!lock) {
        return 0;
    }
    const auto loaded = load_unlocked();
    if (loaded.status != LoadStatus::Ok) {
        return 0;
    }
    return loaded.entries.size();
}

FileToolRecoveryMemory::LoadedStore
FileToolRecoveryMemory::load_unlocked() const {
    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return {}; // A missing store is the normal first-run state.
    }
    const std::string json((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    if (json.empty()) {
        return {};
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    Element root;
    Object root_object;
    if (parser.parse(padded).get(root) != simdjson::SUCCESS
        || root.get(root_object) != simdjson::SUCCESS) {
        core::logging::warn(
            "[ToolRecovery] Ignoring unreadable store at {}", path_.string());
        return {};
    }

    // Forward compatibility: a file written by a newer layout is left intact
    // and simply not used, rather than partially misinterpreted.
    int64_t version = kVersion;
    core::utils::json::ignore_error(root_object["version"].get(version));
    if (version > kVersion) {
        core::logging::warn(
            "[ToolRecovery] Ignoring store at {} written by a newer version ({})",
            path_.string(),
            version);
        return LoadedStore{.status = LoadStatus::UnsupportedVersion};
    }

    simdjson::dom::array lessons;
    if (root_object["lessons"].get(lessons) != simdjson::SUCCESS) {
        return {};
    }

    std::vector<Entry> entries;
    for (Element lesson : lessons) {
        Object lesson_object;
        if (lesson.get(lesson_object) != simdjson::SUCCESS) continue;

        RecoveryKey key{
            .tool = core::utils::json::string_field(lesson_object, "tool", ""),
            .schema_fingerprint = core::utils::json::string_field(
                lesson_object, "fingerprint", ""),
            .issue_code = core::utils::json::string_field(
                lesson_object, "issue", ""),
            .parameter = core::utils::json::string_field(
                lesson_object, "parameter", ""),
        };
        std::string hint = clamp_hint(
            core::utils::json::string_field(lesson_object, "hint", ""));
        if (key.tool.empty() || key.issue_code.empty() || hint.empty()) {
            continue;
        }
        // Tolerate hand-edited or merged files: keep the first record of a key
        // so recall, eviction, and counting all agree on one entry per key.
        if (std::ranges::any_of(entries, [&](const Entry& existing) {
                return existing.key == key;
            })) {
            continue;
        }

        int64_t evidence = 1;
        core::utils::json::ignore_error(lesson_object["evidence"].get(evidence));
        entries.push_back(Entry{
            .key = std::move(key),
            .hint = std::move(hint),
            .evidence = static_cast<int>(
                std::clamp<int64_t>(evidence, 1, kMaxEvidence)),
            .updated_at = core::utils::json::string_field(
                lesson_object, "updated_at", ""),
        });
    }
    evict_overflow(entries);
    return LoadedStore{.status = LoadStatus::Ok, .entries = std::move(entries)};
}

void FileToolRecoveryMemory::save_unlocked(
    const std::vector<Entry>& entries) const {
    core::utils::JsonWriter writer(256 + entries.size() * 192);
    {
        auto root = writer.object();
        writer.kv_num("version", kVersion).comma().key("lessons");
        {
            auto lessons = writer.array();
            for (std::size_t i = 0; i < entries.size(); ++i) {
                if (i > 0) writer.comma();
                const auto& entry = entries[i];
                auto lesson = writer.object();
                writer.kv_str("tool", entry.key.tool).comma()
                      .kv_str("fingerprint", entry.key.schema_fingerprint).comma()
                      .kv_str("issue", entry.key.issue_code).comma()
                      .kv_str("parameter", entry.key.parameter).comma()
                      .kv_str("hint", entry.hint).comma()
                      .kv_num("evidence", entry.evidence).comma()
                      .kv_str("updated_at", entry.updated_at);
            }
        }
    }
    std::string json = std::move(writer).take();
    json.push_back('\n');

    std::error_code ec;
    std::filesystem::create_directories(path_.parent_path(), ec);

    std::string error;
    if (!core::utils::atomic_write_file(path_, json, &error)) {
        core::logging::warn(
            "[ToolRecovery] Could not persist lessons to {}: {}",
            path_.string(), error);
    }
}

void FileToolRecoveryMemory::evict_overflow(std::vector<Entry>& entries) {
    if (entries.size() <= kMaxEntries) {
        return;
    }
    // Rank weakest-first: least evidence, then least recently verified.
    // Stable ordering keeps equal-ranked eviction deterministic.
    std::vector<std::size_t> order(entries.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::ranges::stable_sort(order, [&](std::size_t lhs, std::size_t rhs) {
        const auto& a = entries[lhs];
        const auto& b = entries[rhs];
        if (a.evidence != b.evidence) return a.evidence < b.evidence;
        return a.updated_at < b.updated_at;
    });

    std::vector<bool> evicted(entries.size(), false);
    for (std::size_t i = 0; i < entries.size() - kMaxEntries; ++i) {
        evicted[order[i]] = true;
    }
    std::vector<Entry> kept;
    kept.reserve(kMaxEntries);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (!evicted[i]) {
            kept.push_back(std::move(entries[i]));
        }
    }
    entries = std::move(kept);
}

std::shared_ptr<ToolRecoveryMemory> make_tool_recovery_memory(bool enabled) {
    if (!enabled) {
        return std::make_shared<NullToolRecoveryMemory>();
    }
    return std::make_shared<FileToolRecoveryMemory>();
}

} // namespace core::memory
