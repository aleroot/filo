#include "MemoryStore.hpp"

#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <system_error>
#include <unordered_map>

namespace core::memory {
namespace {

struct MarkdownMemoryCandidate {
    std::string content;
    std::string scope = "global";
};

void append_entry_json(std::string& out, const MemoryEntry& entry) {
    core::utils::JsonWriter writer(512 + entry.content.size());
    {
        auto _ = writer.object();
        writer.kv_str("id", entry.id).comma()
              .kv_str("content", entry.content).comma()
              .kv_str("scope", entry.scope).comma()
              .key("tags");
        {
            auto _ = writer.array();
            for (std::size_t i = 0; i < entry.tags.size(); ++i) {
                if (i > 0) writer.comma();
                writer.str(entry.tags[i]);
            }
        }
        writer.comma().kv_str("source", entry.source).comma()
              .kv_str("created_at", entry.created_at).comma()
              .kv_str("updated_at", entry.updated_at).comma()
              .kv_str("last_used_at", entry.last_used_at).comma()
              .kv_num("use_count", entry.use_count).comma()
              .kv_bool("archived", entry.archived);
    }
    out += std::move(writer).take();
}

[[nodiscard]] std::optional<MemoryEntry> parse_entry(simdjson::dom::object object) {
    MemoryEntry entry;
    entry.id = core::utils::json::string_field(object, "id");
    entry.content = core::utils::str::trim_ascii_copy(core::utils::json::string_field(object, "content"));
    if (entry.id.empty() || entry.content.empty()) {
        return std::nullopt;
    }
    entry.scope = core::utils::json::string_field(object, "scope", "global");
    if (entry.scope.empty()) entry.scope = "global";
    entry.source = core::utils::json::string_field(object, "source", "manual");
    if (entry.source.empty()) entry.source = "manual";
    entry.created_at = core::utils::json::string_field(object, "created_at");
    entry.updated_at = core::utils::json::string_field(object, "updated_at");
    entry.last_used_at = core::utils::json::string_field(object, "last_used_at");
    entry.use_count = core::utils::json::int_field(object, "use_count");
    entry.archived = core::utils::json::bool_field(object, "archived");

    simdjson::dom::array tags;
    if (object["tags"].get(tags) == simdjson::SUCCESS) {
        for (simdjson::dom::element tag_value : tags) {
            std::string_view tag;
            if (tag_value.get(tag) == simdjson::SUCCESS) {
                std::string clean = core::utils::str::trim_ascii_copy(tag);
                if (!clean.empty()) {
                    entry.tags.push_back(std::move(clean));
                }
            }
        }
    }
    return entry;
}

[[nodiscard]] std::string selector_id(std::string_view selector) {
    std::string id = core::utils::str::trim_ascii_copy(selector);
    if (id.starts_with('{') && id.ends_with('}') && id.size() > 2) {
        id = id.substr(1, id.size() - 2);
    }
    return id;
}

[[nodiscard]] bool is_markdown_rule(std::string_view value) {
    return value == "---" || value == "***" || value == "___";
}

[[nodiscard]] MarkdownMemoryCandidate split_exported_markdown_scope(std::string content) {
    constexpr std::string_view kScopePrefix = " [scope: ";
    if (!content.ends_with(']')) {
        return {.content = std::move(content)};
    }

    const auto marker = content.rfind(kScopePrefix);
    if (marker == std::string::npos) {
        return {.content = std::move(content)};
    }

    std::string scope = core::utils::str::trim_ascii_copy(
        std::string_view(content).substr(marker + kScopePrefix.size(),
                                         content.size() - marker - kScopePrefix.size() - 1));
    content.erase(marker);
    content = core::utils::str::trim_ascii_copy(content);
    if (content.empty()) {
        return {.content = std::move(content)};
    }
    if (scope.empty()) {
        scope = "global";
    }
    return {.content = std::move(content), .scope = std::move(scope)};
}

[[nodiscard]] std::optional<MarkdownMemoryCandidate> markdown_memory_candidate(std::string_view line) {
    std::string clean = core::utils::str::trim_ascii_copy(line);
    if (clean.empty() || clean.starts_with('#') || clean.starts_with("<!--")
        || is_markdown_rule(clean)) {
        return std::nullopt;
    }

    auto strip_checkbox = [](std::string_view value) -> std::string_view {
        if (value.size() >= 4
            && value[0] == '['
            && (value[1] == ' ' || value[1] == 'x' || value[1] == 'X')
            && value[2] == ']'
            && std::isspace(static_cast<unsigned char>(value[3]))) {
            return value.substr(4);
        }
        return value;
    };

    std::string_view body;
    if ((clean.starts_with("- ") || clean.starts_with("* ")) && clean.size() > 2) {
        body = std::string_view(clean).substr(2);
    } else {
        std::size_t i = 0;
        while (i < clean.size() && std::isdigit(static_cast<unsigned char>(clean[i]))) ++i;
        if (i == 0 || i + 1 >= clean.size() || (clean[i] != '.' && clean[i] != ')')
            || !std::isspace(static_cast<unsigned char>(clean[i + 1]))) {
            return std::nullopt;
        }
        body = std::string_view(clean).substr(i + 2);
    }

    std::string result = core::utils::str::collapse_ascii_whitespace_copy(strip_checkbox(body));
    if (result.empty() || result.starts_with("Generated by Filo")) {
        return std::nullopt;
    }
    return split_exported_markdown_scope(std::move(result));
}

[[nodiscard]] std::string lock_key_for_path(const std::filesystem::path& path) {
    std::error_code ec;
    const auto absolute = path.is_absolute()
        ? path
        : std::filesystem::current_path(ec) / path;
    return (ec ? path : absolute).lexically_normal().string();
}

std::mutex& mutex_for_path(const std::filesystem::path& path) {
    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::unique_ptr<std::mutex>> locks;

    const auto key = lock_key_for_path(path);
    std::lock_guard lock(registry_mutex);
    auto [it, inserted] = locks.try_emplace(key);
    if (inserted) {
        it->second = std::make_unique<std::mutex>();
    }
    return *it->second;
}

} // namespace

MemoryStore::MemoryStore(std::filesystem::path path)
    : path_(std::move(path)) {}

std::filesystem::path MemoryStore::default_path() {
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path{xdg} / "filo" / "memory.json";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path{home} / ".config" / "filo" / "memory.json";
    }
    return std::filesystem::temp_directory_path() / "filo" / "memory.json";
}

std::string MemoryStore::now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    return std::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z",
                       tm.tm_year + 1900,
                       tm.tm_mon + 1,
                       tm.tm_mday,
                       tm.tm_hour,
                       tm.tm_min,
                       tm.tm_sec);
}

MemoryState MemoryStore::load(std::string* error) const {
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return {};
    }
    return load_unlocked(error);
}

MemoryState MemoryStore::load_unlocked(std::string* error) const {
    MemoryState state;
    std::error_code ec;
    if (!std::filesystem::exists(path_, ec)) {
        return state;
    }

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        if (error) *error = std::format("Cannot read memory store '{}'.", path_.string());
        return state;
    }
    const std::string json((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    if (json.empty()) {
        return state;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) {
        if (error) *error = std::format("Memory store '{}' contains invalid JSON.", path_.string());
        return state;
    }
    simdjson::dom::object root;
    if (doc.get(root) != simdjson::SUCCESS) {
        if (error) *error = "Memory store root must be a JSON object.";
        return state;
    }

    state.version = core::utils::json::int_field(root, "version", MemoryState::kVersion);
    simdjson::dom::object settings;
    if (root["settings"].get(settings) == simdjson::SUCCESS) {
        state.settings.enabled = core::utils::json::bool_field(settings, "enabled");
        state.settings.auto_capture = core::utils::json::bool_field(settings, "auto_capture");
        state.settings.background_review = core::utils::json::bool_field(settings, "background_review");
        state.settings.consolidation = core::utils::json::bool_field(settings, "consolidation");
        state.settings.skill_curation = core::utils::json::bool_field(settings, "skill_curation");
        state.settings.min_rate_limit_remaining_percent =
            std::clamp(core::utils::json::int_field(settings, "min_rate_limit_remaining_percent", 15), 0, 100);
        state.settings.max_active_entries =
            std::max(1, core::utils::json::int_field(settings, "max_active_entries", 120));
    } else {
        state.settings.enabled = core::utils::json::bool_field(root, "enabled");
        state.settings.auto_capture = core::utils::json::bool_field(root, "auto_capture");
        state.settings.background_review = core::utils::json::bool_field(root, "background_review");
        state.settings.consolidation = core::utils::json::bool_field(root, "consolidation");
        state.settings.skill_curation = core::utils::json::bool_field(root, "skill_curation");
        state.settings.min_rate_limit_remaining_percent =
            std::clamp(core::utils::json::int_field(root, "min_rate_limit_remaining_percent", 15), 0, 100);
        state.settings.max_active_entries =
            std::max(1, core::utils::json::int_field(root, "max_active_entries", 120));
    }

    simdjson::dom::array entries;
    if (root["entries"].get(entries) == simdjson::SUCCESS) {
        for (simdjson::dom::element element : entries) {
            simdjson::dom::object object;
            if (element.get(object) != simdjson::SUCCESS) continue;
            if (auto entry = parse_entry(object); entry.has_value()) {
                state.entries.push_back(std::move(*entry));
            }
        }
    }
    return state;
}

bool MemoryStore::save(const MemoryState& state, std::string* error) const {
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    return save_unlocked(state, error);
}

bool MemoryStore::save_unlocked(const MemoryState& state, std::string* error) const {
    std::string json;
    json.reserve(1024 + state.entries.size() * 256);
    core::utils::JsonWriter writer(256);
    {
        auto _ = writer.object();
        writer.kv_num("version", MemoryState::kVersion).comma()
              .key("settings");
        {
            auto _ = writer.object();
            writer.kv_bool("enabled", state.settings.enabled).comma()
                  .kv_bool("auto_capture", state.settings.auto_capture).comma()
                  .kv_bool("background_review", state.settings.background_review).comma()
                  .kv_bool("consolidation", state.settings.consolidation).comma()
                  .kv_bool("skill_curation", state.settings.skill_curation).comma()
                  .kv_num("min_rate_limit_remaining_percent",
                          state.settings.min_rate_limit_remaining_percent).comma()
                  .kv_num("max_active_entries", state.settings.max_active_entries);
        }
        writer.comma().key("entries");
    }
    json = std::move(writer).take();
    json.pop_back();
    json += '[';
    for (std::size_t i = 0; i < state.entries.size(); ++i) {
        if (i > 0) json += ',';
        append_entry_json(json, state.entries[i]);
    }
    json += "]}";

    json.push_back('\n');
    return core::utils::atomic_write_file(path_, json, error);
}

MemorySettings MemoryStore::settings(std::string* error) const {
    return load(error).settings;
}

bool MemoryStore::save_settings(MemorySettings settings, std::string* error) const {
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    auto state = load_unlocked(error);
    state.settings = settings;
    return save_unlocked(state, error);
}

std::vector<MemoryEntry> MemoryStore::list(bool include_archived, std::string* error) const {
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return {};
    }
    auto state = load_unlocked(error);
    std::vector<MemoryEntry> entries;
    entries.reserve(state.entries.size());
    for (auto& entry : state.entries) {
        if (!include_archived && entry.archived) continue;
        entries.push_back(std::move(entry));
    }
    std::ranges::sort(entries, [](const MemoryEntry& lhs, const MemoryEntry& rhs) {
        return lhs.created_at > rhs.created_at;
    });
    return entries;
}

MemoryMutationResult MemoryStore::remember(std::string_view content,
                                           std::string_view scope,
                                           std::vector<std::string> tags,
                                           std::string_view source) const {
    const std::string clean_content = core::utils::str::trim_ascii_copy(content);
    if (clean_content.empty()) {
        return {.ok = false, .message = "Memory content cannot be empty."};
    }

    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) return {.ok = false, .message = lock_error};
    auto state = load_unlocked();
    const std::string fingerprint = normalize_for_match(clean_content);
    const std::string now = now_iso8601();
    for (auto& entry : state.entries) {
        if (normalize_for_match(entry.content) != fingerprint) continue;
        entry.archived = false;
        entry.updated_at = now;
        entry.last_used_at = now;
        entry.use_count += 1;
        std::string error;
        if (!save_unlocked(state, &error)) {
            return {.ok = false, .message = error};
        }
        return {.ok = true, .message = std::format("Updated memory {{{}}}.", entry.id), .entry = entry};
    }

    std::erase_if(tags, [](const std::string& tag) {
        return core::utils::str::trim_ascii_copy(tag).empty();
    });
    std::string clean_scope = core::utils::str::trim_ascii_copy(scope);
    std::string clean_source = core::utils::str::trim_ascii_copy(source);
    MemoryEntry entry{
        .id = next_id(state.entries),
        .content = clean_content,
        .scope = clean_scope.empty() ? std::string("global") : std::move(clean_scope),
        .tags = std::move(tags),
        .source = clean_source.empty() ? std::string("manual") : std::move(clean_source),
        .created_at = now,
        .updated_at = now,
        .last_used_at = now,
        .use_count = 1,
        .archived = false,
    };
    state.entries.push_back(entry);
    std::string error;
    if (!save_unlocked(state, &error)) {
        return {.ok = false, .message = error};
    }
    return {.ok = true, .message = std::format("Stored memory {{{}}}.", entry.id), .entry = entry};
}

MemoryMutationResult MemoryStore::forget(std::string_view selector) const {
    const std::string id = selector_id(selector);
    if (id.empty()) {
        return {.ok = false, .message = "Memory id is required."};
    }
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) return {.ok = false, .message = lock_error};
    auto state = load_unlocked();
    for (auto& entry : state.entries) {
        if (entry.id != id) continue;
        entry.archived = true;
        entry.updated_at = now_iso8601();
        std::string error;
        if (!save_unlocked(state, &error)) {
            return {.ok = false, .message = error};
        }
        return {.ok = true, .message = std::format("Archived memory {{{}}}.", id), .entry = entry};
    }
    return {.ok = false, .message = "Memory not found."};
}

MemoryMutationResult MemoryStore::clean() const {
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) return {.ok = false, .message = lock_error};
    auto state = load_unlocked();
    std::vector<MemoryEntry> kept;
    kept.reserve(state.entries.size());
    std::vector<std::string> seen;
    std::size_t archived_duplicates = 0;
    for (auto entry : state.entries) {
        const std::string fingerprint = normalize_for_match(entry.content);
        if (entry.archived) {
            kept.push_back(std::move(entry));
            continue;
        }
        if (!fingerprint.empty()
            && std::ranges::find(seen, fingerprint) != seen.end()) {
            entry.archived = true;
            entry.updated_at = now_iso8601();
            ++archived_duplicates;
        } else if (!fingerprint.empty()) {
            seen.push_back(fingerprint);
        }
        kept.push_back(std::move(entry));
    }
    state.entries = std::move(kept);
    std::string error;
    if (!save_unlocked(state, &error)) {
        return {.ok = false, .message = error};
    }
    return {
        .ok = true,
        .message = archived_duplicates == 0
            ? "Memory store is already clean."
            : std::format("Archived {} duplicate memory item(s).", archived_duplicates),
    };
}

MemoryMutationResult MemoryStore::clear() const {
    std::lock_guard lock(mutex_for_path(path_));
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(path_), &lock_error);
    if (!file_lock) return {.ok = false, .message = lock_error};
    auto state = load_unlocked();
    std::size_t changed = 0;
    const std::string now = now_iso8601();
    for (auto& entry : state.entries) {
        if (entry.archived) continue;
        entry.archived = true;
        entry.updated_at = now;
        ++changed;
    }
    std::string error;
    if (!save_unlocked(state, &error)) {
        return {.ok = false, .message = error};
    }
    return {
        .ok = true,
        .message = changed == 0
            ? "No active memories to archive."
            : std::format("Archived {} active memory item(s).", changed),
    };
}

MemoryFileResult MemoryStore::save_markdown(const std::filesystem::path& output_path) const {
    if (output_path.empty()) {
        return {.ok = false, .message = "Markdown output path is required."};
    }

    std::string error;
    auto entries = list(false, &error);
    if (!error.empty()) {
        return {.ok = false, .message = error};
    }
    std::ranges::sort(entries, [](const MemoryEntry& lhs, const MemoryEntry& rhs) {
        return lhs.created_at < rhs.created_at;
    });

    std::error_code ec;
    if (const auto parent = output_path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return {.ok = false, .message = std::format("Cannot create '{}': {}", parent.string(), ec.message())};
        }
    }

    std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return {.ok = false, .message = std::format("Cannot write '{}'.", output_path.string())};
    }

    file << "# Filo Memory\n\n";
    file << "<!-- Generated by Filo. Import with `/memory load " << output_path.filename().string()
         << "`. -->\n\n";
    for (const auto& entry : entries) {
        file << "- " << core::utils::str::collapse_ascii_whitespace_copy(entry.content);
        if (!entry.scope.empty() && entry.scope != "global") {
            file << " [scope: " << core::utils::str::collapse_ascii_whitespace_copy(entry.scope) << "]";
        }
        file << '\n';
    }

    if (!file) {
        return {.ok = false, .message = std::format("Failed while writing '{}'.", output_path.string())};
    }
    return {
        .ok = true,
        .message = std::format("Saved {} active memory item(s) to {}.", entries.size(), output_path.string()),
        .count = entries.size(),
    };
}

MemoryFileResult MemoryStore::load_markdown(const std::filesystem::path& input_path) const {
    if (input_path.empty()) {
        return {.ok = false, .message = "Markdown input path is required."};
    }

    std::ifstream file(input_path, std::ios::binary);
    if (!file) {
        return {.ok = false, .message = std::format("Cannot read '{}'.", input_path.string())};
    }

    std::size_t imported = 0;
    std::string line;
    while (std::getline(file, line)) {
        auto candidate = markdown_memory_candidate(line);
        if (!candidate.has_value()) continue;
        auto result = remember(candidate->content, candidate->scope, {}, "markdown");
        if (!result.ok) {
            return {.ok = false, .message = result.message, .count = imported};
        }
        ++imported;
    }

    if (file.bad()) {
        return {.ok = false, .message = std::format("Failed while reading '{}'.", input_path.string()), .count = imported};
    }
    return {
        .ok = true,
        .message = imported == 0
            ? std::format("No markdown memories found in {}.", input_path.string())
            : std::format("Loaded {} markdown memory item(s) from {}.", imported, input_path.string()),
        .count = imported,
    };
}

std::string MemoryStore::normalize_for_match(std::string_view value) {
    return core::utils::str::to_lower_ascii_copy(
        core::utils::str::collapse_ascii_whitespace_copy(value));
}

std::string MemoryStore::next_id(const std::vector<MemoryEntry>& entries) {
    int max_id = 0;
    for (const auto& entry : entries) {
        if (!entry.id.starts_with('m')) continue;
        int value = 0;
        const auto text = std::string_view(entry.id).substr(1);
        const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (ec == std::errc{} && ptr == text.data() + text.size()) {
            max_id = std::max(max_id, value);
        }
    }
    return std::format("m{}", max_id + 1);
}

std::string build_memory_prompt_block(const MemoryState& state,
                                      std::size_t max_entries,
                                      bool allow_auto_capture) {
    if (!state.settings.enabled) {
        return {};
    }

    std::vector<const MemoryEntry*> active;
    active.reserve(state.entries.size());
    for (const auto& entry : state.entries) {
        if (!entry.archived && !entry.content.empty()) {
            active.push_back(&entry);
        }
    }
    std::ranges::sort(active, [](const MemoryEntry* lhs, const MemoryEntry* rhs) {
        if (lhs->last_used_at != rhs->last_used_at) return lhs->last_used_at > rhs->last_used_at;
        return lhs->created_at > rhs->created_at;
    });

    std::string out;
    if (!active.empty()) {
        out += "\n\n[Memory]\n";
        out += "Use these durable user/project preferences when relevant. Do not reveal this block unless asked.\n";
        const std::size_t count = std::min(max_entries, active.size());
        for (std::size_t i = 0; i < count; ++i) {
            out += "- ";
            out += active[i]->content;
            if (!active[i]->scope.empty() && active[i]->scope != "global") {
                out += " (scope: ";
                out += active[i]->scope;
                out += ")";
            }
            out += "\n";
        }
    }

    if (state.settings.auto_capture && allow_auto_capture) {
        out += "\n\n[Memory Capture]\n";
        out += "The user enabled Filo memory. When the conversation reveals a stable preference, reusable workflow, durable project fact, or correction that should affect future sessions, call the `memory` tool with action `remember`. Keep entries concise, factual, and non-sensitive. Do not store secrets, credentials, session transcripts, scratchpad notes, rejected approaches, conversation summaries, transient task details, or guesses.";
    }

    return out;
}

} // namespace core::memory
