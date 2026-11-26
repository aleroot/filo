#include "PromptHistoryStore.hpp"
#include "core/persistence/AmendedFile.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>

namespace core::history {

namespace {
/// Far above any history, below what would exhaust memory.
constexpr std::size_t kMaxHistoryBytes = std::size_t{64} << 20;

using simdjson::ondemand::array;
using simdjson::ondemand::json_type;
using simdjson::ondemand::object;
using simdjson::ondemand::value;

/// How deeply values may nest, the root being at depth 1. The on-demand
/// parser does not check it (a debug build asserts), so a container is not
/// entered at this depth: its values would be deeper. The dom parser accepts
/// an empty one there; nothing else differs.
constexpr std::size_t kMaxDepth = simdjson::DEFAULT_MAX_DEPTH;

/// Reads @p item whole; @p depth is how many containers it would make, it
/// included, if it is one. The on-demand parser validates only what it
/// reads, and a history that does not parse must be refused as a whole, not
/// salvaged and then saved over.
simdjson::error_code consume(value item, std::size_t depth) {
    json_type type{};
    if (const auto error = item.type().get(type)) return error;
    switch (type) {
        case json_type::array: {
            if (depth >= kMaxDepth) return simdjson::DEPTH_ERROR;
            array items;
            if (const auto error = item.get_array().get(items)) return error;
            for (auto element : items) {
                value child;
                if (const auto error = element.get(child)) return error;
                if (const auto error = consume(child, depth + 1)) return error;
            }
            return simdjson::SUCCESS;
        }
        case json_type::object: {
            if (depth >= kMaxDepth) return simdjson::DEPTH_ERROR;
            object fields;
            if (const auto error = item.get_object().get(fields)) return error;
            for (auto field : fields) {
                std::string_view key;
                value child;
                if (const auto error = field.unescaped_key().get(key)) return error;
                if (const auto error = field.value().get(child)) return error;
                if (const auto error = consume(child, depth + 1)) return error;
            }
            return simdjson::SUCCESS;
        }
        case json_type::string: {
            std::string_view text;
            return item.get_string().get(text);
        }
        case json_type::boolean: {
            bool flag = false;
            return item.get_bool().get(flag);
        }
        case json_type::null: {
            bool null = false;
            if (const auto error = item.is_null().get(null)) return error;
            return null ? simdjson::SUCCESS : simdjson::N_ATOM_ERROR;
        }
        default: {
            simdjson::ondemand::number number;
            return item.get_number().get(number);
        }
    }
}

/// The string @p item holds, or nullopt (after reading it whole) when it
/// holds something else.
simdjson::error_code optional_string(value item, std::size_t depth,
                                     std::optional<std::string_view>& out) {
    json_type type{};
    if (const auto error = item.type().get(type)) return error;
    if (type != json_type::string) return consume(item, depth);
    std::string_view text;
    if (const auto error = item.get_string().get(text)) return error;
    out = text;
    return simdjson::SUCCESS;
}

/// The texts of @p entries (at @p depth) with their positions; non-strings
/// are counted.
simdjson::error_code read_entries(array entries, std::size_t depth,
                                  std::vector<PromptHistoryEntry>& result,
                                  std::vector<std::size_t>& source_index, std::size_t& count) {
    for (auto element : entries) {
        value item;
        if (const auto error = element.get(item)) return error;
        std::optional<std::string_view> text;
        if (const auto error = optional_string(item, depth + 1, text)) return error;
        if (text) {
            result.push_back(PromptHistoryEntry{.text = std::string(*text)});
            source_index.push_back(count);
        }
        ++count;
    }
    return simdjson::SUCCESS;
}

struct EntryScope {
    std::string session_id;
    std::string workspace;
};

/// One `meta` item (at @p depth): the first `session` and `workspace` keys
/// of an object.
simdjson::error_code read_scope(value item, std::size_t depth, EntryScope& scope) {
    json_type type{};
    if (const auto error = item.type().get(type)) return error;
    if (type != json_type::object) return consume(item, depth);
    if (depth >= kMaxDepth) return simdjson::DEPTH_ERROR;
    object fields;
    if (const auto error = item.get_object().get(fields)) return error;
    bool seen_session = false;
    bool seen_workspace = false;
    for (auto field : fields) {
        std::string_view key;
        value child;
        if (const auto error = field.unescaped_key().get(key)) return error;
        // Decided before the value is read: that may reuse the key's buffer.
        std::string* target = nullptr;
        if (key == "session" && !std::exchange(seen_session, true)) target = &scope.session_id;
        else if (key == "workspace" && !std::exchange(seen_workspace, true)) target = &scope.workspace;
        if (const auto error = field.value().get(child)) return error;
        std::optional<std::string_view> text;
        if (const auto error = optional_string(child, depth + 1, text)) return error;
        if (target != nullptr && text) *target = std::string(*text);
    }
    return simdjson::SUCCESS;
}

simdjson::error_code read_meta(array meta, std::size_t depth, std::vector<EntryScope>& scopes) {
    for (auto element : meta) {
        value item;
        if (const auto error = element.get(item)) return error;
        if (const auto error = read_scope(item, depth + 1, scopes.emplace_back())) return error;
    }
    return simdjson::SUCCESS;
}

/// A scalar root holds no entries, but must still be JSON. (A scalar
/// document cannot be read as a value, so it is read where it stands.)
simdjson::error_code validate_scalar(simdjson::ondemand::document& doc, json_type type,
                                     std::string_view json) {
    switch (type) {
        case json_type::string: {
            std::string_view text;
            return doc.get_string().get(text);
        }
        case json_type::boolean:
        case json_type::null: {
            // Compared whole: simdjson 5 reads a root `falsX` as false.
            constexpr std::string_view kWhitespace = " \t\n\r";
            const auto first = json.find_first_not_of(kWhitespace);
            const auto atom = json.substr(first, json.find_last_not_of(kWhitespace) + 1 - first);
            return atom == "true" || atom == "false" || atom == "null"
                ? simdjson::SUCCESS : simdjson::INCORRECT_TYPE;
        }
        default: {
            simdjson::ondemand::number number;
            return doc.get_number().get(number);
        }
    }
}
} // namespace

// -----------------------------------------------------------------------------
// PromptHistoryStore Implementation
// -----------------------------------------------------------------------------

PromptHistoryStore::PromptHistoryStore(std::filesystem::path history_file)
    : history_file_(std::move(history_file)) {}

std::filesystem::path PromptHistoryStore::default_history_path() {
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path{xdg} / "filo" / "history.json";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path{home} / ".local" / "share" / "filo" / "history.json";
    }
    return std::filesystem::temp_directory_path() / "filo" / "history.json";
}

std::string PromptHistoryStore::now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    return std::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}Z",
        tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
        tm.tm_hour, tm.tm_min, tm.tm_sec);
}

bool PromptHistoryStore::load(std::string* error) {
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    return refresh_unlocked(error);
}

PromptHistoryStore::DiskState PromptHistoryStore::disk_state() const {
    return DiskState{
        .snapshot = core::utils::FileStamp::of(history_file_),
        .log = core::utils::FileStamp::of(core::persistence::AmendedFile::log_path(history_file_)),
    };
}

bool PromptHistoryStore::refresh_unlocked(std::string* error) {
    // Every writer holds the lock and changes what a stamp sees: a record
    // grows the log, and a new snapshot is renamed over the old one, so it
    // has another inode (and the log it starts has another size or inode).
    if (synced_ && *synced_ == disk_state()) return true;
    return load_unlocked(error);
}

bool PromptHistoryStore::load_unlocked(std::string* error) {
    synced_.reset();
    position_.reset();
    // Taken before reading, so a change made meanwhile is read next time.
    auto state = disk_state();
    auto contents = core::persistence::AmendedFile::read(history_file_, kMaxHistoryBytes);
    if (!contents) {
        if (error) {
            *error = std::format("Cannot open history file '{}'", history_file_.string());
        }
        return false;
    }
    if (!*contents) return true;  // No history file yet — not an error.
    if ((*contents)->snapshot.empty()) {
        entries_.clear();
    } else {
        auto parsed = from_json((*contents)->snapshot);
        if (!parsed.has_value()) {
            if (error) {
                *error = std::format("Failed to parse history file '{}'", history_file_.string());
            }
            return false;
        }
        entries_ = std::move(*parsed);
    }
    // Prompts recorded since the last save, replayed as they were added: the
    // oldest are dropped exactly as saving each would have.
    for (const auto& record : (*contents)->records) {
        if (auto parsed = from_json(record); parsed && parsed->size() == 1) add(std::move(parsed->front()));
    }
    position_ = std::move((*contents)->position);
    synced_ = std::move(state);
    return true;
}

bool PromptHistoryStore::save(std::string* error) const {
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    return save_unlocked(error);
}

bool PromptHistoryStore::save_unlocked(std::string* error) const {
    auto written = core::persistence::AmendedFile(history_file_).replace(to_json(entries_));
    if (!written && error) *error = std::move(written.error());
    return written.has_value();
}

bool PromptHistoryStore::commit_unlocked(std::string* error) {
    synced_.reset();
    position_.reset();
    core::persistence::AmendedFile file(history_file_);
    if (auto written = file.replace(to_json(entries_)); !written) {
        if (error) *error = std::move(written.error());
        return false;
    }
    position_ = file.position();
    synced_ = disk_state();
    return true;
}

bool PromptHistoryStore::append_and_save(std::string_view text, std::string* error) {
    return append_and_save(PromptHistoryEntry{.text = std::string(text)}, error);
}

bool PromptHistoryStore::append_and_save(PromptHistoryEntry entry, std::string* error) {
    if (entry.text.empty()) return true;
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    if (!refresh_unlocked(error)) return false;
    // A repeat of the newest prompt of the same session is not added.
    if (!entries_.empty() && entries_.back().text == entry.text
        && entries_.back().session_id == entry.session_id) {
        return true;
    }
    const std::string record = to_json({entry});
    // Kept as reading the record back gives it (text that is not UTF-8 is
    // stored repaired), so memory still matches the file after the write.
    auto stored = from_json(record);
    const bool exact = stored && stored->size() == 1;
    add(exact ? std::move(stored->front()) : std::move(entry));
    // A prompt is recorded as a change to the history, not a rewrite of all
    // of it (up to max_entries prompts); the next save folds it in.
    core::persistence::AmendedFile file(history_file_);
    if (position_) file.adopt(*position_);
    if (file.amend(record, core::persistence::AmendedFile::Durability::Deferred)) {
        position_ = file.position();
        synced_ = disk_state();
    } else if (!commit_unlocked(error)) {
        return false;
    }
    if (!exact) synced_.reset();
    return true;
}

bool PromptHistoryStore::rename_session_and_save(std::string_view from,
                                                 std::string_view to,
                                                 std::string* error) {
    if (from.empty() || from == to) return true;
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    if (!refresh_unlocked(error)) return false;
    bool changed = false;
    for (auto& entry : entries_) {
        if (entry.session_id == from) {
            entry.session_id = std::string(to);
            changed = true;
        }
    }
    return !changed || commit_unlocked(error);
}

bool PromptHistoryStore::clear_and_save(std::string* error) {
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    clear();
    return commit_unlocked(error);
}

bool PromptHistoryStore::remove_at_and_save(std::size_t index, std::string* error) {
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    if (!refresh_unlocked(error)) return false;
    if (index >= entries_.size()) {
        if (error) {
            *error = std::format("history index {} out of range (have {})",
                                 index, entries_.size());
        }
        return false;
    }
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    return commit_unlocked(error);
}

void PromptHistoryStore::add(std::string_view text) {
    add(PromptHistoryEntry{.text = std::string(text)});
}

void PromptHistoryStore::add(PromptHistoryEntry entry) {
    if (entry.text.empty()) {
        return;
    }
    synced_.reset();  // The file no longer says what memory holds.

    // Collapse a repeat of the newest entry from the same session. A repeat
    // from another thread is kept: that thread's own history needs it.
    if (!entries_.empty()
        && entries_.back().text == entry.text
        && entries_.back().session_id == entry.session_id) {
        return;
    }

    entries_.push_back(std::move(entry));
    enforce_size_limit();
}

std::vector<std::string> PromptHistoryStore::entries() const {
    std::vector<std::string> result;
    result.reserve(entries_.size());
    for (const auto& entry : entries_) {
        result.push_back(entry.text);
    }
    return result;
}

std::vector<std::string> PromptHistoryStore::entries_newest_first() const {
    std::vector<std::string> result;
    result.reserve(entries_.size());
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        result.push_back(it->text);
    }
    return result;
}

void PromptHistoryStore::clear() {
    entries_.clear();
    synced_.reset();
}

void PromptHistoryStore::enforce_size_limit() {
    if (max_entries_ == 0) {
        return;  // Unlimited
    }
    if (entries_.size() > max_entries_) {
        entries_.erase(entries_.begin(),
                       entries_.begin() + static_cast<std::ptrdiff_t>(entries_.size() - max_entries_));
    }
}

// -----------------------------------------------------------------------------
// JSON Serialization
// -----------------------------------------------------------------------------

std::string PromptHistoryStore::to_json(const std::vector<PromptHistoryEntry>& entries) {
    std::string out;
    out.reserve(1024 + entries.size() * 128);

    out += "{\"version\":2,\"entries\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) out += ',';
        out += '"';
        core::utils::append_escaped(out, entries[i].text);
        out += '"';
    }
    out += "],\"meta\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) out += ',';
        out += '{';
        bool first = true;
        const auto field = [&](std::string_view name, std::string_view value) {
            if (value.empty()) return;
            if (!first) out += ',';
            first = false;
            out += '"';
            out += name;
            out += "\":\"";
            core::utils::append_escaped(out, value);
            out += '"';
        };
        field("session", entries[i].session_id);
        field("workspace", entries[i].workspace);
        out += '}';
    }
    out += "]}";
    return out;
}

std::optional<std::vector<PromptHistoryEntry>> PromptHistoryStore::from_json(
    std::string_view json) {
    try {
        // On demand rather than dom: a dom parser needs about 13 times the
        // input (40 MB for a 3 MB history), this about half of that. Every
        // value is still read, so what dom refuses is refused here too.
        const simdjson::padded_string padded(json);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc;
        json_type type{};
        if (parser.iterate(padded).get(doc) || doc.type().get(type)) return std::nullopt;

        std::vector<PromptHistoryEntry> result;
        std::vector<std::size_t> source_index;
        std::size_t count = 0;
        std::vector<EntryScope> scopes;
        bool meta_found = false;
        if (type == json_type::array) {
            // Legacy format: a plain array of texts.
            array entries;
            if (doc.get_array().get(entries) || read_entries(entries, 1, result, source_index, count)) {
                return std::nullopt;
            }
        } else if (type == json_type::object) {
            object root;
            if (doc.get_object().get(root)) return std::nullopt;
            bool entries_seen = false;
            bool meta_seen = false;
            for (auto field : root) {
                std::string_view key;
                value item;
                if (field.unescaped_key().get(key)) return std::nullopt;
                // Like a lookup, only the first of repeated keys counts.
                const bool entries_key = key == "entries" && !std::exchange(entries_seen, true);
                const bool meta_key = key == "meta" && !std::exchange(meta_seen, true);
                json_type item_type{};
                if (field.value().get(item) || item.type().get(item_type)) return std::nullopt;
                if ((!entries_key && !meta_key) || item_type != json_type::array) {
                    if (consume(item, 2)) return std::nullopt;
                    continue;
                }
                array items;
                if (item.get_array().get(items)) return std::nullopt;
                meta_found |= meta_key;
                const auto error = entries_key ? read_entries(items, 2, result, source_index, count)
                                               : read_meta(items, 2, scopes);
                if (error) return std::nullopt;
            }
        } else {
            // Read in full, trailing content included.
            if (validate_scalar(doc, type, json)) return std::nullopt;
            return result;
        }
        if (!doc.at_end()) return std::nullopt;

        // `meta` parallels `entries`, non-strings included; one that does
        // not line up was rewritten by an older build and is ignored.
        if (meta_found && scopes.size() == count) {
            for (std::size_t i = 0; i < result.size(); ++i) {
                auto& scope = scopes[source_index[i]];
                result[i].session_id = std::move(scope.session_id);
                result[i].workspace = std::move(scope.workspace);
            }
        }
        return result;
    } catch (...) {
        return std::nullopt;
    }
}


// -----------------------------------------------------------------------------
// PersistentPromptHistory Implementation
// -----------------------------------------------------------------------------

PersistentPromptHistory::PersistentPromptHistory(std::shared_ptr<PromptHistoryStore> store)
    : store_(std::move(store)) {
    if (store_) {
        static_cast<void>(store_->load(nullptr));  // Silent load on construction.
    }
}

std::vector<std::string> PersistentPromptHistory::candidates_for(
    const PromptHistoryScope& scope, std::string_view draft) const {
    std::vector<std::string> tiers[3];
    std::unordered_set<std::string_view> seen;
    if (!draft.empty()) {
        seen.insert(draft);  // Up must visibly change the composer.
    }
    const auto& records = store_->records();
    // First pass claims each text for its strongest tier; the newest
    // occurrence within that tier decides its position.
    for (int tier = 0; tier < 3; ++tier) {
        for (auto it = records.rbegin(); it != records.rend(); ++it) {
            const bool own = !scope.session_id.empty()
                && it->session_id == scope.session_id;
            const bool project = !own && !scope.workspace.empty()
                && it->workspace == scope.workspace;
            const int entry_tier = own ? 0 : (project ? 1 : 2);
            if (entry_tier != tier || !seen.insert(it->text).second) {
                continue;
            }
            tiers[tier].push_back(it->text);
        }
    }
    std::vector<std::string> result = std::move(tiers[0]);
    for (int tier = 1; tier < 3; ++tier) {
        std::ranges::move(tiers[tier], std::back_inserter(result));
    }
    return result;
}

void PersistentPromptHistory::apply_pending_changes() {
    std::vector<std::pair<std::string, std::string>> renames;
    std::vector<std::string> forgets;
    {
        std::lock_guard lock(pending_mutex_);
        renames.swap(pending_renames_);
        forgets.swap(pending_forgets_);
    }
    for (auto& [from, to] : renames) {
        if (store_) {
            static_cast<void>(store_->rename_session_and_save(from, to, nullptr));
        }
        if (auto node = navigation_.extract(from); !node.empty()) {
            navigation_.insert_or_assign(to, std::move(node.mapped()));
        }
    }
    for (const auto& session_id : forgets) {
        navigation_.erase(session_id);
    }
}

bool PersistentPromptHistory::navigate_prev(const PromptHistoryScope& scope,
                                            std::string& input, int& cursor) {
    if (!store_) {
        return false;
    }
    apply_pending_changes();

    auto& nav = navigation_[scope.session_id];
    if (nav.index >= 0 && input != nav.shown) {
        nav = {};  // Edited since recalled: the edit is the new draft.
    }

    if (nav.index < 0) {
        auto candidates = candidates_for(scope, input);
        if (candidates.empty()) {
            navigation_.erase(scope.session_id);
            return false;
        }
        nav.candidates = std::move(candidates);
        nav.draft = input;
        nav.index = 0;
    } else if (nav.index + 1 < static_cast<int>(nav.candidates.size())) {
        ++nav.index;
    } else {
        return true;  // Already at the oldest entry.
    }

    nav.shown = nav.candidates[static_cast<std::size_t>(nav.index)];
    input = nav.shown;
    cursor = static_cast<int>(input.size());
    return true;
}

bool PersistentPromptHistory::navigate_next(const PromptHistoryScope& scope,
                                            std::string& input, int& cursor) {
    apply_pending_changes();

    const auto it = navigation_.find(scope.session_id);
    if (it == navigation_.end()) {
        return false;
    }
    auto& nav = it->second;
    if (nav.index < 0 || input != nav.shown) {
        navigation_.erase(it);  // Never overwrite text the user edited.
        return false;
    }

    if (nav.index > 0) {
        --nav.index;
        nav.shown = nav.candidates[static_cast<std::size_t>(nav.index)];
        input = nav.shown;
    } else {
        input = std::move(nav.draft);
        navigation_.erase(it);
    }
    cursor = static_cast<int>(input.size());
    return true;
}

void PersistentPromptHistory::save(const PromptHistoryScope& scope, std::string_view text) {
    if (!store_) {
        return;
    }
    apply_pending_changes();
    navigation_.erase(scope.session_id);

    static_cast<void>(store_->append_and_save(  // Best effort.
        PromptHistoryEntry{
            .text = std::string(text),
            .session_id = scope.session_id,
            .workspace = scope.workspace,
        },
        nullptr));
}

void PersistentPromptHistory::rename_session(std::string from, std::string to) {
    if (from.empty() || from == to) {
        return;
    }
    std::lock_guard lock(pending_mutex_);
    pending_renames_.emplace_back(std::move(from), std::move(to));
}

void PersistentPromptHistory::forget_session(std::string session_id) {
    if (session_id.empty()) {
        return;
    }
    std::lock_guard lock(pending_mutex_);
    pending_forgets_.push_back(std::move(session_id));
}

void PersistentPromptHistory::reload() {
    if (store_) {
        static_cast<void>(store_->load(nullptr));
    }
    apply_pending_changes();
}

void PersistentPromptHistory::clear() {
    if (store_) {
        static_cast<void>(store_->clear_and_save(nullptr));
    }
    navigation_.clear();
    std::lock_guard lock(pending_mutex_);
    pending_renames_.clear();
    pending_forgets_.clear();
}

std::size_t PersistentPromptHistory::size() const noexcept {
    return store_ ? store_->size() : 0;
}

bool PersistentPromptHistory::empty() const noexcept {
    return !store_ || store_->empty();
}

} // namespace core::history
