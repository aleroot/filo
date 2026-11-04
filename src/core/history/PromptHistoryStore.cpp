#include "PromptHistoryStore.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <unordered_set>

namespace core::history {

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
    return load_unlocked(error);
}

bool PromptHistoryStore::load_unlocked(std::string* error) {
    std::error_code ec;
    if (!std::filesystem::exists(history_file_, ec)) {
        // No history file yet — not an error.
        return true;
    }

    std::ifstream in(history_file_, std::ios::binary);
    if (!in) {
        if (error) {
            *error = std::format("Cannot open history file '{}'", history_file_.string());
        }
        return false;
    }

    const std::string content{
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>()};

    if (content.empty()) {
        entries_.clear();
        return true;
    }

    auto parsed = from_json(content);
    if (!parsed.has_value()) {
        if (error) {
            *error = std::format("Failed to parse history file '{}'", history_file_.string());
        }
        return false;
    }

    entries_ = std::move(*parsed);
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
    const std::string json = to_json(entries_);
    return core::utils::atomic_write_file(history_file_, json, error);
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
    if (!load_unlocked(error)) return false;
    add(std::move(entry));
    return save_unlocked(error);
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
    if (!load_unlocked(error)) return false;
    bool changed = false;
    for (auto& entry : entries_) {
        if (entry.session_id == from) {
            entry.session_id = std::string(to);
            changed = true;
        }
    }
    return !changed || save_unlocked(error);
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
    return save_unlocked(error);
}

bool PromptHistoryStore::remove_at_and_save(std::size_t index, std::string* error) {
    std::string lock_error;
    auto file_lock = core::utils::InterprocessFileLock::acquire(
        core::utils::lock_path_for(history_file_), &lock_error);
    if (!file_lock) {
        if (error) *error = lock_error;
        return false;
    }
    if (!load_unlocked(error)) return false;
    if (index >= entries_.size()) {
        if (error) {
            *error = std::format("history index {} out of range (have {})",
                                 index, entries_.size());
        }
        return false;
    }
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(index));
    return save_unlocked(error);
}

void PromptHistoryStore::add(std::string_view text) {
    add(PromptHistoryEntry{.text = std::string(text)});
}

void PromptHistoryStore::add(PromptHistoryEntry entry) {
    if (entry.text.empty()) {
        return;
    }

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
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        if (parser.parse_unpadded(json).get(doc) != simdjson::SUCCESS) {
            return std::nullopt;
        }

        // `entries` under an object, or a plain array (legacy format).
        simdjson::dom::array entries_arr;
        if (doc["entries"].get(entries_arr) != simdjson::SUCCESS
            && doc.get(entries_arr) != simdjson::SUCCESS) {
            return std::vector<PromptHistoryEntry>{};
        }

        // Keep `meta` aligned with `entries`: skipped non-string entries drop
        // their meta slot too.
        std::vector<PromptHistoryEntry> result;
        std::vector<std::size_t> source_index;
        std::size_t position = 0;
        for (simdjson::dom::element entry_el : entries_arr) {
            std::string_view sv;
            if (entry_el.get(sv) == simdjson::SUCCESS) {
                result.push_back(PromptHistoryEntry{.text = std::string(sv)});
                source_index.push_back(position);
            }
            ++position;
        }

        simdjson::dom::array meta_arr;
        if (doc.is_object()
            && doc["meta"].get(meta_arr) == simdjson::SUCCESS
            && meta_arr.size() == position) {
            std::vector<simdjson::dom::element> meta;
            meta.reserve(position);
            for (simdjson::dom::element item : meta_arr) {
                meta.push_back(item);
            }
            for (std::size_t i = 0; i < result.size(); ++i) {
                const auto& item = meta[source_index[i]];
                std::string_view value;
                if (item["session"].get(value) == simdjson::SUCCESS) {
                    result[i].session_id = std::string(value);
                }
                if (item["workspace"].get(value) == simdjson::SUCCESS) {
                    result[i].workspace = std::string(value);
                }
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
