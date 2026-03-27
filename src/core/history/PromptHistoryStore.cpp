#include "PromptHistoryStore.hpp"
#include "core/utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>

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
    // Ensure parent directory exists.
    std::error_code ec;
    std::filesystem::create_directories(history_file_.parent_path(), ec);
    if (ec) {
        if (error) {
            *error = std::format("Cannot create history directory: {}", ec.message());
        }
        return false;
    }

    const auto target_path = history_file_;
    const auto tmp_path    = std::filesystem::path(target_path.string() + ".tmp");

    const std::string json = to_json(entries_);
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            if (error) {
                *error = std::format("Cannot write history file '{}'", tmp_path.string());
            }
            return false;
        }
        out.write(json.data(), static_cast<std::streamsize>(json.size()));
        if (!out) {
            if (error) {
                *error = std::format("Failed to write history data to '{}'", tmp_path.string());
            }
            return false;
        }
    }

    std::filesystem::rename(tmp_path, target_path, ec);
    if (ec) {
        if (error) {
            *error = std::format("Cannot finalize history file '{}': {}",
                                 target_path.string(), ec.message());
        }
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

void PromptHistoryStore::add(std::string_view text) {
    // Ignore empty strings.
    if (text.empty()) {
        return;
    }

    // Collapse consecutive duplicates.
    if (!entries_.empty() && entries_.back() == text) {
        return;
    }

    entries_.push_back(std::string(text));
    enforce_size_limit();
}

std::vector<std::string> PromptHistoryStore::entries_newest_first() const {
    std::vector<std::string> result;
    result.reserve(entries_.size());
    result.insert(result.end(), entries_.rbegin(), entries_.rend());
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

std::string PromptHistoryStore::to_json(const std::vector<std::string>& entries) {
    std::string out;
    out.reserve(1024 + entries.size() * 64);

    out += "{\"version\":1,\"entries\":[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) out += ',';
        out += "\"";
        core::utils::append_escaped(out, entries[i]);
        out += "\"";
    }
    out += "]}";
    return out;
}

std::optional<std::vector<std::string>> PromptHistoryStore::from_json(std::string_view json) {
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        if (parser.parse(json.data(), json.size()).get(doc) != simdjson::SUCCESS) {
            return std::nullopt;
        }

        std::vector<std::string> result;

        simdjson::dom::array entries_arr;
        if (doc["entries"].get(entries_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element entry_el : entries_arr) {
                std::string_view sv;
                if (entry_el.get(sv) == simdjson::SUCCESS) {
                    result.emplace_back(sv);
                }
            }
        } else {
            // Fallback: try parsing as a plain array (legacy or simple format).
            if (doc.get(entries_arr) == simdjson::SUCCESS) {
                for (simdjson::dom::element entry_el : entries_arr) {
                    std::string_view sv;
                    if (entry_el.get(sv) == simdjson::SUCCESS) {
                        result.emplace_back(sv);
                    }
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

bool PersistentPromptHistory::navigate_prev(std::string& input, int& cursor) {
    if (!store_ || store_->empty()) {
        return false;
    }

    const auto entries = store_->entries_newest_first();
    if (entries.empty()) {
        return false;
    }

    if (idx_ == -1) {
        saved_input_ = input;
        idx_ = 0;  // Start at most recent entry.
    } else if (idx_ < static_cast<int>(entries.size()) - 1) {
        ++idx_;
    } else {
        return true;  // Already at oldest entry.
    }

    input  = entries[static_cast<std::size_t>(idx_)];
    cursor = static_cast<int>(input.size());
    return true;
}

bool PersistentPromptHistory::navigate_next(std::string& input, int& cursor) {
    if (idx_ == -1) {
        return false;
    }

    const auto entries = store_->entries_newest_first();

    if (idx_ > 0) {
        --idx_;
        input = entries[static_cast<std::size_t>(idx_)];
    } else {
        idx_ = -1;
        input = saved_input_;
    }
    cursor = static_cast<int>(input.size());
    return true;
}

void PersistentPromptHistory::save(std::string_view text) {
    if (!store_) {
        return;
    }

    // Reset navigation state.
    idx_ = -1;
    saved_input_.clear();

    // Add to store and persist.
    store_->add(text);
    static_cast<void>(store_->save(nullptr));  // Silent save (best effort).
}

void PersistentPromptHistory::reload() {
    if (store_) {
        static_cast<void>(store_->load(nullptr));
    }
    idx_ = -1;
    saved_input_.clear();
}

void PersistentPromptHistory::clear() {
    if (store_) {
        store_->clear();
        static_cast<void>(store_->save(nullptr));
    }
    idx_ = -1;
    saved_input_.clear();
}

std::size_t PersistentPromptHistory::size() const noexcept {
    return store_ ? store_->size() : 0;
}

bool PersistentPromptHistory::empty() const noexcept {
    return !store_ || store_->empty();
}

} // namespace core::history
