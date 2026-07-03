#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::memory {

struct MemorySettings {
    bool enabled = false;
    bool auto_capture = false;
    bool background_review = false;
    bool consolidation = false;
    bool skill_curation = false;
    int min_rate_limit_remaining_percent = 15;
    int max_active_entries = 120;
};

struct MemoryEntry {
    std::string id;
    std::string content;
    std::string scope = "global";
    std::vector<std::string> tags;
    std::string source = "manual";
    std::string created_at;
    std::string updated_at;
    std::string last_used_at;
    int use_count = 0;
    bool archived = false;
};

struct MemoryState {
    static constexpr int kVersion = 1;

    int version = kVersion;
    MemorySettings settings;
    std::vector<MemoryEntry> entries;
};

struct MemoryMutationResult {
    bool ok = false;
    std::string message;
    std::optional<MemoryEntry> entry;
};

struct MemoryFileResult {
    bool ok = false;
    std::string message;
    std::size_t count = 0;
};

class MemoryStore {
public:
    explicit MemoryStore(std::filesystem::path path = default_path());

    [[nodiscard]] static std::filesystem::path default_path();
    [[nodiscard]] static std::string now_iso8601();

    [[nodiscard]] MemoryState load(std::string* error = nullptr) const;
    [[nodiscard]] bool save(const MemoryState& state, std::string* error = nullptr) const;

    [[nodiscard]] MemorySettings settings(std::string* error = nullptr) const;
    [[nodiscard]] bool save_settings(MemorySettings settings, std::string* error = nullptr) const;

    [[nodiscard]] std::vector<MemoryEntry> list(bool include_archived = false,
                                                std::string* error = nullptr) const;

    [[nodiscard]] MemoryMutationResult remember(std::string_view content,
                                                std::string_view scope = "global",
                                                std::vector<std::string> tags = {},
                                                std::string_view source = "manual") const;
    [[nodiscard]] MemoryMutationResult forget(std::string_view selector) const;
    [[nodiscard]] MemoryMutationResult clean() const;
    [[nodiscard]] MemoryMutationResult clear() const;
    [[nodiscard]] MemoryFileResult save_markdown(const std::filesystem::path& output_path) const;
    [[nodiscard]] MemoryFileResult load_markdown(const std::filesystem::path& input_path) const;

    [[nodiscard]] std::filesystem::path path() const noexcept { return path_; }

private:
    [[nodiscard]] MemoryState load_unlocked(std::string* error = nullptr) const;
    [[nodiscard]] bool save_unlocked(const MemoryState& state,
                                     std::string* error = nullptr) const;

    [[nodiscard]] static std::string normalize_for_match(std::string_view value);
    [[nodiscard]] static std::string next_id(const std::vector<MemoryEntry>& entries);

    std::filesystem::path path_;
};

[[nodiscard]] std::string build_memory_prompt_block(const MemoryState& state,
                                                    std::size_t max_entries = 24,
                                                    bool allow_auto_capture = true);

} // namespace core::memory
