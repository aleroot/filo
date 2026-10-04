#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::context { struct SessionContext; }

namespace core::memory {

struct MemorySettings {
    bool enabled = true;
    bool auto_capture = true;
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
    std::vector<std::string> tags = {};
    std::string source = "manual";
    std::string created_at = {};
    std::string updated_at = {};
    std::string last_used_at = {};
    int use_count = 0;
    bool archived = false;
    std::string project_root = {};
    std::string session_id = {};
    // Archived originals retained by an atomic revision or merge.
    std::vector<std::string> supersedes = {};
};

struct MemoryScopeUsage {
    std::string scope;
    std::size_t active_entries = 0;
    std::size_t limit = 120;

    [[nodiscard]] bool near_capacity() const noexcept {
        return active_entries >= limit - limit / 5;
    }
};

struct MemoryState {
    static constexpr int kVersion = 1;

    int version = kVersion;
    MemorySettings settings;
    std::vector<MemoryEntry> entries;
    // Derived before prompt selection; never persisted as store data.
    std::vector<MemoryScopeUsage> scope_usage = {};
};

// Only new automatic captures/revisions are bounded. Existing entries and
// explicit manual/imported memories remain readable without truncation.
inline constexpr std::size_t kMaxAutoMemoryBytes = 1200;

struct MemoryRevisionTarget {
    std::string id;
    std::string expected_content;
};

/// How many memories one prompt carries at most.
inline constexpr std::size_t kDefaultPromptEntries = 24;

/// Total memory content one prompt carries, in characters.
inline constexpr std::size_t kDefaultPromptBlockChars = 24'000;

/// What one prompt may carry from semantic memory, and what it is ranked
/// against. See MemoryRelevance.hpp for the ordering and budget rules.
struct PromptProjection {
    std::size_t max_entries = kDefaultPromptEntries;
    std::size_t max_block_chars = kDefaultPromptBlockChars;
    std::string relevance_query;
};

struct MemoryMutationResult {
    bool ok = false;
    std::string message;
    std::optional<MemoryEntry> entry = {};
    std::string code = {};
    std::optional<MemoryScopeUsage> capacity = {};
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

    // A session-bound view of the same file. All entry reads and mutations are
    // restricted to this checkout; settings remain shared across projects.
    [[nodiscard]] MemoryStore for_context(const core::context::SessionContext& context) const;

    [[nodiscard]] MemoryState load(std::string* error = nullptr) const;
    /// Selects and orders the active entries one prompt may carry. The returned
    /// state holds only those entries, in render order (best match first).
    [[nodiscard]] MemoryState load_for_prompt(const PromptProjection& projection = {},
                                              std::string* error = nullptr) const;
    /// Records one user-turn recall for ids that were actually included in a
    /// submitted prompt. Inactive or out-of-scope entries are left untouched.
    [[nodiscard]] bool record_prompt_recall(
        const std::vector<std::string>& entry_ids,
        std::string* error = nullptr) const;
    [[nodiscard]] bool save(const MemoryState& state, std::string* error = nullptr) const;

    [[nodiscard]] MemorySettings settings(std::string* error = nullptr) const;
    [[nodiscard]] bool save_settings(MemorySettings settings, std::string* error = nullptr) const;

    [[nodiscard]] std::vector<MemoryEntry> list(bool include_archived = false,
                                                std::string* error = nullptr) const;

    [[nodiscard]] MemoryMutationResult remember(std::string_view content,
                                                std::string_view scope = {},
                                                std::vector<std::string> tags = {},
                                                std::string_view source = "manual") const;
    /// Replaces one fact, or merges overlapping facts in one scope, retaining
    /// archived originals. Expected contents guard against concurrent edits.
    [[nodiscard]] MemoryMutationResult revise(
        const std::vector<MemoryRevisionTarget>& targets,
        std::string_view content,
        std::string_view source = "agent") const;
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
    [[nodiscard]] bool visible(const MemoryEntry& entry) const;
    void populate_scope_usage(MemoryState& state) const;

    std::filesystem::path path_;
    bool context_bound_ = false;
    std::string project_root_;
    std::string session_id_;
};

[[nodiscard]] std::string build_memory_prompt_block(const MemoryState& state,
                                                    std::size_t max_entries = 24,
                                                    bool allow_auto_capture = true);

} // namespace core::memory
