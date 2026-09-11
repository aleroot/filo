#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::workspace {

/// One workspace-relative path candidate offered for `@` mention completion.
struct MentionSuggestion {
    std::string display_path;
    std::string insertion_text;
    std::string search_path;
    std::string search_basename;
    int depth = 0;
    bool is_directory = false;
};

/// Walks `root` once, honouring agent-ignore visibility and pruning generated
/// directories, producing a path-sorted index suitable for repeated queries.
std::vector<MentionSuggestion> build_mention_index(const std::filesystem::path& root,
                                                   std::stop_token stop_token = {});

/// Ranks `index` against `query`: path prefix, basename prefix, segment prefix,
/// then substring; ties break on shallower depth, shorter path, then name.
std::vector<MentionSuggestion> search_mention_index(const std::vector<MentionSuggestion>& index,
                                                    std::string_view query,
                                                    std::size_t max_results);

/// Thread-safe, TTL-bounded cache of per-root mention indexes.
///
/// Completion is keystroke-driven, so one walk is amortised across a burst of
/// requests. Entries expire instead of being invalidated by a filesystem
/// watcher, which keeps workspace edits visible without any background state.
class MentionIndexCache {
public:
    using Clock = std::chrono::steady_clock;
    using Index = std::vector<MentionSuggestion>;

    explicit MentionIndexCache(Clock::duration ttl) noexcept : ttl_(ttl) {}

    /// Returns the index for `root`, rebuilding it when absent or expired.
    [[nodiscard]] std::shared_ptr<const Index> index_for(const std::filesystem::path& root);

private:
    struct Entry {
        std::shared_ptr<const Index> index;
        Clock::time_point expires_at;
    };

    Clock::duration ttl_;
    std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
};

} // namespace core::workspace
