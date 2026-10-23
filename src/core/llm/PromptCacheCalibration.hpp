#pragma once

#include "Models.hpp"
#include "PromptCacheProfile.hpp"

#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace core::llm {

/// Learns how long an undocumented prompt cache really survives.
///
/// Every successful request reports how many prompt tokens came from the
/// cache. A request that was mostly served from cache after an idle gap
/// proves the cache outlives that gap, so an estimated lifetime shorter than
/// the gap is extended. Misses are deliberately ignored: providers evict
/// best-effort under load, so a miss says nothing about the configured
/// lifetime, and shortening on it would only add false warnings.
///
/// Documented profiles are never changed — the provider's contract is the
/// authority there. Thread-safe; one instance is shared by every session in
/// the process so a lesson learned in one thread spares the others.
class PromptCacheCalibration {
public:
    /// Requests below this size say little: providers do not cache them.
    static constexpr std::int32_t kMinimumObservedPromptTokens = 1024;

    [[nodiscard]] static PromptCacheCalibration& shared();

    /// Records one completed request that followed an idle gap of @p gap.
    void observe(std::string_view key,
                 std::chrono::seconds gap,
                 const TokenUsage& usage);

    /// The profile with any observed evidence applied.
    [[nodiscard]] PromptCacheProfile refine(std::string_view key,
                                            PromptCacheProfile profile) const;

    /// A stable key for one cache namespace: caches are per provider and model.
    [[nodiscard]] static std::string key(std::string_view provider,
                                         std::string_view model);

    void clear();

private:
    /// Lets lookups take the string_view key without allocating.
    struct KeyHash : std::hash<std::string_view> {
        using is_transparent = void;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::chrono::seconds, KeyHash, std::equal_to<>>
        longest_hit_gap_;
};

} // namespace core::llm
