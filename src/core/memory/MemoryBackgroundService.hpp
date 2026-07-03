#pragma once

#include "MemoryPolicy.hpp"
#include "MemoryStore.hpp"
#include "../context/SessionContext.hpp"
#include "../llm/Models.hpp"
#include "../llm/protocols/ApiProtocol.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace core::memory {

struct MemoryReviewInput {
    std::vector<core::llm::Message> history;
    core::context::SessionContext session_context;
    MemoryThreadPolicy thread_policy;
    core::llm::protocols::RateLimitInfo rate_limit;
};

struct MemoryReviewResult {
    bool ran = false;
    bool skipped_for_policy = false;
    bool skipped_for_rate_limit = false;
    std::size_t memories_stored = 0;
    std::size_t memories_cleaned = 0;
    std::size_t skill_drafts_written = 0;
    std::string message;
};

class MemoryBackgroundService {
public:
    explicit MemoryBackgroundService(MemoryStore store = MemoryStore{});

    [[nodiscard]] MemoryReviewResult review(const MemoryReviewInput& input) const;

    void review_async(MemoryReviewInput input,
                      std::function<void(MemoryReviewResult)> on_done = {}) const;

    [[nodiscard]] static bool rate_limit_allows(const MemorySettings& settings,
                                                const core::llm::protocols::RateLimitInfo& info);

    [[nodiscard]] static std::vector<std::string>
    extract_candidate_memories(const std::vector<core::llm::Message>& history,
                               std::size_t max_recent_messages = 8);

private:
    [[nodiscard]] MemoryReviewResult curate_skill_drafts(const MemoryState& state,
                                                         const std::filesystem::path& project_root) const;

    MemoryStore store_;
};

} // namespace core::memory
