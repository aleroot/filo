#pragma once

/**
 * @file AntigravityConversationState.hpp
 * @brief Per-trajectory execution-handle state for the Antigravity envelope.
 *
 * The real `antigravity/hub` client echoes the previous response's
 * `responseId` back as `labels.last_execution_id` on the next request of the
 * same trajectory. Streaming protocol instances are per-request clones, so the
 * handle has to outlive them; this registry keys it by the same conversation
 * anchor the envelope ids are derived from (session id, else first user
 * message).
 *
 * The registry is process-wide and bounded: the oldest anchors are dropped
 * once `kMaxTrackedConversations` is exceeded so long-lived processes cannot
 * grow it without limit.
 */

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace core::llm::protocols {

class AntigravityConversationState {
public:
    static constexpr std::size_t kMaxTrackedConversations = 256;

    static AntigravityConversationState& instance();

    /// Remember `response_id` as the newest execution handle for the anchor.
    void record_execution_id(std::string_view conversation_anchor,
                             std::string_view response_id);

    /// Last recorded execution handle for the anchor, or empty when unknown.
    [[nodiscard]] std::string
    execution_id(std::string_view conversation_anchor) const;

    /// Drop all tracked state (used by tests).
    void clear();

private:
    AntigravityConversationState() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> execution_ids_;
    std::deque<std::string> insertion_order_;
};

} // namespace core::llm::protocols
