#include "AntigravityConversationState.hpp"

#include <algorithm>
#include <utility>

namespace core::llm::protocols {

AntigravityConversationState& AntigravityConversationState::instance() {
    static AntigravityConversationState state;
    return state;
}

void AntigravityConversationState::record_execution_id(
    std::string_view conversation_anchor,
    std::string_view response_id) {
    if (conversation_anchor.empty() || response_id.empty()) return;

    std::lock_guard lock(mutex_);
    const std::string anchor(conversation_anchor);
    const auto [it, inserted] =
        execution_ids_.insert_or_assign(anchor, std::string(response_id));
    (void)it;
    if (inserted) {
        insertion_order_.push_back(anchor);
        while (insertion_order_.size() > kMaxTrackedConversations) {
            execution_ids_.erase(insertion_order_.front());
            insertion_order_.pop_front();
        }
    }
}

std::string AntigravityConversationState::execution_id(
    std::string_view conversation_anchor) const {
    if (conversation_anchor.empty()) return {};

    std::lock_guard lock(mutex_);
    const auto it = execution_ids_.find(std::string(conversation_anchor));
    return it == execution_ids_.end() ? std::string{} : it->second;
}

void AntigravityConversationState::clear() {
    std::lock_guard lock(mutex_);
    execution_ids_.clear();
    insertion_order_.clear();
}

} // namespace core::llm::protocols
