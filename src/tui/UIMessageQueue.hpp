#pragma once

#include "Conversation.hpp"
#include "core/utils/ConcurrentQueue.hpp"
#include <functional>
#include <string>

namespace tui {

/**
 * @brief Simplified UI message queue for thread-safe UI updates.
 * 
 * This replaces the complex mutex/cv/promise system with a simple
 * producer-consumer queue. Background threads enqueue messages,
 * the UI thread dequeues and renders them.
 */
class UIMessageQueue {
public:
    struct Message {
        enum class Type {
            AddText,           // Add text message to history
            UpdateAssistant,   // Update assistant message
            ToolStart,         // Tool execution started
            ToolFinish,        // Tool execution finished
            PermissionPrompt,  // Show permission prompt
            PermissionResolve, // Resolve permission (approve/deny)
        };
        
        Type type;
        std::string data;  // JSON or text payload
        std::size_t index; // For updates (e.g., assistant message index)
    };
    
    void push(Message msg) {
        queue_.push(std::move(msg));
        if (notify_fn_) {
            notify_fn_();
        }
    }
    
    std::optional<Message> try_pop() {
        return queue_.try_pop();
    }
    
    void set_notify_fn(std::function<void()> fn) {
        notify_fn_ = std::move(fn);
    }
    
    void clear() {
        queue_.clear();
    }
    
    bool empty() const {
        return queue_.empty();
    }

private:
    core::utils::ConcurrentQueue<Message> queue_;
    std::function<void()> notify_fn_;
};

} // namespace tui
