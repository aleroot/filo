#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace core::permissions {

// ---------------------------------------------------------------------------
// Permission Request - immutable data for a single permission check
// ---------------------------------------------------------------------------
struct PermissionRequest {
    std::string tool_name;
    std::string tool_args;
    std::string allow_key;      // For "don't ask again" feature
    std::string allow_label;    // Display label for the UI
    
    // Callback invoked when user makes a decision
    // bool = approved
    std::function<void(bool approved)> on_decision;
};

// ---------------------------------------------------------------------------
// Permission Queue - thread-safe queue for pending requests
// Single mutex, minimal locking, no condition variables
// ---------------------------------------------------------------------------
class PermissionQueue {
public:
    struct Entry {
        PermissionRequest request;
        uint64_t sequence_number{0};
    };
    
    // Add a request to the queue
    uint64_t enqueue(PermissionRequest request);
    
    // Get the next pending request (called by UI thread)
    std::optional<Entry> dequeue();
    
    // Check if queue is empty
    bool empty() const;
    
    // Clear all pending requests
    void clear();
    
private:
    mutable std::mutex mutex_;
    std::queue<Entry> queue_;
    uint64_t next_sequence_{1};
};

// ---------------------------------------------------------------------------
// SessionAllowList - thread-safe "don't ask again" storage
// Uses shared_mutex for read-heavy workload
// ---------------------------------------------------------------------------
class SessionAllowList {
public:
    bool contains(std::string_view key) const;
    void insert(std::string key);
    void clear();
    std::vector<std::string> snapshot() const;
    
private:
    mutable std::shared_mutex mutex_;
    std::unordered_set<std::string> allowed_;
};

// ---------------------------------------------------------------------------
// PermissionSystem - main interface
// Thread-safe, minimal locking, callback-based (no blocking)
// ---------------------------------------------------------------------------
class PermissionSystem {
public:
    static PermissionSystem& instance();
    
    // Worker thread API
    // Returns true immediately if auto-approved, false if async decision needed
    bool check_permission(std::string_view tool_name,
                          std::string_view tool_args,
                          bool yolo_enabled,
                          std::function<void(bool approved)> on_result);
    
    // Synchronous check for auto-approved tools
    bool is_auto_approved(std::string_view tool_name,
                          std::string_view tool_args,
                          bool yolo_enabled) const;
    
    // UI thread API
    // Process pending requests - call each frame from UI thread
    bool process_pending_requests();
    
    // Handle user decision - selected: 0=Yes, 1=Yes+Remember, 2=YOLO, 3=No
    void handle_decision(int selected);
    
    // Cancel current prompt
    void cancel_current();
    
    // State access
    bool prompt_active() const { return prompt_active_.load(); }
    void set_prompt_active(bool active) { prompt_active_ = active; }
    
    int selected_index() const { return selected_index_.load(); }
    void set_selected_index(int idx) { selected_index_ = idx; }
    
    // Current request info (only valid when prompt_active)
    std::string current_tool_name() const;
    std::string current_tool_args() const;
    std::string current_allow_label() const;
    
    bool yolo_mode() const { return yolo_mode_.load(); }
    void set_yolo_mode(bool enabled) { yolo_mode_ = enabled; }
    
    SessionAllowList& allow_list() { return allow_list_; }
    
    void reset();
    
private:
    PermissionSystem() = default;
    
    std::atomic<bool> yolo_mode_{false};
    std::atomic<bool> prompt_active_{false};
    std::atomic<int> selected_index_{0};
    
    mutable std::mutex current_mutex_;
    std::string current_tool_name_;
    std::string current_tool_args_;
    std::string current_allow_label_;
    
    PermissionQueue queue_;
    SessionAllowList allow_list_;
    
    std::optional<PermissionQueue::Entry> current_request_;
};

// Utility functions
std::string make_allow_key(std::string_view tool_name, std::string_view tool_args);
std::string make_allow_label(std::string_view tool_name, std::string_view tool_args);

} // namespace core::permissions
