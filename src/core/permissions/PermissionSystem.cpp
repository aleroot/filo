#include "PermissionSystem.hpp"
#include "../agent/PermissionGate.hpp"
#include "../agent/SafetyPolicy.hpp"
#include "../tools/ToolNames.hpp"
#include <array>
#include <cctype>
#include <format>

namespace core::permissions {

namespace {

enum class AllowKeyStrategy {
    ToolName,
    ShellProgram
};

struct AllowRule {
    std::string_view tool_name;
    std::string_view allow_label;
    AllowKeyStrategy key_strategy;
};

constexpr std::array<AllowRule, 7> kAllowRules{{
    {core::tools::names::kRunTerminalCommand, "terminal commands", AllowKeyStrategy::ShellProgram},
    {core::tools::names::kWriteFile,          "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kApplyPatch,         "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kReplace,            "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kReplaceInFile,      "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kDeleteFile,         "file deletions",     AllowKeyStrategy::ToolName},
    {core::tools::names::kMoveFile,           "file moves",         AllowKeyStrategy::ToolName},
}};

const AllowRule* find_allow_rule(std::string_view tool_name) {
    for (const auto& rule : kAllowRules) {
        if (rule.tool_name == tool_name) {
            return &rule;
        }
    }
    return nullptr;
}

std::string_view trim_left(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    return text;
}

std::string_view next_shell_token(std::string_view& rest) {
    rest = trim_left(rest);
    if (rest.empty()) {
        return {};
    }

    std::size_t end = 0;
    while (end < rest.size() && !std::isspace(static_cast<unsigned char>(rest[end]))) {
        ++end;
    }
    const auto token = rest.substr(0, end);
    rest.remove_prefix(end);
    return token;
}

bool is_env_assignment(std::string_view token) {
    if (token.empty()) {
        return false;
    }
    if (!std::isalpha(static_cast<unsigned char>(token.front())) && token.front() != '_') {
        return false;
    }
    for (const char ch : token) {
        if (ch == '=') {
            return true;
        }
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
            return false;
        }
    }
    return false;
}

std::string strip_path_prefix(std::string_view token) {
    const auto separator = token.find_last_of("/\\");
    if (separator != std::string_view::npos) {
        token.remove_prefix(separator + 1);
    }
    return std::string(token);
}

std::string extract_shell_program(std::string_view tool_args) {
    const auto command = core::agent::CommandSafetyPolicy::extract_shell_command(tool_args);
    if (command.empty()) {
        return {};
    }

    std::string_view rest = command;
    auto token = next_shell_token(rest);

    // Skip env assignments: VAR=VALUE cmd ...
    while (!token.empty() && is_env_assignment(token)) {
        token = next_shell_token(rest);
    }

    if (token.empty()) {
        return {};
    }
    return strip_path_prefix(token);
}

std::string make_allow_key_from_rule(const AllowRule& rule, std::string_view tool_args) {
    if (rule.key_strategy == AllowKeyStrategy::ShellProgram) {
        const auto program = extract_shell_program(tool_args);
        if (!program.empty()) {
            return std::format("{}:{}", rule.tool_name, program);
        }
    }
    return std::string(rule.tool_name);
}

std::string make_allow_label_from_rule(const AllowRule& rule, std::string_view tool_args) {
    if (rule.key_strategy == AllowKeyStrategy::ShellProgram) {
        const auto program = extract_shell_program(tool_args);
        if (!program.empty()) {
            return std::format("'{}' commands", program);
        }
    }
    return std::string(rule.allow_label);
}

} // namespace

// ---------------------------------------------------------------------------
// PermissionQueue implementation
// ---------------------------------------------------------------------------
uint64_t PermissionQueue::enqueue(PermissionRequest request) {
    std::lock_guard lock(mutex_);
    Entry entry{.request = std::move(request), .sequence_number = next_sequence_++};
    queue_.push(std::move(entry));
    return next_sequence_ - 1;
}

std::optional<PermissionQueue::Entry> PermissionQueue::dequeue() {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) {
        return std::nullopt;
    }
    auto entry = std::move(queue_.front());
    queue_.pop();
    return entry;
}

bool PermissionQueue::empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
}

void PermissionQueue::clear() {
    std::lock_guard lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}

// ---------------------------------------------------------------------------
// SessionAllowList implementation
// ---------------------------------------------------------------------------
bool SessionAllowList::contains(std::string_view key) const {
    std::shared_lock lock(mutex_);
    return allowed_.contains(std::string(key));
}

void SessionAllowList::insert(std::string key) {
    std::lock_guard lock(mutex_);
    allowed_.insert(std::move(key));
}

void SessionAllowList::clear() {
    std::lock_guard lock(mutex_);
    allowed_.clear();
}

std::vector<std::string> SessionAllowList::snapshot() const {
    std::shared_lock lock(mutex_);
    return std::vector<std::string>(allowed_.begin(), allowed_.end());
}

// ---------------------------------------------------------------------------
// PermissionSystem implementation
// ---------------------------------------------------------------------------
PermissionSystem& PermissionSystem::instance() {
    static PermissionSystem instance;
    return instance;
}

bool PermissionSystem::is_auto_approved(std::string_view tool_name,
                                        std::string_view tool_args,
                                        bool yolo_enabled) const {
    // YOLO mode approves everything
    if (yolo_enabled) {
        return true;
    }
    
    // Check session allow list
    auto key = make_allow_key(tool_name, tool_args);
    if (allow_list_.contains(key)) {
        return true;
    }
    
    // Check if tool needs permission at all
    // Use agent::needs_permission from SafetyPolicy
    using core::agent::needs_permission;
    using core::agent::PermissionProfile;
    
    // For subagents, check with Interactive profile
    if (!needs_permission(tool_name, PermissionProfile::Interactive, tool_args)) {
        return true;
    }
    
    return false;
}

bool PermissionSystem::check_permission(std::string_view tool_name,
                                        std::string_view tool_args,
                                        bool yolo_enabled,
                                        std::function<void(bool approved)> on_result) {
    // Fast path: auto-approved
    if (is_auto_approved(tool_name, tool_args, yolo_enabled)) {
        on_result(true);
        return true;
    }
    
    // Slow path: enqueue request
    PermissionRequest request;
    request.tool_name = std::string(tool_name);
    request.tool_args = std::string(tool_args);
    request.allow_key = make_allow_key(tool_name, tool_args);
    request.allow_label = make_allow_label(tool_name, tool_args);
    request.on_decision = std::move(on_result);
    
    queue_.enqueue(std::move(request));
    return false;
}

bool PermissionSystem::process_pending_requests() {
    // If already showing a prompt, don't start a new one
    if (prompt_active_.load()) {
        return false;
    }
    
    auto entry = queue_.dequeue();
    if (!entry) {
        return false;
    }
    
    // Set up the new prompt
    {
        std::lock_guard lock(current_mutex_);
        current_tool_name_ = entry->request.tool_name;
        current_tool_args_ = entry->request.tool_args;
        current_allow_label_ = entry->request.allow_label;
    }
    
    current_request_ = std::move(*entry);
    selected_index_ = 0;
    prompt_active_ = true;
    
    return true;
}

void PermissionSystem::handle_decision(int selected) {
    if (!current_request_) {
        return;
    }
    
    bool approved = (selected != 3);  // 3 = No
    bool remember = (selected == 1);  // 1 = Yes+Remember
    bool enable_yolo = (selected == 2);  // 2 = YOLO
    
    // Handle YOLO mode
    if (enable_yolo) {
        yolo_mode_ = true;
    }
    
    // Handle remember choice
    if (remember && !current_request_->request.allow_key.empty()) {
        allow_list_.insert(current_request_->request.allow_key);
    }
    
    // Invoke callback
    if (current_request_->request.on_decision) {
        current_request_->request.on_decision(approved);
    }
    
    // Clear state
    current_request_.reset();
    prompt_active_ = false;
    selected_index_ = 0;
}

void PermissionSystem::cancel_current() {
    if (current_request_ && current_request_->request.on_decision) {
        current_request_->request.on_decision(false);  // Deny on cancel
    }
    current_request_.reset();
    prompt_active_ = false;
    selected_index_ = 0;
}

std::string PermissionSystem::current_tool_name() const {
    std::lock_guard lock(current_mutex_);
    return current_tool_name_;
}

std::string PermissionSystem::current_tool_args() const {
    std::lock_guard lock(current_mutex_);
    return current_tool_args_;
}

std::string PermissionSystem::current_allow_label() const {
    std::lock_guard lock(current_mutex_);
    return current_allow_label_;
}

void PermissionSystem::reset() {
    cancel_current();
    queue_.clear();
    allow_list_.clear();
    yolo_mode_ = false;
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------
std::string make_allow_key(std::string_view tool_name, std::string_view tool_args) {
    if (const auto* rule = find_allow_rule(tool_name)) {
        return make_allow_key_from_rule(*rule, tool_args);
    }
    return std::string(tool_name);
}

std::string make_allow_label(std::string_view tool_name, std::string_view tool_args) {
    if (const auto* rule = find_allow_rule(tool_name)) {
        return make_allow_label_from_rule(*rule, tool_args);
    }
    return std::string(tool_name);
}

} // namespace core::permissions
