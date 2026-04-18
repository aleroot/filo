#include "PermissionSystem.hpp"
#include "../agent/PermissionGate.hpp"
#include "../agent/SafetyPolicy.hpp"
#include "../tools/ToolNames.hpp"
#include <algorithm>
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

constexpr std::array<AllowRule, 9> kAllowRules{{
    {core::tools::names::kRunTerminalCommand, "terminal commands", AllowKeyStrategy::ShellProgram},
    {core::tools::names::kWriteFile,          "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kApplyPatch,         "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kReplace,            "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kReplaceInFile,      "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kSearchReplace,      "file modifications", AllowKeyStrategy::ToolName},
    {core::tools::names::kCreateDirectory,    "file modifications", AllowKeyStrategy::ToolName},
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

std::string_view trim_ascii(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

std::string to_lower_ascii_copy(std::string_view text) {
    std::string lowered;
    lowered.reserve(text.size());
    for (const unsigned char ch : text) {
        lowered.push_back(static_cast<char>(std::tolower(ch)));
    }
    return lowered;
}

bool iequals_ascii(std::string_view lhs, std::string_view rhs) {
    return to_lower_ascii_copy(lhs) == to_lower_ascii_copy(rhs);
}

enum class SessionRuleKind {
    ExactAllowKey,
    ToolName,
    ShellAny,
    ShellProgram,
    FilesAny,
    FilesWrite,
    FilesDelete,
    FilesMove,
};

struct ParsedSessionRule {
    SessionRuleKind kind = SessionRuleKind::ExactAllowKey;
    std::string value;
};

bool is_write_like_file_tool(std::string_view tool_name) {
    return tool_name == core::tools::names::kWriteFile
        || tool_name == core::tools::names::kApplyPatch
        || core::tools::names::is_replace_tool(tool_name)
        || tool_name == core::tools::names::kCreateDirectory
        || tool_name == core::tools::names::kSearchReplace;
}

std::string normalize_files_scope(std::string_view value) {
    const std::string normalized = to_lower_ascii_copy(trim_ascii(value));
    if (normalized.empty()) {
        return {};
    }
    if (normalized == "*" || normalized == "all" || normalized == "any") {
        return "*";
    }
    if (normalized == "write" || normalized == "modify" || normalized == "modification"
        || normalized == "modifications" || normalized == "edit" || normalized == "edits") {
        return "write";
    }
    if (normalized == "delete" || normalized == "deletion" || normalized == "deletions"
        || normalized == "remove" || normalized == "removal" || normalized == "removals") {
        return "delete";
    }
    if (normalized == "move" || normalized == "moves" || normalized == "rename"
        || normalized == "renames") {
        return "move";
    }
    return {};
}

ParsedSessionRule parse_session_rule(std::string_view raw_rule) {
    const std::string_view trimmed_raw = trim_ascii(raw_rule);
    if (trimmed_raw.empty()) {
        return {};
    }
    const std::string trimmed_lower = to_lower_ascii_copy(trimmed_raw);
    const std::string_view trimmed = trimmed_lower;

    auto parse_prefixed = [&](std::string_view prefix) -> std::string_view {
        if (!trimmed.starts_with(prefix)) {
            return {};
        }
        return trim_ascii(trimmed.substr(prefix.size()));
    };

    if (const auto value = parse_prefixed("tool:"); !value.empty()) {
        return ParsedSessionRule{.kind = SessionRuleKind::ToolName,
                                 .value = to_lower_ascii_copy(value)};
    }
    if (const auto value = parse_prefixed("shell:"); !value.empty()) {
        if (value == "*") {
            return ParsedSessionRule{.kind = SessionRuleKind::ShellAny, .value = "*"};
        }
        return ParsedSessionRule{
            .kind = SessionRuleKind::ShellProgram,
            .value = to_lower_ascii_copy(strip_path_prefix(value)),
        };
    }
    if (const auto value = parse_prefixed("files:"); !value.empty()) {
        const std::string scope = normalize_files_scope(value);
        if (scope == "*") {
            return ParsedSessionRule{.kind = SessionRuleKind::FilesAny, .value = "*"};
        }
        if (scope == "write") {
            return ParsedSessionRule{.kind = SessionRuleKind::FilesWrite, .value = "write"};
        }
        if (scope == "delete") {
            return ParsedSessionRule{.kind = SessionRuleKind::FilesDelete, .value = "delete"};
        }
        if (scope == "move") {
            return ParsedSessionRule{.kind = SessionRuleKind::FilesMove, .value = "move"};
        }
    }

    // Backward-compatible legacy keys:
    //   run_terminal_command:git  -> shell:git
    //   run_terminal_command      -> exact allow key match
    //   delete_file               -> files:delete
    //   move_file                 -> files:move
    //
    // Legacy write keys (write_file/apply_patch/replace/...) intentionally stay
    // exact to preserve historical "this tool only" allow-key semantics.
    const std::string legacy_shell_prefix =
        std::string(core::tools::names::kRunTerminalCommand) + ":";
    if (trimmed.starts_with(legacy_shell_prefix)) {
        const std::string_view suffix =
            trim_ascii(trimmed.substr(core::tools::names::kRunTerminalCommand.size() + 1));
        if (suffix.empty()) {
            return ParsedSessionRule{
                .kind = SessionRuleKind::ExactAllowKey,
                .value = std::string(trimmed_raw),
            };
        }
        return ParsedSessionRule{
            .kind = SessionRuleKind::ShellProgram,
            .value = to_lower_ascii_copy(strip_path_prefix(suffix)),
        };
    }
    if (trimmed == core::tools::names::kRunTerminalCommand) {
        return ParsedSessionRule{
            .kind = SessionRuleKind::ExactAllowKey,
            .value = std::string(trimmed_raw),
        };
    }
    if (trimmed == core::tools::names::kDeleteFile) {
        return ParsedSessionRule{.kind = SessionRuleKind::FilesDelete, .value = "delete"};
    }
    if (trimmed == core::tools::names::kMoveFile) {
        return ParsedSessionRule{.kind = SessionRuleKind::FilesMove, .value = "move"};
    }
    return ParsedSessionRule{
        .kind = SessionRuleKind::ExactAllowKey,
        .value = std::string(trimmed_raw),
    };
}

std::string session_rule_to_string(const ParsedSessionRule& rule) {
    switch (rule.kind) {
        case SessionRuleKind::ToolName:
            return rule.value.empty() ? std::string{} : std::format("tool:{}", rule.value);
        case SessionRuleKind::ShellAny:
            return "shell:*";
        case SessionRuleKind::ShellProgram:
            return rule.value.empty() ? std::string{} : std::format("shell:{}", rule.value);
        case SessionRuleKind::FilesAny:
            return "files:*";
        case SessionRuleKind::FilesWrite:
            return "files:write";
        case SessionRuleKind::FilesDelete:
            return "files:delete";
        case SessionRuleKind::FilesMove:
            return "files:move";
        case SessionRuleKind::ExactAllowKey:
            return rule.value;
    }
    return {};
}

bool parsed_session_rule_matches(const ParsedSessionRule& rule,
                                 std::string_view tool_name,
                                 std::string_view tool_args) {
    switch (rule.kind) {
        case SessionRuleKind::ToolName:
            return iequals_ascii(tool_name, rule.value);
        case SessionRuleKind::ShellAny:
            return tool_name == core::tools::names::kRunTerminalCommand;
        case SessionRuleKind::ShellProgram: {
            if (tool_name != core::tools::names::kRunTerminalCommand) {
                return false;
            }
            const auto current_program = extract_shell_program(tool_args);
            return !current_program.empty() && iequals_ascii(current_program, rule.value);
        }
        case SessionRuleKind::FilesAny:
            return core::tools::names::is_file_modification_tool(tool_name);
        case SessionRuleKind::FilesWrite:
            return is_write_like_file_tool(tool_name);
        case SessionRuleKind::FilesDelete:
            return tool_name == core::tools::names::kDeleteFile;
        case SessionRuleKind::FilesMove:
            return tool_name == core::tools::names::kMoveFile;
        case SessionRuleKind::ExactAllowKey:
            return !rule.value.empty() && make_allow_key(tool_name, tool_args) == rule.value;
    }
    return false;
}

std::string describe_parsed_session_rule(const ParsedSessionRule& rule) {
    switch (rule.kind) {
        case SessionRuleKind::ToolName:
            return rule.value.empty()
                ? std::string("Specific tool")
                : std::format("Tool `{}`", rule.value);
        case SessionRuleKind::ShellAny:
            return "All terminal commands";
        case SessionRuleKind::ShellProgram:
            return rule.value.empty()
                ? std::string("Terminal commands")
                : std::format("Terminal program `{}`", rule.value);
        case SessionRuleKind::FilesAny:
            return "All file operations";
        case SessionRuleKind::FilesWrite:
            return "File modifications (write/apply/replace/create)";
        case SessionRuleKind::FilesDelete:
            return "File deletions";
        case SessionRuleKind::FilesMove:
            return "File moves";
        case SessionRuleKind::ExactAllowKey:
            return rule.value.empty()
                ? std::string("Exact rule")
                : std::format("Exact allow key `{}`", rule.value);
    }
    return {};
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
    
    // Check session allow list (supports both exact legacy keys and the new
    // canonical session trust rules such as shell:git / files:write / tool:x).
    const auto allow_rules = allow_list_.snapshot();
    for (const auto& allow_rule : allow_rules) {
        if (session_allow_rule_matches(allow_rule, tool_name, tool_args)) {
            return true;
        }
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
    if (remember) {
        const std::string remember_rule = make_session_allow_rule(
            current_request_->request.tool_name,
            current_request_->request.tool_args);
        if (!remember_rule.empty()) {
            allow_list_.insert(remember_rule);
        }
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

std::string make_session_allow_rule(std::string_view tool_name,
                                    std::string_view tool_args) {
    if (tool_name == core::tools::names::kRunTerminalCommand) {
        const auto program = to_lower_ascii_copy(extract_shell_program(tool_args));
        if (!program.empty()) {
            return std::format("shell:{}", program);
        }

        // Fail-safe default: if we cannot confidently scope to a single shell
        // program, preserve the historical exact allow-key behavior instead of
        // broadening to shell:*.
        return make_allow_key(tool_name, tool_args);
    }

    if (tool_name == core::tools::names::kDeleteFile) {
        return "files:delete";
    }
    if (tool_name == core::tools::names::kMoveFile) {
        return "files:move";
    }
    if (is_write_like_file_tool(tool_name)) {
        return "files:write";
    }
    if (core::tools::names::is_file_modification_tool(tool_name)) {
        return "files:*";
    }

    const std::string normalized_tool = to_lower_ascii_copy(trim_ascii(tool_name));
    if (normalized_tool.empty()) {
        return {};
    }
    return std::format("tool:{}", normalized_tool);
}

std::string normalize_session_allow_rule(std::string_view rule) {
    return session_rule_to_string(parse_session_rule(rule));
}

bool session_allow_rule_matches(std::string_view rule,
                                std::string_view tool_name,
                                std::string_view tool_args) {
    const ParsedSessionRule parsed = parse_session_rule(rule);
    if (parsed.kind == SessionRuleKind::ExactAllowKey && parsed.value.empty()) {
        return false;
    }
    return parsed_session_rule_matches(parsed, tool_name, tool_args);
}

std::string describe_session_allow_rule(std::string_view rule) {
    const ParsedSessionRule parsed = parse_session_rule(rule);
    if (parsed.kind == SessionRuleKind::ExactAllowKey && parsed.value.empty()) {
        return "Empty rule";
    }
    return describe_parsed_session_rule(parsed);
}

} // namespace core::permissions
