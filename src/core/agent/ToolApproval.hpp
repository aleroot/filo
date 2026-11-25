#pragma once

#include <optional>
#include <string>

namespace core::agent {

/// What the approval gates decided about one tool call: the PreToolUse hook,
/// then the permission check. A direct call and a call from an `exec` script
/// pass the same gates.
struct ToolApproval {
    bool approved = false;
    /// A delegated task that only reads: approved without asking, and
    /// scheduled as a reader.
    bool read_only_task = false;
    /// Set when a PreToolUse hook blocked the call: the hook's reason, which
    /// may be empty.
    std::optional<std::string> hook_block = {};
};

} // namespace core::agent
