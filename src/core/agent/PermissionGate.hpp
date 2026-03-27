#pragma once

#include "SafetyPolicy.hpp"

#include <string>
#include <string_view>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>

namespace core::agent {

// ---------------------------------------------------------------------------
// PendingPermission — represents one in-flight permission request.
//
// The Agent worker thread creates this on the stack, moves it into
// PermissionGate, then blocks on its future.  The TUI thread calls
// resolve(true/false) to unblock the worker.
// ---------------------------------------------------------------------------
struct PendingPermission {
    std::string tool_name;
    std::string args_preview;   // first 300 chars of the JSON args
    std::shared_ptr<std::promise<bool>> promise;

    [[nodiscard]] bool valid() const noexcept { return promise != nullptr; }
};

// ---------------------------------------------------------------------------
// PermissionGate — singleton mediator between the Agent's worker thread
// (which needs approval) and the TUI thread (which shows the prompt).
//
// Thread-safety contract:
//   • request()  — called from a worker thread; blocks until resolve() is called.
//   • pop_pending() — called from TUI thread to retrieve the pending request.
//   • resolve()  — called from TUI thread to answer the pending request.
//   • set_request_ui_fn() — called once from TUI setup thread.
// ---------------------------------------------------------------------------
class PermissionGate {
public:
    static PermissionGate& get_instance() noexcept {
        static PermissionGate instance;
        return instance;
    }

    // Called once by MainApp to register a UI notification callback.
    // The callback is invoked (from the worker thread) when a permission
    // request arrives.  It must return quickly — it only signals the UI.
    void set_notify_fn(std::function<void()> fn) {
        std::lock_guard lock(mutex_);
        notify_fn_ = std::move(fn);
    }

    // Called by the Agent worker thread for each dangerous tool call.
    // Blocks until the TUI resolves the request.
    // Returns true  → proceed with the tool call.
    // Returns false → skip the tool call.
    [[nodiscard]] bool request(std::string_view tool_name, std::string_view raw_args) {
        auto prom = std::make_shared<std::promise<bool>>();
        auto fut  = prom->get_future();

        PendingPermission perm;
        perm.tool_name    = std::string(tool_name);
        perm.args_preview = raw_args.size() > 300
                            ? std::string(raw_args.substr(0, 300)) + "…"
                            : std::string(raw_args);
        perm.promise      = prom;

        std::function<void()> notify;
        {
            std::lock_guard lock(mutex_);
            pending_ = std::move(perm);
            notify   = notify_fn_;
        }

        if (notify) notify();   // signal the TUI (non-blocking)

        return fut.get();       // BLOCKS until TUI calls resolve()
    }

    // TUI thread: returns true if there is a pending request.
    [[nodiscard]] bool has_pending() const noexcept {
        std::lock_guard lock(mutex_);
        return pending_.has_value();
    }

    // TUI thread: retrieve a copy of the pending request for rendering.
    [[nodiscard]] std::optional<PendingPermission> peek_pending() const {
        std::lock_guard lock(mutex_);
        return pending_;
    }

    // TUI thread: resolve the pending request.  Must only be called when
    // has_pending() returns true.
    void resolve(bool allowed) {
        std::shared_ptr<std::promise<bool>> prom;
        {
            std::lock_guard lock(mutex_);
            if (!pending_.has_value()) return;
            prom = pending_->promise;
            pending_.reset();
        }
        if (prom) prom->set_value(allowed);
    }

private:
    PermissionGate() noexcept = default;

    mutable std::mutex mutex_;
    std::optional<PendingPermission> pending_;
    std::function<void()> notify_fn_;
};

// ---------------------------------------------------------------------------
// Permission Profiles
// ---------------------------------------------------------------------------

enum class PermissionProfile {
    Restricted,  // Read-only logic (usually enforced by tool filtering, but acts as "Always Ask" here)
    Interactive, // Ask for all destructive actions (Default)
    Standard,    // Allow file edits, ask for shell/task/lifecycle
    Autonomous   // Allow everything
};

// ---------------------------------------------------------------------------
// Helper: Checks if a tool requires permission under the given profile.
//
// For run_terminal_command the optional `tool_args` JSON blob is inspected by
// CommandSafetyPolicy so that read-only commands (ls, grep, git log, …) are
// auto-approved even in Interactive mode, while state-changing commands
// (rm, git push, npm install, …) always prompt the user.
// ---------------------------------------------------------------------------

[[nodiscard]] inline bool needs_permission(std::string_view tool_name,
                                           PermissionProfile profile = PermissionProfile::Interactive,
                                           std::string_view tool_args = {}) noexcept {
    // 1. Autonomous mode: Trust everything.
    if (profile == PermissionProfile::Autonomous) {
        return false;
    }

    // 2. Identify tool categories.
    const bool is_file_mod = (tool_name == "write_file"
                           || tool_name == "apply_patch"
                           || tool_name == "replace"
                           || tool_name == "replace_in_file"
                           || tool_name == "delete_file"
                           || tool_name == "move_file"
                           || tool_name == "create_directory");

    const bool is_task = (tool_name == "task");
    const bool is_shell = (tool_name == "run_terminal_command");

    // 3. Standard mode: Allow file mods, gate shell/task.
    if (profile == PermissionProfile::Standard) {
        if (is_file_mod) return false;
        if (is_task)    return true;
        if (is_shell) {
            // Even in Standard mode, purely read-only shell commands are safe.
            if (!tool_args.empty()) {
                const auto cmd = CommandSafetyPolicy::extract_shell_command(tool_args);
                if (!cmd.empty() && !CommandSafetyPolicy::command_needs_permission(cmd))
                    return false;
            }
            return true;
        }
        return false; // Safe tools (read_file, etc.) are always allowed.
    }

    // 4. Restricted: Ask for everything.
    if (profile == PermissionProfile::Restricted) {
        return is_file_mod || is_shell || is_task;
    }

    // 5. Interactive: Ask for all side-effects, but apply SafetyPolicy to shell
    //    commands so purely read-only commands are auto-approved.
    if (is_file_mod || is_task) return true;
    if (is_shell) {
        if (!tool_args.empty()) {
            const auto cmd = CommandSafetyPolicy::extract_shell_command(tool_args);
            if (!cmd.empty() && !CommandSafetyPolicy::command_needs_permission(cmd))
                return false;
        }
        return true;
    }
    return false;
}

} // namespace core::agent
