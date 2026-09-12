#pragma once

// Thread-owned overlays, following FTXUI's Modal contract.
//
// FTXUI 7's Modal is a visibility-gated overlay: `show_modal` chooses which
// child of a Tab receives events, and the hidden child is neither painted nor
// interactive. Filo's chrome uses a bottom panel rather than a centered
// `dbox`, so we do not wrap the tree in `ftxui::Modal`. The same contract
// still applies, keyed by the visible thread:
//
//   show_modal  ≡  overlay.session_id == visible_session_id
//
// Each live thread can wait on its own question or permission (tmux/screen
// windows). A hidden thread keeps its overlay state and blocked worker; the
// physical screen only paints and routes keys to the visible thread's overlay.
//
// Threading: workers call post_* / acquire_*. The TUI thread calls handle_*,
// render_*, and waiting(). Controller methods are never invoked while the
// host mutex is held (lock order: host, then controller — never nested).

#include "DiffPreview.hpp"
#include "QuestionDialogController.hpp"

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tui {

struct PermissionPrompt {
    std::string session_id;
    std::string tool_name;
    std::string args_preview;
    ToolDiffPreview diff_preview;
    int selected = 0;
    std::string remember_rule;
    std::string allow_label;
    std::shared_ptr<std::promise<bool>> promise;
};

/// Immutable copy for rendering. Never an interior pointer into the host.
struct PermissionView {
    std::string session_id;
    std::string tool_name;
    std::string args_preview;
    ToolDiffPreview diff_preview;
    int selected = 0;
    std::string allow_label;
};

struct PermissionEventResult {
    bool handled = false;
    bool restore_main_input_focus = false;
    bool stop_agent = false;
    std::string origin_session_id;
    std::shared_ptr<std::promise<bool>> promise;
    std::optional<bool> approved;
    bool always_allow = false;
    bool enable_yolo = false;
    std::string remember_rule;
    std::string allow_label;

    [[nodiscard]] bool has_resolution() const noexcept {
        return promise != nullptr && approved.has_value();
    }

    void resolve();
};

class ThreadModalHost;

/// Occupies one session's permission slot until destroyed. The worker holds
/// this across the blocking `future.get()` so a second tool call on the same
/// thread cannot overwrite the prompt; other sessions proceed independently.
class PermissionSlotGuard {
public:
    PermissionSlotGuard() noexcept = default;
    ~PermissionSlotGuard();

    PermissionSlotGuard(const PermissionSlotGuard&) = delete;
    PermissionSlotGuard& operator=(const PermissionSlotGuard&) = delete;
    PermissionSlotGuard(PermissionSlotGuard&& other) noexcept;
    PermissionSlotGuard& operator=(PermissionSlotGuard&& other) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept {
        return host_ != nullptr;
    }

private:
    friend class ThreadModalHost;
    PermissionSlotGuard(ThreadModalHost& host, std::string session_id);

    ThreadModalHost* host_ = nullptr;
    std::string session_id_;
};

class ThreadModalHost final {
public:
    ThreadModalHost() = default;
    ~ThreadModalHost();
    ThreadModalHost(const ThreadModalHost&) = delete;
    ThreadModalHost& operator=(const ThreadModalHost&) = delete;

    /// Bind a question to its origin session. Same-session callers wait until
    /// the previous dialog resolves; other sessions are independent. On
    /// shutdown the request is cancelled without opening a dialog.
    void post_question(core::tools::QuestionRequest request);

    [[nodiscard]] bool question_visible(std::string_view visible_session_id) const;
    [[nodiscard]] ftxui::Element render_question(std::string_view visible_session_id);

    /// Forwards keys only while the visible thread owns an active dialog.
    /// Hidden-thread dialogs keep their state and do not steal the keyboard.
    [[nodiscard]] QuestionDialogEventResult handle_question_event(
        std::string_view visible_session_id,
        const ftxui::Event& event,
        bool is_interrupt);

    [[nodiscard]] std::vector<QuestionDialogEventResult> interrupt_all_questions();
    [[nodiscard]] QuestionDialogEventResult interrupt_question(
        std::string_view session_id);

    /// Block until this session's permission slot is free. An empty guard
    /// means the host is shutting down and the caller must deny the request.
    [[nodiscard]] PermissionSlotGuard acquire_permission_slot(std::string session_id);

    void post_permission(PermissionPrompt prompt);
    [[nodiscard]] bool permission_visible(std::string_view visible_session_id) const;
    [[nodiscard]] std::optional<PermissionView> permission_view(
        std::string_view visible_session_id) const;

    /// Arrow keys update selection; confirming keys resolve the promise via
    /// the returned result. Hidden-thread prompts do not steal the keyboard.
    [[nodiscard]] PermissionEventResult handle_permission_event(
        std::string_view visible_session_id,
        const ftxui::Event& event,
        bool is_interrupt,
        bool yolo_shortcut = false);

    [[nodiscard]] bool waiting(std::string_view session_id) const;
    [[nodiscard]] std::unordered_set<std::string> waiting_session_ids() const;

    /// Waiters blocked in post_question / acquire_permission_slot for this
    /// session. Useful for happens-before tests and diagnostics.
    [[nodiscard]] std::size_t blocked_question_waiters(
        std::string_view session_id) const;
    [[nodiscard]] std::size_t blocked_permission_waiters(
        std::string_view session_id) const;

    /// Drop a closed thread's slot. Interrupts a pending question (caller
    /// must resolve the result) and denies a pending permission.
    [[nodiscard]] QuestionDialogEventResult erase_session(
        std::string_view session_id);

    void request_shutdown();
    [[nodiscard]] bool shutting_down() const;

private:
    friend class PermissionSlotGuard;

    struct SessionIdHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view value) const noexcept {
            return std::hash<std::string_view>{}(value);
        }
        std::size_t operator()(const std::string& value) const noexcept {
            return std::hash<std::string_view>{}(value);
        }
    };

    struct SessionSlot {
        std::shared_ptr<QuestionDialogController> question;
        bool question_busy = false;
        std::size_t question_waiters = 0;
        std::optional<PermissionPrompt> permission;
        bool permission_busy = false;
        std::size_t permission_waiters = 0;
    };

    using SlotMap = std::unordered_map<std::string,
                                       SessionSlot,
                                       SessionIdHash,
                                       std::equal_to<>>;

    SessionSlot& slot_locked(const std::string& session_id);
    [[nodiscard]] SessionSlot* find_slot_locked(std::string_view session_id);
    [[nodiscard]] const SessionSlot* find_slot_locked(
        std::string_view session_id) const;
    void release_permission_slot(std::string_view session_id);
    void mark_question_idle(std::string_view session_id);
    [[nodiscard]] std::shared_ptr<QuestionDialogController> busy_question_locked(
        std::string_view session_id) const;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    SlotMap slots_;
    bool shutdown_ = false;
};

} // namespace tui
