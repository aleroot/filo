#pragma once

#include <filesystem>
#include <functional>
#include <stop_token>
#include <string>
#include <string_view>

namespace tui::editor {

// How a backend takes control of the machine while the user edits.
enum class LaunchMode {
    // Blocking and terminal-bound (vi, nano, ...): the host suspends the TUI
    // and runs the session inline on the UI thread.
    Terminal,
    // Non-blocking and out-of-process (a GUI app): the host runs the session on
    // a worker thread and keeps the TUI responsive behind a status overlay.
    Detached,
};

// Static, user-facing identity of a backend. String views point at storage with
// static lifetime, so descriptors are trivially copyable and cheap to pass by
// value.
struct EditorDescriptor {
    std::string_view id;           // Stable `prompt_editor` config value.
    std::string_view label;        // Settings-pane label.
    std::string_view description;  // Settings-pane help line.
    LaunchMode launch = LaunchMode::Terminal;
};

enum class EditStatus {
    Saved,
    Cancelled,
    Failed,
};

class EditResult {
public:
    [[nodiscard]] static EditResult saved() noexcept { return EditResult{EditStatus::Saved, {}}; }

    [[nodiscard]] static EditResult cancelled() noexcept {
        return EditResult{EditStatus::Cancelled, {}};
    }

    [[nodiscard]] static EditResult failed(std::string reason) {
        return EditResult{EditStatus::Failed, std::move(reason)};
    }

    [[nodiscard]] EditStatus status() const noexcept { return status_; }
    [[nodiscard]] const std::string& error() const noexcept { return error_; }

private:
    EditResult(EditStatus status, std::string error)
        : status_(status), error_(std::move(error)) {}

    EditStatus status_ = EditStatus::Cancelled;
    std::string error_;
};

// Lifecycle signals a backend may report while a session is running.
enum class EditProgress {
    Launching,
    Editing,
};

// Everything a backend is allowed to touch during one edit. The host injects
// these capabilities so backends never depend on the TUI, the config, or on
// each other.
struct EditContext {
    // Buffer the backend must edit in place; the host owns its lifetime.
    std::filesystem::path file;
    // Opaque per-session token. Backends that speak a wire protocol use it to
    // correlate their messages with this edit.
    std::string session_id;
    // Signalled when the user cancels. Detached backends must poll it.
    std::stop_token cancellation;
    // Optional progress sink; may be empty. Prefer report_progress().
    std::function<void(EditProgress)> on_progress;
    // Hands the terminal to a child process. Only populated (and only
    // meaningful) for LaunchMode::Terminal backends.
    std::function<void(const std::function<void()>&)> with_terminal;

    void report_progress(EditProgress progress) const {
        if (on_progress) {
            on_progress(progress);
        }
    }
};

// A backend able to edit the current prompt draft.
//
// Implementations are single-use per session and are created through the
// PromptEditorCatalog, so adding an IDE integration never requires touching the
// TUI: implement this interface and register a descriptor.
class PromptEditor {
public:
    virtual ~PromptEditor() = default;

    PromptEditor(const PromptEditor&) = delete;
    PromptEditor& operator=(const PromptEditor&) = delete;
    PromptEditor(PromptEditor&&) = delete;
    PromptEditor& operator=(PromptEditor&&) = delete;

    [[nodiscard]] virtual EditorDescriptor descriptor() const noexcept = 0;

    // Runs one edit to completion. Terminal backends are called on the UI
    // thread; detached backends are called on a worker thread.
    [[nodiscard]] virtual EditResult edit(const EditContext& context) = 0;

protected:
    PromptEditor() = default;
};

} // namespace tui::editor
