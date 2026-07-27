#pragma once

#include "PromptBuffer.hpp"
#include "PromptEditor.hpp"
#include "PromptEditorCatalog.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace tui::editor {

struct EditorNotice {
    bool success = false;
    std::string message;
};

// What the host applies once an edit finishes.
struct EditorOutcome {
    std::optional<std::string> text;    // Replacement draft; set only when saved.
    std::optional<EditorNotice> notice; // User-facing message, if any.
};

// Drives one prompt-edit session at a time, whichever backend is configured.
//
// The controller owns the whole lifecycle — scratch buffer, backend selection,
// worker thread, cancellation — so the TUI only has to ask three questions:
// is a session running, what should the overlay say, and is there an outcome to
// apply.
class ExternalEditorController {
public:
    // Capabilities the controller borrows from the TUI.
    struct Host {
        // Requests a redraw; called from the worker thread.
        std::function<void()> wake_ui;
        // Restores the terminal for the duration of a child process.
        std::function<void(const std::function<void()>&)> with_terminal;
    };

    explicit ExternalEditorController(
        Host host,
        const PromptEditorCatalog& catalog = builtin_prompt_editors());

    ~ExternalEditorController();

    ExternalEditorController(const ExternalEditorController&) = delete;
    ExternalEditorController& operator=(const ExternalEditorController&) = delete;

    // Starts an edit of `draft` using the configured backend.
    // Terminal backends finish inline and return their outcome; detached
    // backends return nullopt and publish through take_outcome() later.
    [[nodiscard]] std::optional<EditorOutcome> open(
        std::string_view configured_editor,
        std::string_view draft);

    // True while a detached session holds the prompt.
    [[nodiscard]] bool busy() const;

    // Overlay text for the running session; empty when idle.
    [[nodiscard]] std::string status_label() const;

    // Hands over a finished session's outcome exactly once.
    [[nodiscard]] std::optional<EditorOutcome> take_outcome();

    // Asks the running session to stop. Returns false when nothing was running.
    bool cancel();

private:
    struct Session {
        PromptBuffer buffer;
        std::unique_ptr<PromptEditor> editor;
        std::string label;
        std::string warning;  // Set when the configured id was not usable verbatim.
    };

    void run_detached(std::shared_ptr<Session> session);
    void publish(EditorOutcome outcome);
    void set_progress(EditProgress progress);
    [[nodiscard]] EditorOutcome finish(const Session& session, const EditResult& result) const;
    void join_worker();

    Host host_;
    const PromptEditorCatalog& catalog_;

    mutable std::mutex mutex_;
    bool running_ = false;
    bool cancelling_ = false;
    EditProgress progress_ = EditProgress::Launching;
    std::string label_;
    std::optional<EditorOutcome> outcome_;

    std::jthread worker_;
};

} // namespace tui::editor
