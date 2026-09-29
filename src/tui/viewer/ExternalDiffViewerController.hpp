#pragma once

#include "DiffViewer.hpp"
#include "DiffViewerCatalog.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

namespace tui::viewer {

struct ViewerNotice {
    bool success = false;
    std::string message;
};

// What the host applies once a comparison settles.
struct ViewerOutcome {
    // The configured comparer is the transcript itself, so the host shows the
    // diff instead of waiting for an application.
    bool inline_view = false;
    // An external comparer put the diff on screen. The host attaches what it
    // knows about the comparison (such as files the patch could not carry)
    // only then, so a failed or cancelled open never talks about a diff the
    // user is not looking at.
    bool shown = false;
    // User-facing message, if any. Success is silent: the diff on screen is the
    // confirmation.
    std::optional<ViewerNotice> notice;
};

// Drives one comparison at a time, whichever backend is configured.
//
// The controller owns the whole lifecycle — backend selection, session id,
// worker thread, cancellation — so the TUI only has to ask three questions: is
// a comparison running, what should the overlay say, and is there an outcome to
// apply.
class ExternalDiffViewerController {
public:
    // Capabilities the controller borrows from the TUI.
    struct Host {
        // Requests a redraw; called from the worker thread.
        std::function<void()> wake_ui;
        // Restores the terminal for the duration of a child process.
        std::function<void(const std::function<void()>&)> with_terminal;
    };

    explicit ExternalDiffViewerController(
        Host host,
        const DiffViewerCatalog& catalog = builtin_diff_comparers());

    ~ExternalDiffViewerController();

    ExternalDiffViewerController(const ExternalDiffViewerController&) = delete;
    ExternalDiffViewerController& operator=(const ExternalDiffViewerController&) = delete;

    // Shows `patch` in the configured comparer. Backends that settle
    // synchronously — the built-in view and a terminal pager — return their
    // outcome; a detached backend returns nullopt and publishes through
    // take_outcome() once the other application has answered.
    [[nodiscard]] std::optional<ViewerOutcome> open(
        std::string_view configured_comparer,
        std::string_view patch);

    // True while a detached session holds the diff.
    [[nodiscard]] bool busy() const;

    // Overlay text for the running session; empty when idle.
    [[nodiscard]] std::string status_label() const;

    // Hands over a finished session's outcome exactly once.
    [[nodiscard]] std::optional<ViewerOutcome> take_outcome();

    // Asks the running session to stop. Returns false when nothing was running.
    bool cancel();

private:
    struct Session {
        std::string patch;
        std::string session_id;
        std::unique_ptr<DiffViewer> viewer;
        std::string label;
    };

    void run_detached(std::shared_ptr<Session> session);
    void publish(ViewerOutcome outcome);
    [[nodiscard]] ViewerOutcome finish(const Session& session, const ViewResult& result) const;
    void join_worker();

    Host host_;
    const DiffViewerCatalog& catalog_;

    mutable std::mutex mutex_;
    bool running_ = false;
    bool cancelling_ = false;
    std::string label_;
    std::optional<ViewerOutcome> outcome_;

    std::jthread worker_;
};

} // namespace tui::viewer
