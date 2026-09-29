#pragma once

#include <functional>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>

namespace tui::viewer {

// How a backend takes control of the machine while the user reads a diff.
enum class ViewerLaunch {
    // Nothing is launched: the transcript already shows every diff. This is the
    // launch mode of the built-in choice, which has no backend behind it.
    Inline,
    // Blocking and terminal-bound (a pager): the host suspends the TUI and runs
    // the session inline on the UI thread.
    Terminal,
    // Non-blocking and out-of-process (a GUI app): the host runs the session on
    // a worker thread and keeps the TUI responsive behind a status overlay.
    Detached,
};

// Static, user-facing identity of a backend. String views point at storage with
// static lifetime, so descriptors are trivially copyable and cheap to pass by
// value.
struct ViewerDescriptor {
    std::string_view id;           // Stable `diff_comparer` config value.
    std::string_view label;        // Settings-pane label.
    std::string_view description;  // Settings-pane help line.
    ViewerLaunch launch = ViewerLaunch::Terminal;
};

enum class ViewStatus {
    Shown,
    Cancelled,
    Failed,
};

class ViewResult {
public:
    [[nodiscard]] static ViewResult shown() noexcept { return ViewResult{ViewStatus::Shown, {}}; }

    [[nodiscard]] static ViewResult cancelled() noexcept {
        return ViewResult{ViewStatus::Cancelled, {}};
    }

    [[nodiscard]] static ViewResult failed(std::string reason) {
        return ViewResult{ViewStatus::Failed, std::move(reason)};
    }

    [[nodiscard]] ViewStatus status() const noexcept { return status_; }
    [[nodiscard]] const std::string& error() const noexcept { return error_; }

private:
    ViewResult(ViewStatus status, std::string error)
        : status_(status), error_(std::move(error)) {}

    ViewStatus status_ = ViewStatus::Cancelled;
    std::string error_;
};

// Everything a backend is allowed to touch during one comparison. The host
// injects these capabilities so backends never depend on the TUI, the config,
// or on each other.
struct ViewContext {
    /// The unified diff to present. The host owns it for the whole call.
    std::string_view patch;
    /// Opaque per-session token: 16 hexadecimal characters, the shape a wire
    /// protocol correlates its messages by.
    std::string session_id;
    /// Signalled when the user cancels. Detached backends must poll it.
    std::stop_token cancellation;
    /// Hands the terminal to a child process. Only populated (and only
    /// meaningful) for ViewerLaunch::Terminal backends.
    std::function<void(const std::function<void()>&)> with_terminal;
};

/// A backend able to present a diff outside the transcript.
///
/// Unlike a prompt editor, a viewer is one-way: it takes the patch and reports
/// whether the user got to see it. Implementations are single-use per session
/// and are created through the DiffViewerCatalog, so adding a comparer never
/// requires touching the TUI: implement this interface and register it.
class DiffViewer {
public:
    virtual ~DiffViewer() = default;

    DiffViewer(const DiffViewer&) = delete;
    DiffViewer& operator=(const DiffViewer&) = delete;
    DiffViewer(DiffViewer&&) = delete;
    DiffViewer& operator=(DiffViewer&&) = delete;

    [[nodiscard]] virtual ViewerDescriptor descriptor() const noexcept = 0;

    // Runs one comparison to completion. Terminal backends are called on the UI
    // thread; detached backends are called on a worker thread.
    [[nodiscard]] virtual ViewResult view(const ViewContext& context) = 0;

protected:
    DiffViewer() = default;
};

} // namespace tui::viewer
