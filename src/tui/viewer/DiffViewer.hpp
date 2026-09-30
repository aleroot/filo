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
    /// The diff reached the screen, and the comparison ended there.
    [[nodiscard]] static ViewResult shown() noexcept {
        return ViewResult{ViewStatus::Shown, {}, {}};
    }

    /// The diff reached the screen and the reviewer handed comments back, for
    /// the host to put where a draft goes. Hosts treat it as shown.
    [[nodiscard]] static ViewResult reviewed(std::string comments) noexcept {
        return ViewResult{ViewStatus::Shown, {}, std::move(comments)};
    }

    [[nodiscard]] static ViewResult cancelled() noexcept {
        return ViewResult{ViewStatus::Cancelled, {}, {}};
    }

    [[nodiscard]] static ViewResult failed(std::string reason) {
        return ViewResult{ViewStatus::Failed, std::move(reason), {}};
    }

    [[nodiscard]] ViewStatus status() const noexcept { return status_; }
    [[nodiscard]] const std::string& error() const noexcept { return error_; }
    /// What a reviewer returned. Empty for every backend that cannot review.
    [[nodiscard]] const std::string& comments() const noexcept { return comments_; }

private:
    ViewResult(ViewStatus status, std::string error, std::string comments)
        : status_(status), error_(std::move(error)), comments_(std::move(comments)) {}

    ViewStatus status_ = ViewStatus::Cancelled;
    std::string error_;
    std::string comments_;
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
    /// A detached backend calls this once the diff is on screen and it is
    /// waiting for the reviewer, so the host can say so. May be empty.
    std::function<void()> report_viewing;
};

/// A backend able to present a diff outside the transcript.
///
/// A viewer takes the patch and reports whether the user got to see it. One
/// that supports review may also hand back the reviewer's comments
/// (ViewResult::comments()), which the host appends to the prompt draft.
/// Implementations are single-use per session and are created through the
/// DiffViewerCatalog, so adding a comparer never requires touching the TUI:
/// implement this interface and register it.
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
