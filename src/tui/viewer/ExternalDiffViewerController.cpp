#include "ExternalDiffViewerController.hpp"

#include "core/session/SessionStore.hpp"

#include <format>
#include <utility>

namespace tui::viewer {
namespace {

[[nodiscard]] ViewerNotice failure(std::string message) {
    return ViewerNotice{.success = false, .message = std::move(message)};
}

// Two session ids produce the 16 hexadecimal characters a wire protocol
// requires of a session identifier.
[[nodiscard]] std::string make_session_id() {
    return core::session::SessionStore::generate_id() + core::session::SessionStore::generate_id();
}

} // namespace

ExternalDiffViewerController::ExternalDiffViewerController(
    Host host,
    const DiffViewerCatalog& catalog)
    : host_(std::move(host)), catalog_(catalog) {}

ExternalDiffViewerController::~ExternalDiffViewerController() {
    join_worker();
}

std::optional<ViewerOutcome> ExternalDiffViewerController::open(
    std::string_view configured_comparer,
    std::string_view patch) {
    if (busy()) {
        return std::nullopt;
    }
    // A previous detached session may have finished without being joined yet.
    join_worker();

    auto selection = select_diff_viewer(catalog_, configured_comparer);
    if (selection.inline_view()) {
        // Nothing is launched, so the host renders the diff itself. A selection
        // warning travels with that answer, since this is the only path that
        // can produce one.
        return ViewerOutcome{
            .inline_view = true,
            .notice = selection.warning.empty()
                ? std::nullopt
                : std::optional(failure(std::move(selection.warning))),
            .comments = {},
        };
    }

    const ViewerDescriptor descriptor = selection.viewer->descriptor();
    auto session = std::make_shared<Session>(Session{
        .patch = std::string(patch),
        .session_id = make_session_id(),
        .viewer = std::move(selection.viewer),
        .label = std::string(descriptor.label),
    });

    if (descriptor.launch == ViewerLaunch::Terminal) {
        const ViewContext context{
            .patch = session->patch,
            .session_id = session->session_id,
            .cancellation = {},
            .with_terminal = host_.with_terminal,
            .report_viewing = {},
        };
        return finish(*session, session->viewer->view(context));
    }

    {
        std::lock_guard lock(mutex_);
        running_ = true;
        viewing_ = false;
        cancelling_ = false;
        label_ = session->label;
        outcome_.reset();
    }
    run_detached(std::move(session));
    return std::nullopt;
}

void ExternalDiffViewerController::run_detached(std::shared_ptr<Session> session) {
    worker_ = std::jthread([this, session = std::move(session)](std::stop_token stop) {
        const ViewContext context{
            .patch = session->patch,
            .session_id = session->session_id,
            .cancellation = std::move(stop),
            .with_terminal = {},
            .report_viewing = [this] {
                {
                    std::lock_guard lock(mutex_);
                    viewing_ = true;
                }
                if (host_.wake_ui) { host_.wake_ui(); }
            },
        };
        publish(finish(*session, session->viewer->view(context)));
    });
}

ViewerOutcome ExternalDiffViewerController::finish(
    const Session& session,
    const ViewResult& result) const {
    switch (result.status()) {
        case ViewStatus::Shown:
            // The diff is on screen; a transcript line would only repeat it.
            return ViewerOutcome{
                .inline_view = false, .shown = true, .notice = std::nullopt,
                .comments = result.comments(),
            };
        case ViewStatus::Cancelled:
            // Cancelling is a deliberate, silent user action.
            return ViewerOutcome{};
        case ViewStatus::Failed:
            break;
    }
    return ViewerOutcome{
        .inline_view = false,
        .shown = false,
        .notice = failure(std::format("{} comparer: {}", session.label, result.error())),
        .comments = {},
    };
}

void ExternalDiffViewerController::publish(ViewerOutcome outcome) {
    {
        std::lock_guard lock(mutex_);
        outcome_ = std::move(outcome);
        running_ = false;
        cancelling_ = false;
        label_.clear();
    }
    if (host_.wake_ui) {
        host_.wake_ui();
    }
}

bool ExternalDiffViewerController::busy() const {
    std::lock_guard lock(mutex_);
    return running_;
}

std::string ExternalDiffViewerController::status_label() const {
    std::lock_guard lock(mutex_);
    if (!running_) {
        return {};
    }
    if (cancelling_) {
        return std::format("Cancelling the {} comparison…", label_);
    }
    return viewing_
        ? std::format("Reviewing in {}…", label_)
        : std::format("Opening {}…", label_);
}

std::optional<ViewerOutcome> ExternalDiffViewerController::take_outcome() {
    std::lock_guard lock(mutex_);
    return std::exchange(outcome_, std::nullopt);
}

bool ExternalDiffViewerController::cancel() {
    {
        std::lock_guard lock(mutex_);
        if (!running_ || cancelling_) {
            return false;
        }
        cancelling_ = true;
    }
    worker_.request_stop();
    if (host_.wake_ui) {
        host_.wake_ui();
    }
    return true;
}

void ExternalDiffViewerController::join_worker() {
    if (!worker_.joinable()) {
        return;
    }
    worker_.request_stop();
    worker_.join();
}

} // namespace tui::viewer
