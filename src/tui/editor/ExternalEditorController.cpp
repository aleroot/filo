#include "ExternalEditorController.hpp"

#include <format>
#include <utility>

namespace tui::editor {
namespace {

[[nodiscard]] EditorNotice failure(std::string message) {
    return EditorNotice{.success = false, .message = std::move(message)};
}

} // namespace

ExternalEditorController::ExternalEditorController(Host host, const PromptEditorCatalog& catalog)
    : host_(std::move(host)), catalog_(catalog) {}

ExternalEditorController::~ExternalEditorController() {
    join_worker();
}

std::optional<EditorOutcome> ExternalEditorController::open(
    std::string_view configured_editor,
    std::string_view draft) {
    if (busy()) {
        return std::nullopt;
    }
    // A previous detached session may have finished without being joined yet.
    join_worker();

    auto selection = select_prompt_editor(catalog_, configured_editor);
    auto buffer = PromptBuffer::create(draft);
    if (!buffer) {
        return EditorOutcome{
            .text = std::nullopt,
            .notice = failure(std::format(
                "Could not prepare the editor buffer: {}",
                buffer.error())),
        };
    }

    const EditorDescriptor descriptor = selection.editor->descriptor();
    auto session = std::make_shared<Session>(Session{
        .buffer = std::move(*buffer),
        .editor = std::move(selection.editor),
        .label = std::string(descriptor.label),
        .warning = std::move(selection.warning),
    });

    if (descriptor.launch == LaunchMode::Terminal) {
        const EditContext context{
            .file = session->buffer.file(),
            .session_id = session->buffer.session_id(),
            .cancellation = {},
            .on_progress = {},
            .with_terminal = host_.with_terminal,
        };
        return finish(*session, session->editor->edit(context));
    }

    {
        std::lock_guard lock(mutex_);
        running_ = true;
        cancelling_ = false;
        progress_ = EditProgress::Launching;
        label_ = session->label;
        outcome_.reset();
    }
    run_detached(std::move(session));
    return std::nullopt;
}

void ExternalEditorController::run_detached(std::shared_ptr<Session> session) {
    worker_ = std::jthread([this, session = std::move(session)](std::stop_token stop) {
        const EditContext context{
            .file = session->buffer.file(),
            .session_id = session->buffer.session_id(),
            .cancellation = std::move(stop),
            .on_progress = [this](EditProgress progress) { set_progress(progress); },
            .with_terminal = {},
        };
        publish(finish(*session, session->editor->edit(context)));
    });
}

EditorOutcome ExternalEditorController::finish(
    const Session& session,
    const EditResult& result) const {
    // A selection warning is reported even when the edit itself succeeded, so
    // the user learns their configured value was not honoured verbatim.
    auto with_warning = [&session](EditorOutcome outcome) {
        if (!session.warning.empty() && !outcome.notice) {
            outcome.notice = failure(session.warning);
        }
        return outcome;
    };

    switch (result.status()) {
        case EditStatus::Saved: {
            auto text = session.buffer.read();
            if (!text) {
                return EditorOutcome{
                    .text = std::nullopt,
                    .notice = failure(std::format(
                        "{} editor: {}",
                        session.label,
                        text.error())),
                };
            }
            return with_warning(EditorOutcome{.text = std::move(*text), .notice = std::nullopt});
        }
        case EditStatus::Cancelled:
            // Cancelling is a deliberate, silent user action.
            return with_warning(EditorOutcome{});
        case EditStatus::Failed:
            break;
    }
    return EditorOutcome{
        .text = std::nullopt,
        .notice = failure(std::format("{} editor: {}", session.label, result.error())),
    };
}

void ExternalEditorController::publish(EditorOutcome outcome) {
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

void ExternalEditorController::set_progress(EditProgress progress) {
    {
        std::lock_guard lock(mutex_);
        if (!running_) {
            return;
        }
        progress_ = progress;
    }
    if (host_.wake_ui) {
        host_.wake_ui();
    }
}

bool ExternalEditorController::busy() const {
    std::lock_guard lock(mutex_);
    return running_;
}

std::string ExternalEditorController::status_label() const {
    std::lock_guard lock(mutex_);
    if (!running_) {
        return {};
    }
    if (cancelling_) {
        return std::format("Cancelling the {} edit…", label_);
    }
    return progress_ == EditProgress::Launching
        ? std::format("Opening {}…", label_)
        : std::format("Editing in {}…", label_);
}

std::optional<EditorOutcome> ExternalEditorController::take_outcome() {
    std::lock_guard lock(mutex_);
    return std::exchange(outcome_, std::nullopt);
}

bool ExternalEditorController::cancel() {
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

void ExternalEditorController::join_worker() {
    if (!worker_.joinable()) {
        return;
    }
    worker_.request_stop();
    worker_.join();
}

} // namespace tui::editor
