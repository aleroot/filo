#pragma once

#include "QuestionDialog.hpp"
#include "core/tools/AskUserQuestionTool.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace tui {

using QuestionDialogAnswers =
    std::vector<std::pair<std::string, std::string>>;
using QuestionDialogPromise =
    std::shared_ptr<std::promise<std::optional<QuestionDialogAnswers>>>;

struct QuestionDialogEventResult {
    bool handled = false;
    bool stop_agent = false;
    bool restore_main_input_focus = false;
    QuestionDialogPromise promise;
    std::optional<QuestionDialogAnswers> answers;
    /// Session that owns the waiting tool call. Needed to interrupt a hidden
    /// thread instead of whichever thread happens to be visible.
    std::string origin_session_id;

    [[nodiscard]] bool has_resolution() const noexcept {
        return promise != nullptr;
    }

    void resolve();
};

class QuestionDialogController final {
public:
    QuestionDialogController();

    QuestionDialogController(const QuestionDialogController&) = delete;
    QuestionDialogController& operator=(const QuestionDialogController&) = delete;
    QuestionDialogController(QuestionDialogController&&) = delete;
    QuestionDialogController& operator=(QuestionDialogController&&) = delete;

    [[nodiscard]] QuestionDialogPromise open(
        core::tools::QuestionRequest request,
        std::string origin_label = {});

    /// Thread label captured by open(); rendered next to the dialog so users
    /// know which conversation is waiting on them.
    [[nodiscard]] std::string origin_label() const;

    /// Force-dismiss an active dialog with an interrupt. Used at shutdown so
    /// the waiting tool thread never blocks process exit; returns the result
    /// whose promise must still be resolved by the caller.
    [[nodiscard]] QuestionDialogEventResult force_interrupt();

    [[nodiscard]] bool active() const;
    [[nodiscard]] ftxui::Component editor_component() const;
    [[nodiscard]] ftxui::Element render();

    [[nodiscard]] QuestionDialogEventResult handle_event(
        const ftxui::Event& event,
        bool is_interrupt);

private:
    void focus_other_editor_if_selected();
    void sync_selection_focus(QuestionDialogEventResult& result);
    void apply_answer_progress(
        QuestionDialogAnswerProgress progress,
        QuestionDialogEventResult& result);
    void dismiss(bool is_interrupt, QuestionDialogEventResult& result);

    mutable std::mutex mutex_;
    QuestionDialogState state_;
    QuestionDialogPromise promise_;
    std::string origin_label_;
    std::string origin_session_id_;
    bool submit_requested_ = false;
    ftxui::Component editor_component_;
};

} // namespace tui
