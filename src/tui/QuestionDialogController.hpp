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
        core::tools::QuestionRequest request);

    [[nodiscard]] bool active() const;
    [[nodiscard]] ftxui::Component editor_component() const;
    [[nodiscard]] ftxui::Element render();

    [[nodiscard]] QuestionDialogEventResult handle_event(
        const ftxui::Event& event,
        bool is_interrupt);

private:
    void apply_answer_progress(
        QuestionDialogAnswerProgress progress,
        QuestionDialogEventResult& result);
    void dismiss(bool is_interrupt, QuestionDialogEventResult& result);

    mutable std::mutex mutex_;
    QuestionDialogState state_;
    QuestionDialogPromise promise_;
    bool submit_requested_ = false;
    ftxui::Component editor_component_;
};

} // namespace tui
