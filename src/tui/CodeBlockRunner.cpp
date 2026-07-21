#include "CodeBlockRunner.hpp"

#include <algorithm>
#include <format>

namespace tui {

namespace {

enum class CodeBlockRunnerAction {
    None,
    Run,
    CopySource,
    AttachOutput,
    CopyOutput,
};

struct CodeBlockRunnerEventResult {
    bool handled = false;
    CodeBlockRunnerAction action = CodeBlockRunnerAction::None;
};

} // namespace

std::expected<void, std::string> CodeBlockRunnerController::open(
    std::string_view assistant_markdown,
    std::optional<std::size_t> one_based_block) {
    auto blocks = core::code::extract_fenced_code_blocks(assistant_markdown);
    if (blocks.empty()) {
        state_ = {};
        return std::unexpected("The latest response contains no fenced code blocks.");
    }

    state_ = {};
    state_.active = true;
    state_.items.reserve(blocks.size());
    for (auto& block : blocks) {
        auto plan = core::code::plan_execution(std::move(block));
        if (plan) {
            state_.items.push_back(CodeBlockRunnerItem{
                .block = plan->block,
                .plan = std::move(*plan),
            });
        } else {
            state_.items.push_back(CodeBlockRunnerItem{
                .block = std::move(plan.error().block),
                .unavailable_reason = std::move(plan.error().reason),
            });
        }
    }

    if (one_based_block.has_value()) {
        if (*one_based_block == 0 || *one_based_block > state_.items.size()) {
            state_ = {};
            return std::unexpected(std::format(
                "Block {} does not exist in the latest response.", *one_based_block));
        }
        state_.selected_block = static_cast<int>(*one_based_block - 1);
        state_.mode = CodeBlockRunnerMode::Confirm;
    }
    return {};
}

const CodeBlockRunnerItem* selected_code_block(const CodeBlockRunnerState& state) {
    if (state.selected_block < 0
        || state.selected_block >= static_cast<int>(state.items.size())) return nullptr;
    return &state.items[static_cast<std::size_t>(state.selected_block)];
}

static CodeBlockRunnerEventResult transition_code_block_runner_state(
    CodeBlockRunnerState& state,
    const ftxui::Event& event,
    bool cancel_event) {
    if (!state.active) return {};
    CodeBlockRunnerEventResult result{.handled = true};

    if (event == ftxui::Event::Escape || cancel_event) {
        if (state.mode == CodeBlockRunnerMode::Select) state = {};
        else if (state.mode == CodeBlockRunnerMode::Confirm) {
            state.mode = CodeBlockRunnerMode::Select;
            state.selected_action = 0;
        } else state = {};
        return result;
    }

    if (state.mode == CodeBlockRunnerMode::Select) {
        const int count = static_cast<int>(state.items.size());
        if (event == ftxui::Event::ArrowUp && count > 0) {
            state.selected_block = (state.selected_block + count - 1) % count;
        } else if (event == ftxui::Event::ArrowDown && count > 0) {
            state.selected_block = (state.selected_block + 1) % count;
        } else if (event == ftxui::Event::Return && count > 0) {
            state.mode = CodeBlockRunnerMode::Confirm;
            state.selected_action = 0;
        } else {
            for (int number = 1; number <= std::min(count, 9); ++number) {
                if (event == ftxui::Event::Character(static_cast<char>('0' + number))) {
                    state.selected_block = number - 1;
                    state.mode = CodeBlockRunnerMode::Confirm;
                    state.selected_action = 0;
                    break;
                }
            }
        }
        return result;
    }

    const auto* item = selected_code_block(state);
    if (state.mode == CodeBlockRunnerMode::Confirm) {
        const int action_count = item != nullptr && item->plan.has_value() ? 3 : 2;
        if (event == ftxui::Event::ArrowUp) {
            state.selected_action = (state.selected_action + action_count - 1) % action_count;
        } else if (event == ftxui::Event::ArrowDown) {
            state.selected_action = (state.selected_action + 1) % action_count;
        } else if (event == ftxui::Event::Return) {
            if (item != nullptr && item->plan.has_value()) {
                if (state.selected_action == 0) result.action = CodeBlockRunnerAction::Run;
                else if (state.selected_action == 1) result.action = CodeBlockRunnerAction::CopySource;
                else state.mode = CodeBlockRunnerMode::Select;
            } else if (state.selected_action == 0) {
                result.action = CodeBlockRunnerAction::CopySource;
            } else {
                state.mode = CodeBlockRunnerMode::Select;
            }
        }
        return result;
    }

    const int action_count = state.result ? 4 : 2;
    if (event == ftxui::Event::ArrowUp) {
        state.selected_action = (state.selected_action + action_count - 1) % action_count;
    } else if (event == ftxui::Event::ArrowDown) {
        state.selected_action = (state.selected_action + 1) % action_count;
    } else if (event == ftxui::Event::Return) {
        if (!state.result) {
            if (state.selected_action == 0) {
                state.mode = CodeBlockRunnerMode::Confirm;
                state.selected_action = 0;
                state.error.clear();
            } else {
                state = {};
            }
        } else if (state.selected_action == 0) {
            result.action = CodeBlockRunnerAction::AttachOutput;
        } else if (state.selected_action == 1) {
            result.action = CodeBlockRunnerAction::CopyOutput;
        } else if (state.selected_action == 2) {
            state.mode = CodeBlockRunnerMode::Confirm;
            state.selected_action = 0;
            state.result.reset();
            state.error.clear();
        } else {
            state = {};
        }
    }
    return result;
}

static void complete_code_block_run(CodeBlockRunnerState& state,
                                    core::code::CodeRunResult result) {
    state.active = true;
    state.mode = CodeBlockRunnerMode::Complete;
    state.selected_action = 0;
    state.error.clear();
    state.result = std::make_shared<const core::code::CodeRunResult>(std::move(result));
}

static void fail_code_block_run(CodeBlockRunnerState& state, std::string error) {
    state.active = true;
    state.mode = CodeBlockRunnerMode::Complete;
    state.selected_action = 0;
    state.result.reset();
    state.error = std::move(error);
}

CodeBlockRunnerOutcome CodeBlockRunnerController::handle(
    const ftxui::Event& event,
    bool cancel_event) {
    const auto transition = transition_code_block_runner_state(
        state_, event, cancel_event);
    CodeBlockRunnerOutcome outcome{.handled = transition.handled};
    if (!transition.handled || transition.action == CodeBlockRunnerAction::None) {
        return outcome;
    }

    const auto* item = selected_code_block(state_);
    switch (transition.action) {
    case CodeBlockRunnerAction::Run:
        if (item != nullptr && item->plan.has_value()) {
            auto result = services_.run(*item->plan);
            if (result) complete_code_block_run(state_, std::move(*result));
            else fail_code_block_run(state_, result.error());
        }
        break;
    case CodeBlockRunnerAction::CopySource:
        if (item != nullptr) {
            auto copied = services_.copy(item->block.source);
            outcome.notice = copied
                ? CodeBlockRunnerNotice{.success = true,
                                        .message = "Code block copied to clipboard."}
                : CodeBlockRunnerNotice{.success = false,
                                        .message = "Could not copy block: " + copied.error()};
        }
        break;
    case CodeBlockRunnerAction::AttachOutput:
        if (state_.result) {
            auto stored = services_.store_transcript(*state_.result);
            if (stored) {
                outcome.attachment = *stored;
                outcome.notice = CodeBlockRunnerNotice{
                    .success = true,
                    .message = "Attached sanitized code-run output: " + stored->string(),
                };
            } else {
                outcome.notice = CodeBlockRunnerNotice{
                    .success = false,
                    .message = "Could not save code-run output: " + stored.error(),
                };
            }
        }
        break;
    case CodeBlockRunnerAction::CopyOutput:
        if (state_.result) {
            auto copied = services_.copy(state_.result->output);
            outcome.notice = copied
                ? CodeBlockRunnerNotice{
                    .success = true,
                    .message = "Sanitized code-run output copied to clipboard.",
                }
                : CodeBlockRunnerNotice{
                    .success = false,
                    .message = "Could not copy output: " + copied.error(),
                };
        }
        break;
    case CodeBlockRunnerAction::None:
        break;
    }
    return outcome;
}

} // namespace tui
