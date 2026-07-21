#pragma once

#include "core/code/CodeBlock.hpp"
#include "core/code/InteractiveCodeRunner.hpp"

#include <ftxui/component/event.hpp>

#include <cstddef>
#include <expected>
#include <filesystem>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace tui {

struct CodeBlockRunnerItem {
    core::code::FencedCodeBlock block;
    std::optional<core::code::ExecutionPlan> plan;
    std::string unavailable_reason;
};

enum class CodeBlockRunnerMode {
    Select,
    Confirm,
    Complete,
};

struct CodeBlockRunnerState {
    bool active = false;
    CodeBlockRunnerMode mode = CodeBlockRunnerMode::Select;
    int selected_block = 0;
    int selected_action = 0;
    std::vector<CodeBlockRunnerItem> items;
    std::shared_ptr<const core::code::CodeRunResult> result;
    std::string error;
};

class ICodeBlockRunServices {
public:
    virtual ~ICodeBlockRunServices() = default;

    [[nodiscard]] virtual std::expected<core::code::CodeRunResult, std::string>
    run(const core::code::ExecutionPlan& plan) = 0;

    [[nodiscard]] virtual std::expected<void, std::string>
    copy(std::string_view text) = 0;

    [[nodiscard]] virtual std::expected<std::filesystem::path, std::string>
    store_transcript(const core::code::CodeRunResult& result) = 0;
};

struct CodeBlockRunnerNotice {
    bool success = false;
    std::string message;
};

struct CodeBlockRunnerOutcome {
    bool handled = false;
    std::optional<CodeBlockRunnerNotice> notice;
    std::optional<std::filesystem::path> attachment;
};

class CodeBlockRunnerController {
public:
    explicit CodeBlockRunnerController(ICodeBlockRunServices& services)
        : services_(services) {}

    [[nodiscard]] std::expected<void, std::string> open(
        std::string_view assistant_markdown,
        std::optional<std::size_t> one_based_block = std::nullopt);

    [[nodiscard]] CodeBlockRunnerOutcome handle(
        const ftxui::Event& event,
        bool cancel_event);

    [[nodiscard]] bool active() const noexcept { return state_.active; }
    [[nodiscard]] const CodeBlockRunnerState& state() const noexcept { return state_; }

private:
    ICodeBlockRunServices& services_;
    CodeBlockRunnerState state_;
};

[[nodiscard]] const CodeBlockRunnerItem* selected_code_block(
    const CodeBlockRunnerState& state);

} // namespace tui
