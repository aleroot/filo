#pragma once

#include "CodeBlockRunner.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace tui {

struct CodeBlockRunDependencies {
    std::unique_ptr<core::code::IInteractiveCodeRunner> runner;
    std::function<std::filesystem::path()> working_directory;
    std::filesystem::path script_directory;
    std::function<core::landrun::LandrunPolicy()> landrun_policy;
    std::function<void(std::function<void()>)> with_restored_terminal;
    std::function<std::optional<std::string>(std::string_view)> copy_to_clipboard;
    std::filesystem::path transcript_directory;
};

[[nodiscard]] std::filesystem::path select_code_run_script_directory(
    core::landrun::LandrunMode mode,
    const std::filesystem::path& runtime_root,
    const std::filesystem::path& process_temp_directory);

class CodeBlockRunServices final : public ICodeBlockRunServices {
public:
    explicit CodeBlockRunServices(CodeBlockRunDependencies dependencies);

    [[nodiscard]] std::expected<core::code::CodeRunResult, std::string>
    run(const core::code::ExecutionPlan& plan) override;

    [[nodiscard]] std::expected<void, std::string>
    copy(std::string_view text) override;

    [[nodiscard]] std::expected<std::filesystem::path, std::string>
    store_transcript(const core::code::CodeRunResult& result) override;

private:
    CodeBlockRunDependencies dependencies_;
};

} // namespace tui
