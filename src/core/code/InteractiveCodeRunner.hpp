#pragma once

#include "CodeBlock.hpp"
#include "core/landrun/LandrunPolicy.hpp"

#include <chrono>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace core::code {

struct CodeRunResult {
    int exit_code = -1;
    int terminating_signal = 0;
    std::string output;
    std::size_t output_bytes = 0;
    bool output_truncated = false;
    std::chrono::milliseconds duration{0};

    [[nodiscard]] bool succeeded() const noexcept {
        return exit_code == 0 && terminating_signal == 0;
    }
};

class IInteractiveCodeRunner {
public:
    virtual ~IInteractiveCodeRunner() = default;

    [[nodiscard]] virtual std::expected<CodeRunResult, std::string> run(
        const ExecutionPlan& plan,
        const std::filesystem::path& working_directory,
        const std::filesystem::path& temporary_directory,
        const core::landrun::LandrunPolicy& policy) const = 0;
};

[[nodiscard]] std::unique_ptr<IInteractiveCodeRunner> make_interactive_code_runner();

} // namespace core::code
