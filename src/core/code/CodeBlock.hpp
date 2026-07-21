#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace core::code {

struct FencedCodeBlock {
    std::size_t ordinal = 0;
    std::size_t first_line = 0;
    std::size_t last_line = 0;
    std::string language;
    std::string info;
    std::string source;
};

struct ExecutionPlan {
    FencedCodeBlock block;
    std::string interpreter;
    std::filesystem::path interpreter_path;
    std::vector<std::string> interpreter_arguments;
    std::string script_extension;
    std::string prepared_source;
};

struct ExecutionPlanError {
    FencedCodeBlock block;
    std::string reason;
};

[[nodiscard]] std::vector<FencedCodeBlock> extract_fenced_code_blocks(
    std::string_view markdown);

[[nodiscard]] std::expected<ExecutionPlan, ExecutionPlanError> plan_execution(
    FencedCodeBlock block);

[[nodiscard]] std::string code_block_preview(std::string_view source,
                                             std::size_t max_characters = 72);

[[nodiscard]] std::string sanitize_terminal_output(std::string_view output);

} // namespace core::code
