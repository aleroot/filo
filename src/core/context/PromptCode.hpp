#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace core::context {

struct PromptCodeRange {
    std::size_t begin;
    std::size_t end;
};

/// Sorted ranges of Markdown fenced blocks and matched inline code spans.
/// Compute once when expanding a prompt, rather than scanning per reference.
[[nodiscard]] std::vector<PromptCodeRange> prompt_code_ranges(std::string_view input);

[[nodiscard]] bool is_inside_prompt_code(std::string_view input, std::size_t pos);

} // namespace core::context
