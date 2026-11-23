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

/// The code ranges of one prompt, walked left to right. Every scanner that
/// skips code (`@`, `#`, inline commands) shares it, so the ranges are computed
/// once and the "am I inside code" test stays in one place.
class PromptCodeCursor {
public:
    explicit PromptCodeCursor(std::string_view input) : ranges_(prompt_code_ranges(input)) {}

    /// True when @p pos lies inside a code range. Calls must come in
    /// non-decreasing order: each one advances past the ranges already passed.
    [[nodiscard]] bool contains(std::size_t pos) noexcept {
        while (index_ < ranges_.size() && ranges_[index_].end <= pos) ++index_;
        return index_ < ranges_.size() && ranges_[index_].begin <= pos;
    }

    /// One past the code range containing @p pos, or @p pos when it is outside.
    [[nodiscard]] std::size_t end_at(std::size_t pos) noexcept {
        return contains(pos) ? ranges_[index_].end : pos;
    }

private:
    std::vector<PromptCodeRange> ranges_;
    std::size_t index_ = 0;
};

} // namespace core::context
