#include "PromptCode.hpp"

#include "../utils/AsciiUtils.hpp"

#include <algorithm>

namespace core::context {
namespace {

[[nodiscard]] constexpr bool is_space(char ch) noexcept {
    return core::utils::ascii::is_space(static_cast<unsigned char>(ch));
}

struct Fence {
    char marker = 0;
    std::size_t length = 0;
};

// CommonMark fences start after at most three spaces. Closing runs must
// match the opening marker, be at least as long, and have no info string.
[[nodiscard]] Fence fence_on_line(std::string_view line, Fence open) noexcept {
    std::size_t at = 0;
    while (at < line.size() && line[at] == ' ' && at < 4) ++at;
    if (at == 4 || at == line.size()) return {};
    const char marker = line[at];
    if (marker != '`' && marker != '~') return {};
    const auto start = at;
    while (at < line.size() && line[at] == marker) ++at;
    const auto length = at - start;
    if (length < 3) return {};
    const auto rest = line.substr(at);
    if (open.marker != 0) {
        if (marker != open.marker || length < open.length
            || !std::ranges::all_of(rest, is_space)) return {};
    } else if (marker == '`' && rest.contains('`')) {
        return {};
    }
    return {marker, length};
}

} // namespace

std::vector<PromptCodeRange> prompt_code_ranges(std::string_view input) {
    std::vector<PromptCodeRange> fences;
    Fence open;
    std::size_t begin = 0;
    for (std::size_t at = 0; at < input.size();) {
        const auto newline = input.find('\n', at);
        const auto end = newline == std::string_view::npos ? input.size() : newline + 1;
        if (const auto fence = fence_on_line(input.substr(at, end - at), open);
            fence.marker != 0) {
            if (open.marker == 0) {
                begin = at;
                open = fence;
            } else {
                fences.push_back({begin, end});
                open = {};
            }
        }
        at = end;
    }
    if (open.marker != 0) fences.push_back({begin, input.size()});

    std::vector<PromptCodeRange> ranges;
    const auto inline_ranges = [&](std::size_t from, std::size_t end) {
        while (from < end) {
            if (input[from] == '\\' && from + 1 < end) {
                from += 2;
                continue;
            }
            if (input[from] != '`') { ++from; continue; }
            const auto start = from;
            while (from < end && input[from] == '`') ++from;
            const auto length = from - start;
            for (std::size_t close = from; close < end;) {
                // A blank line ends the paragraph, including unclosed spans.
                if (input[close] == '\n') {
                    auto next = close + 1;
                    while (next < end && (input[next] == ' ' || input[next] == '\t'
                                          || input[next] == '\r')) ++next;
                    if (next < end && input[next] == '\n') break;
                }
                if (input[close] != '`') { ++close; continue; }
                const auto run = close;
                while (close < end && input[close] == '`') ++close;
                if (close - run == length) {
                    ranges.push_back({start, close});
                    from = close;
                    break;
                }
            }
        }
    };
    std::size_t from = 0;
    for (const auto& fence : fences) {
        inline_ranges(from, fence.begin);
        ranges.push_back(fence);
        from = fence.end;
    }
    inline_ranges(from, input.size());
    return ranges;
}

bool is_inside_prompt_code(std::string_view input, std::size_t pos) {
    const auto ranges = prompt_code_ranges(input);
    return std::ranges::any_of(ranges, [pos](const auto& range) {
        return range.begin <= pos && pos < range.end;
    });
}

} // namespace core::context
