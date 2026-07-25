#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

/// Split a string on newline characters. An input ending with '\n' produces a
/// trailing empty string (matching the semantics used by the editor widget and
/// the conversation renderer).
inline std::vector<std::string> split_lines(const std::string& input) {
    std::vector<std::string> output;
    if (input.empty()) {
        return output;
    }

    std::size_t start = 0;
    while (true) {
        const std::size_t end = input.find('\n', start);
        if (end == std::string::npos) {
            output.push_back(input.substr(start));
            break;
        }
        output.push_back(input.substr(start, end - start));
        start = end + 1;
    }

    if (!input.empty() && input.back() == '\n') {
        output.emplace_back();
    }

    return output;
}

/// Non-owning variant of `split_lines` for read-only consumers such as the
/// transcript renderer, which splits multi-megabyte tool output every frame and
/// must not copy it. Views alias `input`, which must outlive the result.
/// Unlike `split_lines`, a trailing '\n' does not yield a trailing empty view.
inline std::vector<std::string_view> split_lines_view(std::string_view input) {
    std::vector<std::string_view> output;
    if (input.empty()) {
        return output;
    }

    std::size_t start = 0;
    while (start < input.size()) {
        const std::size_t end = input.find('\n', start);
        if (end == std::string_view::npos) {
            output.push_back(input.substr(start));
            break;
        }
        output.push_back(input.substr(start, end - start));
        start = end + 1;
    }

    return output;
}

/// Counts the lines in `content`, ignoring trailing blank lines, without
/// materialising the split. Matches `split_lines` semantics after popping
/// trailing empties.
inline std::size_t visible_line_count(std::string_view content) {
    while (!content.empty() && content.back() == '\n') {
        content.remove_suffix(1);
    }
    if (content.empty()) {
        return 0;
    }
    std::size_t lines = 1;
    for (const char ch : content) {
        lines += ch == '\n' ? 1 : 0;
    }
    return lines;
}

} // namespace tui
