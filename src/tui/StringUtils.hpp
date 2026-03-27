#pragma once

#include <string>
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

} // namespace tui
