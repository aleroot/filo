#include "Text.hpp"

#include <algorithm>
#include <cctype>

namespace tui {

std::string to_lower_ascii(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

std::string_view trim_ascii(std::string_view input) {
    const auto start = input.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

std::string compact_single_line(std::string_view input, std::size_t max_len) {
    std::string normalized;
    normalized.reserve(input.size());
    bool prev_space = false;
    for (const char ch : input) {
        const bool is_space = (ch == '\n' || ch == '\r' || ch == '\t'
                               || std::isspace(static_cast<unsigned char>(ch)));
        if (is_space) {
            if (!prev_space) {
                normalized.push_back(' ');
                prev_space = true;
            }
            continue;
        }
        normalized.push_back(ch);
        prev_space = false;
    }

    std::string_view trimmed = trim_ascii(normalized);
    if (trimmed.size() <= max_len) {
        return std::string(trimmed);
    }

    constexpr std::string_view kEllipsis = "...";
    if (max_len <= kEllipsis.size()) {
        return std::string(kEllipsis.substr(0, max_len));
    }
    std::string out(trimmed.substr(0, max_len - kEllipsis.size()));
    out += kEllipsis;
    return out;
}

bool visibility_setting_enabled(std::string_view value, bool fallback) {
    std::string normalized;
    normalized.reserve(value.size());
    for (const char ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch))
            || ch == '-'
            || ch == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }

    if (normalized.empty()) {
        return fallback;
    }
    if (normalized == "show" || normalized == "on"
        || normalized == "true" || normalized == "yes"
        || normalized == "visible") {
        return true;
    }
    if (normalized == "hide" || normalized == "off"
        || normalized == "false" || normalized == "no"
        || normalized == "hidden") {
        return false;
    }
    return fallback;
}

bool starts_with_line_break(std::string_view text) {
    return !text.empty() && (text.front() == '\n' || text.front() == '\r');
}

bool ends_with_line_break(std::string_view text) {
    return !text.empty() && (text.back() == '\n' || text.back() == '\r');
}

std::string normalize_newlines(std::string input) {
    std::string output;
    output.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '\r') {
            if (i + 1 < input.size() && input[i + 1] == '\n') {
                ++i;
            }
            output.push_back('\n');
            continue;
        }
        output.push_back(input[i]);
    }
    return output;
}

bool erase_last_utf8_codepoint(std::string& text) {
    if (text.empty()) {
        return false;
    }

    std::size_t pos = text.size() - 1;
    while (pos > 0
           && (static_cast<unsigned char>(text[pos]) & 0b1100'0000) == 0b1000'0000) {
        --pos;
    }
    text.erase(pos);
    return true;
}

std::string flatten_whitespace(std::string_view input) {
    std::string out;
    out.reserve(input.size());

    bool last_space = false;
    for (const unsigned char ch : input) {
        if (std::isspace(ch)) {
            if (!last_space) {
                out.push_back(' ');
                last_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(ch));
        last_space = false;
    }

    const auto start = out.find_first_not_of(' ');
    if (start == std::string::npos) {
        return {};
    }
    const auto end = out.find_last_not_of(' ');
    return out.substr(start, end - start + 1);
}

std::string search_role_label(MessageType type) {
    switch (type) {
        case MessageType::User: return "User";
        case MessageType::Assistant: return "Assistant";
        case MessageType::Info: return "Info";
        case MessageType::Warning: return "Warning";
        case MessageType::Error: return "Error";
        case MessageType::ToolGroup: return "Tool";
        case MessageType::System: return "System";
    }
    return "Message";
}

std::string search_text_for_message(const UiMessage& msg) {
    std::string text = msg.text;

    if (!msg.secondary_text.empty()) {
        if (!text.empty()) {
            text.push_back('\n');
        }
        text += msg.secondary_text;
    }

    for (const auto& tool : msg.tools) {
        if (!text.empty()) {
            text.push_back('\n');
        }
        text += tool.name;
        if (!tool.description.empty()) {
            text += " ";
            text += tool.description;
        }
        if (!tool.result.empty()) {
            text += "\n";
            text += tool.result.summary;
        }
    }

    return flatten_whitespace(text);
}

std::string build_search_snippet(std::string_view text,
                                 std::size_t match_pos,
                                 std::size_t query_len) {
    constexpr std::size_t kPreviewRadius = 80;
    constexpr std::size_t kMaxPreview = 200;

    const std::size_t start = match_pos > kPreviewRadius
        ? match_pos - kPreviewRadius
        : 0;

    const std::size_t desired_len = std::max<std::size_t>(kPreviewRadius * 2 + query_len, 80);
    const std::size_t len = std::min(kMaxPreview, std::min(desired_len, text.size() - start));
    std::string snippet(text.substr(start, len));
    if (start > 0) {
        snippet = "..." + snippet;
    }
    if (start + len < text.size()) {
        snippet += "...";
    }
    return snippet;
}

} // namespace tui
