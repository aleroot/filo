#include "MentionPathUtils.hpp"

#include "../utils/StringUtils.hpp"
#include "../utils/UriUtils.hpp"

#include <cctype>
#include <filesystem>
#include <system_error>
#include <utility>
#include <vector>

namespace core::context {

namespace {

std::optional<std::string> decode_file_uri_path(std::string_view token) {
    if (!token.starts_with("file://")) {
        return std::nullopt;
    }

    std::string_view path = token.substr(7);
    if (path.starts_with("localhost/")) {
        path.remove_prefix(std::string_view("localhost").size());
    }

    std::string decoded;
    if (!core::utils::uri::percent_decode(path, decoded)) {
        return std::nullopt;
    }
    return decoded;
}

std::optional<std::string> normalize_pasted_path_token(std::string token) {
    if (auto decoded = decode_file_uri_path(token)) {
        token = std::move(*decoded);
    }

    const std::filesystem::path path(token);
    if (!path.is_absolute()
        && !token.starts_with("./")
        && !token.starts_with("../")
        && token.find('/') == std::string::npos) {
        return std::nullopt;
    }

    std::error_code ec;
    const bool exists = std::filesystem::is_regular_file(path, ec)
        || std::filesystem::is_directory(path, ec);
    if (!exists) {
        return std::nullopt;
    }
    return path.string();
}

std::vector<std::string> split_pasted_shell_words(std::string_view input) {
    std::vector<std::string> words;
    std::size_t cursor = 0;
    while (cursor < input.size()) {
        while (cursor < input.size()
               && std::isspace(static_cast<unsigned char>(input[cursor]))) {
            ++cursor;
        }
        if (cursor >= input.size()) {
            break;
        }

        std::string word;
        while (cursor < input.size()
               && !std::isspace(static_cast<unsigned char>(input[cursor]))) {
            const char ch = input[cursor];
            if (ch == '\'' || ch == '"') {
                const char quote = ch;
                ++cursor;
                while (cursor < input.size() && input[cursor] != quote) {
                    if (input[cursor] == '\\' && cursor + 1 < input.size() && quote == '"') {
                        word.push_back(input[cursor + 1]);
                        cursor += 2;
                    } else {
                        word.push_back(input[cursor++]);
                    }
                }
                if (cursor < input.size() && input[cursor] == quote) {
                    ++cursor;
                }
                continue;
            }
            if (ch == '\\' && cursor + 1 < input.size()) {
                word.push_back(input[cursor + 1]);
                cursor += 2;
                continue;
            }
            word.push_back(ch);
            ++cursor;
        }
        if (!word.empty()) {
            words.push_back(std::move(word));
        }
    }
    return words;
}

} // namespace

bool should_escape_unquoted_mention_path_char(char ch) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isspace(uch)) {
        return true;
    }

    switch (ch) {
        case '\\':
        case '"':
        case '\'':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
        case ',':
        case '.':
        case ';':
        case ':':
        case '!':
        case '?':
            return true;
        default:
            return false;
    }
}

std::string escape_unquoted_mention_path(std::string_view path) {
    std::string escaped;
    escaped.reserve(path.size());
    for (const char ch : path) {
        if (should_escape_unquoted_mention_path_char(ch)) {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

std::string format_unquoted_mention(std::string_view path) {
    std::string mention = "@";
    mention.reserve(path.size() + 1);
    mention += escape_unquoted_mention_path(path);
    return mention;
}

std::optional<std::string> pasted_paths_to_mentions(std::string_view pasted) {
    pasted = core::utils::str::trim_ascii_view(pasted);
    if (pasted.empty()) {
        return std::nullopt;
    }

    std::vector<std::string> paths;
    for (auto& word : split_pasted_shell_words(pasted)) {
        auto normalized = normalize_pasted_path_token(std::move(word));
        if (!normalized.has_value()) {
            return std::nullopt;
        }
        paths.push_back(std::move(*normalized));
    }
    if (paths.empty()) {
        return std::nullopt;
    }

    std::string replacement;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        if (i > 0) {
            replacement.push_back(' ');
        }
        replacement += format_unquoted_mention(paths[i]);
    }
    return replacement;
}

} // namespace core::context
