#include "ContextMentions.hpp"
#include "../config/ConfigManager.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <sstream>
#include <vector>

namespace core::context {

namespace {

bool is_boundary_char(char ch) {
    return std::isspace(static_cast<unsigned char>(ch))
        || ch == '(' || ch == '[' || ch == '{'
        || ch == '"' || ch == '\'';
}

bool is_trailing_mention_suffix(char ch) {
    return ch == ',' || ch == '.' || ch == ';' || ch == ':' || ch == '!'
        || ch == '?' || ch == ')' || ch == ']' || ch == '}';
}

struct ParsedUnquotedMention {
    std::string raw_path;
    std::string trailing_suffix;
    std::vector<std::size_t> source_ends;
    std::vector<bool> escaped;
    std::size_t token_end = 0;
};

bool is_shell_escaped_character(char ch) {
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

ParsedUnquotedMention parse_unquoted_mention(std::string_view input, std::size_t start) {
    ParsedUnquotedMention parsed;
    parsed.raw_path.reserve(input.size() - start);
    parsed.source_ends.reserve(input.size() - start);
    parsed.escaped.reserve(input.size() - start);

    std::size_t cursor = start;
    while (cursor < input.size()) {
        const unsigned char ch = static_cast<unsigned char>(input[cursor]);
        if (ch == '\\'
            && cursor + 1 < input.size()
            && is_shell_escaped_character(input[cursor + 1])) {
            parsed.raw_path.push_back(input[cursor + 1]);
            parsed.source_ends.push_back(cursor + 2);
            parsed.escaped.push_back(true);
            cursor += 2;
            continue;
        }
        if (std::isspace(ch)) {
            break;
        }
        parsed.raw_path.push_back(static_cast<char>(ch));
        parsed.source_ends.push_back(cursor + 1);
        parsed.escaped.push_back(false);
        ++cursor;
    }

    parsed.token_end = cursor;

    while (!parsed.raw_path.empty()
           && !parsed.escaped.empty()
           && !parsed.escaped.back()
           && is_trailing_mention_suffix(parsed.raw_path.back())) {
        parsed.trailing_suffix.insert(parsed.trailing_suffix.begin(), parsed.raw_path.back());
        parsed.raw_path.pop_back();
        parsed.source_ends.pop_back();
        parsed.escaped.pop_back();
    }

    return parsed;
}

std::string normalize_subagent_tag(std::string_view tag) {
    std::string normalized;
    normalized.reserve(tag.size());

    std::size_t start = 0;
    while (start < tag.size() && std::isspace(static_cast<unsigned char>(tag[start]))) {
        ++start;
    }
    if (start < tag.size() && tag[start] == '@') {
        ++start;
    }

    for (std::size_t i = start; i < tag.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(tag[i]);
        if (std::isspace(ch)) break;
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

bool is_reserved_subagent_mention(std::string_view raw_path) {
    const std::string normalized = normalize_subagent_tag(raw_path);
    if (normalized == "general" || normalized == "explore") {
        return true;
    }

    const auto& config = core::config::ConfigManager::get_instance().get_config();
    return config.subagents.contains(normalized);
}

std::string render_file_contents(const std::filesystem::path& path,
                                 std::string_view display_path,
                                 std::size_t max_file_bytes) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::format("\n[Context for {}]\nError: file is not readable.\n[/Context]\n",
                           display_path);
    }

    std::string content;
    content.resize(max_file_bytes);
    input.read(content.data(), static_cast<std::streamsize>(max_file_bytes));
    content.resize(static_cast<std::size_t>(input.gcount()));
    const bool truncated = input.peek() != std::char_traits<char>::eof();

    std::ostringstream rendered;
    rendered << "\n[Context for " << display_path << "]\n";

    std::istringstream lines(content);
    std::string line;
    std::size_t line_no = 1;
    while (std::getline(lines, line)) {
        rendered << std::format("{:>4} | {}\n", line_no++, line);
    }
    if (!content.empty() && content.back() == '\n' && lines.fail()) {
        // No extra line needed when the file ends with a newline.
    }

    if (truncated) {
        rendered << std::format("[truncated after {} bytes]\n", max_file_bytes);
    }
    rendered << "[/Context]\n";
    return rendered.str();
}

std::string render_directory_listing(const std::filesystem::path& path,
                                     std::string_view display_path,
                                     std::size_t max_directory_entries) {
    std::vector<std::filesystem::path> entries;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (ec) break;
        entries.push_back(entry.path().filename());
    }
    std::sort(entries.begin(), entries.end());

    std::ostringstream rendered;
    rendered << "\n[Context for " << display_path << "]\n";
    rendered << "Directory listing:\n";
    const auto count = std::min(entries.size(), max_directory_entries);
    for (std::size_t i = 0; i < count; ++i) {
        rendered << " - " << entries[i].string() << '\n';
    }
    if (entries.size() > max_directory_entries) {
        rendered << std::format("[truncated after {} entries]\n", max_directory_entries);
    }
    rendered << "[/Context]\n";
    return rendered.str();
}

std::string render_missing_path(std::string_view display_path) {
    return std::format("\n[Context for {}]\nError: path not found.\n[/Context]\n", display_path);
}

std::string render_mention(std::string_view raw_path,
                           const std::filesystem::path& base_dir,
                           const MentionExpansionOptions& options) {
    if (raw_path.empty()) return "@";

    std::filesystem::path resolved = raw_path;
    if (!resolved.is_absolute()) {
        resolved = base_dir / resolved;
    }
    resolved = resolved.lexically_normal();

    std::error_code ec;
    if (std::filesystem::is_regular_file(resolved, ec)) {
        return render_file_contents(resolved, raw_path, options.max_file_bytes);
    }
    if (std::filesystem::is_directory(resolved, ec)) {
        return render_directory_listing(resolved, raw_path, options.max_directory_entries);
    }
    return render_missing_path(raw_path);
}

} // namespace

std::optional<ActiveMention> find_active_mention(std::string_view input,
                                                 std::size_t cursor) {
    cursor = std::min(cursor, input.size());

    for (std::size_t i = 0; i <= input.size(); ++i) {
        if (i == input.size() || input[i] != '@') continue;

        const bool at_boundary = (i == 0) || is_boundary_char(input[i - 1]);
        if (!at_boundary) continue;

        std::size_t path_start = i + 1;
        if (path_start > input.size()) continue;

        if (path_start < input.size() && (input[path_start] == '"' || input[path_start] == '\'')) {
            const char quote = input[path_start++];
            std::size_t path_end = path_start;
            while (path_end < input.size() && input[path_end] != quote) {
                ++path_end;
            }

            const bool has_closing_quote = path_end < input.size() && input[path_end] == quote;
            const std::size_t replace_end = has_closing_quote ? path_end + 1 : input.size();
            if (cursor < path_start || cursor > replace_end) {
                i = replace_end;
                continue;
            }
            if (has_closing_quote && cursor > path_end) {
                i = replace_end;
                continue;
            }

            return ActiveMention{
                .replace_begin = i,
                .replace_end = replace_end,
                .raw_path = std::string(input.substr(path_start, path_end - path_start)),
                .quoted = true,
            };
        }

        const auto parsed = parse_unquoted_mention(input, path_start);
        const std::size_t path_end = parsed.token_end;
        const std::size_t replace_end = parsed.source_ends.empty()
            ? path_start
            : parsed.source_ends.back();

        if (parsed.raw_path.empty()) {
            i = path_end;
            continue;
        }

        if (cursor < path_start || cursor > replace_end) {
            i = path_end;
            continue;
        }

        return ActiveMention{
            .replace_begin = i,
            .replace_end = replace_end,
            .raw_path = parsed.raw_path,
            .quoted = false,
        };
    }

    return std::nullopt;
}

MentionCompletion apply_mention_completion(std::string_view input,
                                           const ActiveMention& mention,
                                           std::string_view replacement) {
    const bool keep_open_quote = !replacement.empty() && replacement.back() == '/';
    const bool needs_quotes = mention.quoted
        || replacement.find_first_of(" \t") != std::string_view::npos;

    std::string mention_text = "@";
    if (needs_quotes) {
        mention_text += '"';
        mention_text += replacement;
        if (!keep_open_quote) {
            mention_text += '"';
        }
    } else {
        mention_text += replacement;
    }

    MentionCompletion out;
    out.text.reserve(input.size() + mention_text.size() + 8);
    out.text.append(input.substr(0, mention.replace_begin));
    out.text.append(mention_text);
    out.text.append(input.substr(std::min(mention.replace_end, input.size())));
    out.cursor = mention.replace_begin + mention_text.size();

    // For unquoted non-directory completions the cursor lands exactly at
    // replace_end.  find_active_mention uses `cursor > replace_end` (strict),
    // so the cursor is still considered inside the mention and the picker
    // re-opens showing the just-completed file.  Insert a trailing space so
    // the cursor is unambiguously outside the token.
    //
    // Quoted paths: find_active_mention already closes via the
    //   `has_closing_quote && cursor > path_end` branch — no space needed.
    // Directories: intentionally keep the picker open so the user can
    //   continue navigating inside the selected directory.
    const bool is_directory = !replacement.empty() && replacement.back() == '/';
    if (!is_directory && !needs_quotes) {
        if (out.cursor >= out.text.size() || out.text[out.cursor] != ' ') {
            out.text.insert(out.cursor, 1, ' ');
        }
        ++out.cursor;  // place cursor after the space, not on it
    }

    return out;
}

std::string expand_mentions(std::string_view input,
                            const std::filesystem::path& base_dir,
                            const MentionExpansionOptions& options) {
    std::string output;
    output.reserve(input.size() + 256);

    std::size_t i = 0;
    while (i < input.size()) {
        if (input[i] != '@') {
            output.push_back(input[i++]);
            continue;
        }

        const bool at_boundary = (i == 0) || is_boundary_char(input[i - 1]);
        if (!at_boundary || i + 1 >= input.size()) {
            output.push_back(input[i++]);
            continue;
        }

        std::size_t cursor = i + 1;
        std::string raw_path;
        std::string trailing_suffix;
        bool quoted = false;
        if (input[cursor] == '"' || input[cursor] == '\'') {
            const char quote = input[cursor++];
            quoted = true;
            const std::size_t start = cursor;
            while (cursor < input.size() && input[cursor] != quote) {
                ++cursor;
            }
            if (cursor >= input.size()) {
                output.push_back(input[i++]);
                continue;
            }
            raw_path = std::string(input.substr(start, cursor - start));
            ++cursor;
        } else {
            const auto parsed = parse_unquoted_mention(input, cursor);
            raw_path = parsed.raw_path;
            trailing_suffix = parsed.trailing_suffix;
            cursor = parsed.token_end;
        }

        if (raw_path.empty()) {
            output.push_back(input[i++]);
            continue;
        }

        if (!quoted && is_reserved_subagent_mention(raw_path)) {
            output.push_back('@');
            output += raw_path;
            output += trailing_suffix;
            i = cursor;
            continue;
        }

        std::string expansion = render_mention(raw_path, base_dir, options);
        if (!trailing_suffix.empty() && !expansion.empty() && expansion.back() == '\n') {
            expansion.pop_back();
            expansion += trailing_suffix;
            expansion.push_back('\n');
        } else {
            expansion += trailing_suffix;
        }
        output += expansion;
        i = cursor;
    }

    return output;
}

} // namespace core::context
