#include "ShellCommandParser.hpp"

#include "../utils/AsciiUtils.hpp"

#include <algorithm>
#include <format>
#include <utility>

namespace core::permissions {

namespace {

/// Raised anywhere inside the parser; parse_shell_commands reports it.
struct Unsupported {
    std::string reason;
};

[[noreturn]] void reject(std::string reason) {
    throw Unsupported{std::move(reason)};
}

/// `$( $( $( …` nests the parser; bound it so input cannot exhaust the stack.
constexpr int kMaxSubstitutionDepth = 16;

using core::utils::ascii::is_digit;

/// Bash's <blank>: the only whitespace that separates words. A newline is an
/// operator, and \r, \f, \v are ordinary word characters — which is why the
/// broader ascii::is_space would be wrong here.
constexpr bool is_blank(char c) noexcept { return c == ' ' || c == '\t'; }

/// Shell variable names: [A-Za-z_][A-Za-z0-9_]*
constexpr bool is_name_start(char c) noexcept { return core::utils::ascii::is_alpha(c) || c == '_'; }

constexpr bool is_name_char(char c) noexcept { return core::utils::ascii::is_alnum(c) || c == '_'; }

/// $?, $#, $@, $*, $$, $!, $-, $0…$9
constexpr bool is_special_parameter(char c) noexcept {
    return is_digit(c) || std::string_view{"?#@*$!-"}.contains(c);
}

/// Characters that end an unquoted word.
constexpr bool is_word_break(char c) noexcept {
    return is_blank(c) || std::string_view{"\n;&|<>()"}.contains(c);
}

[[nodiscard]] bool is_plain_parameter(std::string_view name) noexcept {
    // NAME[index] with a literal index: `${PIPESTATUS[0]}`, `${args[@]}`. Any
    // other index is arithmetic, which can run a command: `${a[$(id)]}`.
    if (name.ends_with(']')) {
        const auto open = name.find('[');
        if (open == std::string_view::npos) return false;
        const auto index = name.substr(open + 1, name.size() - open - 2);
        const bool literal_index = index == "@" || index == "*"
            || (!index.empty() && std::ranges::all_of(index, is_digit));
        name = name.substr(0, open);
        if (!literal_index || name.empty() || !is_name_start(name.front())) return false;
        return std::ranges::all_of(name, is_name_char);
    }
    if (name.size() == 1 && is_special_parameter(name.front())) return true;
    if (!name.empty() && std::ranges::all_of(name, is_digit)) return true;
    return !name.empty() && is_name_start(name.front()) && std::ranges::all_of(name, is_name_char);
}

/// Accumulates one word while the parser walks its quoting.
class WordBuilder {
public:
    void literal(char c) {
        started_ = true;
        if (!word_.expands) word_.text += c;
    }

    void literal(std::string_view text) {
        started_ = true;
        if (!word_.expands) word_.text += text;
    }

    /// Quotes make a word even when they enclose nothing: `''` is an argument.
    void quoted() {
        started_ = true;
        quoted_ = true;
    }

    void expansion(bool quoted) {
        started_ = true;
        word_.expands = true;
        word_.splits = word_.splits || !quoted;
    }

    [[nodiscard]] bool started() const noexcept { return started_; }

    /// An unquoted run of digits directly before `<` or `>` names a file
    /// descriptor (`2>`), not an argument.
    [[nodiscard]] bool is_descriptor() const noexcept {
        return started_ && !quoted_ && word_.is_literal() && !word_.text.empty()
            && std::ranges::all_of(word_.text, is_digit);
    }

    [[nodiscard]] ShellWord take() && { return std::move(word_); }

private:
    ShellWord word_;
    bool started_ = false;
    bool quoted_ = false;
};

class Parser {
public:
    explicit Parser(std::string_view source) noexcept : source_(source) {}

    [[nodiscard]] std::vector<ShellCommand> run() && {
        parse_list(0);
        return std::move(commands_);
    }

private:
    [[nodiscard]] bool at_end() const noexcept { return pos_ >= source_.size(); }

    [[nodiscard]] char peek(std::size_t ahead = 0) const noexcept {
        return pos_ + ahead < source_.size() ? source_[pos_ + ahead] : '\0';
    }

    void skip_blanks() noexcept {
        while (!at_end() && is_blank(peek())) ++pos_;
    }

    // A list is commands joined by operators. At depth 0 it runs to the end
    // of input; inside $( it runs to the `)` that closes the substitution.
    void parse_list(int depth) {
        ShellCommand current;
        std::string_view dangling_operator;  // set after && || | |& until a command follows

        const auto end_command = [&](std::string_view op, bool needs_command_after) {
            if (current.empty()) {
                reject(std::format("`{}` has no command before it", op));
            }
            commands_.push_back(std::move(current));
            current.clear();
            dangling_operator = needs_command_after ? op : std::string_view{};
        };

        while (true) {
            skip_blanks();
            if (at_end()) {
                if (depth > 0) reject("unterminated `$(`");
                break;
            }
            const char c = peek();
            if (c == '\n') {
                ++pos_;
                if (!current.empty()) end_command("newline", false);
                continue;
            }
            if (c == ';') {
                if (peek(1) == ';') reject("`;;` is only valid inside `case`");
                ++pos_;
                end_command(";", false);
                continue;
            }
            if (c == '&') {
                if (peek(1) == '&') {
                    pos_ += 2;
                    end_command("&&", true);
                    continue;
                }
                if (peek(1) == '>') {
                    parse_redirection(depth);
                    continue;
                }
                reject("background jobs (`&`) are not analysed");
            }
            if (c == '|') {
                const bool two_chars = peek(1) == '|' || peek(1) == '&';
                const std::string_view op = source_.substr(pos_, two_chars ? 2 : 1);
                pos_ += op.size();
                end_command(op, true);
                continue;
            }
            if (c == '<' || c == '>') {
                parse_redirection(depth);
                continue;
            }
            if (c == '(') reject("subshells and `( … )` are not analysed");
            if (c == ')') {
                if (depth == 0) reject("unbalanced `)`");
                ++pos_;
                break;
            }
            if (c == '#') {  // a comment: only reachable where a word would start
                while (!at_end() && peek() != '\n') ++pos_;
                continue;
            }

            WordBuilder word;
            parse_word(word, depth);
            if (!word.started()) continue;  // a bare line continuation
            if (word.is_descriptor() && (peek() == '<' || peek() == '>')) {
                parse_redirection(depth);
                continue;
            }
            current.push_back(std::move(word).take());
            dangling_operator = {};
        }

        if (!dangling_operator.empty()) {
            reject(std::format("`{}` has no command after it", dangling_operator));
        }
        if (!current.empty()) commands_.push_back(std::move(current));
    }

    // At `<`, `>` or `&>`. Accepts only redirections that cannot write a file.
    void parse_redirection(int depth) {
        if (peek() == '&') {  // &> and &>> redirect stdout and stderr together
            pos_ += 2;
            if (peek() == '>') ++pos_;
            require_null_device(parse_target(depth));
            return;
        }
        if (peek() == '<') {
            ++pos_;
            switch (peek()) {
                case '<': reject("heredocs and here-strings are not analysed");
                case '(': reject("process substitution is not analysed");
                case '>': reject("`<>` opens a file for writing");
                case '&': ++pos_; parse_descriptor_target(); return;
                default: (void)parse_target(depth); return;  // reading is as harmless as `cat`
            }
        }
        ++pos_;  // '>'
        switch (peek()) {
            case '(': reject("process substitution is not analysed");
            case '|': reject("`>|` overwrites a file");
            case '&': ++pos_; parse_descriptor_target(); return;
            case '>': ++pos_; break;
            default: break;
        }
        require_null_device(parse_target(depth));
    }

    [[nodiscard]] ShellWord parse_target(int depth) {
        skip_blanks();
        if (at_end() || is_word_break(peek())) reject("a redirection has no target");
        WordBuilder word;
        parse_word(word, depth);
        return std::move(word).take();
    }

    // The target of >& or <&: a descriptor number or `-` (close).
    void parse_descriptor_target() {
        skip_blanks();
        const auto start = pos_;
        while (!at_end() && !is_word_break(peek())) ++pos_;
        const auto target = source_.substr(start, pos_ - start);
        const bool descriptor = !target.empty() && std::ranges::all_of(target, is_digit);
        if (!descriptor && target != "-") {
            reject(std::format("`>&{}` writes to a file", target));
        }
    }

    static void require_null_device(const ShellWord& target) {
        if (!target.is_literal() || target.text != "/dev/null") {
            reject("output redirection to a file is not analysed");
        }
    }

    void parse_word(WordBuilder& word, int depth) {
        while (!at_end() && !is_word_break(peek())) {
            const char c = peek();
            switch (c) {
                case '\'': {
                    const auto close = source_.find('\'', pos_ + 1);
                    if (close == std::string_view::npos) reject("unterminated `'`");
                    word.quoted();
                    word.literal(source_.substr(pos_ + 1, close - pos_ - 1));
                    pos_ = close + 1;
                    break;
                }
                case '"':
                    ++pos_;
                    parse_double_quoted(word, depth);
                    break;
                case '\\':
                    ++pos_;
                    if (at_end()) {
                        word.literal('\\');
                    } else if (peek() == '\n') {
                        ++pos_;  // line continuation
                    } else {
                        word.literal(peek());
                        ++pos_;
                    }
                    break;
                case '$':
                    ++pos_;
                    parse_dollar(word, depth, /*quoted=*/false);
                    break;
                case '`':
                    reject("backtick substitution is not analysed");
                case '*':
                case '?':
                case '{':
                    word.expansion(/*quoted=*/false);  // glob or brace expansion
                    ++pos_;
                    break;
                case '[':
                    // `[` alone is the test command; inside a word it opens a glob class.
                    if (pos_ + 1 >= source_.size() || is_word_break(peek(1))) {
                        word.literal(c);
                    } else {
                        word.expansion(/*quoted=*/false);
                    }
                    ++pos_;
                    break;
                default:
                    word.literal(c);
                    ++pos_;
                    break;
            }
        }
    }

    // After the opening `"`; consumes through the closing one.
    void parse_double_quoted(WordBuilder& word, int depth) {
        word.quoted();
        while (true) {
            if (at_end()) reject("unterminated `\"`");
            const char c = source_[pos_++];
            switch (c) {
                case '"':
                    return;
                case '\\':
                    if (at_end()) reject("unterminated `\"`");
                    if (std::string_view{"$`\"\\"}.contains(peek())) {
                        word.literal(peek());
                        ++pos_;
                    } else if (peek() == '\n') {
                        ++pos_;
                    } else {
                        word.literal('\\');
                    }
                    break;
                case '$':
                    parse_dollar(word, depth, /*quoted=*/true);
                    break;
                case '`':
                    reject("backtick substitution is not analysed");
                default:
                    word.literal(c);
                    break;
            }
        }
    }

    // After a `$`.
    void parse_dollar(WordBuilder& word, int depth, bool quoted) {
        const char c = peek();
        if (c == '(') {
            if (peek(1) == '(') reject("arithmetic expansion `$((…))` is not analysed");
            if (depth + 1 > kMaxSubstitutionDepth) reject("command substitutions nest too deeply");
            ++pos_;
            parse_list(depth + 1);
            word.expansion(quoted);
            return;
        }
        if (c == '{') {
            const auto close = source_.find('}', pos_);
            if (close == std::string_view::npos) reject("unterminated `${`");
            const auto name = source_.substr(pos_ + 1, close - pos_ - 1);
            if (!is_plain_parameter(name)) {
                reject(std::format("`${{{}}}` is not a plain parameter expansion", name));
            }
            pos_ = close + 1;
            word.expansion(quoted);
            return;
        }
        if (!quoted && (c == '\'' || c == '"')) {
            reject("`$'…'` and `$\"…\"` quoting is not analysed");
        }
        if (is_name_start(c)) {
            while (!at_end() && is_name_char(peek())) ++pos_;
            word.expansion(quoted);
            return;
        }
        if (!at_end() && is_special_parameter(c)) {
            ++pos_;
            word.expansion(quoted);
            return;
        }
        word.literal('$');  // a `$` that starts nothing is literal
    }

    std::string_view source_;
    std::size_t pos_ = 0;
    std::vector<ShellCommand> commands_;
};

} // namespace

std::expected<std::vector<ShellCommand>, std::string>
parse_shell_commands(std::string_view line) {
    try {
        return Parser{line}.run();
    } catch (Unsupported& unsupported) {
        return std::unexpected(std::move(unsupported.reason));
    }
}

} // namespace core::permissions
