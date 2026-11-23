#include "InlineCommands.hpp"

#include "../context/MentionSyntax.hpp"
#include "../context/PromptCode.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <ranges>
#include <unordered_set>
#include <utility>

namespace core::commands {

namespace {

namespace ascii = core::utils::ascii;
namespace str = core::utils::str;

[[nodiscard]] constexpr bool is_space(char ch) noexcept {
    return ascii::is_space(static_cast<unsigned char>(ch));
}

[[nodiscard]] constexpr bool is_horizontal_space(char ch) noexcept {
    return ch == ' ' || ch == '\t';
}

[[nodiscard]] std::size_t end_of_word(std::string_view text, std::size_t from) noexcept {
    while (from < text.size() && !is_space(text[from])) ++from;
    return from;
}

[[nodiscard]] std::size_t start_of_word(std::string_view text, std::size_t to) noexcept {
    while (to > 0 && !is_space(text[to - 1])) --to;
    return to;
}

/// `!ls /tmp` runs in a shell; nothing in it is a Filo command.
[[nodiscard]] bool is_shell_prompt(std::string_view prompt) noexcept {
    const auto first = prompt.find_first_not_of(" \t\r\n");
    return first != std::string_view::npos && prompt[first] == '!';
}

[[nodiscard]] const InlineCommandValue* find_value(const InlineCommandSpec& spec,
                                                   std::string_view typed) noexcept {
    const auto found = std::ranges::find_if(spec.values, [&](const InlineCommandValue& value) {
        return ascii::iequals(value.value, typed);
    });
    return found == spec.values.end() ? nullptr : &*found;
}

[[nodiscard]] bool starts_some_value(const InlineCommandSpec& spec, std::string_view typed) noexcept {
    return std::ranges::any_of(spec.values, [&](const InlineCommandValue& value) {
        return ascii::istarts_with(value.value, typed);
    });
}

/// The token whose name starts at @p at, or nullopt when the word there is
/// prose. @p at holds a '/' at a word boundary outside code.
[[nodiscard]] std::optional<InlineCommandToken> parse_token_at(std::string_view prompt,
                                                               std::size_t at,
                                                               const InlineCommandCatalog& catalog) {
    const std::size_t name_end = end_of_word(prompt, at);
    const InlineCommandSpec* spec = catalog.find(prompt.substr(at, name_end - at));
    if (spec == nullptr) return std::nullopt;

    std::size_t value_begin = name_end;
    while (value_begin < prompt.size() && is_horizontal_space(prompt[value_begin])) ++value_begin;
    const std::size_t word_end = end_of_word(prompt, value_begin);
    std::size_t value_end = word_end;
    while (value_end > value_begin
           && core::context::is_trailing_mention_punctuation(prompt[value_end - 1])) {
        --value_end;
    }

    InlineCommandToken token{.begin = at, .name = spec->name};
    const std::string_view typed = prompt.substr(value_begin, value_end - value_begin);
    if (word_end == value_begin) {
        token.end = token.value_begin = name_end;
        token.state = InlineCommandState::AwaitingValue;
        return token;
    }
    if (typed.empty()) return std::nullopt;  // `/effort ...`: punctuation is prose

    token.value_begin = value_begin;
    token.end = value_end;
    if (const auto* value = find_value(*spec, typed)) {
        token.value = value->value;
        token.state = InlineCommandState::Complete;
    } else if (starts_some_value(*spec, typed)) {
        token.value = std::string(typed);
        token.state = InlineCommandState::PartialValue;
    } else {
        return std::nullopt;
    }
    return token;
}

} // namespace

InlineCommandCatalog::InlineCommandCatalog(std::vector<InlineCommandSpec> specs)
    : specs_(std::move(specs)) {}

const InlineCommandSpec* InlineCommandCatalog::find(std::string_view name) const noexcept {
    const auto found = std::ranges::find_if(specs_, [&](const InlineCommandSpec& spec) {
        return ascii::iequals(spec.name, name);
    });
    return found == specs_.end() ? nullptr : &*found;
}

std::vector<InlineCommandToken> scan_inline_commands(std::string_view prompt,
                                                     const InlineCommandCatalog& catalog) {
    std::vector<InlineCommandToken> tokens;
    if (catalog.specs().empty() || is_shell_prompt(prompt)) return tokens;

    core::context::PromptCodeCursor code(prompt);
    for (std::size_t at = prompt.find('/'); at != std::string_view::npos;
         at = prompt.find('/', at + 1)) {
        if (code.contains(at)) {
            at = code.end_at(at) - 1;
            continue;
        }
        if (at > 0 && !is_space(prompt[at - 1])) continue;
        if (auto token = parse_token_at(prompt, at, catalog)) {
            at = std::max(at, token->end - 1);
            tokens.push_back(std::move(*token));
        }
    }

    std::unordered_set<std::string_view> applied;
    for (auto& token : tokens | std::views::reverse) {
        if (token.state != InlineCommandState::Complete) continue;
        token.superseded = !applied.insert(token.name).second;
    }
    return tokens;
}

InlineCommandExtraction extract_inline_commands(std::string_view prompt,
                                                const InlineCommandCatalog& catalog) {
    InlineCommandExtraction extraction;
    const auto tokens = scan_inline_commands(prompt, catalog);

    std::size_t copied = 0;
    for (const auto& token : tokens) {
        if (token.state != InlineCommandState::Complete) continue;
        if (!token.superseded) {
            extraction.invocations.push_back({.name = token.name, .value = token.value});
        }

        // Take the token out with the blanks after it, or, at the end of a
        // line, with the blanks before it, so no double space is left behind.
        std::size_t begin = token.begin;
        std::size_t end = token.end;
        while (end < prompt.size() && is_horizontal_space(prompt[end])) ++end;
        if (end == prompt.size() || prompt[end] == '\n' || prompt[end] == '\r') {
            while (begin > copied && is_horizontal_space(prompt[begin - 1])) --begin;
        }
        extraction.prompt.append(prompt.substr(copied, begin - copied));
        copied = end;
    }
    if (extraction.invocations.empty()) {
        extraction.prompt = std::string(prompt);
        return extraction;
    }
    extraction.prompt.append(prompt.substr(copied));
    if (str::trim_ascii_view(extraction.prompt).empty()) extraction.prompt.clear();
    return extraction;
}

std::optional<ActiveInlineCommand> find_active_inline_command(std::string_view prompt,
                                                              std::size_t cursor,
                                                              const InlineCommandCatalog& catalog) {
    cursor = std::min(cursor, prompt.size());
    if (catalog.specs().empty() || is_shell_prompt(prompt)) return std::nullopt;

    const std::size_t word_begin = start_of_word(prompt, cursor);
    const std::size_t word_end = end_of_word(prompt, cursor);
    if (core::context::is_inside_prompt_code(prompt, word_begin)) return std::nullopt;

    if (word_begin < prompt.size() && prompt[word_begin] == '/') {
        const std::string_view query = prompt.substr(word_begin, cursor - word_begin);
        const bool letters_only = std::ranges::all_of(query.substr(1), [](char ch) {
            return ascii::is_alpha(static_cast<unsigned char>(ch));
        });
        if (!letters_only || word_begin == prompt.find_first_not_of(" \t\r\n")) {
            return std::nullopt;
        }
        return ActiveInlineCommand{
            .part = ActiveInlineCommand::Part::Name,
            .replace_begin = word_begin,
            .replace_end = word_end,
            .query = std::string(query),
        };
    }

    // A value: the word before this one, on the same line, names a command.
    std::size_t name_end = word_begin;
    while (name_end > 0 && is_horizontal_space(prompt[name_end - 1])) --name_end;
    if (name_end == word_begin) return std::nullopt;
    const std::size_t name_begin = start_of_word(prompt, name_end);
    const InlineCommandSpec* spec = catalog.find(prompt.substr(name_begin, name_end - name_begin));
    if (spec == nullptr) return std::nullopt;

    return ActiveInlineCommand{
        .part = ActiveInlineCommand::Part::Value,
        .replace_begin = word_begin,
        .replace_end = word_end,
        .query = std::string(prompt.substr(word_begin, cursor - word_begin)),
        .command = spec->name,
    };
}

std::vector<InlineCommandSuggestion> suggest_inline_commands(const ActiveInlineCommand& active,
                                                             const InlineCommandCatalog& catalog,
                                                             std::size_t max_results) {
    std::vector<InlineCommandSuggestion> suggestions;
    if (active.part == ActiveInlineCommand::Part::Name) {
        for (const auto& spec : catalog.specs()) {
            if (!ascii::istarts_with(spec.name, active.query)) continue;
            suggestions.push_back({.insertion = spec.name, .description = spec.description});
        }
    } else if (const auto* spec = catalog.find(active.command)) {
        for (const auto& value : spec->values) {
            if (!ascii::istarts_with(value.value, active.query)) continue;
            suggestions.push_back({
                .insertion = value.value,
                .description = value.description,
                .current = ascii::iequals(value.value, spec->current_value),
            });
        }
    }
    if (suggestions.size() > max_results) suggestions.resize(max_results);
    return suggestions;
}

InlineCommandCompletion apply_inline_completion(std::string_view prompt,
                                                const ActiveInlineCommand& active,
                                                std::string_view insertion) {
    const std::size_t begin = std::min(active.replace_begin, prompt.size());
    const std::size_t end = std::clamp(active.replace_end, begin, prompt.size());

    InlineCommandCompletion out;
    out.text.reserve(prompt.size() + insertion.size() + 1);
    out.text.append(prompt.substr(0, begin));
    out.text.append(insertion);
    if (end >= prompt.size() || !is_horizontal_space(prompt[end])) out.text.push_back(' ');
    out.cursor = out.text.size() + (end < prompt.size() && is_horizontal_space(prompt[end]) ? 1 : 0);
    out.text.append(prompt.substr(end));
    return out;
}

} // namespace core::commands
