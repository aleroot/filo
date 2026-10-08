#include "CommandPolicy.hpp"

#include "../config/ConfigManager.hpp"
#include "../logging/Logger.hpp"
#include "../utils/JsonUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <regex>
#include <span>

namespace core::permissions {

namespace {

/// How a program's leading options are spelled, so they can be stepped over.
struct OptionGrammar {
    std::string program = {};
    std::vector<std::string> flags = {};               // take no value
    std::vector<std::string> options_with_value = {};  // take the next word, `=value`, or an attached value
};

/// A program that runs another command: `timeout 5 make` is judged as `make`.
struct Wrapper {
    OptionGrammar options = {};
    std::size_t positionals = 0;  // literal words between the options and the command
};

/// One position of a rule's pattern: a set of literal words, or a regex.
struct PatternToken {
    /// std::regex recurses per character; a longer word never matches rather
    /// than risking the stack on hostile input.
    static constexpr std::size_t kMaxRegexInput = 1024;

    std::vector<std::string> words = {};
    std::optional<std::regex> regex = {};

    [[nodiscard]] bool matches(const ShellWord& word) const {
        if (!word.is_literal()) return false;
        if (!regex) return std::ranges::contains(words, word.text);
        return word.text.size() <= kMaxRegexInput && std::regex_match(word.text, *regex);
    }
};

struct Rule {
    std::vector<PatternToken> pattern;
    CommandDecision decision = CommandDecision::Ask;
    std::vector<std::string> unless_options;
    std::optional<std::size_t> max_positionals;
    std::string label;  // where the rule is and its pattern, for messages
};

} // namespace

struct CommandPolicy::Definition {
    std::vector<Wrapper> wrappers;
    std::vector<OptionGrammar> global_options;
    std::vector<Rule> rules;
};

namespace {

using Definition = CommandPolicy::Definition;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

template <typename Entry, typename Program>
[[nodiscard]] const Entry* find_program(const std::vector<Entry>& entries,
                                        const ShellCommand& command,
                                        Program program_of) {
    if (command.empty() || !command.front().is_literal()) return nullptr;
    const auto it = std::ranges::find(entries, command.front().text, program_of);
    return it == entries.end() ? nullptr : &*it;
}

/// Steps over the option at `index`: the index after it, or nullopt when the
/// word is not an option the grammar declares (or its value is not literal).
[[nodiscard]] std::optional<std::size_t> skip_option(const OptionGrammar& grammar,
                                                     const ShellCommand& words,
                                                     std::size_t index) {
    const ShellWord& word = words[index];
    if (!word.is_literal()) return std::nullopt;
    const std::string_view text = word.text;
    if (std::ranges::contains(grammar.flags, text)) return index + 1;
    for (const std::string& option : grammar.options_with_value) {
        if (text == option) {
            const bool has_value = index + 1 < words.size() && words[index + 1].is_literal();
            return has_value ? std::optional{index + 2} : std::nullopt;
        }
        if (!text.starts_with(option) || text.size() == option.size()) continue;
        const bool long_option = option.starts_with("--");
        if (long_option && text[option.size()] == '=') return index + 1;  // --signal=KILL
        if (!long_option && option.size() == 2) return index + 1;          // -sKILL
    }
    return std::nullopt;
}

[[nodiscard]] std::expected<ShellCommand, std::string> unwrap(const Definition& definition,
                                                              ShellCommand command) {
    while (const Wrapper* wrapper =
               find_program(definition.wrappers, command, [](const Wrapper& w) -> const std::string& {
                   return w.options.program;
               })) {
        const std::string& program = wrapper->options.program;
        std::size_t index = 1;
        while (index < command.size() && command[index].text.starts_with('-')) {
            const auto next = skip_option(wrapper->options, command, index);
            if (!next) {
                return std::unexpected(std::format(
                    "cannot tell what `{}` runs: `{}` is not an option it is known to take",
                    program, command[index].text));
            }
            index = *next;
        }
        for (std::size_t skipped = 0; skipped < wrapper->positionals; ++skipped, ++index) {
            if (index >= command.size() || !command[index].is_literal()) {
                return std::unexpected(std::format("cannot tell what `{}` runs", program));
            }
        }
        if (index >= command.size()) {
            return std::unexpected(std::format("`{}` is given no command to run", program));
        }
        command.erase(command.begin(), command.begin() + static_cast<std::ptrdiff_t>(index));
    }
    return command;
}

[[nodiscard]] ShellCommand without_global_options(const Definition& definition,
                                                  const ShellCommand& command) {
    const OptionGrammar* grammar = find_program(
        definition.global_options, command, &OptionGrammar::program);
    if (grammar == nullptr) return command;
    std::size_t index = 1;
    while (index < command.size()) {
        const auto next = skip_option(*grammar, command, index);
        if (!next) break;
        index = *next;
    }
    ShellCommand result{command.front()};
    result.insert(result.end(), command.begin() + static_cast<std::ptrdiff_t>(index), command.end());
    return result;
}

/// Whether `argument` may set `option`, as a rule spells it:
///   --long  also matches `--long=value` and abbreviations such as `--lo`
///           (GNU getopt and git both accept unambiguous prefixes);
///   -x      matches any short-option cluster containing x (`-nxv`);
///   -word   (single-dash long, like find's -exec) matches by prefix.
/// An argument whose text is decided at run time may be anything the shell
/// produces, so it matches unless its known prefix rules the option out.
[[nodiscard]] bool may_set_option(const ShellWord& argument, std::string_view option) {
    std::string_view text = argument.text;
    if (!argument.is_literal()) {
        if (!text.starts_with('-')) return text.empty();
        const auto equals = text.find('=');
        if (equals == std::string_view::npos) return true;  // the option's name is unknown
        text = text.substr(0, equals);
    }
    if (text.size() < 2 || !text.starts_with('-')) return false;
    if (option.starts_with("--")) {
        if (!text.starts_with("--")) return false;
        const std::string_view name = text.substr(0, text.find('='));
        return name == option || (name.size() > 2 && option.starts_with(name));
    }
    if (option.size() == 2) {
        return !text.starts_with("--") && text.find(option[1], 1) != std::string_view::npos;
    }
    return text.starts_with(option);
}

/// Positional arguments, or nullopt when an unquoted expansion makes the
/// count unknowable. An option's separate value counts as positional, which
/// can only make a `max_positionals` rule stricter.
[[nodiscard]] std::optional<std::size_t> count_positionals(std::span<const ShellWord> arguments) {
    std::size_t count = 0;
    bool options_ended = false;
    for (const ShellWord& argument : arguments) {
        if (argument.splits) return std::nullopt;
        if (!options_ended && argument.is_literal() && argument.text == "--") {
            options_ended = true;
            continue;
        }
        const bool option = !options_ended && argument.text.size() > 1 && argument.text.starts_with('-');
        if (!option) ++count;
    }
    return count;
}

[[nodiscard]] bool rule_matches(const Rule& rule, const ShellCommand& command) {
    if (command.size() < rule.pattern.size()) return false;
    for (std::size_t i = 0; i < rule.pattern.size(); ++i) {
        if (!rule.pattern[i].matches(command[i])) return false;
    }
    const std::span<const ShellWord> arguments{command.begin() + static_cast<std::ptrdiff_t>(rule.pattern.size()),
                                               command.end()};
    for (const ShellWord& argument : arguments) {
        for (const std::string& option : rule.unless_options) {
            if (may_set_option(argument, option)) return false;
        }
    }
    if (!rule.max_positionals) return true;
    const auto positionals = count_positionals(arguments);
    return positionals && *positionals <= *rule.max_positionals;
}

[[nodiscard]] std::expected<std::vector<ShellCommand>, std::string>
analyse(const Definition& definition, std::string_view line) {
    auto parsed = parse_shell_commands(line);
    if (!parsed) return std::unexpected(std::move(parsed.error()));
    std::vector<ShellCommand> commands;
    commands.reserve(parsed->size());
    for (ShellCommand& command : *parsed) {
        auto unwrapped = unwrap(definition, std::move(command));
        if (!unwrapped) return std::unexpected(std::move(unwrapped.error()));
        commands.push_back(std::move(*unwrapped));
    }
    return commands;
}

[[nodiscard]] CommandDecision decide_command(const Definition& definition, const ShellCommand& command) {
    const ShellCommand normalized = without_global_options(definition, command);
    bool allowed = false;
    for (const Rule& rule : definition.rules) {
        if (!rule_matches(rule, normalized)) continue;
        if (rule.decision == CommandDecision::Ask) return CommandDecision::Ask;
        allowed = true;
    }
    return allowed ? CommandDecision::Allow : CommandDecision::Ask;
}

// ---------------------------------------------------------------------------
// Reading a rules document
// ---------------------------------------------------------------------------

/// Raised while reading a layer; with_layer reports it.
struct InvalidLayer {
    std::string reason;
};

[[noreturn]] void invalid(std::string reason) {
    throw InvalidLayer{std::move(reason)};
}

using simdjson::dom::array;
using simdjson::dom::element;
using simdjson::dom::object;

[[nodiscard]] object as_object(element value, std::string_view where) {
    object result;
    if (value.get(result) != simdjson::SUCCESS) invalid(std::format("{} must be an object", where));
    return result;
}

[[nodiscard]] array as_array(element value, std::string_view where) {
    array result;
    if (value.get(result) != simdjson::SUCCESS) invalid(std::format("{} must be an array", where));
    return result;
}

[[nodiscard]] std::string as_string(element value, std::string_view where) {
    std::string_view result;
    if (value.get(result) != simdjson::SUCCESS || result.empty()) {
        invalid(std::format("{} must be a non-empty string", where));
    }
    return std::string(result);
}

[[nodiscard]] std::vector<std::string> as_strings(element value, std::string_view where) {
    std::vector<std::string> result;
    for (element item : as_array(value, where)) result.push_back(as_string(item, where));
    return result;
}

/// Typos must not pass silently: `"unless_option"` would drop a safeguard.
void require_known_keys(object value, std::initializer_list<std::string_view> known, std::string_view where) {
    for (const auto field : value) {
        if (!std::ranges::contains(known, field.key)) {
            invalid(std::format("{}: unknown key \"{}\"", where, field.key));
        }
    }
}

[[nodiscard]] std::optional<element> field(object value, std::string_view key) {
    element result;
    if (value[key].get(result) != simdjson::SUCCESS) return std::nullopt;
    return result;
}

[[nodiscard]] element required_field(object value, std::string_view key, std::string_view where) {
    auto result = field(value, key);
    if (!result) invalid(std::format("{}: missing \"{}\"", where, key));
    return *result;
}

[[nodiscard]] std::vector<std::string> optional_strings(object value, std::string_view key, std::string_view where) {
    const auto list = field(value, key);
    return list ? as_strings(*list, std::format("{}.{}", where, key)) : std::vector<std::string>{};
}

[[nodiscard]] std::size_t optional_count(object value, std::string_view key, std::string_view where) {
    const auto number = field(value, key);
    if (!number) return 0;
    std::uint64_t count = 0;
    if (number->get(count) != simdjson::SUCCESS) {
        invalid(std::format("{}.{} must be a non-negative integer", where, key));
    }
    return static_cast<std::size_t>(count);
}

[[nodiscard]] OptionGrammar read_option_grammar(object value, std::string_view where) {
    return OptionGrammar{
        .program = as_string(required_field(value, "program", where), std::format("{}.program", where)),
        .flags = optional_strings(value, "flags", where),
        .options_with_value = optional_strings(value, "options_with_value", where),
    };
}

[[nodiscard]] PatternToken read_pattern_token(element value, bool first, std::string& label,
                                              std::string_view where) {
    if (value.is_string()) {
        auto word = as_string(value, where);
        label += word;
        return PatternToken{.words = {std::move(word)}};
    }
    if (value.is_array()) {
        auto words = as_strings(value, where);
        if (words.empty()) invalid(std::format("{}: an alternatives list must not be empty", where));
        label += std::format("[{}]", words.size() == 1 ? words.front() : std::format("{}|…", words.front()));
        return PatternToken{.words = std::move(words)};
    }
    const object regex_object = as_object(value, where);
    require_known_keys(regex_object, {"regex"}, where);
    if (first) invalid(std::format("{}: the program must be named literally, not by a regex", where));
    const std::string source = as_string(required_field(regex_object, "regex", where), where);
    label += std::format("/{}/", source);
    try {
        return PatternToken{.regex = std::regex(source, std::regex::ECMAScript)};
    } catch (const std::regex_error& error) {
        invalid(std::format("{}: invalid regex /{}/: {}", where, source, error.what()));
    }
}

[[nodiscard]] CommandDecision read_decision(element value, std::string_view where) {
    const std::string decision = as_string(value, where);
    if (decision == "allow") return CommandDecision::Allow;
    if (decision == "ask") return CommandDecision::Ask;
    invalid(std::format("{}: decision must be \"allow\" or \"ask\", not \"{}\"", where, decision));
}

struct Examples {
    std::vector<std::string> match;
    std::vector<std::string> not_match;
};

[[nodiscard]] std::pair<Rule, Examples> read_rule(object value, std::string_view where) {
    require_known_keys(value,
                       {"pattern", "decision", "unless_options", "max_positionals",
                        "justification", "match", "not_match"},
                       where);
    Rule rule;
    std::string written;
    const array pattern = as_array(required_field(value, "pattern", where), std::format("{}.pattern", where));
    for (element token : pattern) {
        if (!rule.pattern.empty()) written += ' ';
        rule.pattern.push_back(read_pattern_token(token, rule.pattern.empty(), written,
                                                  std::format("{}.pattern", where)));
    }
    if (rule.pattern.empty()) invalid(std::format("{}.pattern must not be empty", where));
    const std::string named = std::format("{} `{}`", where, written);
    rule.label = named;

    rule.decision = read_decision(required_field(value, "decision", named), std::format("{}.decision", named));
    rule.unless_options = optional_strings(value, "unless_options", named);
    for (const std::string& option : rule.unless_options) {
        if (!option.starts_with('-') || option == "-" || option == "--") {
            invalid(std::format("{}: \"{}\" in unless_options is not an option", named, option));
        }
    }
    if (field(value, "max_positionals")) {
        rule.max_positionals = optional_count(value, "max_positionals", named);
    }
    if (const auto justification = field(value, "justification")) {
        (void)as_string(*justification, std::format("{}.justification", named));
    }

    Examples examples{
        .match = optional_strings(value, "match", named),
        .not_match = optional_strings(value, "not_match", named),
    };
    if (examples.match.empty()) {
        invalid(std::format("{}: needs at least one \"match\" example", named));
    }
    return {std::move(rule), std::move(examples)};
}

template <typename Entry, typename Program>
void upsert(std::vector<Entry>& entries, Entry entry, Program program_of) {
    const auto existing = std::ranges::find(entries, program_of(entry), program_of);
    if (existing == entries.end()) {
        entries.push_back(std::move(entry));
    } else {
        *existing = std::move(entry);  // a later layer redefines the program
    }
}

/// Appends the document's definitions to `definition`; returns the new
/// rules' examples, in rule order.
[[nodiscard]] std::vector<Examples> read_layer(std::string_view json, Definition& definition) {
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(json);
    element document;
    if (const auto error = parser.parse(padded).get(document); error != simdjson::SUCCESS) {
        invalid(std::format("not valid JSON: {}", simdjson::error_message(error)));
    }
    const object root = as_object(document, "the document");
    require_known_keys(root, {"wrappers", "global_options", "rules"}, "the document");

    if (const auto wrappers = field(root, "wrappers")) {
        std::size_t index = 0;
        for (element item : as_array(*wrappers, "wrappers")) {
            const std::string where = std::format("wrappers[{}]", index++);
            const object value = as_object(item, where);
            require_known_keys(value, {"program", "flags", "options_with_value", "positionals", "justification"},
                               where);
            upsert(definition.wrappers,
                   Wrapper{.options = read_option_grammar(value, where),
                           .positionals = optional_count(value, "positionals", where)},
                   [](const Wrapper& w) -> const std::string& { return w.options.program; });
        }
    }
    if (const auto global_options = field(root, "global_options")) {
        std::size_t index = 0;
        for (element item : as_array(*global_options, "global_options")) {
            const std::string where = std::format("global_options[{}]", index++);
            const object value = as_object(item, where);
            require_known_keys(value, {"program", "flags", "options_with_value", "justification"}, where);
            upsert(definition.global_options, read_option_grammar(value, where),
                   [](const OptionGrammar& g) -> const std::string& { return g.program; });
        }
    }

    std::vector<Examples> examples;
    if (const auto rules = field(root, "rules")) {
        std::size_t index = 0;
        for (element item : as_array(*rules, "rules")) {
            const std::string where = std::format("rules[{}]", index++);
            auto [rule, rule_examples] = read_rule(as_object(item, where), where);
            definition.rules.push_back(std::move(rule));
            examples.push_back(std::move(rule_examples));
        }
    }
    return examples;
}

/// Each example is one command; it must (not) match its own rule once the
/// combined policy has removed wrappers and global options.
void check_examples(const Definition& definition, std::size_t first_rule, const std::vector<Examples>& examples) {
    for (std::size_t i = 0; i < examples.size(); ++i) {
        const Rule& rule = definition.rules[first_rule + i];
        const auto matches = [&](const std::string& example) -> std::optional<bool> {
            const auto commands = analyse(definition, example);
            if (!commands) return std::nullopt;
            if (commands->size() != 1) {
                invalid(std::format("{}: example \"{}\" is not a single command", rule.label, example));
            }
            return rule_matches(rule, without_global_options(definition, commands->front()));
        };
        for (const std::string& example : examples[i].match) {
            if (matches(example) != true) {
                invalid(std::format("{}: example \"{}\" should match but does not", rule.label, example));
            }
        }
        for (const std::string& example : examples[i].not_match) {
            if (matches(example) == true) {
                invalid(std::format("{}: example \"{}\" should not match but does", rule.label, example));
            }
        }
    }
}

[[nodiscard]] std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

} // namespace

// ---------------------------------------------------------------------------
// CommandPolicy
// ---------------------------------------------------------------------------

CommandPolicy::CommandPolicy() : definition_(std::make_shared<const Definition>()) {}

CommandPolicy::CommandPolicy(std::shared_ptr<const Definition> definition) noexcept
    : definition_(std::move(definition)) {}

std::expected<CommandPolicy, std::string> CommandPolicy::with_layer(std::string_view json) const {
    try {
        auto combined = std::make_shared<Definition>(*definition_);
        const std::size_t first_new_rule = combined->rules.size();
        const auto examples = read_layer(json, *combined);
        check_examples(*combined, first_new_rule, examples);
        return CommandPolicy{std::move(combined)};
    } catch (InvalidLayer& error) {
        return std::unexpected(std::move(error.reason));
    } catch (const std::exception& error) {  // e.g. std::regex_error while checking an example
        return std::unexpected(std::format("cannot be checked: {}", error.what()));
    }
}

std::expected<std::vector<ShellCommand>, std::string> CommandPolicy::commands(std::string_view line) const {
    return analyse(*definition_, line);
}

// std::regex_match may throw (error_complexity, error_stack). A decision that
// cannot be completed is an Ask, never an exception into the permission path.
CommandDecision CommandPolicy::decide(const ShellCommand& command) const noexcept {
    try {
        return decide_command(*definition_, command);
    } catch (...) {
        return CommandDecision::Ask;
    }
}

CommandDecision CommandPolicy::decide(std::string_view line) const noexcept {
    try {
        const auto commands = analyse(*definition_, line);
        if (!commands) return CommandDecision::Ask;
        const bool all_allowed = std::ranges::all_of(*commands, [&](const ShellCommand& command) {
            return decide_command(*definition_, command) == CommandDecision::Allow;
        });
        return all_allowed ? CommandDecision::Allow : CommandDecision::Ask;
    } catch (...) {
        return CommandDecision::Ask;
    }
}

// ---------------------------------------------------------------------------
// The enforced policy
// ---------------------------------------------------------------------------

std::string user_command_rules_path() {
    const std::string config_dir = core::config::ConfigManager::get_instance().get_config_dir();
    return config_dir.empty() ? std::string{} : config_dir + "/command_rules.json";
}

const CommandPolicy& command_policy() {
    static const CommandPolicy policy = [] {
        auto built_in = CommandPolicy{}.with_layer(default_command_rules());
        if (!built_in) {
            core::logging::error("Built-in command rules are invalid, every command will ask: {}",
                                 built_in.error());
            return CommandPolicy{};
        }
        const std::string path = user_command_rules_path();
        std::error_code ignored;
        if (path.empty() || !std::filesystem::exists(path, ignored)) return *built_in;

        const auto text = read_file(path);
        if (!text) {
            core::logging::warn("Cannot read command rules from '{}'; using the built-in rules", path);
            return *built_in;
        }
        auto layered = built_in->with_layer(*text);
        if (!layered) {
            core::logging::warn("Ignoring command rules in '{}': {}", path, layered.error());
            return *built_in;
        }
        core::logging::info("Loaded command rules from '{}'", path);
        return *layered;
    }();
    return policy;
}

std::optional<std::string> shell_command_argument(std::string_view tool_args_json) {
    return core::utils::json::first_string_field(tool_args_json, {"command"});
}

bool shell_call_is_allowed(std::string_view tool_args_json) noexcept {
    try {
        const auto command = shell_command_argument(tool_args_json);
        return command && command_policy().decide(*command) == CommandDecision::Allow;
    } catch (...) {
        return false;  // whatever went wrong, asking is the safe answer
    }
}

} // namespace core::permissions
