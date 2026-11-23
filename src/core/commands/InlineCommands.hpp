#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// ---------------------------------------------------------------------------
// Inline commands: settings written anywhere in a prompt.
//
// `fix the parser /effort high` applies `/effort high` on send and sends
// `fix the parser`. Only a few commands qualify, and only with one of a closed
// set of values, because a slash word is just as often prose: `/settings` is a
// route, `/usr/bin` a path, `s/a/b/g` a substitution. A token is a command only
// when all of these hold, so anything else reaches the model untouched:
//
//   * the '/' starts the prompt or follows whitespace;
//   * it is not inside a code fence or an inline code span;
//   * the word is the name of an inline command, ignoring ASCII case;
//   * it is followed, on the same line, by one of that command's values.
//
// A prompt that starts with `!` is a shell command and holds none. To send a
// match as text, put it in backticks.
// ---------------------------------------------------------------------------

namespace core::commands {

/// One value an inline command accepts.
struct InlineCommandValue {
    std::string value;            ///< canonical spelling, matched ignoring ASCII case
    std::string description = {};
};

struct InlineCommandSpec {
    std::string name;                        ///< "/effort"
    std::string description;                 ///< picker row
    std::vector<InlineCommandValue> values;  ///< the closed set
    std::string current_value = {};          ///< the setting now; empty when unknown
};

/// The inline commands a session offers, rebuilt whenever their values may
/// have changed (a provider switch changes the advisors, for example).
class InlineCommandCatalog {
public:
    InlineCommandCatalog() = default;
    explicit InlineCommandCatalog(std::vector<InlineCommandSpec> specs);

    [[nodiscard]] const InlineCommandSpec* find(std::string_view name) const noexcept;
    [[nodiscard]] std::span<const InlineCommandSpec> specs() const noexcept { return specs_; }

private:
    std::vector<InlineCommandSpec> specs_;
};

enum class InlineCommandState : std::uint8_t {
    Complete,       ///< name and an accepted value: applied on send
    AwaitingValue,  ///< the name alone, value not typed yet
    PartialValue,   ///< a value that is the start of an accepted one
};

struct InlineCommandToken {
    std::size_t begin = 0;        ///< the '/'
    std::size_t end = 0;          ///< one past the value, or the name when there is none
    std::size_t value_begin = 0;  ///< == end when there is no value
    std::string name;             ///< spec name
    std::string value;            ///< canonical when Complete, as typed otherwise
    InlineCommandState state = InlineCommandState::Complete;
    bool superseded = false;      ///< a later Complete token sets the same command
};

/// Every inline command token in @p prompt, in order. Incomplete tokens are
/// reported so an editor can show them while they are typed; they are prose
/// when sent.
[[nodiscard]] std::vector<InlineCommandToken> scan_inline_commands(
    std::string_view prompt,
    const InlineCommandCatalog& catalog);

struct InlineCommandInvocation {
    std::string name;
    std::string value;

    [[nodiscard]] std::string command_line() const { return name + ' ' + value; }
    friend bool operator==(const InlineCommandInvocation&, const InlineCommandInvocation&) = default;
};

struct InlineCommandExtraction {
    /// One per command, in prompt order; a repeated command keeps its last value.
    std::vector<InlineCommandInvocation> invocations;
    /// The prompt without its Complete tokens; empty when nothing else was written.
    std::string prompt;
};

[[nodiscard]] InlineCommandExtraction extract_inline_commands(
    std::string_view prompt,
    const InlineCommandCatalog& catalog);

// ---- While typing: the inline picker ---------------------------------------

/// The part of an inline command the cursor is typing.
struct ActiveInlineCommand {
    enum class Part : std::uint8_t { Name, Value };

    Part part = Part::Name;
    std::size_t replace_begin = 0;
    std::size_t replace_end = 0;
    std::string query;    ///< "/eff" for a name, "hi" for a value
    std::string command;  ///< the command whose value is typed (Part::Value)
};

/// The inline command the cursor is in, or nullopt. A name is only offered
/// away from the start of the prompt, which the full command picker owns;
/// values are offered anywhere, the start included.
[[nodiscard]] std::optional<ActiveInlineCommand> find_active_inline_command(
    std::string_view prompt,
    std::size_t cursor,
    const InlineCommandCatalog& catalog);

struct InlineCommandSuggestion {
    std::string insertion;    ///< replaces [replace_begin, replace_end)
    std::string description;
    bool current = false;     ///< the value the setting has now
};

[[nodiscard]] std::vector<InlineCommandSuggestion> suggest_inline_commands(
    const ActiveInlineCommand& active,
    const InlineCommandCatalog& catalog,
    std::size_t max_results);

/// Replaces the active part with @p insertion followed by a space, so a name
/// moves straight on to its values and a value closes the picker.
struct InlineCommandCompletion {
    std::string text;
    std::size_t cursor = 0;
};

[[nodiscard]] InlineCommandCompletion apply_inline_completion(std::string_view prompt,
                                                              const ActiveInlineCommand& active,
                                                              std::string_view insertion);

} // namespace core::commands
