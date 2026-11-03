#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::frontmatter {

/**
 * @brief The YAML frontmatter of a Markdown manifest (`SKILL.md`,
 * `agents/<name>.md`) and the body that follows it.
 *
 * This is not a full YAML library, but a recursive-descent parser for the
 * YAML that manifests are written in: a top-level block mapping whose values
 * are read as text or lists of text. Malformed input is rejected with a line
 * number instead of being guessed at:
 *
 *  - Scalars: plain (multi-line, folded as YAML folds them), single-quoted
 *    (`''` escape), double-quoted (every YAML escape, including `\x`, `\u`,
 *    `\U` and escaped line breaks), literal `|` and folded `>` block scalars
 *    with chomping (`-`, `+`) and indentation indicators.
 *  - Collections: block sequences (indented, or compact at their key's
 *    column), flow sequences (may span lines), block and flow mappings, at
 *    any depth. A top-level sequence of scalars is a
 *    @ref Value::Kind::Sequence; anything deeper — a nested mapping, a
 *    sequence of mappings or of sequences — is parsed and validated, then
 *    reported as @ref Value::Kind::Nested. Its children never reach the top
 *    level, so a nested `name:` cannot replace the real one.
 *  - Properties: anchors (`&a`) and tags (`!tag`, `!!str`) are accepted;
 *    every value is text, so tags change nothing. Aliases (`*a`) resolve to
 *    the anchored node.
 *  - Errors: duplicate keys at any depth, tabs used as indentation,
 *    unterminated quotes or flow collections, invalid escapes, unexpected
 *    indentation, content after a closed quote, a block scalar line indented
 *    less than its block, undefined aliases, and complex (`? `) or
 *    non-scalar keys, which manifests never need.
 *
 * Two leniencies match what libyaml and js-yaml accept in practice:
 *  - a plain mapping value may contain `": "` (`description: Use when: X`),
 *    which strict YAML rejects but hand-written manifests do constantly;
 *  - a quoted scalar or flow collection may continue on lines at any
 *    indentation until it closes.
 *
 * The document starts with a `---` line (after an optional UTF-8 BOM and blank
 * lines) and ends at the next `---` or `...` line. Line endings may be LF or
 * CRLF.
 */
struct Value {
    enum class Kind { Scalar, Sequence, Nested };

    Kind kind = Kind::Scalar;
    /// Kind::Scalar: the value; an empty value (`key:`) is the empty string.
    std::string scalar;
    /// Kind::Sequence: the items, in order.
    std::vector<std::string> items;
};

struct ParseError {
    /// 1-based line number in the parsed text.
    std::size_t line = 0;
    std::string message;
};

struct Document {
    std::map<std::string, Value, std::less<>> fields;
    /// Everything after the closing delimiter, without leading blank lines
    /// and trailing whitespace.
    std::string body;

    [[nodiscard]] const Value* find(std::string_view key) const;

    /// A scalar value; nullopt when the key is absent or not a scalar.
    [[nodiscard]] std::optional<std::string> get(std::string_view key) const;

    /// A list value: a sequence's items, or a scalar split by @ref split_list.
    /// nullopt when the key is absent or nested. Empty items are dropped.
    [[nodiscard]] std::optional<std::vector<std::string>> list(std::string_view key) const;
};

[[nodiscard]] std::expected<Document, ParseError> parse(std::string_view content);

/// Reads and parses @p path. Failures are logged as `<path>:<line>: <reason>`.
[[nodiscard]] std::optional<Document> parse_file(const std::filesystem::path& path);

/// A boolean: `true`/`false`, `yes`/`no`, `on`/`off` (YAML's spellings) or
/// `1`/`0`, case-insensitive. nullopt for anything else, so the caller decides
/// what an invalid flag means.
[[nodiscard]] std::optional<bool> parse_bool(std::string_view value);

/// Splits a scalar that holds a list: comma-separated when it contains a
/// comma (`a, b`), otherwise whitespace-separated (`a b`, the Agent Skills
/// `allowed-tools` form). Items are trimmed; empty items are dropped.
[[nodiscard]] std::vector<std::string> split_list(std::string_view value);

/**
 * @brief Typed reads of a manifest's fields, each value of the wrong shape
 * logged against the file it came from.
 *
 * Every accessor takes the key's accepted spellings (`{"max-steps",
 * "max_steps"}`) and reads the first one present; the first spelling names
 * the key in warnings. A value that is present but unusable yields nullopt,
 * the same as an absent one, so callers state only their defaults.
 */
class FieldReader {
public:
    using Keys = std::initializer_list<std::string_view>;

    /// @p label prefixes every warning, e.g. "Agent definition".
    FieldReader(const Document& document, std::filesystem::path source, std::string_view label);

    [[nodiscard]] bool has_any(Keys keys) const;
    /// A scalar, trimmed.
    [[nodiscard]] std::optional<std::string> text(Keys keys) const;
    /// A scalar that @ref parse_bool accepts.
    [[nodiscard]] std::optional<bool> flag(Keys keys) const;
    /// A scalar holding a decimal integer greater than zero.
    [[nodiscard]] std::optional<int> positive_integer(Keys keys) const;
    /// A list, see @ref Document::list.
    [[nodiscard]] std::optional<std::vector<std::string>> list(Keys keys) const;

    /// Logs @p problem as `<label> '<source>': <problem>`.
    void warn(std::string_view problem) const;

private:
    /// The first spelling in @p keys that is present, or nullopt.
    [[nodiscard]] std::optional<std::string_view> present(Keys keys) const;

    const Document& document_;
    std::filesystem::path source_;
    std::string label_;
};

} // namespace core::tools::frontmatter
