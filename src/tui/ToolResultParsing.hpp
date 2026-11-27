#pragma once

#include <simdjson.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Tool result payloads parsed into what the transcript draws. Pure data: no
// FTXUI here, so each shape can be read (and tested) apart from its card.
namespace tui::tool_results {

/// Parses `payload` as a JSON object and hands it to `fn`. Returns false when
/// the payload is absent or not an object, which is the normal case for tools
/// that report plain-text errors.
template <typename Fn>
bool with_json_object(std::string_view payload, Fn&& fn) {
    if (payload.empty()) return false;
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(payload).get(document) != simdjson::SUCCESS) return false;
    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) return false;
    fn(object);
    return true;
}

struct SearchResultRow {
    std::string  path;
    std::int64_t line = 0;
    std::string  text;
};

struct PathResultRow {
    std::string_view marker;  // Points at a string literal.
    std::string      path;
};

struct WebResultRow {
    std::string title;
    std::string url;
    std::string snippet;
};

struct TodoRow {
    std::string content;
    std::string status;
};

struct FetchMetadata {
    std::string  url;
    std::string  content_type;
    std::string  title;
    std::int64_t status_code = 0;
    bool         truncated = false;
    bool         parsed = false;
};

struct ReviewFindingRow {
    std::string  title;
    std::string  severity;
    std::string  category;
    std::string  path;
    std::int64_t line_start = 0;
    std::int64_t line_end = 0;
    std::string  body;
    std::string  rationale;
};

struct UnreviewedUnit {
    std::string unit;
    std::string reason;
};

/// The `review` tool's JSON, shaped for reading: blocking findings first,
/// paths relative to the reviewed repository.
struct ReviewResult {
    std::vector<ReviewFindingRow> blocking;
    std::vector<ReviewFindingRow> follow_ups;
    std::vector<UnreviewedUnit>   unreviewed;
    std::string                   error;
    bool                          empty = false;
    bool                          incomplete = false;
};

struct MemoryEntryRow {
    std::string id;
    std::string content;
    std::string scope;
};

/// A `memory` call that saved, revised, or listed facts. Other actions
/// (`status`, `clean`) have no shape of their own and read as plain fields.
struct MemoryResult {
    /// What the store said it did, e.g. "Stored memory m905".
    std::string                 outcome;
    std::vector<MemoryEntryRow> entries;
    bool                        listing = false;
    std::int64_t                total = 0;
    bool                        has_more = false;
};

struct AnswerRow {
    std::string header;
    std::string question;
    std::string answer;
};

/// `AskUserQuestion`'s answers, matched back to the questions that were asked.
struct QuestionAnswers {
    std::vector<AnswerRow> answers;
    std::string            note;
};

/// One line of a JSON value laid out as labelled fields. A row with an empty
/// `value` heads the rows nested under it at `depth + 1`.
struct JsonFieldRow {
    int         depth = 0;
    std::string label;
    std::string value;
};

[[nodiscard]] bool parse_search_rows(std::string_view payload, std::vector<SearchResultRow>& out);
[[nodiscard]] bool parse_path_rows(std::string_view payload, std::vector<PathResultRow>& out);
[[nodiscard]] bool parse_web_rows(std::string_view payload, std::vector<WebResultRow>& out);
[[nodiscard]] bool parse_todo_rows(std::string_view payload, std::vector<TodoRow>& out);
[[nodiscard]] bool parse_fetch_metadata(std::string_view payload, FetchMetadata& out);
[[nodiscard]] bool parse_review_result(std::string_view payload, ReviewResult& out);
[[nodiscard]] bool parse_memory_result(std::string_view payload, MemoryResult& out);
/// `arguments` are the call's own, used to show each question's text even
/// when the model keyed its answer by id.
[[nodiscard]] bool parse_question_answers(std::string_view payload,
                                          std::string_view arguments,
                                          QuestionAnswers& out);
/// Any JSON object or array as readable fields: labels lose their
/// snake_case/camelCase, booleans read yes/no, JSON carried inside a string
/// (an MCP `result`) is opened, and a lone wrapper member is skipped.
[[nodiscard]] bool parse_json_fields(std::string_view payload, std::vector<JsonFieldRow>& out);

} // namespace tui::tool_results
