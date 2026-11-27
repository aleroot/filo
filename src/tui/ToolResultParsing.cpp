#include "ToolResultParsing.hpp"

#include "core/utils/JsonUtils.hpp"

#include <algorithm>
#include <format>
#include <memory>
#include <optional>

namespace tui::tool_results {
namespace {

/// "/repo/src/a.cpp" -> "src/a.cpp" when the finding lies inside `repository`.
std::string repository_relative_path(std::string path, std::string_view repository) {
    while (repository.size() > 1 && repository.back() == '/') repository.remove_suffix(1);
    if (!repository.empty()
        && path.size() > repository.size() + 1
        && path.starts_with(repository)
        && path[repository.size()] == '/') {
        path.erase(0, repository.size() + 1);
    }
    return path;
}

/// Drops the reviewer's "[P1] " priority tag: the card shows the severity.
std::string review_title_without_priority(std::string title) {
    if (title.size() > 5 && title[0] == '[' && (title[1] == 'P' || title[1] == 'p')
        && title[2] >= '0' && title[2] <= '9' && title[3] == ']' && title[4] == ' ') {
        title.erase(0, 5);
    }
    return title;
}

} // namespace

bool parse_search_rows(std::string_view payload, std::vector<SearchResultRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array matches;
        if (object["matches"].get_array().get(matches) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(matches.size());
        for (const auto item : matches) {
            simdjson::dom::object match;
            if (item.get_object().get(match) != simdjson::SUCCESS) continue;
            const auto path = core::utils::json::first_string_field(match, {"path"});
            if (!path) continue;
            const auto text = core::utils::json::first_string_field(match, {"text"});
            std::int64_t line = 0;
            core::utils::json::ignore_error(match["line"].get(line));
            out.push_back({
                .path = *path,
                .line = line,
                .text = text.value_or(std::string{}),
            });
        }
    });
    return parsed && found;
}

bool parse_path_rows(std::string_view payload, std::vector<PathResultRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array files;
        if (object["files"].get_array().get(files) == simdjson::SUCCESS) {
            found = true;
            out.reserve(files.size());
            for (const auto item : files) {
                std::string_view path;
                if (item.get(path) == simdjson::SUCCESS) {
                    out.push_back({.marker = "·", .path = std::string(path)});
                }
            }
            return;
        }

        simdjson::dom::array entries;
        if (object["entries"].get_array().get(entries) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(entries.size());
        for (const auto item : entries) {
            simdjson::dom::object entry;
            if (item.get_object().get(entry) != simdjson::SUCCESS) continue;
            const auto name = core::utils::json::first_string_field(entry, {"name"});
            if (!name) continue;
            const auto type = core::utils::json::first_string_field(entry, {"type"});
            out.push_back({
                .marker = type && *type == "dir" ? "▸" : "·",
                .path = *name,
            });
        }
    });
    return parsed && found;
}

bool parse_web_rows(std::string_view payload, std::vector<WebResultRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array results;
        if (object["results"].get_array().get(results) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(results.size());
        for (const auto item : results) {
            simdjson::dom::object result;
            if (item.get_object().get(result) != simdjson::SUCCESS) continue;
            out.push_back({
                .title = core::utils::json::first_string_field(result, {"title"})
                             .value_or(std::string{"Untitled"}),
                .url = core::utils::json::first_string_field(result, {"url"})
                           .value_or(std::string{}),
                .snippet = core::utils::json::first_string_field(result, {"snippet", "content"})
                               .value_or(std::string{}),
            });
        }
    });
    return parsed && found;
}

bool parse_todo_rows(std::string_view payload, std::vector<TodoRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array array;
        if (object["todos"].get_array().get(array) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(array.size());
        for (const auto item : array) {
            simdjson::dom::object todo;
            if (item.get_object().get(todo) != simdjson::SUCCESS) continue;
            const auto content = core::utils::json::first_string_field(todo, {"content"});
            if (!content) continue;
            const auto status = core::utils::json::first_string_field(todo, {"status"});
            out.push_back({
                .content = *content,
                .status = status.value_or(std::string{"pending"}),
            });
        }
    });
    return parsed && found;
}

bool parse_review_result(std::string_view payload, ReviewResult& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array blocking;
        simdjson::dom::array follow_ups;
        const bool has_lists =
            object["blocking"].get_array().get(blocking) == simdjson::SUCCESS
            && object["follow_ups"].get_array().get(follow_ups) == simdjson::SUCCESS;
        out.error = core::utils::json::first_string_field(object, {"error"})
                        .value_or(std::string{});
        // A failed review with nothing found carries only `ok` and `error`.
        if (!has_lists && out.error.empty()) return;
        found = true;
        core::utils::json::ignore_error(object["empty"].get(out.empty));
        core::utils::json::ignore_error(object["incomplete"].get(out.incomplete));
        out.reviewer = core::utils::json::first_string_field(object, {"reviewer"})
                           .value_or(std::string{});
        const std::string repository =
            core::utils::json::first_string_field(object, {"repository"})
                .value_or(std::string{});
        auto read_findings = [&](simdjson::dom::array array,
                                 std::vector<ReviewFindingRow>& rows) {
            rows.reserve(array.size());
            for (const auto item : array) {
                simdjson::dom::object finding;
                if (item.get_object().get(finding) != simdjson::SUCCESS) continue;
                auto field = [&](const char* name) {
                    return core::utils::json::first_string_field(finding, {name})
                        .value_or(std::string{});
                };
                ReviewFindingRow row{
                    .title = review_title_without_priority(field("title")),
                    .severity = field("severity"),
                    .category = field("category"),
                    .path = repository_relative_path(field("path"), repository),
                    .body = field("body"),
                    .rationale = field("rationale"),
                };
                core::utils::json::ignore_error(finding["line_start"].get(row.line_start));
                core::utils::json::ignore_error(finding["line_end"].get(row.line_end));
                rows.push_back(std::move(row));
            }
        };
        if (has_lists) {
            read_findings(blocking, out.blocking);
            read_findings(follow_ups, out.follow_ups);
        }
        simdjson::dom::array unreviewed;
        if (object["unreviewed"].get_array().get(unreviewed) == simdjson::SUCCESS) {
            for (const auto item : unreviewed) {
                simdjson::dom::object unit;
                if (item.get_object().get(unit) != simdjson::SUCCESS) continue;
                out.unreviewed.push_back({
                    .unit = core::utils::json::first_string_field(unit, {"unit"})
                                .value_or(std::string{}),
                    .reason = core::utils::json::first_string_field(unit, {"reason"})
                                  .value_or(std::string{}),
                });
            }
        }
    });
    return parsed && found;
}

bool parse_fetch_metadata(std::string_view payload, FetchMetadata& out) {
    out.parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        out.url = core::utils::json::first_string_field(object, {"url"})
                      .value_or(std::string{});
        out.content_type = core::utils::json::first_string_field(object, {"content_type"})
                               .value_or(std::string{});
        out.title = core::utils::json::first_string_field(object, {"title"})
                        .value_or(std::string{});
        core::utils::json::ignore_error(object["status_code"].get(out.status_code));
        core::utils::json::ignore_error(object["truncated"].get(out.truncated));
    });
    return out.parsed;
}

namespace {

/// "Stored memory {m905}." -> "Stored memory m905": the store's message is
/// written for the model, the header only needs the act and the id.
std::string memory_outcome(std::string message) {
    std::erase_if(message, [](char c) { return c == '{' || c == '}'; });
    while (!message.empty() && (message.back() == '.' || message.back() == ' ')) {
        message.pop_back();
    }
    return message;
}

MemoryEntryRow memory_entry(simdjson::dom::object entry) {
    using core::utils::json::first_string_field;
    return {
        .id = first_string_field(entry, {"id"}).value_or(std::string{}),
        .content = first_string_field(entry, {"content"}).value_or(std::string{}),
        .scope = first_string_field(entry, {"scope"}).value_or(std::string{}),
    };
}

} // namespace

bool parse_memory_result(std::string_view payload, MemoryResult& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array entries;
        if (object["entries"].get_array().get(entries) == simdjson::SUCCESS) {
            found = true;
            out.listing = true;
            core::utils::json::ignore_error(object["total"].get(out.total));
            core::utils::json::ignore_error(object["has_more"].get(out.has_more));
            out.entries.reserve(entries.size());
            for (const auto item : entries) {
                simdjson::dom::object entry;
                if (item.get_object().get(entry) == simdjson::SUCCESS) {
                    out.entries.push_back(memory_entry(entry));
                }
            }
            return;
        }
        const auto message = core::utils::json::first_string_field(object, {"message"});
        if (!message) return;
        found = true;
        out.outcome = memory_outcome(*message);
        simdjson::dom::object entry;
        if (object["entry"].get_object().get(entry) == simdjson::SUCCESS) {
            out.entries.push_back(memory_entry(entry));
        }
    });
    return parsed && found;
}

bool parse_question_answers(std::string_view payload,
                            std::string_view arguments,
                            QuestionAnswers& out) {
    struct Asked {
        std::string key;
        std::string header;
        std::string question;
    };
    std::vector<Asked> asked;
    (void)with_json_object(arguments, [&](simdjson::dom::object object) {
        simdjson::dom::array questions;
        if (object["questions"].get_array().get(questions) != simdjson::SUCCESS) return;
        for (const auto item : questions) {
            simdjson::dom::object question;
            if (item.get_object().get(question) != simdjson::SUCCESS) continue;
            auto field = [&](const char* name) {
                return core::utils::json::first_string_field(question, {name})
                    .value_or(std::string{});
            };
            Asked entry{.key = field("id"), .header = field("header"), .question = field("question")};
            // The tool keys an answer by the question's id when one was given.
            if (entry.key.empty()) entry.key = entry.question;
            asked.push_back(std::move(entry));
        }
    });

    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::object answers;
        if (object["answers"].get_object().get(answers) != simdjson::SUCCESS) return;
        found = true;
        out.note = core::utils::json::first_string_field(object, {"note"})
                       .value_or(std::string{});
        for (const auto field : answers) {
            std::string_view answer;
            if (field.value.get(answer) != simdjson::SUCCESS) continue;
            AnswerRow row{.question = std::string(field.key), .answer = std::string(answer)};
            const auto match = std::ranges::find(asked, field.key, &Asked::key);
            if (match != asked.end()) {
                row.header = match->header;
                row.question = match->question;
            }
            out.answers.push_back(std::move(row));
        }
    });
    return parsed && found;
}

namespace {

/// "primary_directory" and "primaryDirectory" both read "Primary directory".
std::string humanize_label(std::string_view key) {
    std::string label;
    label.reserve(key.size() + 4);
    for (std::size_t i = 0; i < key.size(); ++i) {
        const char c = key[i];
        if (c == '_' || c == '-') {
            if (!label.empty() && label.back() != ' ') label += ' ';
            continue;
        }
        const bool upper = c >= 'A' && c <= 'Z';
        const bool after_lower = i > 0 && key[i - 1] >= 'a' && key[i - 1] <= 'z';
        if (upper && after_lower) {
            label += ' ';
            label += static_cast<char>(c - 'A' + 'a');
            continue;
        }
        label += c;
    }
    if (!label.empty() && label.front() >= 'a' && label.front() <= 'z') {
        label.front() = static_cast<char>(label.front() - 'a' + 'A');
    }
    return label;
}

constexpr int kMaxFieldDepth = 6;

class JsonFieldFlattener {
public:
    explicit JsonFieldFlattener(std::vector<JsonFieldRow>& out) : out_(out) {}

    /// Lays out `value`'s members (or items) at `depth`. A string holding
    /// JSON is opened in place, which is how MCP servers wrap their results.
    void members(simdjson::dom::element value, int depth) {
        if (const auto nested = embedded_container(value)) {
            members(*nested, depth);
            return;
        }
        simdjson::dom::object object;
        if (value.get(object) == simdjson::SUCCESS) {
            for (const auto field : object) {
                add(humanize_label(field.key), field.value, depth);
            }
            return;
        }
        simdjson::dom::array array;
        if (value.get(array) == simdjson::SUCCESS) {
            std::size_t index = 0;
            for (const auto item : array) {
                add(std::format("{}.", ++index), item, depth);
            }
        }
    }

private:
    void add(std::string label, simdjson::dom::element value, int depth) {
        if (const auto nested = embedded_container(value)) value = *nested;
        if (const auto scalar = scalar_text(value)) {
            out_.push_back({depth, std::move(label), *scalar});
            return;
        }
        if (depth + 1 >= kMaxFieldDepth) {
            out_.push_back({depth, std::move(label), simdjson::to_string(value)});
            return;
        }
        out_.push_back({depth, std::move(label), {}});
        members(value, depth + 1);
    }

    /// Scalars, empty containers, and arrays of scalars fit on one row.
    std::optional<std::string> scalar_text(simdjson::dom::element value) {
        switch (value.type()) {
            case simdjson::dom::element_type::STRING:
                return std::string(value.get_string().value_unsafe());
            case simdjson::dom::element_type::BOOL:
                return std::string(value.get_bool().value_unsafe() ? "yes" : "no");
            case simdjson::dom::element_type::NULL_VALUE:
                return std::string("—");
            case simdjson::dom::element_type::INT64:
            case simdjson::dom::element_type::UINT64:
            case simdjson::dom::element_type::DOUBLE:
            case simdjson::dom::element_type::BIGINT:
                return simdjson::to_string(value);
            case simdjson::dom::element_type::OBJECT:
                return value.get_object().value_unsafe().size() == 0
                    ? std::optional<std::string>("—") : std::nullopt;
            case simdjson::dom::element_type::ARRAY: {
                const auto array = value.get_array().value_unsafe();
                if (array.size() == 0) return std::string("—");
                std::string joined;
                for (const auto item : array) {
                    const auto type = item.type();
                    if (type == simdjson::dom::element_type::OBJECT
                        || type == simdjson::dom::element_type::ARRAY) {
                        return std::nullopt;
                    }
                    if (!joined.empty()) joined += '\n';
                    joined += scalar_text(item).value_or(std::string{});
                }
                return joined;
            }
        }
        return std::nullopt;
    }

    std::optional<simdjson::dom::element> embedded_container(simdjson::dom::element value) {
        std::string_view text;
        if (value.get(text) != simdjson::SUCCESS) return std::nullopt;
        const auto first = text.find_first_not_of(" \n\r\t");
        if (first == std::string_view::npos || (text[first] != '{' && text[first] != '[')) {
            return std::nullopt;
        }
        // Each opened string needs a parser of its own: the element points
        // into the parser's document.
        auto& parser = parsers_.emplace_back(std::make_unique<simdjson::dom::parser>());
        simdjson::dom::element nested;
        if (parser->parse(text).get(nested) != simdjson::SUCCESS) return std::nullopt;
        return nested;
    }

    std::vector<JsonFieldRow>& out_;
    std::vector<std::unique_ptr<simdjson::dom::parser>> parsers_;
};

} // namespace

bool parse_json_fields(std::string_view payload, std::vector<JsonFieldRow>& out) {
    const auto first = payload.find_first_not_of(" \n\r\t");
    if (first == std::string_view::npos || (payload[first] != '{' && payload[first] != '[')) {
        return false;
    }
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(payload).get(document) != simdjson::SUCCESS) return false;

    // A lone member wrapping everything ({"result": {...}}) adds a level of
    // indentation and nothing else.
    simdjson::dom::object object;
    if (document.get(object) == simdjson::SUCCESS && object.size() == 1) {
        const auto only = *object.begin();
        std::string_view text;
        const bool container = only.value.type() == simdjson::dom::element_type::OBJECT
            || only.value.type() == simdjson::dom::element_type::ARRAY
            || (only.value.get(text) == simdjson::SUCCESS
                && text.find_first_not_of(" \n\r\t") != std::string_view::npos
                && (text[text.find_first_not_of(" \n\r\t")] == '{'
                    || text[text.find_first_not_of(" \n\r\t")] == '['));
        if (container) document = only.value;
    }
    JsonFieldFlattener(out).members(document, 0);
    return !out.empty();
}

} // namespace tui::tool_results
