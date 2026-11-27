#include "OpenDocumentTool.hpp"

#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <cstdint>
#include <format>
#include <limits>
#include <string_view>
#include <system_error>
#include <utility>

namespace core::tools {

namespace {

[[nodiscard]] std::string error_json(std::string_view message) {
    return std::format(R"({{"error":"{}"}})", core::utils::escape_json_string(message));
}

[[nodiscard]] std::string_view display_name(DocumentDisplay display) noexcept {
    switch (display) {
        case DocumentDisplay::shown:       return "shown";
        case DocumentDisplay::behind:      return "behind";
        case DocumentDisplay::hidden:      return "hidden";
        case DocumentDisplay::elsewhere:   return "elsewhere";
        case DocumentDisplay::unavailable: return "unavailable";
    }
    return "unavailable";
}

// Written for the model: what the user can see now, and what to do when it
// is nothing.
[[nodiscard]] std::string_view display_note(DocumentDisplay display) noexcept {
    switch (display) {
        case DocumentDisplay::shown:
            return "The file is open beside the conversation.";
        case DocumentDisplay::behind:
            return "The user is reading another file in the pane; this one was added as a tab.";
        case DocumentDisplay::hidden:
            return "The terminal is too narrow to show it beside the conversation; it was "
                   "added as a tab the user can open with F6.";
        case DocumentDisplay::elsewhere:
            return "The user is viewing another conversation, so nothing was opened. "
                   "Link the file in your reply instead.";
        case DocumentDisplay::unavailable:
            return "The screen is not available. Link the file in your reply instead.";
    }
    return {};
}

} // namespace

ToolDefinition OpenDocumentTool::get_definition() const {
    return {
        .name = std::string(names::kOpenDocument),
        .title = "Open",
        .description =
            "Show a file to the user in Filo's document pane, beside the conversation, "
            "optionally scrolled to a line. Use it when the user should look at a file "
            "itself: a plan, report or document you wrote, or the code an answer is "
            "about. It returns no content; use read to see a file. It never takes the "
            "keyboard from the user.",
        .parameters = {
            {"path", "string", "File path, absolute or relative to the workspace.", true},
            {
                .name = "line",
                .type = "integer",
                .schema = R"({"type":"integer","minimum":1,"description":"One-based line to show first."})",
            },
        },
        .output_schema =
            R"({"type":"object","properties":{"path":{"type":"string","description":"The file, absolute."},"line":{"type":"integer"},"display":{"type":"string","enum":["shown","behind","hidden","elsewhere","unavailable"],"description":"What the user's screen did with it."},"note":{"type":"string"}},"required":["path","display","note"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint = true,  // changes what the screen shows, nothing else
            .idempotent_hint = true,
        },
        .agent_only = true,
        .primary_argument = "path",
    };
}

void OpenDocumentTool::set_opener(Opener opener) {
    std::lock_guard lock(mutex_);
    opener_ = std::move(opener);
}

std::string OpenDocumentTool::execute(const std::string& json_args,
                                      const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return error_json("Invalid JSON arguments for open_document.");
    }
    if (auto invalid = detail::validate_object_arguments(doc, names::kOpenDocument,
                                                         {"path", "line"})) {
        return *invalid;
    }

    std::string_view path_v;
    if (doc["path"].get(path_v) != simdjson::SUCCESS || path_v.empty()) {
        return error_json("Missing required argument 'path'.");
    }
    std::optional<int> line;
    if (auto field = doc["line"]; field.error() == simdjson::SUCCESS && !field.is_null()) {
        std::int64_t value = 0;
        if (field.get(value) != simdjson::SUCCESS || value < 1
            || value > std::numeric_limits<int>::max()) {
            return error_json("'line' must be a positive integer.");
        }
        line = static_cast<int>(value);
    }

    const std::string path_str(path_v);
    std::filesystem::path resolved;
    if (auto denied = detail::check_workspace_access(
            std::filesystem::path(path_str), path_str, context, &resolved,
            names::kOpenDocument)) {
        return *denied;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(resolved, ec)) {
        return error_json(std::filesystem::exists(resolved, ec)
            ? std::format("'{}' is not a file.", path_str)
            : std::format("'{}' does not exist.", path_str));
    }

    Opener opener;
    {
        std::lock_guard lock(mutex_);
        opener = opener_;
    }
    if (!opener) {
        return error_json("This session has no document pane.");
    }
    const auto display = opener(DocumentRequest{
        .path = resolved,
        .line = line,
        .session_id = context.session_id,
    });

    core::utils::JsonWriter w(256 + resolved.native().size());
    {
        auto _obj = w.object();
        w.kv_str("path", resolved.string());
        if (line) w.comma().kv_num("line", *line);
        w.comma().kv_str("display", display_name(display));
        w.comma().kv_str("note", display_note(display));
    }
    return std::move(w).take();
}

} // namespace core::tools
