#include "TodoTool.hpp"

#include "ToolNames.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

namespace core::tools {
namespace {

[[nodiscard]] std::string todos_json(
    const std::vector<core::session::SessionTodoItem>& todos) {
    core::utils::JsonWriter writer(128 + todos.size() * 128);
    {
        auto object = writer.object();
        writer.kv_bool("ok", true).comma().key("todos");
        {
            auto array = writer.array();
            for (std::size_t i = 0; i < todos.size(); ++i) {
                if (i != 0) writer.comma();
                auto item = writer.object();
                writer.kv_str("id", todos[i].id).comma()
                      .kv_str("content", todos[i].text).comma()
                      .kv_str("status", core::session::to_string(todos[i].status));
            }
        }
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string error_json(std::string_view message) {
    core::utils::JsonWriter writer(128 + message.size());
    {
        auto object = writer.object();
        writer.kv_str("error", message);
    }
    return std::move(writer).take();
}

} // namespace

TodoTool::TodoTool(core::session::TodoManager& manager) noexcept
    : manager_(manager) {}

ToolDefinition TodoTool::get_definition() const {
    return {
        .name = std::string(names::kWriteTodos),
        .title = "Write Todos",
        .description =
            "Replaces the current session task plan. Use for complex multi-step work, keep "
            "exactly one item in_progress while working, and mark items completed immediately. "
            "Do not create a plan for a simple one-step request. Preserve existing ids when updating.",
        .input_schema =
            R"({"type":"object","properties":{"todos":{"type":"array","maxItems":100,"items":{"type":"object","properties":{"id":{"type":"string","description":"Existing stable id; omit for a new todo."},"content":{"type":"string","minLength":1,"description":"Concise actionable step."},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["content","status"],"additionalProperties":false}}},"required":["todos"],"additionalProperties":false})",
        .output_schema =
            R"({"type":"object","properties":{"ok":{"type":"boolean"},"todos":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed"]}},"required":["id","content","status"],"additionalProperties":false}}},"required":["ok","todos"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint = false,
            .destructive_hint = false,
            .idempotent_hint = true,
            .open_world_hint = false,
        },
    };
}

std::string TodoTool::execute(const std::string& json_args,
                              const core::context::SessionContext&) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(json_args).get(document) != simdjson::SUCCESS) {
        return error_json("Invalid JSON arguments for write_todos.");
    }
    simdjson::dom::array input;
    if (document["todos"].get(input) != simdjson::SUCCESS) {
        return error_json("write_todos requires a todos array.");
    }

    std::vector<core::session::TodoDraft> drafts;
    drafts.reserve(input.size());
    for (simdjson::dom::element element : input) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) {
            return error_json("Each todo must be an object.");
        }
        std::string_view id;
        std::string_view content;
        std::string_view status;
        (void)object["id"].get(id);
        if (object["content"].get(content) != simdjson::SUCCESS
            || object["status"].get(status) != simdjson::SUCCESS) {
            return error_json("Each todo requires string content and status.");
        }
        if (status != "pending" && status != "in_progress" && status != "completed") {
            return error_json("Todo status must be pending, in_progress, or completed.");
        }
        drafts.push_back({
            .id = std::string(id),
            .text = std::string(content),
            .status = core::session::todo_status_from_string(status),
        });
    }

    auto result = manager_.replace(std::move(drafts));
    return result.has_value() ? todos_json(*result) : error_json(result.error());
}

} // namespace core::tools
