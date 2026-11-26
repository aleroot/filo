#include "AsyncUserMessages.hpp"

#include "ToolNames.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

namespace core::tools {

ToolDefinition request_user_input_async_definition() {
    return ToolDefinition{
        .name = std::string(names::kRequestUserInputAsync),
        .title = "Question",
        .description =
            "Ask the user one or more questions during ongoing work. Use this tool only to "
            "request missing information, preferences, constraints, clarification, or approval. "
            "The tool returns immediately without ending the turn or waiting for a reply; any "
            "reply arrives asynchronously as a new user message. Keep questions concise, "
            "self-contained, and easy to understand, using a level of detail appropriate to the "
            "user and task. The UI always allows a free-text answer, including when suggested "
            "options are provided. A preselected option is not submitted automatically.",
        .input_schema =
            R"({"type":"object","properties":{"questions":{"type":"array","minItems":1,)"
            R"("description":"One or more self-contained questions to present together, in display order.",)"
            R"("items":{"type":"object","properties":{)"
            R"("title":{"type":"string","description":"The complete question shown to the user, including any context needed to answer it."},)"
            R"("options":{"type":"array","minItems":1,"items":{"type":"string"},)"
            R"("description":"Suggested answers, in display order. Put the recommended answer first; the first option is preselected by default. The user can select one option or enter a free-text answer. Do not include an Other option or a free-text placeholder; the UI provides free-text input automatically. Omit options for a free-text-only question."}},)"
            R"("required":["title"],"additionalProperties":false}}},)"
            R"("required":["questions"],"additionalProperties":false})",
        .annotations = {.read_only_hint = true},
    };
}

ToolDefinition send_message_to_user_async_definition() {
    return ToolDefinition{
        .name = std::string(names::kSendMessageToUserAsync),
        .title = "Message",
        .description =
            "Send a concise message that needs the user's attention during ongoing work. The "
            "tool returns immediately without ending the turn or waiting for a reply; any reply "
            "arrives asynchronously as a new user message. Use this tool to report a critical "
            "blocker or a finding that may change the task's direction, or to answer a user "
            "question or status request received while work is still in progress. Use this tool "
            "when a message needs the user's immediate attention; use commentary for routine "
            "progress and intermediate context. Use clear formatting, such as bolding questions, "
            "to make requests easy to notice and answer.",
        .parameters = {
            {"message", "string", "The concise question or update to send to the user.", true},
        },
        .annotations = {.read_only_hint = true},
    };
}

bool is_async_user_message_tool(std::string_view tool_name) noexcept {
    return tool_name == names::kRequestUserInputAsync
        || tool_name == names::kSendMessageToUserAsync;
}

std::expected<std::string, std::string> async_user_message(std::string_view tool_name,
                                                           std::string_view arguments) {
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(arguments);
    simdjson::dom::object object;
    if (parser.parse(padded).get(object) != simdjson::SUCCESS) {
        return std::unexpected("arguments must be a JSON object");
    }

    if (tool_name == names::kSendMessageToUserAsync) {
        std::string_view message;
        std::ignore = object["message"].get(message);
        message = core::utils::str::trim_ascii_view(message);
        if (message.empty()) return std::unexpected("message must not be empty");
        return std::string(message);
    }

    // codex-rs request_user_input_async.rs: each question is its title, then
    // one "- option" line per suggested answer; questions are blank-line apart.
    simdjson::dom::array questions;
    if (object["questions"].get(questions) != simdjson::SUCCESS || questions.size() == 0) {
        return std::unexpected("questions must not be empty");
    }
    std::string text;
    for (simdjson::dom::element question : questions) {
        std::string_view title;
        std::ignore = question["title"].get(title);
        if (core::utils::str::trim_ascii_view(title).empty()) {
            return std::unexpected("question titles must not be empty");
        }
        if (!text.empty()) text += "\n\n";
        text += title;
        simdjson::dom::array options;
        if (question["options"].get(options) != simdjson::SUCCESS) continue;
        if (options.size() == 0) {
            return std::unexpected("options must contain at least one non-empty answer");
        }
        for (simdjson::dom::element option : options) {
            std::string_view answer;
            if (option.get(answer) != simdjson::SUCCESS
                || core::utils::str::trim_ascii_view(answer).empty()) {
                return std::unexpected("options must contain at least one non-empty answer");
            }
            text += "\n- ";
            text += answer;
        }
    }
    return text;
}

} // namespace core::tools
