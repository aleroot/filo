#include "AskUserQuestionTool.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <format>
#include <random>
#include <sstream>

namespace core::tools {

namespace {

std::string generateUuid() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    const char* hex = "0123456789abcdef";
    std::string uuid;
    uuid.reserve(36);
    
    for (int i = 0; i < 8; ++i) uuid += hex[dis(gen)];
    uuid += '-';
    for (int i = 0; i < 4; ++i) uuid += hex[dis(gen)];
    uuid += '-';
    for (int i = 0; i < 4; ++i) uuid += hex[dis(gen)];
    uuid += '-';
    for (int i = 0; i < 4; ++i) uuid += hex[dis(gen)];
    uuid += '-';
    for (int i = 0; i < 12; ++i) uuid += hex[dis(gen)];
    
    return uuid;
}

} // namespace

AskUserQuestionTool::AskUserQuestionTool() = default;

ToolDefinition AskUserQuestionTool::get_definition() const {
    ToolDefinition def;
    def.name = std::string(names::kAskUserQuestion);
    def.title = "Ask User Question";
    def.description = R"(Ask the user structured question(s) to guide the conversation.

Use this tool when you need user input to make a decision, clarify requirements, or choose between options. The user will see a dialog with your question and the provided options.

Guidelines:
- Ask specific, actionable questions
- Provide 2-4 meaningful, distinct options
- Add brief descriptions to explain trade-offs
- Use multi_select only when multiple answers are valid
- Keep header tags short (max 12 chars, e.g. 'Auth', 'Style'))";

    def.parameters = {
        {"questions", "array",
         "The questions to ask the user (1-4 questions).", true},
    };
    
    // Note: The items_schema for the questions parameter would be set here
    // but for simplicity we're using a simpler definition

    return def;
}

std::string AskUserQuestionTool::execute(
    const std::string& args_json,
    [[maybe_unused]] const core::context::SessionContext& context) {
    if (!questionCallback_) {
        return R"({"error":"AskUserQuestion tool not initialized with a UI callback."})";
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    
    if (parser.parse(args_json).get(document) != simdjson::SUCCESS) {
        return R"({"error":"Failed to parse tool arguments as JSON."})";
    }

    simdjson::dom::array questions_array;
    if (document["questions"].get(questions_array) != simdjson::SUCCESS) {
        return R"({"error":"Missing or invalid 'questions' parameter."})";
    }

    QuestionRequest request;
    request.id = generateUuid();
    request.promise = std::make_shared<std::promise<std::optional<std::vector<std::pair<std::string, std::string>>>>>();

    for (auto question_elem : questions_array) {
        QuestionItem item;
        
        std::string_view question_text;
        if (question_elem["question"].get(question_text) == simdjson::SUCCESS) {
            item.question = std::string(question_text);
        }
        
        std::string_view header;
        if (question_elem["header"].get(header) == simdjson::SUCCESS) {
            item.header = std::string(header);
        }
        
        bool multi_select = false;
        if (question_elem["multi_select"].get(multi_select) == simdjson::SUCCESS) {
            item.multi_select = multi_select;
        }
        
        std::string_view body;
        if (question_elem["body"].get(body) == simdjson::SUCCESS) {
            item.body = std::string(body);
        }

        simdjson::dom::array options_array;
        if (question_elem["options"].get(options_array) == simdjson::SUCCESS) {
            for (auto option_elem : options_array) {
                QuestionOption opt;
                
                std::string_view label;
                if (option_elem["label"].get(label) == simdjson::SUCCESS) {
                    opt.label = std::string(label);
                }
                
                std::string_view description;
                if (option_elem["description"].get(description) == simdjson::SUCCESS) {
                    opt.description = std::string(description);
                }
                
                if (!opt.label.empty()) {
                    item.options.push_back(std::move(opt));
                }
            }
        }

        // Add synthetic "Other" option
        QuestionOption other_opt;
        other_opt.label = "Other";
        other_opt.description = "";
        item.options.push_back(std::move(other_opt));

        if (!item.question.empty() && item.options.size() >= 2) {
            request.questions.push_back(std::move(item));
        }
    }

    if (request.questions.empty()) {
        return R"({"error":"No valid questions provided. Each question needs 'question' text and at least 2 'options'."})";
    }

    // Send the question to the UI
    questionCallback_(request);

    // Wait for the answer
    auto future = request.promise->get_future();
    auto answers = future.get();

    if (!answers.has_value()) {
        // User dismissed/cancelled
        return R"({"answers": {}, "note": "User dismissed the question without answering."})";
    }

    // Format the answers as JSON
    std::ostringstream json;
    json << "{\"answers\": {";
    bool first = true;
    for (const auto& [question, answer] : *answers) {
        if (!first) json << ", ";
        first = false;
        json << "\"" << core::utils::escape_json_string(question) << "\": \""
             << core::utils::escape_json_string(answer) << "\"";
    }
    json << "}}";

    return json.str();
}

void AskUserQuestionTool::setQuestionCallback(std::function<void(QuestionRequest)> callback) {
    questionCallback_ = std::move(callback);
}

} // namespace core::tools
