#pragma once

#include "Tool.hpp"
#include <string>
#include <vector>
#include <functional>
#include <future>
#include <optional>

namespace core::tools {

/**
 * @brief Option for a question.
 */
struct QuestionOption {
    std::string label;
    std::string description;
};

/**
 * @brief A single question to ask the user.
 */
struct QuestionItem {
    std::string question;
    std::string header;
    std::vector<QuestionOption> options;
    bool multi_select = false;
    std::string body;
};

/**
 * @brief Request to ask the user questions interactively.
 */
struct QuestionRequest {
    std::string id;
    std::string tool_call_id;
    std::vector<QuestionItem> questions;
    
    // Promise for async result
    std::shared_ptr<std::promise<std::optional<std::vector<std::pair<std::string, std::string>>>>>
        promise;
};

/**
 * @brief Tool for asking the user structured questions.
 * 
 * This tool allows the AI to ask the user multiple-choice or free-text
 * questions during execution, similar to kimi-cli's AskUserQuestion tool.
 */
class AskUserQuestionTool : public Tool {
public:
    AskUserQuestionTool();
    ~AskUserQuestionTool() override = default;
    
    [[nodiscard]] ToolDefinition get_definition() const override;
    [[nodiscard]] std::string execute(const std::string& args_json, const core::context::SessionContext& context) override;
    
    /**
     * @brief Set the callback for when a question needs to be displayed.
     * 
     * The callback receives the question request and should display the UI.
     * The result is returned via the promise in the request.
     */
    void setQuestionCallback(std::function<void(QuestionRequest)> callback);
    
private:
    std::function<void(QuestionRequest)> questionCallback_;
};

} // namespace core::tools
