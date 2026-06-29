#pragma once

#include "AskUserQuestionTool.hpp"

#include <functional>
#include <memory>

namespace core::tools {

class ToolManager;

struct BuiltinToolRegistrationOptions {
    bool include_get_time = false;
    bool include_workspace_config = false;
    bool include_delegate_task = false;
    bool include_ask_user_question = false;
    bool include_python = false;
    bool include_instruction_skills = true;
    bool discover_python_skills = false;

    std::function<void(QuestionRequest)> ask_user_question_callback = {};
    std::shared_ptr<AskUserQuestionTool>* ask_user_question_tool_out = nullptr;
};

[[nodiscard]] BuiltinToolRegistrationOptions agent_builtin_tool_options();
[[nodiscard]] BuiltinToolRegistrationOptions mcp_builtin_tool_options();

void register_builtin_tools(ToolManager& tool_manager,
                            BuiltinToolRegistrationOptions options);

} // namespace core::tools
