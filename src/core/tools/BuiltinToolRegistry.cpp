#include "BuiltinToolRegistry.hpp"

#include "ActivateSkillTool.hpp"
#include "ApplyPatchTool.hpp"
#include "CreateDirectoryTool.hpp"
#include "DeleteFileTool.hpp"
#include "FileSearchTool.hpp"
#include "GetTimeTool.hpp"
#include "GetWorkspaceConfigTool.hpp"
#include "GrepSearchTool.hpp"
#include "ListDirectoryTool.hpp"
#include "MoveFileTool.hpp"
#include "ReadFileTool.hpp"
#include "ReplaceTool.hpp"
#include "SearchReplaceTool.hpp"
#include "ShellTool.hpp"
#include "SkillRegistry.hpp"
#include "TaskTool.hpp"
#include "ToolManager.hpp"
#include "WebFetchTool.hpp"
#include "WebSearchTool.hpp"
#include "WriteFileTool.hpp"

#ifdef FILO_ENABLE_PYTHON
#include "PythonInterpreterTool.hpp"
#include "SkillLoader.hpp"
#endif

namespace core::tools {

BuiltinToolRegistrationOptions agent_builtin_tool_options() {
    return {
        .include_get_time = true,
        .include_workspace_config = false,
        .include_delegate_task = false,
        .include_ask_user_question = true,
        .include_python = true,
        .include_instruction_skills = true,
        .discover_python_skills = true,
    };
}

BuiltinToolRegistrationOptions mcp_builtin_tool_options() {
    return {
        .include_get_time = false,
        .include_workspace_config = true,
        .include_delegate_task = true,
        .include_ask_user_question = false,
        .include_python = false,
        .include_instruction_skills = true,
        .discover_python_skills = false,
    };
}

void register_builtin_tools(ToolManager& tool_manager,
                            BuiltinToolRegistrationOptions options) {
    if (options.include_get_time) {
        tool_manager.register_tool(std::make_shared<GetTimeTool>());
    }

    tool_manager.register_tool(std::make_shared<ShellTool>());
    tool_manager.register_tool(std::make_shared<ApplyPatchTool>());
    tool_manager.register_tool(std::make_shared<FileSearchTool>());
    tool_manager.register_tool(std::make_shared<ReadFileTool>());
    tool_manager.register_tool(std::make_shared<WriteFileTool>());
    tool_manager.register_tool(std::make_shared<ListDirectoryTool>());
    tool_manager.register_tool(std::make_shared<ReplaceTool>());
    tool_manager.register_tool(std::make_shared<GrepSearchTool>());
    tool_manager.register_tool(std::make_shared<SearchReplaceTool>());
    tool_manager.register_tool(std::make_shared<DeleteFileTool>());
    tool_manager.register_tool(std::make_shared<MoveFileTool>());
    tool_manager.register_tool(std::make_shared<CreateDirectoryTool>());
    tool_manager.register_tool(std::make_shared<WebSearchTool>());
    tool_manager.register_tool(std::make_shared<WebFetchTool>());

    if (options.include_workspace_config) {
        tool_manager.register_tool(std::make_shared<GetWorkspaceConfigTool>());
    }
    if (options.include_delegate_task) {
        tool_manager.register_tool(std::make_shared<TaskTool>());
    }
    if (options.include_ask_user_question) {
        auto ask_user_tool = std::make_shared<AskUserQuestionTool>();
        if (options.ask_user_question_callback) {
            ask_user_tool->setQuestionCallback(
                std::move(options.ask_user_question_callback));
        }
        if (options.ask_user_question_tool_out != nullptr) {
            *options.ask_user_question_tool_out = ask_user_tool;
        }
        tool_manager.register_tool(std::move(ask_user_tool));
    }

    if (options.include_instruction_skills
        && !SkillRegistry::discover_instruction_skills().empty()) {
        tool_manager.register_tool(std::make_shared<ActivateSkillTool>());
    }

#ifdef FILO_ENABLE_PYTHON
    if (options.include_python) {
        tool_manager.register_tool(std::make_shared<PythonInterpreterTool>());
    }
    if (options.discover_python_skills) {
        SkillLoader::discover_and_register(tool_manager);
    }
#else
    (void)options;
#endif
}

} // namespace core::tools
