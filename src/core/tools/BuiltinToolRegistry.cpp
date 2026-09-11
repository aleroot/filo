#include "BuiltinToolRegistry.hpp"

#include "../landrun/LandrunSettings.hpp"
#include "../logging/Logger.hpp"
#include "../workspace/Workspace.hpp"
#include "ActivateSkillTool.hpp"
#include "ApplyPatchTool.hpp"
#include "CreateDirectoryTool.hpp"
#include "DeleteFileTool.hpp"
#include "FileSearchTool.hpp"
#include "GetTimeTool.hpp"
#include "GetWorkspaceConfigTool.hpp"
#include "GrepSearchTool.hpp"
#include "ListDirectoryTool.hpp"
#include "MemoryTool.hpp"
#include "MoveFileTool.hpp"
#include "PathVisibilityToolDecorator.hpp"
#include "ReadTool.hpp"
#include "ReplaceTool.hpp"
#include "SearchReplaceTool.hpp"
#include "ShellTool.hpp"
#include "SkillRegistry.hpp"
#include "TaskTool.hpp"
#include "ToolManager.hpp"
#include "VerificationTool.hpp"
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
    tool_manager.register_tool(std::make_shared<VerificationTool>());
    tool_manager.register_tool(with_path_visibility(std::make_shared<ApplyPatchTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<FileSearchTool>()));
    tool_manager.register_tool(
        with_path_visibility(std::make_shared<ReadTool>(
            read::ResourceReader(options.tool_result_root.has_value()
                ? core::agent::ToolResultStore(*options.tool_result_root)
                : core::agent::ToolResultStore{}),
            read::ReaderWorker{})));
    tool_manager.register_tool(
        with_path_visibility(std::make_shared<WriteFileTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<ListDirectoryTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<ReplaceTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<GrepSearchTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<SearchReplaceTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<DeleteFileTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<MoveFileTool>()));
    tool_manager.register_tool(with_path_visibility(std::make_shared<CreateDirectoryTool>()));
    tool_manager.register_tool(std::make_shared<WebSearchTool>());
    tool_manager.register_tool(std::make_shared<WebFetchTool>());
    if (!core::landrun::LandrunSettings::instance().enabled()) {
        tool_manager.register_tool(std::make_shared<MemoryTool>(
            options.memory_store.value_or(core::memory::MemoryStore{})));
    }

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

    // Skill discovery is workspace-scoped: every root granted with -w may
    // contribute instruction and prompt skills, while executable Python skills
    // stay limited to the user's own directories and the primary (enforced by
    // SkillSearchRoot::permits_tool_skills). Roots come from the caller, or from
    // the process-wide workspace the composition root initialized — never from
    // the ambient cwd, which only coincided with the primary by accident.
    const auto skill_roots = options.workspace_roots.empty()
        ? core::workspace::Workspace::get_instance().ordered_roots()
        : options.workspace_roots;

    if (options.include_instruction_skills
        && !SkillRegistry::discover_instruction_skills(skill_roots).empty()) {
        tool_manager.register_tool(std::make_shared<ActivateSkillTool>(skill_roots));
    }

#ifdef FILO_ENABLE_PYTHON
    const bool secure_mode = !core::landrun::LandrunSettings::instance().permits(
        core::landrun::LandrunCapability::in_process_untrusted_code);
    if (options.include_python && !secure_mode) {
        tool_manager.register_tool(std::make_shared<PythonInterpreterTool>());
    }
    if (options.discover_python_skills && !secure_mode) {
        SkillLoader::discover_and_register(tool_manager, skill_roots);
    }
    if (secure_mode && (options.include_python || options.discover_python_skills)) {
        core::logging::warn(
            "landrun secure mode: embedded Python and executable Python skills are disabled");
    }
#else
    (void)options;
#endif
}

} // namespace core::tools
