#pragma once

#include "AskUserQuestionTool.hpp"
#include "../memory/MemoryStore.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>

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
    /// Semantic store from the execution-root MemorySystem. When unset the
    /// tool uses the default on-disk path handle (same file, separate owner).
    std::optional<core::memory::MemoryStore> memory_store;
    /// Root that `read` resolves `result://` references against. It must match
    /// the root the owning Agent stores oversized tool output in; when unset
    /// both sides use ToolResultStore's default root.
    std::optional<std::filesystem::path> tool_result_root;
};

[[nodiscard]] BuiltinToolRegistrationOptions agent_builtin_tool_options();
[[nodiscard]] BuiltinToolRegistrationOptions mcp_builtin_tool_options();

void register_builtin_tools(ToolManager& tool_manager,
                            BuiltinToolRegistrationOptions options);

} // namespace core::tools
