#include "TaskTool.hpp"

#include "../task/TaskService.hpp"

namespace core::tools {

ToolDefinition TaskTool::get_definition() const {
    return {
        .name = std::string(kToolName),
        .title = "Delegate Task",
        .description =
            "Manage delegated coding work. start requires title and instructions; resume requires "
            "work_id; list accepts filters. start/resume may return an MCP task handle.",
        .input_schema =
            R"({"type":"object","properties":{"action":{"type":"string","enum":["start","resume","list"]},"work_id":{"type":"string","description":"Work id for resume."},"title":{"type":"string","description":"Short label for start."},"instructions":{"type":"string","description":"Task or follow-up instructions."},"worker":{"type":"string","description":"Worker profile or list filter."},"mode":{"type":"string","description":"BUILD, PLAN, RESEARCH, DEBUG, or EXECUTE."},"provider":{"type":"string"},"model":{"type":"string"},"max_steps":{"type":"integer","minimum":1},"cwd":{"type":"string","description":"Workspace-contained project directory or list filter."},"status":{"type":"string","description":"List filter."}},"required":["action"],"additionalProperties":false})",
        .output_schema =
            R"({"type":"object","properties":{"action":{"type":"string"},"work_id":{"type":"string"},"status":{"type":"string"},"title":{"type":"string"},"worker":{"type":"string"},"mode":{"type":"string"},"provider":{"type":"string"},"model":{"type":"string"},"summary":{"type":"string"},"result":{"type":"string"},"steps":{"type":"integer"},"tool_calls":{"type":"integer"},"failed_tool_calls":{"type":"integer"},"files_touched":{"type":"array","items":{"type":"string"}},"commands_run":{"type":"array","items":{"type":"string"}},"handoff_summary":{"type":"string"},"items":{"type":"array","items":{"type":"object","properties":{"work_id":{"type":"string"},"title":{"type":"string"},"status":{"type":"string"},"worker":{"type":"string"},"provider":{"type":"string"},"model":{"type":"string"},"working_dir":{"type":"string"},"updated_at":{"type":"string"}},"required":["work_id","title","status","worker","provider","model","working_dir","updated_at"],"additionalProperties":false}}},"required":["action"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,
            .open_world_hint = true,
        },
    };
}

std::string TaskTool::execute(const std::string& json_args,
                              const core::context::SessionContext& context) {
    return core::task::TaskService::get_instance().execute(json_args, context);
}

} // namespace core::tools
