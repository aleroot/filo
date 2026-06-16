#include "TaskTool.hpp"

#include "../task/TaskService.hpp"

namespace core::tools {

ToolDefinition TaskTool::get_definition() const {
    return {
        .name = std::string(kToolName),
        .title = "Delegate Task",
        .description =
            "Creates, resumes, or lists architect-level delegated coding work items. "
            "When the MCP Tasks extension is available, long-running start/resume calls return an MCP task handle.",
        .input_schema =
            R"({"type":"object","oneOf":[{"type":"object","properties":{"action":{"const":"start"},"title":{"type":"string","description":"Short human label for the work item."},"instructions":{"type":"string","description":"Full worker instructions for the delegated task."},"worker":{"type":"string","description":"Worker profile to use. Defaults to general."},"mode":{"type":"string","description":"Worker mode: BUILD, PLAN, RESEARCH, DEBUG, or EXECUTE."},"provider":{"type":"string","description":"Optional provider alias override."},"model":{"type":"string","description":"Optional explicit model override for the selected provider."},"max_steps":{"type":"integer","minimum":1,"description":"Optional per-run maximum number of model steps."},"cwd":{"type":"string","description":"Optional project directory for the worker. Must be inside the allowed workspace."}},"required":["action","title","instructions"],"additionalProperties":false},{"type":"object","properties":{"action":{"const":"resume"},"work_id":{"type":"string","description":"Existing delegated work item id."},"instructions":{"type":"string","description":"Optional follow-up instructions for the resumed run."},"provider":{"type":"string","description":"Optional provider alias override."},"model":{"type":"string","description":"Optional explicit model override for the selected provider."},"max_steps":{"type":"integer","minimum":1,"description":"Optional per-run maximum number of model steps."}},"required":["action","work_id"],"additionalProperties":false},{"type":"object","properties":{"action":{"const":"list"},"status":{"type":"string","description":"Optional status filter."},"worker":{"type":"string","description":"Optional worker-profile filter."},"cwd":{"type":"string","description":"Optional working-directory filter."}},"required":["action"],"additionalProperties":false}],"additionalProperties":false})",
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
