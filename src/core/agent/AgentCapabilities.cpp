#include "AgentCapabilities.hpp"

#include "AgentMode.hpp"
#include "SubagentOrchestrator.hpp"
#include "../tools/ToolManager.hpp"
#include "../tools/ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"

namespace core::agent::capabilities {

std::expected<core::verification::Receipt, std::string>
execute_verification(core::tools::ToolManager &tools,
                     VerificationRequest request) {
  if (request.recipe_id.empty()) {
    return std::unexpected("verification recipe id is empty");
  }

  core::utils::JsonWriter writer;
  {
    auto object = writer.object();
    writer.kv_str("recipe_id", request.recipe_id);
  }
  const std::string arguments = std::move(writer).take();
  const std::string tool_name(core::tools::names::kRunVerification);
  if (request.permission_check &&
      !request.permission_check(tool_name, arguments)) {
    return std::unexpected("verification permission was denied");
  }

  const std::string result = tools.execute_tool(
      tool_name, arguments,
      core::tools::ToolInvocationContext{
          .session_context = std::move(request.session_context),
          .tool_call_id = std::move(request.tool_call_id),
          .provider_name = std::move(request.provider_name),
          .model_name = std::move(request.model_name),
          .provider = std::move(request.provider),
      });
  return core::verification::parse_receipt(result);
}

std::expected<std::string, std::string>
execute_exploration(SubagentOrchestrator &orchestrator,
                    ExplorationRequest request) {
  if (request.prompt.empty()) {
    return std::unexpected("goal exploration prompt is empty");
  }
  if (!request.provider) {
    return std::unexpected("goal exploration has no active provider");
  }

  core::utils::JsonWriter writer;
  {
    auto object = writer.object();
    writer
        .kv_str("description",
                request.description.empty() ? "goal exploration"
                                            : request.description)
        .comma();
    writer.kv_str("prompt", request.prompt).comma();
    writer.kv_str("subagent_type", request.read_only ? "explore" : "general");
  }
  const std::string raw = orchestrator.execute_task(
      std::move(writer).take(), request.provider,
      SubagentOrchestrator::RunContext{
          .active_provider_name = std::move(request.provider_name),
          .active_model = std::move(request.model_name),
          .parent_mode =
              delegated_mode_name(request.parent_mode, request.read_only),
          .session_context = std::move(request.session_context),
          .permission_check = std::move(request.permission_check),
          .parent_tool_call_id = std::move(request.tool_call_id),
          .on_subagent_event = std::move(request.on_subagent_event),
          .cancellation_requested =
              std::move(request.cancellation_requested),
          .memory_system = std::move(request.memory_system),
          .effort = std::move(request.effort),
      });
  if (const auto error =
          core::utils::json::first_string_field(raw, {"error"});
      error.has_value()) {
    return std::unexpected(*error);
  }
  const std::string result = core::utils::json::string_field(raw, "result");
  if (result.empty()) {
    return std::unexpected("goal exploration returned no evidence");
  }
  return result;
}

} // namespace core::agent::capabilities
