#include "ToolManager.hpp"
#include "shell/ShellSessions.hpp"
#include "ToolArgumentUtils.hpp"
#include "../changes/TurnChangeTracker.hpp"
#include "../changes/WorkspaceMutationGate.hpp"
#include "../policy/PolicySnapshot.hpp"

namespace core::tools {
ToolManager::ToolManager(std::shared_ptr<core::changes::WorkspaceMutationGate> mutations,
                         std::shared_ptr<shell::ShellSessions> shells,
                         std::shared_ptr<core::changes::MutationObservations> observations)
    : mutations_(mutations ? std::move(mutations) : std::make_shared<core::changes::WorkspaceMutationGate>()),
      shells_(shells ? std::move(shells) : std::make_shared<shell::ShellSessions>()),
      observations_(observations ? std::move(observations) : std::make_shared<core::changes::MutationObservations>()) {}

std::string ToolManager::execute_tool(
    const std::string& name,
    std::string_view json_args,
    const core::context::SessionContext& context)
{
    return execute_tool(name, json_args, ToolInvocationContext{
        .session_context = context,
    });
}

std::string ToolManager::execute_tool(
    const std::string& name,
    std::string_view json_args,
    const ToolInvocationContext& incoming)
{
    std::shared_ptr<Tool> tool;
    std::shared_ptr<const core::policy::PolicySnapshot> mandatory_policy;
    ToolDefinition definition;
    {
        std::lock_guard lock(mutex_);
        const auto* registered = find_tool_unlocked(name);
        if (!registered)
            return "{\"error\": \"Tool not found: " + name + "\"}";
        tool = registered->tool;
        definition = tool->get_definition();
    }
    // The policy the process runs under is the one every tool must obey.
    mandatory_policy = policy_snapshot();
    if (mandatory_policy && incoming.policy
        && mandatory_policy->identity() != incoming.policy->identity()) {
        return R"({"error":"Network policy: conflicting policy snapshots cannot be combined safely."})";
    }
    std::optional<ToolInvocationContext> enforced;
    if (mandatory_policy && !incoming.policy) {
        enforced = incoming;
        enforced->policy = mandatory_policy;
    }
    const auto& invocation = enforced ? *enforced : incoming;
    if (!tool_allows_transport(definition, invocation.session_context.transport)) {
        return R"({"error":"Tool is available only to Filo's internal agents."})";
    }
    if (invocation.policy && definition.network_mediation == NetworkMediation::unmediated) {
        // The tool can reach the network in a way the policy cannot
        // intercept. Refusing it is the only honest answer: allowing it
        // would claim a protection that does not exist.
        return R"({"error":"Network policy: this tool reaches the network without mediation and is unavailable while a policy is active."})";
    }
    const auto sandbox_mode =
        core::landrun::LandrunSettings::instance().mode();
    if (!landrun_allows_tool(sandbox_mode, definition.name)) {
        return "{\"error\":\"Tool "
            + core::utils::escape_json_string(name)
            + " is unavailable while sandbox mode is "
            + std::string(core::landrun::landrun_mode_name(sandbox_mode))
            + ".\"}";
    }
    const auto normalized = schema::normalize_arguments(definition, json_args);
    if (!normalized.has_value()) {
        return "{\"error\":\"Invalid arguments for "
            + core::utils::escape_json_string(name)
            + ": "
            + core::utils::escape_json_string(normalized.error())
            + "\"}";
    }
    try {
        if (invocation.cancellation_requested && invocation.cancellation_requested())
            return R"({"error":"Tool execution cancelled."})";
        const auto scope = core::changes::mutation_scope(
            definition.name, *normalized, invocation.session_context, definition.annotations);
        for (const auto& path : scope.paths) {
            if (const auto denied = detail::check_workspace_access(
                    path, path.string(), invocation.session_context, nullptr, definition.name)) return *denied;
        }
        std::shared_ptr<core::changes::WorkspaceMutationGate::Lease> lease;
        if (!scope.paths.empty()) {
            // Reserve declared mutations, never permission grants. Arbitrary
            // commands have no declared targets and cannot lock a workspace.
            auto acquired = mutations_->acquire(
                scope.paths,
                std::chrono::seconds(30), invocation.cancellation_requested);
            if (!acquired) return std::string("{\"error\":\"")
                + core::changes::WorkspaceMutationGate::failure_message(acquired.error()) + "\"}";
            lease = std::move(*acquired);
        }
        // A question the policy asks during this call belongs to the chat
        // that made it, and is withdrawn when that chat stops.
        std::optional<core::policy::ApprovalOriginScope> origin;
        if (invocation.policy) {
            origin.emplace(std::string(invocation.session_context.root_session()),
                           invocation.cancellation_requested);
        }
        const auto execute = [&] { return tool->execute(*normalized, invocation); };
        // MCP and other callers without a turn tracker still participate in
        // attribution without reserving any additional execution resources.
        const auto observation = invocation.change_tracker ? nullptr : observations_->observe({},
            scope.unbounded ? invocation.session_context.workspace_view().ordered_roots() : scope.paths);
        return invocation.change_tracker ? invocation.change_tracker->track(scope, execute)
                                         : execute();
    } catch (const std::exception& e) {
        return "{\"error\": \"Exception executing tool: " + core::utils::escape_json_string(e.what()) + "\"}";
    }
}

} // namespace core::tools
