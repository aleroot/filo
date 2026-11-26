#include "ScriptToolHost.hpp"

#include "ToolCallPlanner.hpp"
#include "ToolCallScheduler.hpp"
#include "../tools/ToolSchema.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/Uuid.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <utility>

namespace core::agent {

struct ScriptToolHost::Admitted {
    core::llm::ToolCall call;
    ScheduledToolTask<core::llm::Message> task;
};

ScriptToolHost::ScriptToolHost(NestedToolServices services)
    : services_(std::move(services)) {}

bool ScriptToolHost::cancelled() const {
    return services_.cancelled && services_.cancelled();
}

std::vector<core::tools::ToolScriptReply>
ScriptToolHost::run(std::span<const core::tools::ToolScriptCall> calls) {
    std::vector<core::tools::ToolScriptReply> replies(calls.size());
    std::vector<ScheduledToolTask<core::llm::Message>> tasks;
    std::vector<core::llm::ToolCall> task_calls;
    std::vector<std::size_t> task_replies;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        auto admitted = admit(calls[i], replies[i]);
        if (!admitted) continue;
        task_calls.push_back(std::move(admitted->call));
        task_replies.push_back(i);
        tasks.push_back(std::move(admitted->task));
    }

    ToolCallScheduler<core::llm::Message> scheduler;
    auto messages = scheduler.run(
        std::move(tasks),
        [&](std::size_t index, const core::llm::Message& message) {
            if (services_.on_finish) services_.on_finish(task_calls[index], message);
        });
    for (std::size_t t = 0; t < messages.size(); ++t) {
        replies[task_replies[t]] = {.ran = true, .value = std::move(messages[t].content)};
    }
    return replies;
}

core::tools::ToolScriptReply ScriptToolHost::refuse(const core::llm::ToolCall& call,
                                                    std::string reason) const {
    // Closes the transcript row the call opened.
    if (services_.on_finish) {
        services_.on_finish(call, core::llm::Message{
            .role = "tool",
            .content = std::format(R"({{"error":"{}"}})", core::utils::escape_json_string(reason)),
            .name = call.function.name,
            .tool_call_id = call.id,
        });
    }
    return {.ran = false, .value = std::move(reason)};
}

std::optional<ScriptToolHost::Admitted>
ScriptToolHost::admit(const core::tools::ToolScriptCall& requested,
                      core::tools::ToolScriptReply& refusal) {
    const auto refuse_unstarted = [&](std::string reason) {
        refusal = {.ran = false, .value = std::move(reason)};
        return std::nullopt;
    };
    if (cancelled()) return refuse_unstarted("the turn was stopped");

    // A script reaches run_terminal_command as Codex's exec_command and
    // write_stdin. The call is checked against the contract the script sees,
    // then becomes the Filo tool's, so every gate and row treats it as one.
    std::optional<core::tools::ScriptToolAlias> alias;
    std::string requested_arguments = requested.arguments;
    if (const auto alias_definition = core::tools::script_alias_definition(requested.tool_name)) {
        auto validated = core::tools::schema::validate_arguments(
            *alias_definition,
            core::tools::nested_tool_arguments(*alias_definition, requested.arguments));
        if (!validated) {
            return refuse_unstarted(std::format("invalid arguments for {}: {}",
                                                requested.tool_name, validated.error().message));
        }
        alias = core::tools::resolve_script_alias(
            requested.tool_name, *validated,
            services_.session_context.workspace_view().primary().string());
        requested_arguments = alias->arguments;
    }
    const std::string& name = alias ? alias->tool_name : requested.tool_name;
    const bool offered = std::ranges::any_of(services_.offered, [&](const core::llm::Tool& tool) {
        return tool.function.name == name;
    });
    if (!offered) return refuse_unstarted(requested.tool_name + " is not available in this step");
    auto definition = services_.resolve(name);
    if (!definition) return refuse_unstarted(requested.tool_name + " is not available");
    // A translated call carries fields only the alias declares.
    std::string arguments = std::move(requested_arguments);
    if (!alias) {
        auto validated = core::tools::schema::validate_arguments(
            *definition, core::tools::nested_tool_arguments(*definition, std::move(arguments)));
        if (!validated) {
            return refuse_unstarted(
                std::format("invalid arguments for {}: {}", name, validated.error().message));
        }
        arguments = std::move(*validated);
    }

    core::llm::ToolCall call;
    call.id = "call_" + core::utils::random_uuid_v4();
    call.function.name = name;
    call.function.arguments = std::move(arguments);
    if (services_.on_start) services_.on_start(call);

    if (denied_) {
        refusal = refuse(call, "skipped after an earlier denial in this script");
        return std::nullopt;
    }
    const ToolApproval approval = alias && alias->pre_approved
        ? ToolApproval{.approved = true}
        : services_.approve(call);
    if (approval.hook_block) {
        denied_ = true;
        refusal = refuse(call, approval.hook_block->empty()
            ? std::string("blocked by PreToolUse hook")
            : "blocked by PreToolUse hook: " + *approval.hook_block);
        return std::nullopt;
    }
    if (!approval.approved) {
        denied_ = true;
        refusal = refuse(call, "denied by the user");
        return std::nullopt;
    }

    auto planned = plan_tool_call(call, services_.session_context);
    if (approval.read_only_task) planned.accesses = read_all_tool_access();
    return Admitted{
        .call = call,
        .task = {
            .accesses = std::move(planned.accesses),
            .run = [this, call, definition = std::move(definition), alias] {
                const auto started = std::chrono::steady_clock::now();
                auto message = services_.execute(call, definition);
                if (alias) {
                    const std::chrono::duration<double> wall =
                        std::chrono::steady_clock::now() - started;
                    message.content = core::tools::script_alias_result(
                        *alias, message.content, wall.count());
                }
                return message;
            },
        },
    };
}

} // namespace core::agent
