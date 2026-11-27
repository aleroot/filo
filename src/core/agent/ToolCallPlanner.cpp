#include "ToolCallPlanner.hpp"

#include "SubagentOrchestrator.hpp"
#include "../permissions/CommandPolicy.hpp"
#include "../tools/ToolNames.hpp"
#include "../tools/read/ReadTypes.hpp"
#include "../tools/read/SerializedPathRecovery.hpp"
#include "../utils/JsonUtils.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace core::agent {

namespace {

[[nodiscard]] std::filesystem::path resolved_arg_path(
    const core::context::SessionContext& context,
    std::string_view raw_args,
    std::string_view field,
    std::filesystem::path fallback = ".") {
    const auto value = core::utils::json::first_string_field(raw_args, {field});
    return context.resolve_path(value && !value->empty()
        ? std::filesystem::path(*value)
        : std::move(fallback));
}

[[nodiscard]] ToolAccessSet single_file_access(ToolFileOperation operation,
                                               const core::context::SessionContext& context,
                                               std::string_view raw_args,
                                               std::string_view field,
                                               bool recursive = false) {
    return {
        ToolAccess::file_access(
            operation,
            resolved_arg_path(context, raw_args, field),
            recursive),
    };
}

/// A command started as its own process (an exec script's exec_command: it
/// carries a yield time) that the built-in command rules let run without
/// asking, which makes it one that changes nothing. Such commands run beside
/// reads and each other, as Codex runs exec_command calls in parallel. The
/// user's rules layer is left out: it allows what the user trusts, builds
/// included, not only what writes nothing. A command for the persistent
/// shell stays exclusive: a `cd` there changes what the next command sees.
[[nodiscard]] bool runs_read_only_command_process(std::string_view raw_args) {
    using core::permissions::CommandPolicy;
    static const CommandPolicy built_in =
        CommandPolicy{}.with_layer(core::permissions::default_command_rules())
            .value_or(CommandPolicy{});
    if (core::utils::json::first_int64_field(raw_args, {"process_id"})
        || !core::utils::json::first_int64_field(raw_args, {"yield_time_ms"})) {
        return false;
    }
    const auto command = core::permissions::shell_command_argument(raw_args);
    return command
        && built_in.decide(*command) == core::permissions::CommandDecision::Allow;
}

} // namespace

PlannedToolCall plan_tool_call(const core::llm::ToolCall& call,
                               const core::context::SessionContext& context) {
    using namespace core::tools::names;

    const std::string_view name = call.function.name;
    const std::string_view args = call.function.arguments;
    ToolAccessSet accesses;

    if (name == kGetCurrentTime || name == kWriteTodos || name == kReadToolResult
        || name == kConversation || name == kRequestUserInputAsync
        || name == kSendMessageToUserAsync) {
        accesses = no_tool_access();
    } else if (name == kAskUserQuestion) {
        accesses = all_tool_access();
    } else if (is_read_tool(name)) {
        auto options = core::tools::read::parse_options(args);
        if (!options) accesses = all_tool_access();
        else {
            // ReadTool reads the recovered path when `path` holds serialized
            // argument text; reserve it too so that read never overlaps a
            // concurrent write to the file it will actually open.
            auto paths = options->paths;
            if (auto recovered = core::tools::read::recover_serialized_path(args))
                paths.insert(paths.end(), recovered->paths.begin(), recovered->paths.end());
            for (const auto& path : paths) {
                if (path.contains("://")) continue;
                // Directory listings also overlap writes to their descendants.
                const auto resolved = context.resolve_path(path);
                accesses.push_back(ToolAccess::file_access(ToolFileOperation::Read,
                    resolved, true));
            }
        }
    } else if (name == kFileSearch || name == kGrepSearch || name == kListDirectory) {
        accesses = single_file_access(ToolFileOperation::Search, context, args, "path", true);
    } else if (name == kWriteFile) {
        accesses = single_file_access(ToolFileOperation::Write, context, args, "file_path");
    } else if (name == kReplace || name == kReplaceInFile || name == kSearchReplace) {
        accesses = single_file_access(ToolFileOperation::ReadWrite, context, args, "path");
    } else if (name == kDeleteFile) {
        accesses = single_file_access(ToolFileOperation::Write, context, args,
                                      "file_path",
                                      core::utils::json::bool_field(args, "recursive"));
    } else if (name == kMoveFile) {
        // A move carries a whole subtree, so a write inside either side
        // overlaps it and must not run alongside.
        accesses = single_file_access(ToolFileOperation::ReadWrite, context, args,
                                      "source", /*recursive=*/true);
        accesses.push_back(ToolAccess::file_access(
            ToolFileOperation::Write,
            resolved_arg_path(context, args, "destination"),
            /*recursive=*/true));
    } else if (name == kCreateDirectory) {
        accesses = single_file_access(ToolFileOperation::Write, context, args,
                                      "dir_path", /*recursive=*/true);
    } else if (name == kApplyPatch) {
        accesses = {
            ToolAccess::file_access(
                ToolFileOperation::ReadWrite,
                resolved_arg_path(context, args, "working_dir"),
                true),
        };
    } else if (name == kActivateSkill) {
        accesses = no_tool_access();
    } else if (name == SubagentOrchestrator::kTaskToolName) {
        accesses = delegated_tool_access();
    } else if (is_terminal_tool(name) && runs_read_only_command_process(args)) {
        accesses = read_all_tool_access();
    } else {
        accesses = all_tool_access();
    }

    return {
        .call = call,
        .accesses = std::move(accesses),
    };
}

} // namespace core::agent
