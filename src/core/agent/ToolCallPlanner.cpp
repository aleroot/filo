#include "ToolCallPlanner.hpp"

#include "SubagentOrchestrator.hpp"
#include "../tools/ToolNames.hpp"

#include <simdjson.h>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::agent {

namespace {

[[nodiscard]] std::optional<std::string> json_string_field(std::string_view raw,
                                                           std::string_view field) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(raw.data(), raw.size()).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    std::string_view value;
    if (doc[field.data()].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return std::nullopt;
}

[[nodiscard]] std::filesystem::path resolved_arg_path(
    const core::context::SessionContext& context,
    std::string_view raw_args,
    std::string_view field,
    std::filesystem::path fallback = ".") {
    const auto value = json_string_field(raw_args, field);
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

} // namespace

PlannedToolCall plan_tool_call(const core::llm::ToolCall& call,
                               const core::context::SessionContext& context) {
    using namespace core::tools::names;

    const std::string_view name = call.function.name;
    const std::string_view args = call.function.arguments;
    ToolAccessSet accesses;

    if (name == kGetCurrentTime) {
        accesses = no_tool_access();
    } else if (name == kAskUserQuestion) {
        accesses = all_tool_access();
    } else if (name == kReadFile) {
        accesses = single_file_access(ToolFileOperation::Read, context, args, "path");
    } else if (name == kFileSearch || name == kGrepSearch || name == kListDirectory) {
        accesses = single_file_access(ToolFileOperation::Search, context, args, "path", true);
    } else if (name == kWriteFile) {
        accesses = single_file_access(ToolFileOperation::Write, context, args, "file_path");
    } else if (name == kReplace || name == kReplaceInFile || name == kSearchReplace) {
        accesses = single_file_access(ToolFileOperation::ReadWrite, context, args, "path");
    } else if (name == kDeleteFile) {
        accesses = single_file_access(ToolFileOperation::Write, context, args, "path");
    } else if (name == kMoveFile) {
        accesses = single_file_access(ToolFileOperation::ReadWrite, context, args, "source");
        accesses.push_back(ToolAccess::file_access(
            ToolFileOperation::Write,
            resolved_arg_path(context, args, "destination")));
    } else if (name == kCreateDirectory) {
        accesses = single_file_access(ToolFileOperation::Write, context, args, "path", true);
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
        accesses = all_tool_access();
    } else {
        accesses = all_tool_access();
    }

    return {
        .call = call,
        .accesses = std::move(accesses),
    };
}

} // namespace core::agent
