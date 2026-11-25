#include "ReviewTool.hpp"

#include "../agent/Agent.hpp"
#include "../review/RunReview.hpp"
#include "../utils/StringUtils.hpp"
#include "ToolNames.hpp"

#include <simdjson.h>

namespace core::tools {
namespace {

using core::utils::str::trim_ascii_copy;

} // namespace

ToolDefinition review_tool_definition() {
    return {
        .name = std::string(names::kReview),
        .title = std::string(names::kReviewTitle),
        .description =
            "Review the current checkout with Filo's review engine and return "
            "blocking findings separately from follow-ups. Blocking means a "
            "critical or high bug or security issue (an unset category at that "
            "severity also blocks). Does not edit files. target is uncommitted "
            "(default) or staged.",
        .input_schema =
            R"({"type":"object","properties":{"target":{"type":"string","enum":["uncommitted","staged"],"description":"Which diff to review. Defaults to uncommitted."}},"additionalProperties":false})",
        .annotations = {
            .read_only_hint = true,
            .idempotent_hint = false,
        },
        .agent_only = true,
    };
}

std::string execute_review_tool(
    core::agent::Agent& agent,
    std::string_view json_args,
    const std::function<bool()>& cancellation_requested) {
    std::string target = "uncommitted";
    const std::string trimmed = trim_ascii_copy(json_args);
    if (!trimmed.empty() && trimmed != "{}") {
        simdjson::dom::parser parser;
        simdjson::padded_string padded{std::string(trimmed)};
        simdjson::dom::object object;
        if (parser.parse(padded).get(object) != simdjson::SUCCESS) {
            core::review::WorkspaceReview failed;
            failed.error = "review arguments are not valid JSON";
            return core::review::render_review_json(failed);
        }
        std::string_view requested;
        if (object["target"].get(requested) == simdjson::SUCCESS && !requested.empty()) {
            target = std::string(requested);
        }
    }
    if (target != "uncommitted" && target != "staged") {
        core::review::WorkspaceReview failed;
        failed.error = "review target must be uncommitted or staged";
        return core::review::render_review_json(failed);
    }
    const auto repository = agent.workspace_snapshot().primary();
    return core::review::render_review_json(core::review::review_workspace(
        agent.shared_from_this(),
        repository,
        target,
        {},
        {},
        cancellation_requested));
}

} // namespace core::tools
