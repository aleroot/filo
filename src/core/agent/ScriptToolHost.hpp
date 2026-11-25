#pragma once

#include "ToolApproval.hpp"
#include "../context/SessionContext.hpp"
#include "../llm/Models.hpp"
#include "../tools/ExecTool.hpp"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace core::agent {

/// What the nested calls of an `exec` script need from the agent running it.
struct NestedToolServices {
    /// The tools the step offered: the only ones a script may call.
    std::span<const core::llm::Tool> offered;
    core::context::SessionContext session_context;
    std::function<std::optional<core::tools::ToolDefinition>(const std::string&)> resolve;
    std::function<ToolApproval(const core::llm::ToolCall&)> approve;
    /// Runs an approved call and returns the result the script reads.
    std::function<core::llm::Message(const core::llm::ToolCall&,
                                     const std::optional<core::tools::ToolDefinition>&)> execute;
    std::function<void(const core::llm::ToolCall&)> on_start;
    std::function<void(const core::llm::ToolCall&, const core::llm::Message&)> on_finish;
    std::function<bool()> cancelled;
};

/// The host an `exec` script reaches its tools through. A nested call passes
/// the gates of a direct call: it must be a tool the step offered, its
/// arguments must validate, then the approval gates decide. As in a direct
/// batch, one denial skips every later call of the script. The approved calls
/// of one batch run under the scheduler's file-access rules, so independent
/// calls run in parallel.
class ScriptToolHost final : public core::tools::ToolScriptHost {
public:
    explicit ScriptToolHost(NestedToolServices services);

    [[nodiscard]] std::vector<core::tools::ToolScriptReply>
    run(std::span<const core::tools::ToolScriptCall> calls) override;
    [[nodiscard]] bool cancelled() const override;

private:
    struct Admitted;
    [[nodiscard]] std::optional<Admitted> admit(const core::tools::ToolScriptCall& requested,
                                                core::tools::ToolScriptReply& refusal);
    [[nodiscard]] core::tools::ToolScriptReply refuse(const core::llm::ToolCall& call,
                                                      std::string reason) const;

    NestedToolServices services_;
    bool denied_ = false;
};

} // namespace core::agent
