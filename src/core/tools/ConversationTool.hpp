#pragma once

#include "Tool.hpp"
#include "../session/SessionTranscriptReader.hpp"

namespace core::tools {

/// Reads provider-neutral saved conversation text; no workspace path access
/// or original-provider credentials are needed.
class ConversationTool final : public Tool {
public:
    explicit ConversationTool(core::session::SessionStore store =
        core::session::SessionStore{core::session::SessionStore::default_sessions_dir()});

    [[nodiscard]] ToolDefinition get_definition() const override;
    [[nodiscard]] std::string execute(const std::string& json_args,
                                     const core::context::SessionContext& context) override;

private:
    core::session::SessionTranscriptReader reader_;
};

} // namespace core::tools
