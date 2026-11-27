#pragma once

#include "Tool.hpp"

#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

namespace core::tools {

/// A file a model asks to show the user.
struct DocumentRequest {
    /// Absolute; an existing regular file the session may read.
    std::filesystem::path path;
    /// One-based line to bring into view.
    std::optional<int> line;
    /// The conversation that asked.
    std::string session_id;
};

/// What the user's screen did with a DocumentRequest.
enum class DocumentDisplay {
    /// In front, beside the conversation.
    shown,
    /// Added as a tab behind the file the user is reading in the pane.
    behind,
    /// Added as a tab; the screen is too narrow to show it beside the
    /// conversation without hiding it.
    hidden,
    /// Not opened: the user is looking at another conversation.
    elsewhere,
    /// The screen did not answer (it is closing).
    unavailable,
};

/// Shows a file to the user in the host's document pane. Filo-internal: only
/// an interactive host that has a pane registers it, and it is never offered
/// to MCP clients (ToolDefinition::agent_only).
///
/// The host decides how much of the screen a request may take; the tool only
/// checks the path and reports what the host did, so the model knows whether
/// to link the file in its reply instead.
class OpenDocumentTool final : public Tool {
public:
    using Opener = std::function<DocumentDisplay(const DocumentRequest&)>;

    [[nodiscard]] ToolDefinition get_definition() const override;
    [[nodiscard]] std::string execute(const std::string& json_args,
                                      const core::context::SessionContext& context) override;

    /// Installs the host's pane; without one, calls fail.
    void set_opener(Opener opener);

private:
    std::mutex mutex_;
    Opener opener_;
};

} // namespace core::tools
