#pragma once

#include "DiffViewer.hpp"

#include <string>
#include <string_view>
#include <utility>

namespace tui::viewer {

// The portable comparer: whatever the user already configured through the
// standard `PAGER` convention, or an explicit command such as "delta".
class TerminalDiffViewer final : public DiffViewer {
public:
    // An empty command defers to $PAGER, then "less", then "more" — resolved at
    // view time so the user can change it without restarting Filo.
    explicit TerminalDiffViewer(std::string command = {}) : command_(std::move(command)) {}

    [[nodiscard]] ViewerDescriptor descriptor() const noexcept override;
    [[nodiscard]] ViewResult view(const ViewContext& context) override;

private:
    std::string command_;
};

[[nodiscard]] ViewerDescriptor terminal_diff_viewer_descriptor() noexcept;

// ── Exposed for testing ──────────────────────────────────────────────────────

// Resolves the pager command, honouring $PAGER then "less" then "more".
[[nodiscard]] std::string resolve_pager_command(std::string_view configured_command);

// Builds the shell invocation that pages one patch file through standard input.
[[nodiscard]] std::string build_pager_invocation(
    std::string_view command,
    std::string_view file_path);

} // namespace tui::viewer
