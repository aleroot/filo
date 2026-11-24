#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools {

/// Paths a patch will touch, relative to the patch working directory.
/// Unified diffs are the `---`/`+++` paths `patch -p1` sees (`a/` and `b/`
/// stripped, `/dev/null` skipped). A Codex envelope (`*** Begin Patch`) yields
/// its Update, Add, Delete, and Move paths verbatim, so a context line cannot
/// be mistaken for a unified header.
[[nodiscard]] std::vector<std::string> patch_target_paths(std::string_view patch_text);

/// True when the first non-empty line is `*** Begin Patch` at column 0.
[[nodiscard]] bool is_codex_patch(std::string_view patch_text) noexcept;

/// A reason to refuse a unified diff before invoking `patch`, or nullopt when
/// the text has no counted hunks or every hunk body agrees with its header.
/// Apple patch 2.0 reports a bad count as "No file found" because the repair
/// prompt reads the rest of the diff. Callers surface this message instead.
[[nodiscard]] std::optional<std::string> unified_diff_diagnostic(std::string_view patch_text);

} // namespace core::tools
