#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools {

/// One filesystem change produced by a Codex `*** Begin Patch` envelope.
/// Paths are the literal paths from the patch, relative to the patch directory
/// unless the model wrote an absolute path.
struct CodexAction {
    enum class Kind { Write, Remove };

    Kind kind = Kind::Write;
    std::string path;
    /// Present for Kind::Write. Empty content is an empty file.
    std::string content;
};

struct CodexPlan {
    std::vector<CodexAction> actions;
    /// Short per-file report for the tool result. Not a diff.
    std::string output;
};

/// Applies the envelope to the files `read_file` returns. nullopt from the
/// reader means the path does not exist. On failure `error` names the hunk
/// that did not match and `plan` is empty: the caller must not write anything.
[[nodiscard]] std::optional<CodexPlan> plan_codex_patch(
    std::string_view text,
    const std::function<std::optional<std::string>(std::string_view)>& read_file,
    std::string& error);

} // namespace core::tools
