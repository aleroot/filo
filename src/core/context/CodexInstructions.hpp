#pragma once

#include <string>
#include <string_view>

namespace core::context {

/// The base instructions and collaboration block a Codex-trained model reads.
/// Empty instructions mean Filo's own doctrine stands.
struct CodexPrompt {
    std::string instructions;
    /// Wrapped `<collaboration_mode>` block. Empty when @c instructions is.
    std::string collaboration;
};

/// Codex prompt text a catalog published for one model.
struct CodexPromptText {
    std::string instructions;
    std::string collaboration_default;
    std::string collaboration_plan;
};

/// Codex's `render_model_instructions`: the catalog template verbatim when the
/// endpoint published one, otherwise the bundled GPT-6.2 template for a model
/// trained on code mode. Anything else keeps Filo's doctrine.
///
/// @p mode selects the collaboration text the way Codex selects Default or
/// Plan. PLAN and RESEARCH are Plan; every other Filo mode is Default.
[[nodiscard]] CodexPrompt codex_prompt_for(std::string_view model_id,
                                           std::string_view mode,
                                           std::string_view catalog_instructions,
                                           std::string_view catalog_collaboration_default,
                                           std::string_view catalog_collaboration_plan,
                                           bool code_mode_only);

} // namespace core::context
