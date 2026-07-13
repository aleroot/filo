#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace core::context {

enum class ContextLayerKind {
    RuntimeInstructions,
    WorkspaceFacts,
    Memory,
    ProjectSteering,
    SkillCatalog,
    ProjectFacts,
    ConversationState,
};

enum class PromptStability {
    Stable,    ///< Product/runtime instructions shared across requests.
    Workspace, ///< Project-scoped material that changes infrequently.
    Session,   ///< Session-scoped memory and user preferences.
    Dynamic,   ///< Mutable facts that must remain at the end of the prompt.
};

struct ContextLayer {
    ContextLayerKind kind;
    PromptStability stability = PromptStability::Dynamic;
    std::string name;
    std::string content;
};

class PromptPlan {
public:
    void append(ContextLayer layer);

    [[nodiscard]] const std::vector<ContextLayer>& layers() const noexcept {
        return layers_;
    }
    [[nodiscard]] bool empty() const noexcept { return layers_.empty(); }
    [[nodiscard]] std::string render() const;
    [[nodiscard]] std::string render_through(PromptStability stability) const;

private:
    std::vector<ContextLayer> layers_;
};

} // namespace core::context
