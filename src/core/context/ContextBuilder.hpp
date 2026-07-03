#pragma once

#include "SessionContext.hpp"

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
};

struct ContextLayer {
    ContextLayerKind kind;
    std::string name;
    std::string content;
};

class ContextBuilder {
public:
    explicit ContextBuilder(const SessionContext& session_context);

    ContextBuilder& with_mode(std::string_view mode);
    ContextBuilder& include_project_context(bool include = true) noexcept;
    ContextBuilder& include_skill_catalog(bool include = true) noexcept;

    [[nodiscard]] std::vector<ContextLayer> build_layers() const;
    [[nodiscard]] std::string build() const;

private:
    const SessionContext& session_context_;
    std::string mode_ = "BUILD";
    bool include_project_context_ = true;
    bool include_skill_catalog_ = true;
};

} // namespace core::context
