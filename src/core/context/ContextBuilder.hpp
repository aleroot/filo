#pragma once

#include "SessionContext.hpp"
#include "PromptPlan.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::context {

struct ModelGuidance;

// Mutable repository observations kept outside the cache-stable prompt plan.
struct ProjectFactsSnapshot {
    std::string status;
    std::string tree;

    [[nodiscard]] bool empty() const noexcept {
        return status.empty() && tree.empty();
    }

    bool operator==(const ProjectFactsSnapshot&) const = default;
};

[[nodiscard]] std::optional<ProjectFactsSnapshot> capture_project_facts(
    const SessionContext& session_context) noexcept;
[[nodiscard]] std::string render_project_facts(const ProjectFactsSnapshot& snapshot);
[[nodiscard]] std::string render_project_facts_update(
    const ProjectFactsSnapshot& previous,
    const ProjectFactsSnapshot& current);

class ContextBuilder {
public:
    explicit ContextBuilder(const SessionContext& session_context);

    ContextBuilder& with_mode(std::string_view mode);
    // The profile of the model the prompt is written for (find_model_guidance);
    // nullptr keeps the shared doctrine as written.
    ContextBuilder& with_model_guidance(const ModelGuidance* guidance) noexcept;
    // Pre-rendered semantic memory block from MemorySystem. ContextBuilder
    // does not open a store: the composition root projects the prompt so the
    // assembler stays independent of persistence.
    ContextBuilder& with_memory_prompt(std::string prompt);
    ContextBuilder& include_project_context(bool include = true) noexcept;
    ContextBuilder& include_skill_catalog(bool include = true) noexcept;
    // Agents disable this and deliver snapshots as append-only synthetic messages.
    ContextBuilder& include_project_facts(bool include = true) noexcept;

    [[nodiscard]] std::vector<ContextLayer> build_layers() const;
    [[nodiscard]] PromptPlan build_plan() const;
    [[nodiscard]] std::string build() const;

private:
    const SessionContext& session_context_;
    std::string memory_prompt_;
    std::string mode_ = "BUILD";
    const ModelGuidance* model_guidance_ = nullptr;
    bool include_project_context_ = true;
    bool include_skill_catalog_ = true;
    bool include_project_facts_ = true;
};

} // namespace core::context
