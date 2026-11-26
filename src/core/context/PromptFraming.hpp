#pragma once

#include "SessionContext.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace core::context {

struct ModelGuidance;
struct SteeringLoadResult;

/// How the prompt presents each kind of context to the model it is written
/// for. ContextBuilder decides which layers a prompt has and in what order;
/// a framing decides what each layer says. A model trained on another
/// harness's prompt gets its own framing instead of branches in the builder.
class PromptFraming {
public:
    virtual ~PromptFraming() = default;

    /// The instructions that open the prompt, for a turn in @p mode.
    [[nodiscard]] virtual std::string runtime(std::string_view mode) const = 0;
    /// The workspace the session works in.
    [[nodiscard]] virtual std::string workspace(const SessionContext& session_context) const = 0;
    /// The project instructions @p loaded carries, and @p withheld_notice
    /// naming the instruction files the policy deliberately withheld.
    [[nodiscard]] virtual std::string steering(const SteeringLoadResult& loaded,
                                               std::string_view withheld_notice) const = 0;
    /// The skills available under @p workspace_roots.
    [[nodiscard]] virtual std::string skills(
        const std::vector<std::filesystem::path>& workspace_roots) const = 0;
};

/// Filo's own prompt: its doctrine and working rules, tailored by @p guidance
/// when given (find_model_guidance), and Filo's section headings.
class FiloPromptFraming final : public PromptFraming {
public:
    explicit FiloPromptFraming(const ModelGuidance* guidance = nullptr) noexcept
        : guidance_(guidance) {}

    [[nodiscard]] std::string runtime(std::string_view mode) const override;
    [[nodiscard]] std::string workspace(const SessionContext& session_context) const override;
    [[nodiscard]] std::string steering(const SteeringLoadResult& loaded,
                                       std::string_view withheld_notice) const override;
    [[nodiscard]] std::string skills(
        const std::vector<std::filesystem::path>& workspace_roots) const override;

private:
    const ModelGuidance* guidance_;
};

/// The `[Workspace]` section every framing shares: the primary and additional
/// directories and how paths resolve. Empty without a primary directory.
[[nodiscard]] std::string render_workspace_paths(const SessionContext& session_context);

/// What a prompt in AUTO mode tells the model about Filo's AUTO controller.
inline constexpr std::string_view kAutoContractInstructions =
    "Let Filo's AUTO controller choose a proportional execution contract for each turn. "
    "Follow the generated AUTO contract exactly: keep simple work direct, decompose complex "
    "work, fan out independent read-only investigations, preserve repository state, and "
    "finish mutations with fresh verification evidence.";

} // namespace core::context
