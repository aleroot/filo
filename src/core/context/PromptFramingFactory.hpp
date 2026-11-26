#pragma once

#include "PromptFraming.hpp"

#include <memory>
#include <string_view>

namespace core::llm {
struct ModelInfo;
}

namespace core::context {

/// The framing the prompt for @p model is written in during a turn in @p mode.
/// A model trained on Codex's harness (its catalog publishes an instructions
/// template, or the registry marks it code-mode only) gets Codex's framing;
/// any other gets Filo's, tailored by its model guidance. @p catalog_card is
/// what the serving provider's live catalog published for the model, if any.
[[nodiscard]] std::shared_ptr<const PromptFraming> prompt_framing_for(
    std::string_view model,
    std::string_view mode,
    const core::llm::ModelInfo* catalog_card);

} // namespace core::context
