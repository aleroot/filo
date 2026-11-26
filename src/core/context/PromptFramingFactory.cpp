#include "PromptFramingFactory.hpp"

#include "CodexInstructions.hpp"
#include "CodexPromptFraming.hpp"
#include "ModelGuidance.hpp"
#include "../llm/CatalogToolMode.hpp"
#include "../llm/ModelRegistry.hpp"

namespace core::context {

std::shared_ptr<const PromptFraming> prompt_framing_for(std::string_view model,
                                                        std::string_view mode,
                                                        const core::llm::ModelInfo* catalog_card) {
    const auto registered = core::llm::ModelRegistry::instance().lookup(model);
    auto codex = codex_prompt_for(
        model,
        mode,
        catalog_card ? std::string_view(catalog_card->instructions_template) : std::string_view{},
        catalog_card ? std::string_view(catalog_card->collaboration_default) : std::string_view{},
        catalog_card ? std::string_view(catalog_card->collaboration_plan) : std::string_view{},
        registered && core::llm::is_code_mode_only(registered->tool_mode));
    if (codex.instructions.empty()) {
        return std::make_shared<const FiloPromptFraming>(find_model_guidance(model));
    }
    // The live card's list wins; the curated card covers a catalog that
    // published none.
    auto experimental_tools = catalog_card && !catalog_card->experimental_supported_tools.empty()
        ? catalog_card->experimental_supported_tools
        : registered ? registered->experimental_supported_tools : std::vector<std::string>{};
    return std::make_shared<const CodexPromptFraming>(std::move(codex),
                                                      std::move(experimental_tools));
}

} // namespace core::context
