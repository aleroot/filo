#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "CodexModelCatalog.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../auth/CodexClientIdentity.hpp"
#include "../../utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <optional>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] bool is_reasoning_model(std::string_view model_id) {
    const std::string lower =
        core::utils::str::to_lower_ascii_copy(model_id);
    return lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("gpt-5")
        || lower.starts_with("gpt-6")
        || catalog::contains_ascii(model_id, "reasoning")
        || catalog::contains_ascii(model_id, "thinking");
}

[[nodiscard]] ModelCapabilities codex_capabilities(
    std::string_view model_id,
    const catalog::JsonObjectView& model) {
    if (catalog::is_embedding_model(model_id)) {
        return static_cast<uint32_t>(ModelCapability::Embeddings);
    }

    ModelCapabilities capabilities =
        catalog::kTextCapabilities
        | static_cast<uint32_t>(ModelCapability::FunctionCalling)
        | static_cast<uint32_t>(ModelCapability::JsonMode);
    if (model.boolean("supports_parallel_tool_calls")) {
        capabilities |= static_cast<uint32_t>(
            ModelCapability::ParallelToolCalls);
    }
    if (model.boolean("supports_image_detail_original")
        || model.string_array_contains("input_modalities", "image")
        || catalog::contains_ascii(model_id, "gpt-4")
        || catalog::contains_ascii(model_id, "gpt-5")
        || catalog::contains_ascii(model_id, "gpt-6")
        || catalog::contains_ascii(model_id, "4o")
        || catalog::contains_ascii(model_id, "vision")) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Vision);
    }

    std::string default_reasoning_level;
    if (is_reasoning_model(model_id)
        || model.string("default_reasoning_level", default_reasoning_level)
        || catalog::compatible_reasoning_profile(model).effort.supports_effort()) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    return capabilities;
}

// Mirrors codex-rs: `ultra` drives the model at `multi_agent_reasoning_effort`
// (ModelInfo::resolve_reasoning_effort) and switches a `multi_agent_version:
// v2` model to the proactive multi-agent mode (effective_multi_agent_mode).
// A catalog `model_messages.multi_agent.mode` can replace that mode text
// (`hint_text` for every mode, else `proactive`); an empty replacement
// suppresses it. Replacement text is Codex-specific (it names Codex's tools),
// so Filo keeps only the decision and sends its own words.
[[nodiscard]] std::optional<UltraEffortProfile> codex_ultra_profile(
    const catalog::JsonObjectView& model,
    const ModelReasoningProfile& reasoning) {
    if (std::ranges::find(reasoning.supported_efforts, std::string("ultra"))
        == reasoning.supported_efforts.end()) {
        return std::nullopt;
    }
    UltraEffortProfile ultra;
    std::string effort;
    if (model.string("multi_agent_reasoning_effort", effort) && !effort.empty()) {
        ultra.effort = core::utils::str::to_lower_ascii_copy(effort);
    }
    std::string version;
    ultra.proactive_delegation = model.string("multi_agent_version", version)
        && core::utils::str::to_lower_ascii_copy(version) == "v2";
    simdjson::dom::object mode;
    if (ultra.proactive_delegation
        && model.raw().at_pointer("/model_messages/multi_agent/mode").get(mode)
               == simdjson::SUCCESS) {
        const catalog::JsonObjectView mode_view(mode);
        std::string text;
        if ((mode_view.string("hint_text", text) || mode_view.string("proactive", text))
            && text.empty()) {
            ultra.proactive_delegation = false;
        }
    }
    return ultra;
}

[[nodiscard]] ModelInfo decode_codex_model(
    const catalog::JsonObjectView& model,
    std::string_view provider_name,
    bool codex_controls) {
    ModelInfo info;
    if (!model.string("slug", info.canonical_id)) {
        (void)model.string("id", info.canonical_id);
    }
    if (info.canonical_id.empty()) return info;

    // The Codex picker only lists `visibility: list`. Hidden and unavailable
    // rows stay out of Filo's model list; a missing field stays visible so
    // older catalogs are not emptied.
    std::string visibility;
    if (model.string("visibility", visibility)) {
        const std::string normalized =
            core::utils::str::to_lower_ascii_copy(visibility);
        if (normalized == "hide" || normalized == "none") {
            info.canonical_id.clear();
            return info;
        }
    }

    info.provider = catalog::copy_string(provider_name);
    if (!model.string("display_name", info.display_name)) {
        info.display_name = info.canonical_id;
    }
    info.capabilities =
        catalog::compatible_advertised_capabilities(info.canonical_id, model)
        | codex_capabilities(info.canonical_id, model);
    info.context_window = model.first_integer(
        {"context_window", "max_context_window", "context_length"});
    info.max_output_tokens = model.first_integer(
        {"max_output_tokens", "max_tokens"});
    info.max_reasoning_tokens = model.first_integer(
        {"max_reasoning_tokens", "reasoning_max_tokens"});
    info.reasoning = catalog::compatible_reasoning_profile(model);
    if (codex_controls) {
        if (model.boolean("use_responses_lite")) info.wire_protocol = "responses_lite";
        info.reasoning.summary_supported = model.optional_boolean("supports_reasoning_summary_parameter");
        std::string summary;
        if (model.string("default_reasoning_summary", summary)) {
            info.reasoning.default_summary = std::move(summary);
        }
        if (const auto limit = model.integer("auto_compact_token_limit"); limit > 0) {
            info.auto_compact_token_limit = limit;
        }
        info.reasoning.ultra = codex_ultra_profile(model, info.reasoning);
    }
    if (model.boolean("support_verbosity")) {
        (void)model.string("default_verbosity", info.default_verbosity);
    }
    info.tier = catalog::infer_tier(
        info.canonical_id,
        info.supports(ModelCapability::Reasoning));
    info.constraints = catalog::standard_constraints();
    return info;
}

} // namespace

CodexModelCatalogProvider::CodexModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) provider_name_ = "openai-codex";
}

std::string_view CodexModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string CodexModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/models?client_version="
        + std::string(core::auth::codex_cli::client_version());
}

namespace {
ModelCatalogResult parse_models(
    std::string_view body,
    std::string_view provider_name,
    bool codex_controls) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["models"].get(models) != simdjson::SUCCESS) {
        result.error = "Codex model catalog response is missing models[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        ModelInfo info = decode_codex_model(
            catalog::JsonObjectView(object), provider_name, codex_controls);
        if (!info.canonical_id.empty()) {
            result.models.push_back(std::move(info));
        }
    }
    return result;
}
} // namespace

ModelCatalogResult catalog::parse_codex_style_model_catalog(
    std::string_view body, std::string_view provider_name) {
    return parse_models(body, provider_name, false);
}

ModelCatalogResult CodexModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    return parse_models(body, provider_name_, true);
}

} // namespace core::llm
