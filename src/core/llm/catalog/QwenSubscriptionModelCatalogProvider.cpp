#include "QwenSubscriptionModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../QwenModelTraits.hpp"
#include "core/utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

namespace core::llm {
namespace {

constexpr ModelCapabilities kBaseCapabilities =
    static_cast<uint32_t>(ModelCapability::TextInput)
    | static_cast<uint32_t>(ModelCapability::TextOutput)
    | static_cast<uint32_t>(ModelCapability::Streaming)
    | static_cast<uint32_t>(ModelCapability::SystemPrompts)
    | static_cast<uint32_t>(ModelCapability::FunctionCalling)
    | static_cast<uint32_t>(ModelCapability::Reasoning);

/// Qwen Code's id rule: 1-256 characters, no comma and no control characters.
[[nodiscard]] std::optional<std::string> model_id(simdjson::dom::object model) {
    std::string_view raw;
    if (model["id"].get(raw) != simdjson::SUCCESS) return std::nullopt;
    const std::string_view id = core::utils::str::trim_ascii_view(raw);
    if (id.empty() || id.size() > 256) return std::nullopt;
    const bool unsafe = std::ranges::any_of(id, [](char ch) {
        const auto byte = static_cast<unsigned char>(ch);
        return ch == ',' || byte < 0x20 || byte == 0x7f;
    });
    if (unsafe) return std::nullopt;
    return std::string(id);
}

/// Declared input capabilities, or nullopt when the field is absent or invalid.
[[nodiscard]] std::optional<ModelCapabilities> declared_inputs(simdjson::dom::object model) {
    simdjson::dom::array modalities;
    if (model["input_modalities"].get(modalities) != simdjson::SUCCESS
        || modalities.size() == 0) {
        return std::nullopt;
    }
    ModelCapabilities inputs = 0;
    for (const auto element : modalities) {
        std::string_view modality;
        if (element.get(modality) != simdjson::SUCCESS) return std::nullopt;
        if (modality == "image") {
            inputs |= static_cast<uint32_t>(ModelCapability::Vision);
        } else if (modality == "video") {
            inputs |= static_cast<uint32_t>(ModelCapability::VideoInput);
        } else if (modality == "pdf") {
            inputs |= static_cast<uint32_t>(ModelCapability::PdfInput);
        } else if (modality != "text" && modality != "audio") {
            return std::nullopt;
        }
    }
    return inputs;
}

[[nodiscard]] int32_t declared_context(simdjson::dom::object model) {
    int64_t tokens = 0;
    if (model["context_window_tokens"].get(tokens) != simdjson::SUCCESS || tokens <= 0) return 0;
    return static_cast<int32_t>(std::min<int64_t>(tokens, std::numeric_limits<int32_t>::max()));
}

[[nodiscard]] ModelCapabilities default_inputs(std::string_view id) {
    return qwen_model_supports_vision(id)
        ? static_cast<uint32_t>(ModelCapability::Vision)
            | static_cast<uint32_t>(ModelCapability::VideoInput)
        : 0;
}

} // namespace

QwenSubscriptionModelCatalogProvider::QwenSubscriptionModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {}

std::string_view QwenSubscriptionModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string QwenSubscriptionModelCatalogProvider::model_list_path(std::string_view) const {
    return "/models";
}

ModelCatalogResult QwenSubscriptionModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result = catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) != simdjson::SUCCESS) {
        result.error = "Qwen Subscription model list is missing data[]";
        return result;
    }
    std::unordered_set<std::string> seen;
    for (const auto element : models) {
        simdjson::dom::object model;
        if (element.get(model) != simdjson::SUCCESS) continue;
        auto id = model_id(model);
        if (!id || !seen.insert(*id).second) continue;

        ModelInfo info;
        info.canonical_id = std::move(*id);
        info.display_name = info.canonical_id;
        info.provider = provider_name_;
        info.capabilities = kBaseCapabilities
            | declared_inputs(model).value_or(default_inputs(info.canonical_id));
        info.context_window = declared_context(model);
        if (info.context_window == 0) {
            info.context_window = qwen_endpoint_context_window(info.canonical_id);
        }
        info.tier = catalog::infer_tier(info.canonical_id, true);
        info.constraints = catalog::standard_constraints();
        result.models.push_back(std::move(info));
    }
    if (result.models.empty()) result.error = "Qwen Subscription returned no models";
    return result;
}

} // namespace core::llm
