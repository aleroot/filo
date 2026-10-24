#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../KimiModelTraits.hpp"

#include <simdjson.h>

#include <optional>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] int32_t infer_context_window(std::string_view model_id) {
    if (const int32_t known = kimi_model_context_window(model_id);
        known > 0) {
        return known;
    }
    if (catalog::contains_ascii(model_id, "moonshot-v1-8k")) return 8'192;
    if (catalog::contains_ascii(model_id, "moonshot-v1-32k")) return 32'768;
    if (catalog::contains_ascii(model_id, "moonshot-v1-128k")) return 128'000;
    return 128'000;
}

[[nodiscard]] int32_t infer_max_output_tokens(std::string_view model_id) {
    return kimi_model_max_output_tokens(model_id);
}

/// Reasoning as either Kimi catalog dialect publishes it, normalized.
///
/// The managed Kimi Code catalog nests everything under one `reasoning` object:
///   {"type":"only"|"both"|"none","effort":{"valid":[...],"default":"max"}}
/// The open-platform catalog keeps the older flat shape:
///   {"supports_reasoning":true,"supports_thinking_type":"only",
///    "think_efforts":{"support":true,"valid_efforts":[...],"default_effort":"max"}}
struct KimiCatalogReasoning {
    /// An authoritative reasoning block was published, so `supported` is a
    /// statement of fact rather than an inference from the model id.
    bool published = false;
    bool supported = false;
    /// `type:"only"` — thinking cannot be switched off.
    bool always_on = false;
    std::vector<std::string> efforts;
    std::string default_effort;
};

[[nodiscard]] bool effort_listed(const std::vector<std::string>& efforts,
                                std::string_view needle) {
    return std::ranges::find(efforts, needle) != efforts.end();
}

/// Read the managed dialect's nested `reasoning` object.
///
/// kimi-code treats a missing object and `type:"none"` as the same non-thinking
/// state, and reserves "unrecognized" for a present object whose type it does
/// not know — in that case the legacy fields still get a chance, so this
/// reports `published = false` instead of guessing.
[[nodiscard]] KimiCatalogReasoning parse_managed_reasoning(
    const catalog::JsonObjectView& model) {
    KimiCatalogReasoning out;
    simdjson::dom::object reasoning_obj;
    if (model.raw()["reasoning"].get(reasoning_obj) != simdjson::SUCCESS) {
        return out;
    }
    const catalog::JsonObjectView reasoning(reasoning_obj);
    std::string type;
    if (!reasoning.string("type", type)
        || (type != "only" && type != "both" && type != "none")) {
        return out;
    }
    out.published = true;
    out.always_on = type == "only";
    out.supported = type != "none";

    simdjson::dom::object effort_obj;
    if (reasoning.raw()["effort"].get(effort_obj) == simdjson::SUCCESS) {
        const catalog::JsonObjectView effort(effort_obj);
        out.efforts = effort.string_array("valid");
        std::string fallback;
        // A default that is not listed in `valid` is dropped, as kimi-code does.
        if (effort.string("default", fallback) && !fallback.empty()
            && effort_listed(out.efforts, fallback)) {
            out.default_effort = std::move(fallback);
        }
    }
    return out;
}

/// Read the open-platform dialect's flat reasoning fields.
[[nodiscard]] KimiCatalogReasoning parse_open_platform_reasoning(
    const catalog::JsonObjectView& model) {
    KimiCatalogReasoning out;
    std::string thinking_type;
    const bool has_thinking_type =
        model.string("supports_thinking_type", thinking_type);

    simdjson::dom::object effort_object;
    if (model.raw()["think_efforts"].get(effort_object) == simdjson::SUCCESS) {
        const catalog::JsonObjectView efforts(effort_object);
        // `support` gates the whole object: when it is not true, ignore
        // valid_efforts / default_effort entirely.
        if (efforts.boolean("support")) {
            out.efforts = efforts.string_array("valid_efforts");
            std::string fallback;
            if (efforts.string("default_effort", fallback) && !fallback.empty()
                && (out.efforts.empty() || effort_listed(out.efforts, fallback))) {
                out.default_effort = std::move(fallback);
            }
            out.published = true;
        }
    }

    // Exact values, matching kimi-code's parseSupportsThinkingType. A substring
    // test is wrong here: "only" contains "no", so it would publish an
    // always-thinking model as one that cannot think.
    const bool type_recognized = has_thinking_type
        && (thinking_type == "only" || thinking_type == "both"
            || thinking_type == "no");
    if (type_recognized) {
        out.published = true;
        out.always_on = thinking_type == "only";
        out.supported = thinking_type != "no";
    } else if (const auto supports = model.optional_boolean("supports_reasoning");
               supports.has_value()) {
        out.published = true;
        out.supported = out.supported || *supports;
    }
    return out;
}

[[nodiscard]] ModelReasoningProfile kimi_reasoning_profile(
    const KimiCatalogReasoning& reasoning) {
    ModelReasoningProfile profile;
    if (!reasoning.supported) {
        // A published "cannot think" is worth recording: it lets a refresh
        // clear a stale thinking default instead of leaving it inferred.
        profile.complete = reasoning.published;
        return profile;
    }

    profile.effort = ReasoningCapabilities{ReasoningCapability::Effort};
    profile.manual_thinking = true;
    if (reasoning.always_on) {
        profile.effort = profile.effort | ReasoningCapability::Required;
    }
    if (effort_listed(reasoning.efforts, "max")) {
        profile.effort = profile.effort | ReasoningCapability::MaxEffort;
    }
    if (effort_listed(reasoning.efforts, "xhigh")) {
        profile.effort = profile.effort | ReasoningCapability::XHighEffort;
    }
    profile.supported_efforts = reasoning.efforts;
    if (!reasoning.default_effort.empty()) {
        profile.catalog_default_effort = reasoning.default_effort;
    }
    profile.complete = reasoning.published;
    return profile;
}

/// Input modalities and tool support as either dialect publishes them.
///
/// A `nullopt` means "this catalog said nothing", which lets the caller fall
/// back to what the model id implies instead of reading silence as a denial.
struct KimiCatalogSurface {
    std::optional<bool> reasoning;
    std::optional<bool> image;
    std::optional<bool> video;
    /// Managed dialect only; kimi-code treats an absent key as `true`.
    std::optional<bool> tool_use;
};

[[nodiscard]] KimiCatalogSurface kimi_catalog_surface(
    const catalog::JsonObjectView& model) {
    KimiCatalogSurface surface;
    surface.reasoning = model.optional_boolean("supports_reasoning");
    surface.tool_use = model.optional_boolean("supports_tool_use");

    // The managed catalog lists `modalities.input`; the open platform uses
    // per-modality booleans. Trust a present list as authoritative — one that
    // omits image/video really does mean the model cannot take them.
    simdjson::dom::object modalities_obj;
    if (model.raw()["modalities"].get(modalities_obj) == simdjson::SUCCESS) {
        const catalog::JsonObjectView modalities(modalities_obj);
        simdjson::dom::array input;
        if (modalities.raw()["input"].get(input) == simdjson::SUCCESS) {
            for (const auto value : input) {
                std::string_view modality;
                if (value.get(modality) != simdjson::SUCCESS) continue;
                if (modality == "image") surface.image = true;
                else if (modality == "video") surface.video = true;
            }
            if (!surface.image.has_value()) surface.image = false;
            if (!surface.video.has_value()) surface.video = false;
        }
    }
    if (!surface.image.has_value()) {
        surface.image = model.optional_boolean("supports_image_in");
    }
    if (!surface.video.has_value()) {
        surface.video = model.optional_boolean("supports_video_in");
    }
    return surface;
}

[[nodiscard]] ModelCapabilities kimi_capabilities(
    std::string_view model_id,
    const KimiCatalogSurface& surface,
    ModelCapabilities advertised) {
    ModelCapabilities capabilities =
        advertised |
        catalog::kToolCapabilities |
        static_cast<uint32_t>(ModelCapability::JsonMode);

    const bool inferred_reasoning =
        is_kimi_k3_model(model_id)
        || catalog::contains_ascii(model_id, "thinking")
        || catalog::contains_ascii(model_id, "reason")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (surface.reasoning.value_or(inferred_reasoning)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }

    const bool inferred_image =
        is_kimi_k3_model(model_id)
        || catalog::contains_ascii(model_id, "vl")
        || catalog::contains_ascii(model_id, "vision")
        || catalog::contains_ascii(model_id, "kimi-k2")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (surface.image.value_or(inferred_image)
        || surface.video.value_or(false)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Vision);
    }

    const bool inferred_video =
        (is_kimi_k3_model(model_id) && !is_kimi_k3_256k_model(model_id))
        || catalog::contains_ascii(model_id, "kimi-k2")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (surface.video.value_or(inferred_video)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::VideoInput);
    }

    // Applied last: an explicit denial outranks anything the row advertised.
    if (surface.tool_use.has_value() && !*surface.tool_use) {
        capabilities &= ~static_cast<uint32_t>(ModelCapability::FunctionCalling);
        capabilities &= ~static_cast<uint32_t>(ModelCapability::ParallelToolCalls);
    }
    return capabilities;
}

/// Managed rows carry the window as `limit.context`, which kimi-code treats as
/// mandatory; open-platform rows carry `context_length`. Both are honored
/// before falling back to what the model id implies.
[[nodiscard]] int32_t kimi_catalog_context_window(
    const catalog::JsonObjectView& model,
    std::string_view model_id) {
    simdjson::dom::object limit_obj;
    if (model.raw()["limit"].get(limit_obj) == simdjson::SUCCESS) {
        const catalog::JsonObjectView limit(limit_obj);
        if (const int32_t context = limit.integer("context", 0); context > 0) {
            return context;
        }
    }
    return model.first_integer(
        {"context_length", "context_window", "max_input_tokens"},
        infer_context_window(model_id));
}

/// The wire the endpoint declares for this model, normalized the way kimi-code
/// does: `anthropic` stays, `response` becomes the Responses API, and anything
/// unrecognized is dropped so the provider default applies.
[[nodiscard]] std::string kimi_catalog_wire_protocol(
    const catalog::JsonObjectView& model) {
    std::string protocol;
    if (!model.string("protocol", protocol)) return {};
    if (protocol == "anthropic") return "anthropic";
    if (protocol == "response") return "openai_responses";
    return {};
}

} // namespace

KimiModelCatalogProvider::KimiModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "kimi";
    }
}

std::string_view KimiModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string KimiModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/models";
}

ModelCatalogResult KimiModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) != simdjson::SUCCESS) {
        result.error = "Kimi model catalog response is missing data[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        const catalog::JsonObjectView model(object);

        ModelInfo info;
        if (!model.string("id", info.canonical_id)
            || info.canonical_id.empty()) {
            continue;
        }
        info.provider = provider_name_;
        if (!model.string("display_name", info.display_name)) {
            info.display_name = info.canonical_id;
        }
        KimiCatalogReasoning reasoning = parse_managed_reasoning(model);
        if (!reasoning.published) {
            reasoning = parse_open_platform_reasoning(model);
        }
        KimiCatalogSurface surface = kimi_catalog_surface(model);
        // A published reasoning block is authoritative. The model id is only a
        // guess for rows that say nothing, so it must not override the block
        // and must not drop it when the id is one Filo has never heard of.
        if (reasoning.published) {
            surface.reasoning = reasoning.supported;
        }
        info.context_window = kimi_catalog_context_window(
            model,
            info.canonical_id);
        info.max_output_tokens = model.first_integer(
            {"max_output_tokens", "max_tokens"},
            infer_max_output_tokens(info.canonical_id));
        if (info.context_window > 0
            && info.max_output_tokens > info.context_window) {
            info.max_output_tokens = info.context_window;
        }
        info.capabilities = kimi_capabilities(
            info.canonical_id,
            surface,
            catalog::compatible_advertised_capabilities(
                info.canonical_id,
                model));
        if (!reasoning.published
            && info.supports(ModelCapability::Reasoning)) {
            reasoning.supported = true;
        }
        info.reasoning = kimi_reasoning_profile(reasoning);
        info.wire_protocol = kimi_catalog_wire_protocol(model);
        info.tier = catalog::infer_tier(
            info.canonical_id,
            info.supports(ModelCapability::Reasoning));
        info.constraints = catalog::standard_constraints();
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
