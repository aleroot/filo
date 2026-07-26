#include "ModelCatalogTraits.hpp"

#include "../../utils/StringUtils.hpp"

namespace core::llm::catalog {

bool contains_ascii(std::string_view haystack, std::string_view needle) {
    const std::string lower_haystack =
        core::utils::str::to_lower_ascii_copy(haystack);
    const std::string lower_needle =
        core::utils::str::to_lower_ascii_copy(needle);
    return lower_haystack.find(lower_needle) != std::string::npos;
}

std::string strip_prefix(std::string_view value, std::string_view prefix) {
    if (value.starts_with(prefix)) {
        value.remove_prefix(prefix.size());
    }
    return std::string(value);
}

bool is_embedding_model(std::string_view model_id) {
    return contains_ascii(model_id, "embedding")
        || contains_ascii(model_id, "embed");
}

ModelTier infer_tier(std::string_view model_id, bool reasoning) {
    if (reasoning || contains_ascii(model_id, "thinking")) {
        return ModelTier::Reasoning;
    }
    if (contains_ascii(model_id, "flash")
        || contains_ascii(model_id, "lite")
        || contains_ascii(model_id, "mini")
        || contains_ascii(model_id, "nano")
        || contains_ascii(model_id, "haiku")) {
        return ModelTier::Fast;
    }
    if (contains_ascii(model_id, "pro")
        || contains_ascii(model_id, "opus")
        || contains_ascii(model_id, "max")) {
        return ModelTier::Powerful;
    }
    return ModelTier::Balanced;
}

ParameterConstraints standard_constraints(double max_temperature) {
    ParameterConstraints constraints;
    constraints.temperature = {0.0, max_temperature};
    constraints.top_p = {0.0, 1.0};
    constraints.frequency_penalty = {-2.0, 2.0};
    constraints.presence_penalty = {-2.0, 2.0};
    return constraints;
}

} // namespace core::llm::catalog
