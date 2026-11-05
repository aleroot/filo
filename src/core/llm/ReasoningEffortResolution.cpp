#include "ReasoningEffortResolution.hpp"

#include "../utils/StringUtils.hpp"

#include <cctype>

namespace core::llm {

bool is_reasoning_off_effort(std::string_view effort) {
    const auto normalized = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(effort));
    return normalized == "off" || normalized == "none" || normalized == "disabled";
}

std::optional<std::string> parse_effort_selection(std::string_view requested) {
    std::string normalized;
    normalized.reserve(requested.size());
    for (const char ch : requested) {
        if (ch == '-' || ch == '_' || std::isspace(static_cast<unsigned char>(ch))) continue;
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (normalized.empty()) return std::nullopt;
    if (normalized == "auto" || normalized == "unset" || normalized == "default") {
        return std::string{};
    }
    if (is_reasoning_off_effort(normalized)) return std::string("none");
    if (normalized == "low" || normalized == "medium" || normalized == "high"
        || normalized == "xhigh" || normalized == "max" || normalized == "ultra") {
        return normalized;
    }
    return std::nullopt;
}

ReasoningEffortResolution resolve_reasoning_effort(
    std::string_view configured,
    const ReasoningCapabilities& capabilities) {
    using Kind = ReasoningEffortResolutionKind;

    if (configured.empty()) {
        return {"provider default", Kind::ProviderDefault};
    }

    if (is_reasoning_off_effort(configured)) {
        if (capabilities.supports(ReasoningCapability::Disable)) {
            return {"off", Kind::Exact};
        }
        if (capabilities.supports(ReasoningCapability::Required)
            && capabilities.supports_effort()) {
            return {"low", Kind::RequiredMinimum};
        }
        return {"provider default", Kind::UnsupportedProviderDefault};
    }

    if (capabilities.supports(ReasoningCapability::ToggleOnly)) {
        return {"on", Kind::Mapped};
    }
    if ((configured == "high" || configured == "max" || configured == "ultra")
        && capabilities.supports(ReasoningCapability::MapsHighToXHigh)) {
        return {"xhigh", Kind::Mapped};
    }

    if (configured == "low"
        && capabilities.supports(ReasoningCapability::MapsLowToHigh)) {
        return {"high", Kind::Mapped};
    }
    if (configured == "medium"
        && capabilities.supports(ReasoningCapability::MapsMediumToHigh)) {
        return {"high", Kind::Mapped};
    }
    if (configured == "minimal"
        && capabilities.supports(ReasoningCapability::MapsMinimalToLow)) {
        return {"low", Kind::Mapped};
    }
    if (configured == "minimal"
        && capabilities.supports(ReasoningCapability::MapsMinimalToOff)) {
        return {"off", Kind::Mapped};
    }
    if (configured == "xhigh"
        && capabilities.supports(ReasoningCapability::MapsXHighToMax)) {
        return {"max", Kind::Mapped};
    }
    if (configured == "xhigh"
        && capabilities.supports(ReasoningCapability::MapsXHighToHigh)) {
        return {"high", Kind::Mapped};
    }
    if (configured == "max"
        && !capabilities.supports(ReasoningCapability::MaxEffort)) {
        return {"high", Kind::UnsupportedFallback};
    }
    if (configured == "xhigh"
        && !capabilities.supports(ReasoningCapability::XHighEffort)) {
        return {"high", Kind::UnsupportedFallback};
    }
    if (configured == "ultra"
        && !capabilities.supports(ReasoningCapability::UltraEffort)) {
        return {
            capabilities.supports(ReasoningCapability::MaxEffort) ? "max" : "high",
            Kind::UnsupportedFallback};
    }

    return {std::string(configured), Kind::Exact};
}

} // namespace core::llm
