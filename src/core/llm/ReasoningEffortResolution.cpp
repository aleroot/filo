#include "ReasoningEffortResolution.hpp"

namespace core::llm {

ReasoningEffortResolution resolve_reasoning_effort(
    std::string_view configured,
    const ReasoningCapabilities& capabilities) {
    using Kind = ReasoningEffortResolutionKind;

    if (configured.empty()) {
        return {"provider default", Kind::ProviderDefault};
    }

    if (configured == "none" || configured == "off" || configured == "disabled") {
        if (capabilities.supports(ReasoningCapability::Disable)) {
            return {"off", Kind::Exact};
        }
        if (capabilities.supports(ReasoningCapability::Required)
            && capabilities.supports_effort()) {
            return {"low", Kind::RequiredMinimum};
        }
        return {"provider default", Kind::UnsupportedProviderDefault};
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
