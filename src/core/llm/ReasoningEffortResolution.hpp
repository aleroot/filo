#pragma once

#include "ReasoningCapabilities.hpp"

#include <string>
#include <string_view>

namespace core::llm {

enum class ReasoningEffortResolutionKind {
    ProviderDefault,
    Exact,
    RequiredMinimum,
    Mapped,
    UnsupportedFallback,
    UnsupportedProviderDefault,
};

struct ReasoningEffortResolution {
    std::string effective;
    ReasoningEffortResolutionKind kind = ReasoningEffortResolutionKind::Exact;
};

// Resolve a configured effort to the value supported by a model. Capability
// interpretation belongs to the LLM layer so callers only need to present the
// resulting value and resolution kind.
[[nodiscard]] ReasoningEffortResolution resolve_reasoning_effort(
    std::string_view configured,
    const ReasoningCapabilities& capabilities);

} // namespace core::llm
