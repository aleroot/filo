#pragma once

#include "core/llm/LLMProvider.hpp"

#include <memory>
#include <optional>
#include <string>

namespace exec {

/// The provider a non-interactive host talks to for this process.
///
/// Prompter mode and the Agent Client Protocol both start from the saved
/// configuration. One function builds that runtime so the two hosts cannot
/// drift on provider registration, the `--model` override, or the default mode.
struct ProviderRuntime {
    std::shared_ptr<core::llm::LLMProvider> provider;
    std::string provider_name;
    std::string model_name;
    std::string default_mode = "BUILD";
    std::string error;
};

[[nodiscard]] ProviderRuntime load_provider_runtime(
    std::optional<std::string> model_override = std::nullopt);

} // namespace exec
