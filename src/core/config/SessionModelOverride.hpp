#pragma once

#include "ConfigManager.hpp"

#include <expected>
#include <string>
#include <string_view>

namespace core::config {

// Applies a process-scoped selector to an AppConfig copy. Accepted forms are
// MODEL, PROVIDER, PROVIDER/MODEL, router, and auto. No files are modified.
[[nodiscard]] std::expected<void, std::string> apply_session_model_override(
    AppConfig& config,
    std::string_view selector);

} // namespace core::config
