#pragma once

#include <cstdint>
#include <string_view>

namespace core::llm {

struct ChatRequest;

/// How a model expects its tools to be presented. Codex publishes this as
/// `tool_mode` on the model row. It is a property of the model, not of the
/// serving endpoint: the same slug is trained against the same harness on the
/// public API and on the ChatGPT backend.
enum class CatalogToolMode : std::uint8_t {
    Unspecified,
    Direct,
    /// Direct tools and a code-execution tool together. Filo does not turn
    /// that mixed harness on; only `code_mode_only` is wired.
    CodeMode,
    CodeModeOnly,
};

[[nodiscard]] CatalogToolMode catalog_tool_mode_from_string(std::string_view text) noexcept;

[[nodiscard]] constexpr bool is_code_mode_only(CatalogToolMode mode) noexcept {
    return mode == CatalogToolMode::CodeModeOnly;
}

/// Whether the request's model is code_mode_only: the live catalog row when
/// it named a mode, otherwise the curated card.
[[nodiscard]] bool request_uses_code_mode(const ChatRequest& request);

} // namespace core::llm
