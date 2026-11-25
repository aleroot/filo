#include "CatalogToolMode.hpp"

#include "ModelRegistry.hpp"
#include "Models.hpp"

#include "../utils/StringUtils.hpp"

namespace core::llm {

CatalogToolMode catalog_tool_mode_from_string(std::string_view text) noexcept {
    const std::string normalized = core::utils::str::to_lower_ascii_copy(text);
    if (normalized == "direct") return CatalogToolMode::Direct;
    if (normalized == "code_mode") return CatalogToolMode::CodeMode;
    if (normalized == "code_mode_only") return CatalogToolMode::CodeModeOnly;
    return CatalogToolMode::Unspecified;
}

bool request_uses_code_mode(const ChatRequest& request) {
    if (request.catalog_tool_mode.has_value()) {
        return is_code_mode_only(*request.catalog_tool_mode);
    }
    const auto info = ModelRegistry::instance().lookup(request.model);
    return info && is_code_mode_only(info->tool_mode);
}

} // namespace core::llm
