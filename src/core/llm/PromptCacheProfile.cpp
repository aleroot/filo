#include "PromptCacheProfile.hpp"

#include "../utils/AsciiUtils.hpp"

namespace core::llm {

std::optional<PromptCacheTtl> parse_prompt_cache_ttl(std::string_view value) noexcept {
    if (value.empty()
        || core::utils::ascii::iequals(value, "5m")
        || core::utils::ascii::iequals(value, "standard")) {
        return PromptCacheTtl::Standard;
    }
    if (core::utils::ascii::iequals(value, "1h")
        || core::utils::ascii::iequals(value, "extended")) {
        return PromptCacheTtl::Extended;
    }
    return std::nullopt;
}

} // namespace core::llm
