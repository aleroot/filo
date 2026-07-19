#pragma once

#include "ApiProtocol.hpp"
#include <simdjson.h>

namespace core::llm::protocols {

/** Parse OpenAI-compatible aggregate and detailed usage fields. */
[[nodiscard]] bool parse_openai_usage(simdjson::dom::object usage,
                                      ParseResult& result) noexcept;

} // namespace core::llm::protocols
