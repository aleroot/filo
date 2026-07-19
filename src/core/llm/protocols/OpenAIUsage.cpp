#include "OpenAIUsage.hpp"

#include <algorithm>
#include <climits>

namespace core::llm::protocols {

namespace {

[[nodiscard]] int32_t token_count(int64_t value) noexcept {
    return static_cast<int32_t>(std::clamp<int64_t>(value, 0, INT32_MAX));
}

[[nodiscard]] bool read_first(simdjson::dom::object object,
                              std::string_view primary,
                              std::string_view fallback,
                              int64_t& value) noexcept {
    return object[primary].get(value) == simdjson::SUCCESS
        || object[fallback].get(value) == simdjson::SUCCESS;
}

} // namespace

bool parse_openai_usage(simdjson::dom::object usage,
                        ParseResult& result) noexcept {
    int64_t prompt = 0;
    int64_t completion = 0;
    const bool has_prompt = read_first(usage, "prompt_tokens", "input_tokens", prompt);
    const bool has_completion = read_first(
        usage, "completion_tokens", "output_tokens", completion);

    if (has_prompt) result.prompt_tokens = token_count(prompt);
    if (has_completion) result.completion_tokens = token_count(completion);

    simdjson::dom::object prompt_details;
    if (usage["prompt_tokens_details"].get(prompt_details) == simdjson::SUCCESS
        || usage["input_tokens_details"].get(prompt_details) == simdjson::SUCCESS) {
        int64_t cached = 0;
        int64_t created = 0;
        (void)prompt_details["cached_tokens"].get(cached);
        (void)prompt_details["cache_creation_input_tokens"].get(created);
        result.cached_prompt_tokens = token_count(cached);
        result.cache_creation_prompt_tokens = token_count(created);
    }

    simdjson::dom::object completion_details;
    if (usage["completion_tokens_details"].get(completion_details) == simdjson::SUCCESS
        || usage["output_tokens_details"].get(completion_details) == simdjson::SUCCESS) {
        int64_t reasoning = 0;
        (void)completion_details["reasoning_tokens"].get(reasoning);
        result.reasoning_tokens = token_count(reasoning);
    }

    return has_prompt || has_completion;
}

} // namespace core::llm::protocols
