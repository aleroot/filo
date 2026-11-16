#include "QwenRequestBudget.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/QwenModelTraits.hpp"

#include "core/utils/StringUtils.hpp"

#include <algorithm>

namespace core::llm::providers {
namespace {

[[nodiscard]] bool qwen4_max_model(std::string_view model) {
    auto id = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_view(model));
    if (const auto slash = id.rfind('/'); slash != std::string::npos) id.erase(0, slash + 1);
    return id.starts_with("qwen4-max");
}

} // namespace
std::size_t QwenRequestBudget::estimate_history(std::span<const Message> messages) noexcept {
    // A conservative fallback, not a tokenizer: UTF-8 and code/tool-heavy
    // text need more headroom than chars/4. Never count base64 bytes as text.
    const auto text_tokens = [](std::string_view text) {
        std::size_t ascii = 0, non_ascii = 0;
        for (const unsigned char ch : text) {
            if (ch < 128) ++ascii;
            else ++non_ascii;
        }
        return (ascii + 2) / 3 + (non_ascii + 1) / 2;
    };
    std::size_t tokens = 4096; // Framing plus schemas unavailable to compaction.
    for (const auto& message : messages) {
        tokens += 8 + text_tokens(message.content) + text_tokens(message.reasoning_content);
        for (const auto& tool : message.tool_calls) {
            tokens += 16 + text_tokens(tool.function.name) + text_tokens(tool.function.arguments);
        }
        for (const auto& part : message.content_parts) {
            if (part.type == ContentPartType::Image) tokens += 4096;
            else if (part.type == ContentPartType::Video) tokens += 32768;
            else if (part.type == ContentPartType::Text
                     && message.content.find(part.text) == std::string::npos) {
                tokens += text_tokens(part.text);
            }
        }
    }
    return tokens;
}

std::size_t QwenRequestBudget::estimate_request(const ChatRequest& request) {
    std::size_t tokens = estimate_history(request.messages);
    for (const auto& tool : request.tools) {
        // Include complete JSON schemas, nested properties and descriptions.
        tokens += (core::tools::schema::canonical_input_schema(tool.function).size()
                   + tool.function.name.size() + tool.function.description.size() + 2) / 3 + 16;
    }
    return tokens;
}

void QwenRequestBudget::prepare(ChatRequest& request, const ModelInfo* model,
                                std::size_t calibrated_prompt_tokens) {
    const int endpoint_window = qwen_endpoint_context_window(request.model);
    const int window = model && model->context_window > 0 ? model->context_window
        : endpoint_window > 0 ? endpoint_window : get_max_context_size(request.model);
    const int output_limit = model ? model->max_output_tokens : 0;
    // Other Qwen models default below their advertised output so a short turn
    // does not reserve the whole cap. Qwen4-Max shares max_tokens with its
    // thinking budget (server default 262144); the 32k/64k ceilings cut that
    // off. The endpoint cap below still bounds an explicit request.
    const bool combined_output = qwen4_max_model(request.model);
    const int unset_ceiling = output_limit > 0
        ? (combined_output ? output_limit : std::min(output_limit, 64000))
        : (combined_output ? 262144 : 32000);
    int ceiling = request.max_tokens.value_or(unset_ceiling);
    if (output_limit > 0) ceiling = std::min(ceiling, output_limit);
    // The gateway's hard cap applies even if a live catalog or an explicit
    // request carries the native vendor's larger output limit.
    if (const int cap = qwen_endpoint_output_limit(request.model); cap > 0) {
        ceiling = std::min(ceiling, cap);
    }
    if (window > 0) {
        const auto prompt = std::max(estimate_request(request), calibrated_prompt_tokens);
        const auto margin = std::max(10000, window / 20);
        const auto room = static_cast<int64_t>(window) - static_cast<int64_t>(prompt) - margin;
        const auto floor = std::clamp<int64_t>(static_cast<int64_t>(window) - static_cast<int64_t>(prompt), 1, 4000);
        ceiling = static_cast<int>(std::min<int64_t>(ceiling, std::max(room, floor)));
    }
    request.max_tokens = std::max(1, ceiling);
}
} // namespace core::llm::providers
