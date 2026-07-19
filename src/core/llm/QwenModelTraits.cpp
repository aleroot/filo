#include "QwenModelTraits.hpp"

#include "core/utils/StringUtils.hpp"

#include <limits>

namespace core::llm {

std::vector<int> qwen_model_generation(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    if (!lowered.starts_with("qwen")) return {};

    std::size_t pos = 4;
    std::vector<int> generation;
    while (pos < lowered.size()) {
        if (lowered[pos] < '0' || lowered[pos] > '9') break;
        int component = 0;
        while (pos < lowered.size() && lowered[pos] >= '0' && lowered[pos] <= '9') {
            if (component <= (std::numeric_limits<int>::max() - 9) / 10) {
                component = component * 10 + (lowered[pos] - '0');
            }
            ++pos;
        }
        generation.push_back(component);
        if (pos >= lowered.size() || lowered[pos] != '.') break;
        ++pos;
    }
    return generation;
}

bool is_qwen_text_model(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    return lowered.starts_with("qwen")
        && lowered.find("image") == std::string::npos
        && lowered.find("audio") == std::string::npos
        && lowered.find("embedding") == std::string::npos
        && lowered.find("rerank") == std::string::npos;
}

bool qwen_model_supports_preserve_thinking(std::string_view model_id) {
    const std::vector<int> generation = qwen_model_generation(model_id);
    if (generation >= std::vector<int>{3, 7}) return true;
    if (generation != std::vector<int>{3, 6}) return false;

    const std::string lowered = core::utils::str::to_lower_ascii_copy(model_id);
    return lowered.find("-max") != std::string::npos
        || lowered.find("-plus") != std::string::npos;
}

} // namespace core::llm
