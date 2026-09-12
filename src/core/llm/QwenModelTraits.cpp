#include "QwenModelTraits.hpp"

#include "core/utils/StringUtils.hpp"

#include <array>
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
    // Qwen Code stamps preserve_thinking on every DashScope chat request for
    // the Qwen family (including coder-model). Restricting it to 3.6+/3.7+
    // left qwen3-coder-plus and qwen3.5-plus unable to keep reasoning across
    // tool turns, which is the common Coding Plan / public DashScope path.
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    return is_qwen_text_model(model_id) || lowered == "coder-model";
}

bool qwen_model_supports_vision(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    if (lowered == "coder-model") return true;
    static constexpr std::array<std::string_view, 8> kPrefixes{
        "qwen-vl",
        "qwen3-vl",
        "qwen3.5-plus",
        "qwen3.6-plus",
        "qwen3.7-plus",
        "qwen3.8-plus",
        "qwen3.8-flash",
        "qwen3.8-max",
    };
    for (const auto prefix : kPrefixes) {
        if (lowered.starts_with(prefix)) return true;
    }
    return false;
}

bool qwen_model_supports_token_plan_hosted_tools(
    std::string_view model_id) {
    const std::vector<int> generation = qwen_model_generation(model_id);
    if (generation >= std::vector<int>{3, 7}) return true;
    if (generation != std::vector<int>{3, 6}) return false;

    const std::string lowered =
        core::utils::str::to_lower_ascii_copy(model_id);
    return lowered.find("-max") != std::string::npos
        || lowered.find("-plus") != std::string::npos
        || lowered.find("-flash") != std::string::npos;
}

bool qwen_model_supports_tiered_effort(std::string_view model_id) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model_id));
    return lowered.starts_with("qwen3.8-max")
        || lowered.starts_with("qwen3.8-flash");
}

} // namespace core::llm
