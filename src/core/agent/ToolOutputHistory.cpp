#include "ToolOutputHistory.hpp"

#include "../tools/ToolNames.hpp"
#include "../utils/JsonWriter.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <string>

namespace core::agent::tool_output_history {

namespace {

[[nodiscard]] bool looks_like_error_payload(std::string_view payload) {
    return payload.find("\"error\"") != std::string_view::npos;
}

[[nodiscard]] uint64_t fnv1a64(std::string_view text) noexcept {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

[[nodiscard]] std::string digest_hex(std::string_view text) {
    return std::format("{:016x}", fnv1a64(text));
}

[[nodiscard]] Limits sanitize(Limits limits) {
    if (limits.max_chars == 0) {
        limits.max_chars = 1;
    }
    if (limits.head_chars + limits.tail_chars == 0) {
        limits.head_chars = limits.max_chars;
        limits.tail_chars = 0;
        return limits;
    }

    if (limits.head_chars + limits.tail_chars > limits.max_chars) {
        const auto total = limits.head_chars + limits.tail_chars;
        const auto maxc = limits.max_chars;
        limits.head_chars = (limits.head_chars * maxc) / total;
        limits.tail_chars = maxc - limits.head_chars;
    }
    return limits;
}

[[nodiscard]] std::pair<std::string_view, std::string_view>
split_preview(std::string_view text, Limits limits) {
    limits = sanitize(limits);
    if (text.size() <= limits.max_chars) {
        return {text, {}};
    }

    const std::size_t head = std::min(limits.head_chars, text.size());
    const std::size_t remaining = text.size() - head;
    const std::size_t tail = std::min(limits.tail_chars, remaining);

    std::string_view head_view = text.substr(0, head);
    std::string_view tail_view = tail > 0 ? text.substr(text.size() - tail, tail) : std::string_view{};
    return {head_view, tail_view};
}

} // namespace

Limits limits_for_tool(std::string_view tool_name) {
    if (tool_name == core::tools::names::kReadFile) {
        return Limits{.max_chars = 12 * 1024, .head_chars = 8 * 1024, .tail_chars = 4 * 1024};
    }
    if (tool_name == core::tools::names::kRunTerminalCommand
        || tool_name == core::tools::names::kGrepSearch
        || tool_name == core::tools::names::kFileSearch
        || tool_name == core::tools::names::kListDirectory) {
        return Limits{.max_chars = 10 * 1024, .head_chars = 7 * 1024, .tail_chars = 3 * 1024};
    }
    return Limits{};
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output) {
    return clamp_for_history(tool_name, raw_output, limits_for_tool(tool_name));
}

std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits) {
    limits = sanitize(limits);
    if (raw_output.size() <= limits.max_chars) {
        return std::string(raw_output);
    }

    const bool is_error = looks_like_error_payload(raw_output);
    const auto [head, tail] = split_preview(raw_output, limits);

    core::utils::JsonWriter writer(256 + head.size() + tail.size());
    {
        auto object = writer.object();
        if (is_error) {
            writer.kv_str("error", "Tool output truncated for history (oversized error payload).");
        } else {
            writer.kv_bool("truncated", true);
        }
        writer.comma();
        writer.kv_str("tool", tool_name);
        writer.comma();
        writer.kv_num("original_chars", static_cast<int64_t>(raw_output.size()));
        writer.comma();
        writer.kv_num("kept_chars", static_cast<int64_t>(head.size() + tail.size()));
        writer.comma();
        writer.kv_str("digest_fnv1a64", digest_hex(raw_output));
        writer.comma();
        writer.kv_str("head", head);
        if (!tail.empty()) {
            writer.comma();
            writer.kv_str("tail", tail);
        }
    }
    return std::move(writer).take();
}

} // namespace core::agent::tool_output_history
