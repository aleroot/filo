#include "ToolCallAssembly.hpp"

#include <algorithm>
#include <cctype>

namespace core::llm {
namespace {

/// True for a payload that is exactly an empty JSON object, modulo whitespace.
[[nodiscard]] bool is_empty_object(std::string_view payload) noexcept {
    bool seen_open = false;
    bool seen_close = false;
    for (const unsigned char ch : payload) {
        if (std::isspace(ch) != 0) continue;
        if (ch == '{' && !seen_open) { seen_open = true; continue; }
        if (ch == '}' && seen_open && !seen_close) { seen_close = true; continue; }
        return false;
    }
    return seen_open && seen_close;
}

} // namespace

void append_arguments_fragment(std::string& accumulated,
                               std::string_view fragment) {
    if (fragment.empty()) return;
    if (!accumulated.empty() && is_empty_object(fragment)) return;
    if (is_empty_object(accumulated)) {
        accumulated.assign(fragment);
        return;
    }
    accumulated.append(fragment);
}

void merge_tool_call_fragment(std::vector<ToolCall>& accumulated,
                              const ToolCall& incoming) {
    for (auto& existing : accumulated) {
        const bool same = (incoming.index != -1 && existing.index == incoming.index)
            || (incoming.index == -1 && !incoming.id.empty()
                && existing.id == incoming.id)
            || (incoming.index == -1 && incoming.id.empty()
                && accumulated.size() == 1);
        if (!same) continue;

        if (!incoming.id.empty()) existing.id = incoming.id;
        if (!incoming.type.empty()) existing.type = incoming.type;
        if (!incoming.function.name.empty()) {
            existing.function.name = incoming.function.name;
        }
        append_arguments_fragment(existing.function.arguments,
                                  incoming.function.arguments);
        return;
    }
    accumulated.push_back(incoming);
}

} // namespace core::llm
