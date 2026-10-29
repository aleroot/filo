#include "MiniMaxPromptCache.hpp"

#include "core/utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <string_view>
#include <vector>

namespace core::llm::protocols {
namespace {

constexpr std::string_view kMarker = R"(,"cache_control":{"type":"ephemeral"})";

struct Insertion {
    std::size_t offset = 0;
    std::string_view text;
};

/// Raw JSON of an array's last element; empty for an empty array.
[[nodiscard]] std::string_view last_element(simdjson::ondemand::array array) {
    std::string_view last;
    for (auto element : array) last = element.raw_json();
    return core::utils::str::trim_ascii_view(last);
}

/// Where the markers go, in payload order. Views into @p payload stay valid
/// because nothing is modified until every anchor is found.
[[nodiscard]] std::vector<Insertion> find_breakpoints(std::string& payload) {
    simdjson::ondemand::parser parser;
    auto document = parser.iterate(simdjson::pad_with_reserve(payload));
    const char* const base = payload.data();
    const auto begin_of = [&](std::string_view json) { return static_cast<std::size_t>(json.data() - base); };
    const auto end_of = [&](std::string_view json) { return begin_of(json) + json.size(); };

    std::vector<Insertion> insertions;
    // A marker is the last member of the block it closes.
    const auto mark_block = [&](std::string_view block) {
        if (block.ends_with('}')) insertions.push_back({end_of(block) - 1, kMarker});
    };

    for (auto field : document.get_object()) {
        const std::string_view key = field.unescaped_key();
        simdjson::ondemand::value value = field.value();
        if ((key == "system" || key == "tools") && value.type() == simdjson::ondemand::json_type::array) {
            mark_block(last_element(value.get_array()));
        } else if (key == "messages") {
            bool last_is_user = false;
            bool text_content = false;
            std::string_view last_block;
            for (simdjson::ondemand::object message : value.get_array()) {
                last_is_user = false;
                last_block = {};
                for (auto member : message) {
                    const std::string_view name = member.unescaped_key();
                    simdjson::ondemand::value content = member.value();
                    if (name == "role") {
                        last_is_user = std::string_view(content.get_string()) == "user";
                    } else if (name == "content") {
                        text_content = content.type() == simdjson::ondemand::json_type::string;
                        last_block = text_content ? core::utils::str::trim_ascii_view(content.raw_json_token())
                                                  : last_element(content.get_array());
                    }
                }
            }
            if (!last_is_user || last_block.empty()) continue;
            if (!text_content) {
                mark_block(last_block);
                continue;
            }
            // String content becomes the one text block that can carry a marker.
            insertions.push_back({begin_of(last_block), R"([{"type":"text","text":)"});
            insertions.push_back({end_of(last_block), kMarker});
            insertions.push_back({end_of(last_block), "}]"});
        }
    }
    return insertions;
}

} // namespace

void add_minimax_cache_breakpoints(std::string& payload) {
    std::vector<Insertion> insertions;
    try {
        insertions = find_breakpoints(payload);
    } catch (const simdjson::simdjson_error&) {
        return;
    }
    std::ranges::stable_sort(insertions, {}, &Insertion::offset);
    std::string marked;
    std::size_t extra = 0;
    for (const auto& insertion : insertions) extra += insertion.text.size();
    marked.reserve(payload.size() + extra);
    std::size_t copied = 0;
    for (const auto& insertion : insertions) {
        marked.append(payload, copied, insertion.offset - copied);
        marked += insertion.text;
        copied = insertion.offset;
    }
    marked.append(payload, copied);
    payload = std::move(marked);
}

} // namespace core::llm::protocols
