#include "SerializedPathRecovery.hpp"

#include "../../utils/JsonWriter.hpp"
#include "../../utils/StringUtils.hpp"

#include <simdjson.h>

#include <array>
#include <charconv>
#include <string>
#include <utility>
#include <vector>

namespace core::tools::read {
namespace {

/// JSON structural characters: always present in serialized argument text,
/// never in real-world paths. The colon is deliberately absent because it is
/// legitimate inside URLs.
constexpr std::string_view kStructural = "\"\\[]{},";

/// Line-window arguments a corrupted serialization may carry inside `path`.
constexpr std::array<std::string_view, 2> kWindowKeys = {"offset_line", "limit_lines"};

[[nodiscard]] bool is_window_key(std::string_view text) noexcept {
    return text == kWindowKeys[0] || text == kWindowKeys[1];
}

/// Whitespace-trimmed runs between structural characters; blank runs dropped.
[[nodiscard]] std::vector<std::string_view> fragments(std::string_view text) {
    std::vector<std::string_view> result;
    std::size_t start = 0;
    while (start <= text.size()) {
        auto end = text.find_first_of(kStructural, start);
        if (end == std::string_view::npos) end = text.size();
        const auto fragment = core::utils::str::trim_ascii_view(text.substr(start, end - start));
        if (!fragment.empty()) result.push_back(fragment);
        start = end + 1;
    }
    return result;
}

/// A fragment names a resource when it has a path separator or an interior
/// extension dot; bare words and numbers (`offset_line`, `9600`) never do.
[[nodiscard]] bool names_resource(std::string_view fragment) noexcept {
    if (fragment.find('/') != std::string_view::npos) return true;
    const auto dot = fragment.find('.');
    return dot != std::string_view::npos && dot != 0 && dot + 1 != fragment.size();
}

/// A path taken verbatim from an embedded array must itself be clean: an array
/// whose entries hold structural characters or edge whitespace is a corrupted
/// serialization that merely happens to parse (`["src/x.cpp", "offset_line\": 9, "]`).
[[nodiscard]] bool is_clean_path(std::string_view text) {
    return !text.empty()
        && text.find_first_of(kStructural) == std::string_view::npos
        && core::utils::str::trim_ascii_view(text).size() == text.size();
}

/// The value half of a `key": 42` pair, e.g. `: 42`. Rejects anything but a
/// complete in-range decimal integer, so overflow is impossible.
[[nodiscard]] std::optional<int> window_value(std::string_view fragment) {
    if (!fragment.starts_with(':')) return std::nullopt;
    const auto digits = core::utils::str::trim_ascii_view(fragment.substr(1));
    int value = 0;
    const auto* const end = digits.data() + digits.size();
    const auto [ptr, error] = std::from_chars(digits.data(), end, value);
    if (digits.empty() || error != std::errc{} || ptr != end) return std::nullopt;
    return value;
}

/// JSON text of a clean path array embedded in @p text, or empty.
[[nodiscard]] std::string embedded_path_array(std::string_view text) {
    simdjson::dom::parser parser;
    simdjson::dom::array items;
    if (parser.parse(simdjson::padded_string{text}).get(items) != simdjson::SUCCESS
        || items.size() == 0) {
        return {};
    }
    for (const auto item : items) {
        std::string_view entry;
        if (item.get(entry) != simdjson::SUCCESS || !is_clean_path(entry)) return {};
    }
    return simdjson::to_string(items);
}

} // namespace

std::optional<Options> recover_serialized_path(std::string_view json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::object args;
    if (parser.parse(simdjson::padded_string{json_args}).get(args) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    std::string_view path;
    if (args["path"].get(path) != simdjson::SUCCESS
        || path.find('"') == std::string_view::npos) {
        return std::nullopt;
    }

    const auto parts = fragments(path);
    std::string recovered_path = embedded_path_array(path);
    if (recovered_path.empty()) {
        for (const auto fragment : parts) {
            if (is_window_key(fragment) || !names_resource(fragment)) continue;
            core::utils::JsonWriter quoted;
            quoted.str(fragment);
            recovered_path = std::move(quoted).take();
            break;
        }
    }
    if (recovered_path.empty()) return std::nullopt;

    std::vector<std::pair<std::string_view, int>> window;
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        simdjson::dom::element explicit_value;
        if (!is_window_key(parts[i])
            || args[parts[i]].get(explicit_value) == simdjson::SUCCESS) {
            continue;
        }
        if (const auto value = window_value(parts[i + 1])) {
            window.emplace_back(parts[i], *value);
        }
    }

    core::utils::JsonWriter rebuilt(json_args.size() + recovered_path.size());
    {
        auto object = rebuilt.object();
        rebuilt.kv_raw("path", recovered_path);
        for (const auto field : args) {
            if (field.key == "path") continue;
            rebuilt.comma().kv_raw(field.key, simdjson::to_string(field.value));
        }
        for (const auto& [key, value] : window) {
            rebuilt.comma().kv_num(key, value);
        }
    }
    auto options = parse_options(std::move(rebuilt).take());
    if (!options) return std::nullopt;
    return std::move(*options);
}

} // namespace core::tools::read
