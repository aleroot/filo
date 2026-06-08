#pragma once

#include "../../utils/AsciiUtils.hpp"
#include "../../utils/StringUtils.hpp"

#include <cpr/cpr.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace core::llm::transport {

[[nodiscard]] inline std::optional<std::pair<std::string, std::string>>
parse_header_line(std::string_view line) {
    line = core::utils::str::trim_ascii_view(line);
    if (line.empty()) return std::nullopt;

    const auto colon = line.find(':');
    if (colon == std::string_view::npos) return std::nullopt;

    auto name = core::utils::str::trim_ascii_copy(line.substr(0, colon));
    auto value = core::utils::str::trim_ascii_copy(line.substr(colon + 1));
    if (name.empty()) return std::nullopt;

    return std::pair{std::move(name), std::move(value)};
}

[[nodiscard]] inline std::optional<std::string> find_header(
    const cpr::Header& headers,
    std::string_view name) {
    for (const auto& [key, value] : headers) {
        if (core::utils::ascii::iequals(key, name)) return value;
    }
    return std::nullopt;
}

} // namespace core::llm::transport
