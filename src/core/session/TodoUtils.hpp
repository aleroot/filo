#pragma once

#include "SessionData.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::session::todo {

inline std::string trim_copy(std::string_view value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(start, end - start + 1));
}

inline std::optional<int> parse_positive_int(std::string_view value) {
    if (value.empty()) {
        return std::nullopt;
    }

    int parsed = 0;
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc{} || ptr != value.data() + value.size() || parsed < 1) {
        return std::nullopt;
    }
    return parsed;
}

inline std::optional<std::string> parse_sequence_id(std::string_view id) {
    const std::string trimmed = trim_copy(id);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    if (const auto parsed = parse_positive_int(trimmed); parsed.has_value()) {
        return std::to_string(*parsed);
    }

    if (trimmed.size() > 1 && (trimmed[0] == 't' || trimmed[0] == 'T')) {
        if (const auto parsed = parse_positive_int(std::string_view(trimmed).substr(1));
            parsed.has_value()) {
            return std::to_string(*parsed);
        }
    }

    return std::nullopt;
}

inline std::optional<std::string> strip_braced_id(std::string_view selector) {
    const std::string trimmed = trim_copy(selector);
    if (trimmed.size() >= 2 && trimmed.front() == '{' && trimmed.back() == '}') {
        const std::string inner = trim_copy(
            std::string_view(trimmed).substr(1, trimmed.size() - 2));
        if (!inner.empty()) {
            return inner;
        }
    }
    return std::nullopt;
}

inline std::optional<std::size_t> find_by_id(
    const std::vector<SessionTodoItem>& todos,
    std::string_view selector) {
    const std::string trimmed = trim_copy(selector);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    for (std::size_t i = 0; i < todos.size(); ++i) {
        if (todos[i].id == trimmed) {
            return i;
        }
    }
    return std::nullopt;
}

inline std::optional<std::size_t> resolve_index(
    const std::vector<SessionTodoItem>& todos,
    std::string_view selector) {
    if (const auto explicit_id = strip_braced_id(selector); explicit_id.has_value()) {
        return find_by_id(todos, *explicit_id);
    }

    const std::string trimmed = trim_copy(selector);
    if (trimmed.empty()) {
        return std::nullopt;
    }

    // Prefer exact ID matches first so legacy numeric IDs remain stable even
    // after list reordering or deletions.
    if (const auto id_match = find_by_id(todos, trimmed); id_match.has_value()) {
        return id_match;
    }

    if (const auto parsed_index = parse_positive_int(trimmed); parsed_index.has_value()
        && *parsed_index <= static_cast<int>(todos.size())) {
        return static_cast<std::size_t>(*parsed_index - 1);
    }

    return std::nullopt;
}

inline std::string next_id(const std::vector<SessionTodoItem>& todos) {
    int max_sequence = 0;
    for (const auto& todo : todos) {
        if (const auto parsed = parse_sequence_id(todo.id); parsed.has_value()) {
            if (const auto numeric = parse_positive_int(*parsed); numeric.has_value()) {
                max_sequence = std::max(max_sequence, *numeric);
            }
        }
    }
    return std::format("t{}", max_sequence + 1);
}

} // namespace core::session::todo
