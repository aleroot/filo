#pragma once

#include "../llm/Models.hpp"

#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace core::agent {

inline constexpr std::string_view kRepositoryContextMessageName =
    "filo_repository_context";
inline constexpr std::string_view kLegacyRepositoryContextMessageName =
    "filo.repository_context";
inline constexpr std::string_view kRepositoryContextStatePrefix =
    "filo.repository_context.v1:";

[[nodiscard]] inline bool is_repository_context_message(
    const core::llm::Message& message) noexcept {
    return message.synthetic
        && (message.name == kRepositoryContextMessageName
            || message.name == kLegacyRepositoryContextMessageName);
}

[[nodiscard]] inline std::string encode_repository_context_state(
    std::string_view status,
    std::string_view tree) {
    std::string encoded(kRepositoryContextStatePrefix);
    encoded += std::to_string(status.size());
    encoded += ':';
    encoded += std::to_string(tree.size());
    encoded += ':';
    encoded += status;
    encoded += tree;
    return encoded;
}

[[nodiscard]] inline std::optional<std::pair<std::string, std::string>>
decode_repository_context_state(std::string_view encoded) {
    if (!encoded.starts_with(kRepositoryContextStatePrefix)) return std::nullopt;
    encoded.remove_prefix(kRepositoryContextStatePrefix.size());

    const auto parse_size = [&encoded]() -> std::optional<std::size_t> {
        const auto separator = encoded.find(':');
        if (separator == std::string_view::npos) return std::nullopt;
        std::size_t value = 0;
        const auto [end, error] = std::from_chars(
            encoded.data(), encoded.data() + separator, value);
        if (error != std::errc{} || end != encoded.data() + separator) {
            return std::nullopt;
        }
        encoded.remove_prefix(separator + 1);
        return value;
    };

    const auto status_size = parse_size();
    const auto tree_size = parse_size();
    if (!status_size || !tree_size
        || *status_size > encoded.size()
        || *tree_size > encoded.size() - *status_size
        || *status_size + *tree_size != encoded.size()) {
        return std::nullopt;
    }

    return std::pair{
        std::string(encoded.substr(0, *status_size)),
        std::string(encoded.substr(*status_size, *tree_size)),
    };
}

} // namespace core::agent
