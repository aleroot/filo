#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace core::context {

[[nodiscard]] bool should_escape_unquoted_mention_path_char(char ch);

[[nodiscard]] std::string escape_unquoted_mention_path(std::string_view path);

[[nodiscard]] std::string format_unquoted_mention(std::string_view path);

[[nodiscard]] std::optional<std::string> pasted_paths_to_mentions(std::string_view pasted);

} // namespace core::context
