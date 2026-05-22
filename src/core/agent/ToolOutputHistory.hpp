#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace core::agent::tool_output_history {

struct Context {
    std::string_view tool_arguments;
    std::string_view session_id;
};

struct Limits {
    std::size_t max_chars = 8 * 1024;
    std::size_t head_chars = 5 * 1024;
    std::size_t tail_chars = 3 * 1024;
};

[[nodiscard]] Limits limits_for_tool(std::string_view tool_name);

[[nodiscard]] std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output);

[[nodiscard]] std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    std::string_view compression_mode);

[[nodiscard]] std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    std::string_view compression_mode,
    Context context);

[[nodiscard]] std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits);

[[nodiscard]] std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits,
    std::string_view compression_mode);

[[nodiscard]] std::string clamp_for_history(
    std::string_view tool_name,
    std::string_view raw_output,
    Limits limits,
    std::string_view compression_mode,
    Context context);

} // namespace core::agent::tool_output_history
