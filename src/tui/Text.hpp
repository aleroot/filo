#pragma once

#include "Conversation.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace tui {

std::string to_lower_ascii(std::string_view input);
std::string_view trim_ascii(std::string_view input);
std::string compact_single_line(std::string_view input, std::size_t max_len = 96);
bool visibility_setting_enabled(std::string_view value, bool fallback = true);

bool starts_with_line_break(std::string_view text);
bool ends_with_line_break(std::string_view text);

std::string normalize_newlines(std::string input);
bool erase_last_utf8_codepoint(std::string& text);

std::string flatten_whitespace(std::string_view input);
std::string search_role_label(MessageType type);
std::string search_text_for_message(const UiMessage& msg);
std::string build_search_snippet(std::string_view text,
                                 std::size_t match_pos,
                                 std::size_t query_len);

} // namespace tui
