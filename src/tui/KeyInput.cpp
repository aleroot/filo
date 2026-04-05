#include "KeyInput.hpp"

#include <cctype>
#include <charconv>
#include <optional>
#include <string_view>

namespace tui {
namespace {

bool parse_decimal(std::string_view token, int& out) {
    if (token.empty()) {
        return false;
    }
    int value = 0;
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size()) {
        return false;
    }
    out = value;
    return true;
}

bool is_ctrl_modifier(int encoded_modifier) {
    if (encoded_modifier <= 0) {
        return false;
    }
    const int modifier_bits = encoded_modifier - 1;
    return (modifier_bits & 4) != 0;
}

bool is_letter_codepoint(int codepoint, char letter) {
    const unsigned char lower = static_cast<unsigned char>(
        std::tolower(static_cast<unsigned char>(letter)));
    const unsigned char upper = static_cast<unsigned char>(
        std::toupper(static_cast<unsigned char>(letter)));
    return codepoint == static_cast<int>(lower) || codepoint == static_cast<int>(upper);
}

bool is_ctrl_letter_kitty_sequence(std::string_view input, char letter) {
    if (!input.starts_with("\x1B[") || !input.ends_with("u")) {
        return false;
    }

    const std::string_view body = input.substr(2, input.size() - 3);
    const std::size_t semicolon = body.find(';');
    if (semicolon == std::string_view::npos) {
        return false;
    }

    const std::string_view code_token = body.substr(0, semicolon);
    std::string_view modifier_token = body.substr(semicolon + 1);
    if (const std::size_t colon = modifier_token.find(':');
        colon != std::string_view::npos) {
        modifier_token = modifier_token.substr(0, colon);
    }

    int codepoint = 0;
    int modifier = 0;
    if (!parse_decimal(code_token, codepoint) || !parse_decimal(modifier_token, modifier)) {
        return false;
    }

    return is_letter_codepoint(codepoint, letter) && is_ctrl_modifier(modifier);
}

bool is_ctrl_letter_modify_other_keys_sequence(std::string_view input, char letter) {
    if (!input.starts_with("\x1B[27;") || !input.ends_with("~")) {
        return false;
    }

    const std::string_view body = input.substr(5, input.size() - 6);
    const std::size_t semicolon = body.find(';');
    if (semicolon == std::string_view::npos) {
        return false;
    }

    const std::string_view modifier_token = body.substr(0, semicolon);
    const std::string_view key_token = body.substr(semicolon + 1);

    int modifier = 0;
    int key_code = 0;
    if (!parse_decimal(modifier_token, modifier) || !parse_decimal(key_token, key_code)) {
        return false;
    }

    return is_letter_codepoint(key_code, letter) && is_ctrl_modifier(modifier);
}

std::optional<unsigned char> ctrl_letter_control_byte(char letter) {
    const unsigned char lower = static_cast<unsigned char>(
        std::tolower(static_cast<unsigned char>(letter)));
    if (lower < static_cast<unsigned char>('a') || lower > static_cast<unsigned char>('z')) {
        return std::nullopt;
    }
    return static_cast<unsigned char>(1 + (lower - static_cast<unsigned char>('a')));
}

} // namespace

bool is_ctrl_letter_event(const ftxui::Event& event, char letter) {
    const std::string& input = event.input();
    const auto control_byte = ctrl_letter_control_byte(letter);
    if (!control_byte.has_value()) {
        return false;
    }

    // Some terminals can deliver Ctrl+<letter> as raw control bytes.
    if (input.size() == 1 && static_cast<unsigned char>(input[0]) == *control_byte) {
        return true;
    }

    return is_ctrl_letter_kitty_sequence(input, letter)
        || is_ctrl_letter_modify_other_keys_sequence(input, letter);
}

bool is_ctrl_x_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'x');
}

bool is_ctrl_o_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'o');
}

bool is_ctrl_y_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'y');
}

bool is_ctrl_d_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'd');
}

bool is_ctrl_f_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'f');
}

bool is_ctrl_v_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'v');
}

bool is_ctrl_l_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'l');
}

bool is_ctrl_c_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'c');
}

} // namespace tui
