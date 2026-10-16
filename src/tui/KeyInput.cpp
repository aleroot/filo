#include "KeyInput.hpp"

#include <cctype>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace tui {

bool is_panel_dismiss_event(const ftxui::Event& event) {
    return event == ftxui::Event::Escape
        || event == ftxui::Event::Character('q')
        || event == ftxui::Event::Character('Q');
}

bool is_refresh_notification(const ftxui::Event& event) {
    return event == ftxui::Event::Custom;
}

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

bool is_shift_modifier(int encoded_modifier) {
    return encoded_modifier > 0 && ((encoded_modifier - 1) & 1) != 0;
}

bool is_letter_codepoint(int codepoint, char letter) {
    const unsigned char lower = static_cast<unsigned char>(
        std::tolower(static_cast<unsigned char>(letter)));
    const unsigned char upper = static_cast<unsigned char>(
        std::toupper(static_cast<unsigned char>(letter)));
    return codepoint == static_cast<int>(lower) || codepoint == static_cast<int>(upper);
}

/// The encoded modifier a kitty keyboard protocol sequence (ESC [ code ; mod u)
/// reports for `letter`, or nullopt when it is not that letter.
std::optional<int> kitty_letter_modifier(std::string_view input, char letter) {
    if (!input.starts_with("\x1B[") || !input.ends_with("u")) {
        return std::nullopt;
    }

    const std::string_view body = input.substr(2, input.size() - 3);
    const std::size_t semicolon = body.find(';');
    if (semicolon == std::string_view::npos) {
        return std::nullopt;
    }

    const std::string_view code_token = body.substr(0, semicolon);
    std::string_view modifier_token = body.substr(semicolon + 1);
    if (const std::size_t colon = modifier_token.find(':');
        colon != std::string_view::npos) {
        modifier_token = modifier_token.substr(0, colon);
    }

    int codepoint = 0;
    int modifier = 0;
    if (!parse_decimal(code_token, codepoint) || !parse_decimal(modifier_token, modifier)
        || !is_letter_codepoint(codepoint, letter)) {
        return std::nullopt;
    }
    return modifier;
}

// Parse a modifyOtherKeys CSI sequence: ESC [ 27 ; <modifier> ; <key> ~
// Returns (key_code, modifier) on success.
std::optional<std::pair<int, int>>
parse_modify_other_keys(std::string_view input) {
    if (!input.starts_with("\x1B[27;") || !input.ends_with("~")) {
        return std::nullopt;
    }

    const std::string_view body = input.substr(5, input.size() - 6);
    const std::size_t semicolon = body.find(';');
    if (semicolon == std::string_view::npos) {
        return std::nullopt;
    }

    int modifier = 0;
    int key_code = 0;
    if (!parse_decimal(body.substr(0, semicolon), modifier)
        || !parse_decimal(body.substr(semicolon + 1), key_code)) {
        return std::nullopt;
    }

    return std::make_pair(key_code, modifier);
}

/// The encoded modifier a modifyOtherKeys sequence reports for `letter`, or
/// nullopt when it is not that letter.
std::optional<int> modify_other_keys_letter_modifier(std::string_view input, char letter) {
    const auto parsed = parse_modify_other_keys(input);
    if (!parsed || !is_letter_codepoint(parsed->first, letter)) {
        return std::nullopt;
    }
    return parsed->second;
}

/// The encoded modifier either enhanced-keyboard protocol reports for
/// `letter`. Only these protocols can report Shift alongside Ctrl: a legacy
/// terminal sends the same control byte for Ctrl+X and Ctrl+Shift+X.
std::optional<int> enhanced_letter_modifier(std::string_view input, char letter) {
    if (auto modifier = kitty_letter_modifier(input, letter)) {
        return modifier;
    }
    return modify_other_keys_letter_modifier(input, letter);
}

bool is_ctrl_letter_enhanced_sequence(std::string_view input, char letter) {
    const auto modifier = enhanced_letter_modifier(input, letter);
    return modifier.has_value() && is_ctrl_modifier(*modifier);
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

    return is_ctrl_letter_enhanced_sequence(input, letter);
}

bool is_ctrl_shift_letter_event(const ftxui::Event& event, char letter) {
    const auto modifier = enhanced_letter_modifier(event.input(), letter);
    return modifier.has_value() && is_ctrl_modifier(*modifier) && is_shift_modifier(*modifier);
}

bool is_ctrl_x_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'x') && !is_ctrl_shift_letter_event(event, 'x');
}

bool is_ctrl_shift_x_event(const ftxui::Event& event) {
    return is_ctrl_shift_letter_event(event, 'x');
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

bool is_ctrl_g_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'g');
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

bool is_ctrl_p_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'p');
}

bool is_ctrl_r_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'r');
}

bool is_ctrl_t_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 't');
}

bool is_ctrl_n_event(const ftxui::Event& event) {
    return is_ctrl_letter_event(event, 'n');
}

bool is_ctrl_j_event(const ftxui::Event& event) {
    // FTXUI defines Event::Return as Special({10}), which is the same raw byte
    // as classic Ctrl+J (LF). Matching the raw control byte would make every
    // Enter open the thread browser. Only accept unambiguous encoded forms:
    // kitty keyboard protocol and modifyOtherKeys CSI sequences.
    const std::string& input = event.input();
    return is_ctrl_letter_enhanced_sequence(input, 'j');
}

bool is_ctrl_h_event(const ftxui::Event& event) {
    // Raw Ctrl+H is byte 8, which FTXUI deliberately normalizes to Backspace.
    // Only enhanced keyboard protocols can distinguish the binding without
    // stealing normal text editing.
    const std::string& input = event.input();
    return is_ctrl_letter_enhanced_sequence(input, 'h');
}

bool is_ctrl_enter_event(const ftxui::Event& event) {
    // modifyOtherKeys CSI: ESC[27;<modifier>;<key>~  where key 13 = Enter.
    const auto parsed = parse_modify_other_keys(event.input());
    if (!parsed) {
        return false;
    }
    const auto [key_code, modifier] = *parsed;
    return key_code == 13 && is_ctrl_modifier(modifier);
}

} // namespace tui
