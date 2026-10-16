#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string_view>

#include "TuiTheme.hpp"

namespace tui {

// ---------------------------------------------------------------------------
// Terminal background detection
//
// FTXUI exposes no background colour, so Filo asks the terminal itself, once,
// before the UI takes over the tty:
//
//   OSC 11 ; ? ST   "what is your background colour?"  -> OSC 11 ; rgb:R/G/B ST
//   CSI c           primary device attributes (DA1)    -> CSI ? ... c
//
// Every terminal answers DA1 and replies arrive in request order, so the DA1
// reply marks the end of the exchange: a terminal that ignores OSC 11 costs one
// round trip, not the whole timeout. When the terminal cannot tell, the
// rxvt/Konsole COLORFGBG convention is consulted, and Dark — the palette Filo
// was designed for — remains the final answer.
// ---------------------------------------------------------------------------

enum class ThemePreference : std::uint8_t { Auto, Dark, Light };

struct ThemePreferenceOption {
    ThemePreference preference;
    std::string_view token;
    std::string_view label;
};

// The single list of `ui_theme` values, in the order the settings pane shows
// them. The first entry is the default.
[[nodiscard]] constexpr std::array<ThemePreferenceOption, 3> theme_preference_options() noexcept {
    return {{
        {ThemePreference::Auto, "auto", "Auto (match terminal)"},
        {ThemePreference::Dark, "dark", "Dark"},
        {ThemePreference::Light, "light", "Light"},
    }};
}

// Case-insensitive; an empty value is Auto. nullopt for an unknown token.
[[nodiscard]] std::optional<ThemePreference> parse_theme_preference(std::string_view value) noexcept;

// WCAG 2.x relative luminance (0 = black, 1 = white) and contrast ratio.
[[nodiscard]] double relative_luminance(RgbColor color) noexcept;
[[nodiscard]] double contrast_ratio(RgbColor first, RgbColor second) noexcept;

// Light when dark text contrasts better with `background` than light text.
[[nodiscard]] Theme theme_for_background(RgbColor background) noexcept;

// Extracts the colour from an OSC 11 reply found anywhere in `input`
// (`ESC ] 11 ; rgb:RRRR/GGGG/BBBB` or `rgba:...`, 1–4 hex digits per channel,
// terminated by BEL or ST). nullopt when no complete, well-formed reply.
[[nodiscard]] std::optional<RgbColor> parse_background_color_reply(std::string_view input) noexcept;

// True once `input` holds a complete DA1 reply (`ESC [ ? <params> c`).
[[nodiscard]] bool contains_device_attributes_reply(std::string_view input) noexcept;

// Interprets COLORFGBG ("fg;bg" or "fg;xpm;bg"): background palette slots 0–6
// and 8 are dark, 7 and 9–15 are light. nullopt when unset or unrecognised.
[[nodiscard]] std::optional<Theme> theme_from_colorfgbg(std::string_view value) noexcept;

// theme_from_colorfgbg on this process's COLORFGBG. No terminal I/O, so it is
// safe while the UI owns the tty.
[[nodiscard]] std::optional<Theme> theme_from_environment() noexcept;

inline constexpr std::chrono::milliseconds kBackgroundQueryTimeout{200};

// Sends the OSC 11 + DA1 queries on `tty_fd` (raw mode for the duration, the
// previous termios restored afterwards) and parses the answer. Must run before
// anything else reads the terminal: bytes typed during the exchange are
// consumed with the replies. nullopt when `tty_fd` is not a terminal, the
// terminal does not report a background, or `timeout` expires.
[[nodiscard]] std::optional<RgbColor> query_terminal_background(
    int tty_fd,
    std::chrono::milliseconds timeout = kBackgroundQueryTimeout);

// The controlling terminal's theme: OSC 11 through /dev/tty, then COLORFGBG.
// nullopt when neither knows.
[[nodiscard]] std::optional<Theme> detect_terminal_theme(
    std::chrono::milliseconds timeout = kBackgroundQueryTimeout);

// An explicit preference wins; Auto takes the detected theme, else Dark.
[[nodiscard]] constexpr Theme resolve_theme(ThemePreference preference,
                                            std::optional<Theme> detected) noexcept {
    switch (preference) {
        case ThemePreference::Dark:
            return Theme::Dark;
        case ThemePreference::Light:
            return Theme::Light;
        case ThemePreference::Auto:
            break;
    }
    return detected.value_or(Theme::Dark);
}

} // namespace tui
