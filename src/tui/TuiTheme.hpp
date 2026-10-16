#pragma once

#include <atomic>
#include <cstdint>
#include <utility>
#include <variant>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

namespace tui {

// ---------------------------------------------------------------------------
// Theme
//
// Filo's palette was designed for a dark terminal. On a light background the
// same inks (white body text, saturated yellows) all but disappear, so every
// colour that is drawn ON the terminal background carries a second, light-theme
// value. The Dark value of every colour below is the original, hand-picked one:
// with Theme::Dark (the default) rendering is unchanged.
//
// The theme is process-wide and resolved when a colour is converted to
// ftxui::Color, i.e. at render time, so switching it repaints everything on the
// next frame. Long-lived caches of rendered output must include active_theme()
// in their key.
// ---------------------------------------------------------------------------

enum class Theme : std::uint8_t { Dark, Light };

namespace detail {
// Written by the UI thread, read by any thread that builds elements.
inline std::atomic<Theme> g_active_theme{Theme::Dark};
} // namespace detail

[[nodiscard]] inline Theme active_theme() noexcept {
    return detail::g_active_theme.load(std::memory_order_relaxed);
}

inline void set_active_theme(Theme theme) noexcept {
    detail::g_active_theme.store(theme, std::memory_order_relaxed);
}

struct RgbColor {
    unsigned char red;
    unsigned char green;
    unsigned char blue;

    friend constexpr bool operator==(const RgbColor&, const RgbColor&) = default;

    operator ftxui::Color() const {
        return ftxui::Color::RGB(red, green, blue);
    }
};

// A colour that can live in a constexpr table. ftxui::Color is not a literal
// type, so palette entries name either the terminal default, one of the 16
// terminal palette slots (which the user's terminal theme already adapts), or
// an exact RGB value.
class ColorSpec {
public:
    constexpr ColorSpec(ftxui::Color::Palette1 index) noexcept : value_(index) {}
    constexpr ColorSpec(ftxui::Color::Palette16 index) noexcept : value_(index) {}
    constexpr ColorSpec(RgbColor rgb) noexcept : value_(rgb) {}

    friend constexpr bool operator==(const ColorSpec&, const ColorSpec&) = default;

    // The exact value when this names an RGB colour; null for palette slots,
    // whose rendering depends on the terminal.
    [[nodiscard]] constexpr const RgbColor* rgb() const noexcept {
        return std::get_if<RgbColor>(&value_);
    }

    operator ftxui::Color() const {
        return std::visit([](auto value) { return ftxui::Color(value); }, value_);
    }

private:
    std::variant<ftxui::Color::Palette1, ftxui::Color::Palette16, RgbColor> value_;
};

// A colour with one value per theme, resolved against active_theme().
//
// Text needs 4.5:1 against the background, but borders, bars and other
// non-text glyphs only 3:1 (WCAG 1.4.11). On a light background that gap is
// the difference between olive and gold, so a colour may carry a brighter
// `light_graphic` value for those uses. It defaults to `light`; the dark theme
// has a single value for both.
struct ThemedColor {
    ColorSpec dark;
    ColorSpec light;
    ColorSpec light_graphic = light;

    [[nodiscard]] constexpr const ColorSpec& resolve(Theme theme) const noexcept {
        return theme == Theme::Light ? light : dark;
    }

    [[nodiscard]] constexpr const ColorSpec& resolve_graphic(Theme theme) const noexcept {
        return theme == Theme::Light ? light_graphic : dark;
    }

    operator ftxui::Color() const {
        return resolve(active_theme());
    }
};

// The value of `color` for borders and other non-text glyphs.
[[nodiscard]] inline ftxui::Color graphic(const ThemedColor& color) {
    return color.resolve_graphic(active_theme());
}

// ---------------------------------------------------------------------------
// Fills — backgrounds under black text. A saturated fill behind black text
// reads on any terminal background, so fills do not vary with the theme.
// ---------------------------------------------------------------------------

inline constexpr RgbColor FillYellowDark{255, 192, 32};
inline constexpr RgbColor FillYellowBright{255, 246, 61};
inline constexpr RgbColor FillWarn{255, 120, 30};

// ---------------------------------------------------------------------------
// Inks — text, borders and glyphs drawn on the terminal background. Light
// values keep the hue of the dark ones and reach WCAG AA (>= 4.5:1) on white
// and on a cream background such as Solarized Light's (pinned by
// tests/test_tui_theme.cpp).
// ---------------------------------------------------------------------------

// Text-strength yellow cannot stay yellow on white (every hue that reaches
// 4.5:1 is amber or olive), so the light text values lean warm, towards gold,
// and the graphic values carry the colour.
inline constexpr ThemedColor ColorYellowDark{FillYellowDark, RgbColor{160, 98, 0}, RgbColor{202, 121, 0}};
inline constexpr ThemedColor ColorYellowBright{FillYellowBright, RgbColor{148, 103, 0}, RgbColor{191, 127, 0}};
// The banner's block letters. A logotype is exempt from contrast minimums, so
// on a light background it keeps a saturated gold; its outline glyphs use
// ColorYellowBright and read as a darker edge.
inline constexpr ThemedColor ColorLogo{FillYellowBright, RgbColor{236, 170, 0}};
inline constexpr ThemedColor ColorWarn{FillWarn, RgbColor{186, 72, 0}};
inline constexpr ThemedColor ColorToolDone{RgbColor{255, 221, 92}, RgbColor{148, 103, 0}};
inline constexpr ThemedColor ColorToolFail{RgbColor{255, 150, 92}, RgbColor{184, 64, 20}};
inline constexpr ThemedColor ColorToolPending{RgbColor{255, 205, 110}, RgbColor{156, 96, 0}};
inline constexpr ThemedColor ColorQuestionCyan{RgbColor{0, 180, 220}, RgbColor{0, 118, 158}};

// Body text. On a light terminal the terminal's own foreground is the most
// readable choice there is.
inline constexpr ThemedColor ColorText{ftxui::Color::White, ftxui::Color::Default};
// Supporting text: descriptions, values next to a label.
inline constexpr ThemedColor ColorTextSecondary{ftxui::Color::GrayLight, RgbColor{72, 72, 72}};
// Hints, metadata, separators.
inline constexpr ThemedColor ColorTextMuted{ftxui::Color::GrayDark, RgbColor{104, 104, 104}};
// Background of a selected row whose text keeps its own colours.
inline constexpr ThemedColor FillSelection{ftxui::Color::GrayDark, RgbColor{224, 224, 224}};

// Unified-diff lines.
inline constexpr ThemedColor ColorDiffAdd{RgbColor{120, 220, 130}, RgbColor{16, 120, 40}};
inline constexpr ThemedColor ColorDiffDelete{RgbColor{255, 150, 130}, RgbColor{185, 45, 30}};
inline constexpr ThemedColor ColorDiffHunk{RgbColor{145, 196, 255}, RgbColor{30, 90, 190}};

#if defined(__APPLE__)
inline constexpr auto UiBorderStyle = ftxui::LIGHT;
#else
inline constexpr auto UiBorderStyle = ftxui::ROUNDED;
#endif

inline ftxui::Decorator UiBorder() {
    return ftxui::borderStyled(UiBorderStyle);
}

inline ftxui::Decorator UiBorder(ftxui::Color color) {
    return ftxui::borderStyled(UiBorderStyle, color);
}

inline ftxui::Decorator UiBorder(const ThemedColor& color) {
    return UiBorder(graphic(color));
}

inline ftxui::Element UiWindow(ftxui::Element title, ftxui::Element content) {
    return ftxui::window(std::move(title), std::move(content), UiBorderStyle);
}

} // namespace tui
