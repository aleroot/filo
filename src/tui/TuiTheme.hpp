#pragma once

#include <utility>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>

namespace tui {

inline const auto ColorYellowDark   = ftxui::Color::RGB(255, 192,  32);
inline const auto ColorYellowBright = ftxui::Color::RGB(255, 246,  61);
inline const auto ColorWarn         = ftxui::Color::RGB(255, 120,  30);
inline const auto ColorToolDone     = ftxui::Color::RGB(255, 221,  92);
inline const auto ColorToolFail     = ftxui::Color::RGB(255, 150,  92);
inline const auto ColorToolPending  = ftxui::Color::RGB(255, 205, 110);

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

inline ftxui::Element UiWindow(ftxui::Element title, ftxui::Element content) {
    return ftxui::window(std::move(title), std::move(content), UiBorderStyle);
}

} // namespace tui
