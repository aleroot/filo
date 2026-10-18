#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/terminal.hpp>

#include "tui/Conversation.hpp"
#include "tui/MarkdownRenderer.hpp"
#include "tui/TerminalTheme.hpp"
#include "tui/TuiTheme.hpp"

using namespace std::chrono_literals;

namespace {

// Restores the process-wide theme so a failing test cannot leak Light into
// the rest of the suite.
class ScopedTheme {
public:
    explicit ScopedTheme(tui::Theme theme) : previous_(tui::active_theme()) {
        tui::set_active_theme(theme);
    }
    ~ScopedTheme() { tui::set_active_theme(previous_); }

    ScopedTheme(const ScopedTheme&) = delete;
    ScopedTheme& operator=(const ScopedTheme&) = delete;

private:
    tui::Theme previous_;
};

class ScopedColorSupport {
public:
    explicit ScopedColorSupport(ftxui::Terminal::Color support)
        : previous_(ftxui::Terminal::ColorSupport()) {
        ftxui::Terminal::SetColorSupport(support);
    }
    ~ScopedColorSupport() { ftxui::Terminal::SetColorSupport(previous_); }

    ScopedColorSupport(const ScopedColorSupport&) = delete;
    ScopedColorSupport& operator=(const ScopedColorSupport&) = delete;

private:
    ftxui::Terminal::Color previous_;
};

struct NamedColor {
    std::string_view name;
    const tui::ThemedColor& color;
};

// Every themed ink. Fills are deliberately absent: they never change.
const std::array kThemedInks{
    NamedColor{"ColorYellowDark", tui::ColorYellowDark},
    NamedColor{"ColorYellowBright", tui::ColorYellowBright},
    NamedColor{"ColorWarn", tui::ColorWarn},
    NamedColor{"ColorToolDone", tui::ColorToolDone},
    NamedColor{"ColorToolFail", tui::ColorToolFail},
    NamedColor{"ColorToolPending", tui::ColorToolPending},
    NamedColor{"ColorQuestionCyan", tui::ColorQuestionCyan},
    NamedColor{"ColorTextSecondary", tui::ColorTextSecondary},
    NamedColor{"ColorTextMuted", tui::ColorTextMuted},
    NamedColor{"ColorDiffAdd", tui::ColorDiffAdd},
    NamedColor{"ColorDiffDelete", tui::ColorDiffDelete},
    NamedColor{"ColorDiffHunk", tui::ColorDiffHunk},
};

// A pseudo-terminal: the slave is what Filo talks to, the master plays the
// terminal emulator.
class PseudoTerminal {
public:
    PseudoTerminal() {
        master_ = ::posix_openpt(O_RDWR | O_NOCTTY);
        REQUIRE(master_ >= 0);
        REQUIRE(::grantpt(master_) == 0);
        REQUIRE(::unlockpt(master_) == 0);
        const char* name = ::ptsname(master_);
        REQUIRE(name != nullptr);
        slave_name_ = name;
        slave_ = ::open(name, O_RDWR | O_NOCTTY);
        REQUIRE(slave_ >= 0);
    }
    ~PseudoTerminal() {
        ::close(slave_);
        ::close(master_);
    }

    PseudoTerminal(const PseudoTerminal&) = delete;
    PseudoTerminal& operator=(const PseudoTerminal&) = delete;

    [[nodiscard]] int slave() const noexcept { return slave_; }
    [[nodiscard]] const std::string& slave_name() const noexcept { return slave_name_; }

    // Waits for the DA1 query (always sent last), then sends `reply`.
    std::jthread answer_with(std::string reply) const {
        return std::jthread([master = master_, reply = std::move(reply)] {
            std::string received;
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            while (received.find("\x1b[c") == std::string::npos
                   && std::chrono::steady_clock::now() < deadline) {
                pollfd readable{.fd = master, .events = POLLIN, .revents = 0};
                if (::poll(&readable, 1, 50) <= 0) {
                    continue;
                }
                std::array<char, 128> buffer{};
                const ssize_t count = ::read(master, buffer.data(), buffer.size());
                if (count > 0) {
                    received.append(buffer.data(), static_cast<std::size_t>(count));
                }
            }
            if (!reply.empty()) {
                // Not REQUIRE: Catch2 assertions are not thread-safe. A short
                // write shows up as a failed parse on the test thread.
                static_cast<void>(::write(master, reply.data(), reply.size()));
            }
        });
    }

private:
    int master_ = -1;
    int slave_ = -1;
    std::string slave_name_;
};

} // namespace

TEST_CASE("The dark theme renders the original palette unchanged", "[tui][theme]") {
    const ScopedTheme dark{tui::Theme::Dark};

    // These are the hand-picked values the TUI shipped with before themes
    // existed. Changing one changes how Filo looks on every dark terminal.
    CHECK(ftxui::Color(tui::ColorYellowDark) == ftxui::Color::RGB(255, 192, 32));
    CHECK(ftxui::Color(tui::ColorYellowBright) == ftxui::Color::RGB(255, 246, 61));
    CHECK(ftxui::Color(tui::ColorWarn) == ftxui::Color::RGB(255, 120, 30));
    CHECK(ftxui::Color(tui::ColorToolDone) == ftxui::Color::RGB(255, 221, 92));
    CHECK(ftxui::Color(tui::ColorToolFail) == ftxui::Color::RGB(255, 150, 92));
    CHECK(ftxui::Color(tui::ColorToolPending) == ftxui::Color::RGB(255, 205, 110));
    CHECK(ftxui::Color(tui::ColorQuestionCyan) == ftxui::Color::RGB(0, 180, 220));
    CHECK(ftxui::Color(tui::ColorText) == ftxui::Color(ftxui::Color::White));
    CHECK(ftxui::Color(tui::ColorTextSecondary) == ftxui::Color(ftxui::Color::GrayLight));
    CHECK(ftxui::Color(tui::ColorTextMuted) == ftxui::Color(ftxui::Color::GrayDark));
    CHECK(ftxui::Color(tui::FillSelection) == ftxui::Color(ftxui::Color::GrayDark));
    CHECK(ftxui::Color(tui::ColorDiffAdd) == ftxui::Color::RGB(120, 220, 130));
    CHECK(ftxui::Color(tui::ColorDiffDelete) == ftxui::Color::RGB(255, 150, 130));
    CHECK(ftxui::Color(tui::ColorDiffHunk) == ftxui::Color::RGB(145, 196, 255));

    // The banner logo and every border keep the single dark value.
    CHECK(ftxui::Color(tui::ColorLogo) == ftxui::Color::RGB(255, 246, 61));
    for (const auto& [name, color] : kThemedInks) {
        INFO(name);
        CHECK(tui::graphic(color) == ftxui::Color(color));
    }

    // In the dark theme a fill and its ink are the same colour, as they were.
    CHECK(ftxui::Color(tui::FillYellowDark) == ftxui::Color(tui::ColorYellowDark));
    CHECK(ftxui::Color(tui::FillYellowBright) == ftxui::Color(tui::ColorYellowBright));
    CHECK(ftxui::Color(tui::FillWarn) == ftxui::Color(tui::ColorWarn));
}

TEST_CASE("Themed colours resolve against the active theme at conversion time", "[tui][theme]") {
    REQUIRE(tui::active_theme() == tui::Theme::Dark);
    const ftxui::Color dark_ink = tui::ColorYellowDark;
    {
        const ScopedTheme light{tui::Theme::Light};
        CHECK(ftxui::Color(tui::ColorYellowDark) == ftxui::Color::RGB(160, 98, 0));
        CHECK(ftxui::Color(tui::ColorText) == ftxui::Color(ftxui::Color::Default));
        // Fills are not themed.
        CHECK(ftxui::Color(tui::FillYellowDark) == dark_ink);
    }
    CHECK(ftxui::Color(tui::ColorYellowDark) == dark_ink);
}

TEST_CASE("Light inks meet WCAG AA on white and cream backgrounds", "[tui][theme]") {
    constexpr tui::RgbColor kWhite{255, 255, 255};
    constexpr tui::RgbColor kSolarizedLight{253, 246, 227};
    for (const auto& [name, color] : kThemedInks) {
        INFO(name);
        const tui::RgbColor* light = color.light.rgb();
        REQUIRE(light != nullptr);
        CHECK(tui::contrast_ratio(*light, kWhite) >= 4.5);
        CHECK(tui::contrast_ratio(*light, kSolarizedLight) >= 4.5);
    }
}

TEST_CASE("Light graphic values meet the WCAG non-text minimum", "[tui][theme]") {
    constexpr tui::RgbColor kWhite{255, 255, 255};
    constexpr tui::RgbColor kSolarizedLight{253, 246, 227};
    for (const auto& [name, color] : kThemedInks) {
        INFO(name);
        const tui::RgbColor* light_graphic = color.light_graphic.rgb();
        REQUIRE(light_graphic != nullptr);
        CHECK(tui::contrast_ratio(*light_graphic, kWhite) >= 3.0);
        CHECK(tui::contrast_ratio(*light_graphic, kSolarizedLight) >= 3.0);
    }
}

TEST_CASE("Graphic values default to the text value", "[tui][theme]") {
    const ScopedTheme light{tui::Theme::Light};
    CHECK(tui::graphic(tui::ColorTextMuted) == ftxui::Color(tui::ColorTextMuted));
    CHECK(tui::graphic(tui::ColorYellowBright) == ftxui::Color::RGB(191, 127, 0));
    CHECK(ftxui::Color(tui::ColorYellowBright) == ftxui::Color::RGB(148, 103, 0));
}

TEST_CASE("Fills keep black text readable", "[tui][theme]") {
    constexpr tui::RgbColor kBlack{0, 0, 0};
    for (const tui::RgbColor fill : {tui::FillYellowDark, tui::FillYellowBright, tui::FillWarn}) {
        CHECK(tui::contrast_ratio(fill, kBlack) >= 4.5);
    }
    const tui::RgbColor* light_selection = tui::FillSelection.light.rgb();
    REQUIRE(light_selection != nullptr);
    CHECK(tui::contrast_ratio(*light_selection, kBlack) >= 4.5);
}

TEST_CASE("Contrast follows WCAG 2 relative luminance", "[tui][theme]") {
    CHECK(tui::contrast_ratio({0, 0, 0}, {255, 255, 255}) == 21.0);
    CHECK(tui::contrast_ratio({255, 255, 255}, {0, 0, 0}) == 21.0);
    CHECK(tui::contrast_ratio({90, 90, 90}, {90, 90, 90}) == 1.0);
}

TEST_CASE("The background colour decides the theme", "[tui][theme]") {
    CHECK(tui::theme_for_background({0, 0, 0}) == tui::Theme::Dark);
    CHECK(tui::theme_for_background({255, 255, 255}) == tui::Theme::Light);
    CHECK(tui::theme_for_background({0, 43, 54}) == tui::Theme::Dark);       // Solarized Dark
    CHECK(tui::theme_for_background({253, 246, 227}) == tui::Theme::Light);  // Solarized Light
    CHECK(tui::theme_for_background({40, 40, 40}) == tui::Theme::Dark);      // Gruvbox Dark
    CHECK(tui::theme_for_background({251, 241, 199}) == tui::Theme::Light);  // Gruvbox Light
}

TEST_CASE("OSC 11 replies are parsed in every form terminals send", "[tui][theme]") {
    using tui::parse_background_color_reply;
    using tui::RgbColor;

    SECTION("four hex digits, ST-terminated") {
        CHECK(parse_background_color_reply("\x1b]11;rgb:ffff/ffff/ffff\x1b\\") == RgbColor{255, 255, 255});
    }
    SECTION("BEL-terminated") {
        CHECK(parse_background_color_reply("\x1b]11;rgb:0000/2b2b/3636\x07") == RgbColor{0, 43, 54});
    }
    SECTION("channel widths of one to four digits scale to eight bits") {
        CHECK(parse_background_color_reply("\x1b]11;rgb:f/8/0\x07") == RgbColor{255, 136, 0});
        CHECK(parse_background_color_reply("\x1b]11;rgb:fd/f6/e3\x07") == RgbColor{253, 246, 227});
        CHECK(parse_background_color_reply("\x1b]11;rgb:fff/000/800\x07") == RgbColor{255, 0, 128});
        CHECK(parse_background_color_reply("\x1b]11;rgb:FDFD/F6F6/E3E3\x07") == RgbColor{253, 246, 227});
    }
    SECTION("rgba ignores alpha") {
        CHECK(parse_background_color_reply("\x1b]11;rgba:1e1e/1e1e/2e2e/ffff\x07") == RgbColor{30, 30, 46});
    }
    SECTION("the reply is found among other replies") {
        CHECK(parse_background_color_reply("x\x1b[?62;22c\x1b]11;rgb:ffff/ffff/ffff\x1b\\\x1b[?1;2c")
              == RgbColor{255, 255, 255});
    }
    SECTION("malformed or incomplete replies are rejected") {
        CHECK_FALSE(parse_background_color_reply(""));
        CHECK_FALSE(parse_background_color_reply("\x1b[?1;2c"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;rgb:ffff/ffff/ffff"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;rgb:ffff/ffff\x07"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;rgb:ffff/ffff/ffff/ffff\x07"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;rgb:fffff/0/0\x07"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;rgb:gg/00/00\x07"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;rgb://\x07"));
        CHECK_FALSE(parse_background_color_reply("\x1b]11;#ffffff\x07"));
        CHECK_FALSE(parse_background_color_reply("\x1b]10;rgb:ffff/ffff/ffff\x07"));
    }
}

TEST_CASE("The DA1 reply marks the end of the probe", "[tui][theme]") {
    CHECK(tui::contains_device_attributes_reply("\x1b[?1;2c"));
    CHECK(tui::contains_device_attributes_reply("\x1b]11;rgb:cccc/cccc/cccc\x07\x1b[?62;22;52c"));
    CHECK(tui::contains_device_attributes_reply("\x1b[?6c"));
    CHECK_FALSE(tui::contains_device_attributes_reply(""));
    CHECK_FALSE(tui::contains_device_attributes_reply("\x1b[?1;2"));
    CHECK_FALSE(tui::contains_device_attributes_reply("\x1b]11;rgb:cccc/cccc/cccc\x07"));
    CHECK_FALSE(tui::contains_device_attributes_reply("\x1b[>1;2c"));
}

TEST_CASE("COLORFGBG names the background palette slot last", "[tui][theme]") {
    CHECK(tui::theme_from_colorfgbg("15;0") == tui::Theme::Dark);
    CHECK(tui::theme_from_colorfgbg("0;15") == tui::Theme::Light);
    CHECK(tui::theme_from_colorfgbg("0;default;15") == tui::Theme::Light);
    CHECK(tui::theme_from_colorfgbg("0;7") == tui::Theme::Light);
    CHECK(tui::theme_from_colorfgbg("7;8") == tui::Theme::Dark);
    CHECK_FALSE(tui::theme_from_colorfgbg(""));
    CHECK_FALSE(tui::theme_from_colorfgbg("15"));
    CHECK_FALSE(tui::theme_from_colorfgbg("0;16"));
    CHECK_FALSE(tui::theme_from_colorfgbg("0;default"));
    CHECK_FALSE(tui::theme_from_colorfgbg("0;-1"));
}

TEST_CASE("The ui_theme preference parses and resolves", "[tui][theme]") {
    CHECK(tui::parse_theme_preference("") == tui::ThemePreference::Auto);
    CHECK(tui::parse_theme_preference("auto") == tui::ThemePreference::Auto);
    CHECK(tui::parse_theme_preference("DARK") == tui::ThemePreference::Dark);
    CHECK(tui::parse_theme_preference("light") == tui::ThemePreference::Light);
    CHECK_FALSE(tui::parse_theme_preference("solarized"));

    CHECK(tui::theme_preference_options().front().preference == tui::ThemePreference::Auto);

    using tui::resolve_theme;
    using tui::Theme;
    using tui::ThemePreference;
    CHECK(resolve_theme(ThemePreference::Dark, Theme::Light) == Theme::Dark);
    CHECK(resolve_theme(ThemePreference::Light, Theme::Dark) == Theme::Light);
    CHECK(resolve_theme(ThemePreference::Auto, Theme::Light) == Theme::Light);
    CHECK(resolve_theme(ThemePreference::Auto, std::nullopt) == Theme::Dark);
}

TEST_CASE("The terminal is asked for its background over a tty", "[tui][theme][pty]") {
    PseudoTerminal terminal;
    termios before{};
    REQUIRE(::tcgetattr(terminal.slave(), &before) == 0);

    SECTION("a terminal that reports its background") {
        auto emulator = terminal.answer_with("\x1b]11;rgb:fdfd/f6f6/e3e3\x1b\\\x1b[?62;22c");
        CHECK(tui::query_terminal_background(terminal.slave(), 5s) == tui::RgbColor{253, 246, 227});
    }
    SECTION("a terminal that only answers DA1 is not waited on") {
        auto emulator = terminal.answer_with("\x1b[?1;2c");
        const auto started = std::chrono::steady_clock::now();
        CHECK_FALSE(tui::query_terminal_background(terminal.slave(), 5s));
        CHECK(std::chrono::steady_clock::now() - started < 2s);
    }
    SECTION("a silent terminal times out") {
        auto emulator = terminal.answer_with("");
        CHECK_FALSE(tui::query_terminal_background(terminal.slave(), 50ms));
    }

    termios after{};
    REQUIRE(::tcgetattr(terminal.slave(), &after) == 0);
    // PENDIN is the kernel's own bookkeeping: BSD-derived ttys set it whenever
    // canonical mode is switched back on.
#if defined(PENDIN)
    constexpr tcflag_t kKernelManaged = PENDIN;
#else
    constexpr tcflag_t kKernelManaged = 0;
#endif
    CHECK((after.c_lflag & ~kKernelManaged) == (before.c_lflag & ~kKernelManaged));
    CHECK(after.c_cc[VMIN] == before.c_cc[VMIN]);
    CHECK(after.c_cc[VTIME] == before.c_cc[VTIME]);
}

TEST_CASE("A descriptor that is not a terminal is never queried", "[tui][theme]") {
    std::array<int, 2> pipe_fds{};
    REQUIRE(::pipe(pipe_fds.data()) == 0);
    CHECK_FALSE(tui::query_terminal_background(pipe_fds[1], 50ms));
    ::close(pipe_fds[0]);
    ::close(pipe_fds[1]);
}

// The production path: the pty is the process's controlling terminal and the
// probe reaches it through /dev/tty. That device behaves differently from the
// pty slave itself (macOS poll() rejects it with POLLNVAL), so the direct-fd
// test above cannot stand in for this one.
TEST_CASE("The controlling terminal is probed through /dev/tty", "[tui][theme][pty]") {
    PseudoTerminal terminal;
    auto emulator = terminal.answer_with("\x1b]11;rgb:ffff/ffff/ffff\x07\x1b[?62;22c");

    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        // New session, so the pty can become this process's controlling tty.
        if (::setsid() < 0) {
            ::_exit(10);
        }
        const int controlling = ::open(terminal.slave_name().c_str(), O_RDWR);
        if (controlling < 0 || ::ioctl(controlling, TIOCSCTTY, 0) < 0) {
            ::_exit(11);
        }
        // Only the terminal's answer may decide.
        ::unsetenv("COLORFGBG");
        ::setenv("TERM", "xterm-256color", 1);
        const auto theme = tui::detect_terminal_theme(5s);
        ::_exit(theme == tui::Theme::Light ? 0 : 12);
    }

    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
}

namespace {

struct Glyph {
    std::string character;
    std::string foreground;  // SGR parameters: "97", "38;2;160;98;0", ...
};

// Every visible glyph drawn directly on the terminal background.
std::vector<Glyph> glyphs_on_terminal_background(ftxui::Element element) {
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100), ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);
    std::vector<Glyph> glyphs;
    for (int y = 0; y < screen.dimy(); ++y) {
        for (int x = 0; x < screen.dimx(); ++x) {
            const auto& cell = screen.CellAt(x, y);
            if (cell.character.empty() || cell.character == " "
                || cell.background_color != ftxui::Color::Default || cell.inverted) {
                continue;
            }
            glyphs.push_back({cell.character, cell.foreground_color.Print(false)});
        }
    }
    return glyphs;
}

ftxui::Element sample_transcript() {
    std::vector<tui::UiMessage> messages;
    messages.push_back(tui::make_user_message("Summarise the change", "10:45:46"));
    messages.push_back(tui::make_assistant_message(
        "# Summary\n\n"
        "The **export** command now writes a `bundle.json`. See [the docs](https://example.com).\n\n"
        "### Steps\n\n"
        "- first item\n- second item\n\n"
        "1. numbered\n2. list\n\n"
        "> a quoted remark\n\n"
        "| Key | Value |\n|---|---|\n| ui_theme | auto |\n\n"
        "```cpp\nint main() { return 0; }\n```\n",
        "10:45:55",
        false));
    messages.push_back(tui::make_info_message("Saved"));
    messages.push_back(tui::make_warning_message("Context nearly full"));
    return tui::render_history_content(messages, 0);
}

} // namespace

TEST_CASE("Light-theme transcripts draw no pale ink on the terminal background", "[tui][theme][render]") {
    // Inspect the actual palette and RGB inks even when the test runner sets
    // NO_COLOR or advertises a limited terminal. Otherwise FTXUI strips or
    // quantizes them, invalidating the dark control and skipping RGB checks.
    const ScopedColorSupport true_color{ftxui::Terminal::Color::TrueColor};
    // Palette white / light grey are near-invisible on a light terminal. Exact
    // RGB inks must clear WCAG AA against white: 4.5:1 for text, 3:1 for
    // borders, bars and other non-text glyphs (WCAG 1.4.11).
    const auto is_pale_palette = [](std::string_view sgr) {
        return sgr == "97" || sgr == "37";
    };
    const auto rgb_of = [](std::string_view sgr) -> std::optional<tui::RgbColor> {
        if (!sgr.starts_with("38;2;")) {
            return std::nullopt;
        }
        std::array<int, 3> channels{};
        sgr.remove_prefix(5);
        for (int& channel : channels) {
            const auto [end, error] = std::from_chars(sgr.data(), sgr.data() + sgr.size(), channel);
            REQUIRE(error == std::errc{});
            sgr.remove_prefix(static_cast<std::size_t>(end - sgr.data()));
            if (!sgr.empty()) {
                sgr.remove_prefix(1);
            }
        }
        return tui::RgbColor{static_cast<unsigned char>(channels[0]),
                             static_cast<unsigned char>(channels[1]),
                             static_cast<unsigned char>(channels[2])};
    };

    SECTION("the sample exercises the pale dark-theme inks") {
        const ScopedTheme dark{tui::Theme::Dark};
        const auto glyphs = glyphs_on_terminal_background(sample_transcript());
        CHECK(std::ranges::any_of(glyphs, [&](const Glyph& glyph) { return is_pale_palette(glyph.foreground); }));
    }
    SECTION("the light theme replaces all of them") {
        const ScopedTheme light{tui::Theme::Light};
        const auto glyphs = glyphs_on_terminal_background(sample_transcript());
        REQUIRE_FALSE(glyphs.empty());
        for (const auto& [character, sgr] : glyphs) {
            INFO(character << " " << sgr);
            CHECK_FALSE(is_pale_palette(sgr));
            if (const auto rgb = rgb_of(sgr)) {
                const bool text = character.size() == 1 && std::isalnum(static_cast<unsigned char>(character[0]));
                CHECK(tui::contrast_ratio(*rgb, {255, 255, 255}) >= (text ? 4.5 : 3.0));
            }
        }
    }
}
