#include "TerminalTheme.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string>

#include <fcntl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include "core/utils/AsciiUtils.hpp"

namespace tui {

namespace {

constexpr std::string_view kBackgroundQuery = "\x1b]11;?\x1b\\";
constexpr std::string_view kDeviceAttributesQuery = "\x1b[c";
constexpr std::string_view kBackgroundReplyIntroducer = "\x1b]11;";
constexpr std::string_view kDeviceAttributesReplyIntroducer = "\x1b[?";
// Replies are a few dozen bytes; anything longer is not a terminal answering.
constexpr std::size_t kMaxReplyBytes = 4096;

[[nodiscard]] double linear_channel(unsigned char channel) noexcept {
    const double value = channel / 255.0;
    return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
}

// One X11 colour channel: 1–4 hex digits, scaled from its own range to 0–255
// ("f" and "ffff" are both full intensity).
[[nodiscard]] std::optional<unsigned char> parse_hex_channel(std::string_view digits) noexcept {
    if (digits.empty() || digits.size() > 4) {
        return std::nullopt;
    }
    unsigned value = 0;
    const auto [end, error] = std::from_chars(digits.data(), digits.data() + digits.size(), value, 16);
    if (error != std::errc{} || end != digits.data() + digits.size()) {
        return std::nullopt;
    }
    const unsigned max = (1u << (4 * digits.size())) - 1;
    return static_cast<unsigned char>((value * 255 + max / 2) / max);
}

// "rgb:R/G/B" or "rgba:R/G/B/A" (alpha ignored).
[[nodiscard]] std::optional<RgbColor> parse_color_spec(std::string_view spec) noexcept {
    std::size_t channel_count = 0;
    if (spec.starts_with("rgb:")) {
        spec.remove_prefix(4);
        channel_count = 3;
    } else if (spec.starts_with("rgba:")) {
        spec.remove_prefix(5);
        channel_count = 4;
    } else {
        return std::nullopt;
    }

    std::array<unsigned char, 4> channels{};
    for (std::size_t index = 0; index < channel_count; ++index) {
        const std::size_t slash = spec.find('/');
        const bool last = index + 1 == channel_count;
        if (last != (slash == std::string_view::npos)) {
            return std::nullopt;
        }
        const auto channel = parse_hex_channel(spec.substr(0, slash));
        if (!channel) {
            return std::nullopt;
        }
        channels[index] = *channel;
        spec.remove_prefix(last ? spec.size() : slash + 1);
    }
    return RgbColor{channels[0], channels[1], channels[2]};
}

[[nodiscard]] bool write_all(int fd, std::string_view bytes) noexcept {
    while (!bytes.empty()) {
        const ssize_t written = ::write(fd, bytes.data(), bytes.size());
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        bytes.remove_prefix(static_cast<std::size_t>(written));
    }
    return true;
}

// Restores the saved terminal attributes on every exit path.
class TerminalModeGuard {
public:
    TerminalModeGuard(int fd, const termios& saved) noexcept : fd_(fd), saved_(saved) {}
    ~TerminalModeGuard() { ::tcsetattr(fd_, TCSANOW, &saved_); }

    TerminalModeGuard(const TerminalModeGuard&) = delete;
    TerminalModeGuard& operator=(const TerminalModeGuard&) = delete;

private:
    int fd_;
    termios saved_;
};

class FileDescriptor {
public:
    explicit FileDescriptor(int fd) noexcept : fd_(fd) {}
    ~FileDescriptor() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    [[nodiscard]] int get() const noexcept { return fd_; }

private:
    int fd_;
};

[[nodiscard]] std::string_view environment_value(const char* name) noexcept {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string_view{value} : std::string_view{};
}

} // namespace

std::optional<ThemePreference> parse_theme_preference(std::string_view value) noexcept {
    if (value.empty()) {
        return ThemePreference::Auto;
    }
    for (const auto& option : theme_preference_options()) {
        if (core::utils::ascii::iequals(value, option.token)) {
            return option.preference;
        }
    }
    return std::nullopt;
}

double relative_luminance(RgbColor color) noexcept {
    return 0.2126 * linear_channel(color.red)
         + 0.7152 * linear_channel(color.green)
         + 0.0722 * linear_channel(color.blue);
}

double contrast_ratio(RgbColor first, RgbColor second) noexcept {
    const double first_luminance = relative_luminance(first);
    const double second_luminance = relative_luminance(second);
    const auto [darker, lighter] = std::minmax(first_luminance, second_luminance);
    return (lighter + 0.05) / (darker + 0.05);
}

Theme theme_for_background(RgbColor background) noexcept {
    constexpr RgbColor kBlack{0, 0, 0};
    constexpr RgbColor kWhite{255, 255, 255};
    return contrast_ratio(background, kBlack) > contrast_ratio(background, kWhite)
        ? Theme::Light
        : Theme::Dark;
}

std::optional<RgbColor> parse_background_color_reply(std::string_view input) noexcept {
    for (std::size_t start = input.find(kBackgroundReplyIntroducer);
         start != std::string_view::npos;
         start = input.find(kBackgroundReplyIntroducer, start + 1)) {
        const std::string_view body = input.substr(start + kBackgroundReplyIntroducer.size());
        // BEL or ST (ESC \) ends the reply.
        const std::size_t end = body.find_first_of("\x07\x1b");
        if (end == std::string_view::npos) {
            return std::nullopt;
        }
        const bool terminated = body[end] == '\x07'
            || (end + 1 < body.size() && body[end + 1] == '\\');
        if (!terminated) {
            continue;
        }
        if (const auto color = parse_color_spec(body.substr(0, end))) {
            return color;
        }
    }
    return std::nullopt;
}

bool contains_device_attributes_reply(std::string_view input) noexcept {
    for (std::size_t start = input.find(kDeviceAttributesReplyIntroducer);
         start != std::string_view::npos;
         start = input.find(kDeviceAttributesReplyIntroducer, start + 1)) {
        const std::string_view params = input.substr(start + kDeviceAttributesReplyIntroducer.size());
        const std::size_t end = params.find_first_not_of("0123456789;");
        if (end != std::string_view::npos && params[end] == 'c') {
            return true;
        }
    }
    return false;
}

std::optional<Theme> theme_from_colorfgbg(std::string_view value) noexcept {
    const std::size_t separator = value.rfind(';');
    if (separator == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view background = value.substr(separator + 1);
    int slot = -1;
    const auto [end, error] = std::from_chars(background.data(), background.data() + background.size(), slot);
    if (error != std::errc{} || end != background.data() + background.size() || slot < 0 || slot > 15) {
        return std::nullopt;
    }
    return slot == 7 || slot >= 9 ? Theme::Light : Theme::Dark;
}

std::optional<Theme> theme_from_environment() noexcept {
    return theme_from_colorfgbg(environment_value("COLORFGBG"));
}

std::optional<RgbColor> query_terminal_background(int tty_fd, std::chrono::milliseconds timeout) {
    termios saved{};
    if (tty_fd < 0 || tty_fd >= FD_SETSIZE || !::isatty(tty_fd) || ::tcgetattr(tty_fd, &saved) != 0) {
        return std::nullopt;
    }
    // Unbuffered and unechoed, so the replies neither wait for Enter nor show
    // up on screen. ISIG stays on: Ctrl+C still interrupts.
    termios raw = saved;
    raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (::tcsetattr(tty_fd, TCSANOW, &raw) != 0) {
        return std::nullopt;
    }
    const TerminalModeGuard restore{tty_fd, saved};

    std::string query{kBackgroundQuery};
    query += kDeviceAttributesQuery;
    if (!write_all(tty_fd, query)) {
        return std::nullopt;
    }

    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + timeout;
    std::string replies;
    while (!contains_device_attributes_reply(replies) && replies.size() < kMaxReplyBytes) {
        const auto remaining = std::chrono::ceil<std::chrono::microseconds>(deadline - Clock::now());
        if (remaining.count() <= 0) {
            break;
        }
        // select(), not poll(): macOS poll() rejects device files such as
        // /dev/tty with POLLNVAL, which would end the probe before any reply.
        fd_set readable;
        FD_ZERO(&readable);
        FD_SET(tty_fd, &readable);
        timeval wait{
            .tv_sec = static_cast<decltype(timeval::tv_sec)>(remaining.count() / 1'000'000),
            .tv_usec = static_cast<decltype(timeval::tv_usec)>(remaining.count() % 1'000'000),
        };
        const int ready = ::select(tty_fd + 1, &readable, nullptr, nullptr, &wait);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready <= 0) {
            break;
        }
        std::array<char, 256> buffer{};
        const ssize_t received = ::read(tty_fd, buffer.data(), buffer.size());
        if (received < 0 && (errno == EINTR || errno == EAGAIN)) {
            continue;
        }
        if (received <= 0) {
            break;
        }
        replies.append(buffer.data(), static_cast<std::size_t>(received));
    }
    return parse_background_color_reply(replies);
}

std::optional<Theme> detect_terminal_theme(std::chrono::milliseconds timeout) {
    if (!core::utils::ascii::iequals(environment_value("TERM"), "dumb")) {
        const FileDescriptor tty{::open("/dev/tty", O_RDWR | O_NOCTTY | O_CLOEXEC)};
        if (tty.get() >= 0) {
            if (const auto background = query_terminal_background(tty.get(), timeout)) {
                return theme_for_background(*background);
            }
        }
    }
    return theme_from_environment();
}

} // namespace tui
