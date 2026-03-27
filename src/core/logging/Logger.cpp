#include "Logger.hpp"

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <print>
#include <string>

namespace core::logging {

namespace {

std::string normalize_token(std::string_view raw) {
    std::string normalized;
    normalized.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isspace(ch) || ch == '-' || ch == '_') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

} // namespace

Logger& Logger::get_instance() noexcept {
    static Logger instance;
    return instance;
}

void Logger::set_min_level(Level level) noexcept {
    std::scoped_lock lock(mutex_);
    min_level_ = level;
}

Level Logger::min_level() const noexcept {
    std::scoped_lock lock(mutex_);
    return min_level_;
}

void Logger::enable_timestamps(bool enabled) noexcept {
    std::scoped_lock lock(mutex_);
    timestamps_enabled_ = enabled;
}

bool Logger::timestamps_enabled() const noexcept {
    std::scoped_lock lock(mutex_);
    return timestamps_enabled_;
}

void Logger::use_stderr() noexcept {
    std::scoped_lock lock(mutex_);
    file_sink_.reset();
    sink_ = Sink::Stderr;
}

bool Logger::use_file(const std::filesystem::path& path) {
    std::error_code ec;
    if (const auto parent = path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

    std::FILE* file = std::fopen(path.string().c_str(), "a");
    if (!file) {
        return false;
    }

    std::scoped_lock lock(mutex_);
    file_sink_.reset(file);
    sink_ = Sink::File;
    return true;
}

void Logger::disable() noexcept {
    std::scoped_lock lock(mutex_);
    sink_ = Sink::Null;
    file_sink_.reset();
}

void Logger::configure_from_env() {
    if (const char* raw_level = std::getenv("FILO_LOG_LEVEL");
        raw_level && raw_level[0] != '\0') {
        const std::string normalized = normalize_token(raw_level);
        const bool valid =
            normalized == "trace"
            || normalized == "debug"
            || normalized == "info"
            || normalized == "warn"
            || normalized == "warning"
            || normalized == "error"
            || normalized == "off"
            || normalized == "none"
            || normalized == "quiet";

        if (!valid) {
            std::println(stderr,
                         "[filo][WARN] Invalid FILO_LOG_LEVEL='{}'. Keeping current level.",
                         raw_level);
        } else {
            set_min_level(parse_level(normalized));
        }
    }

    if (const char* raw_timestamps = std::getenv("FILO_LOG_TIMESTAMPS");
        raw_timestamps && raw_timestamps[0] != '\0') {
        enable_timestamps(parse_bool(raw_timestamps, true));
    }

    if (const char* raw_file = std::getenv("FILO_LOG_FILE");
        raw_file && raw_file[0] != '\0') {
        if (!use_file(raw_file)) {
            std::println(stderr,
                         "[filo][WARN] Could not open FILO_LOG_FILE='{}'. Falling back to stderr.",
                         raw_file);
            use_stderr();
        }
    }
}

std::string_view Logger::level_name(Level level) noexcept {
    switch (level) {
        case Level::Trace: return "TRACE";
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO";
        case Level::Warn:  return "WARN";
        case Level::Error: return "ERROR";
        case Level::Off:   return "OFF";
    }
    return "INFO";
}

Level Logger::parse_level(std::string_view raw) noexcept {
    const std::string normalized = normalize_token(raw);
    if (normalized == "trace") return Level::Trace;
    if (normalized == "debug") return Level::Debug;
    if (normalized == "info") return Level::Info;
    if (normalized == "warn" || normalized == "warning") return Level::Warn;
    if (normalized == "error") return Level::Error;
    if (normalized == "off" || normalized == "none" || normalized == "quiet") return Level::Off;
    return Level::Off;
}

bool Logger::parse_bool(std::string_view raw, bool default_value) noexcept {
    const std::string normalized = normalize_token(raw);
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
        return false;
    }
    return default_value;
}

std::string Logger::timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::floor<std::chrono::seconds>(now);
    return std::format("{:%Y-%m-%d %H:%M:%S}", seconds);
}

void Logger::log_formatted(Level level, std::string_view message) noexcept {
    try {
        std::scoped_lock lock(mutex_);

        if (!is_enabled(level) || sink_ == Sink::Null) {
            return;
        }

        if (sink_ == Sink::File && file_sink_) {
            emit_line(file_sink_.get(), level, message);
            std::fflush(file_sink_.get());
            return;
        }

        emit_line(stderr, level, message);
    } catch (...) {
        // Logging must never throw.
    }
}

bool Logger::is_enabled(Level level) const noexcept {
    return level >= min_level_ && min_level_ != Level::Off;
}

void Logger::emit_line(std::FILE* target, Level level, std::string_view message) const {
    if (timestamps_enabled_) {
        std::println(target, "[{}] [{}] {}", timestamp_now(), level_name(level), message);
    } else {
        std::println(target, "[{}] {}", level_name(level), message);
    }
}

} // namespace core::logging
