#pragma once

#include <cstdio>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <string_view>
#include <utility>

namespace core::logging {

enum class Level : int {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Off   = 5,
};

class Logger {
public:
    static Logger& get_instance() noexcept;

    void set_min_level(Level level) noexcept;
    [[nodiscard]] Level min_level() const noexcept;

    void enable_timestamps(bool enabled) noexcept;
    [[nodiscard]] bool timestamps_enabled() const noexcept;

    void use_stderr() noexcept;
    bool use_file(const std::filesystem::path& path);
    void disable() noexcept;

    void configure_from_env();

    template <typename... Args>
    void log(Level level, std::format_string<Args...> fmt, Args&&... args) {
        try {
            log_formatted(level, std::format(fmt, std::forward<Args>(args)...));
        } catch (...) {
            // Logging should never interrupt application control flow.
        }
    }

private:
    enum class Sink {
        Stderr,
        File,
        Null,
    };

    struct FileCloser {
        void operator()(std::FILE* file) const noexcept {
            if (file) {
                std::fclose(file);
            }
        }
    };

    Logger() = default;

    static std::string_view level_name(Level level) noexcept;
    static Level parse_level(std::string_view raw) noexcept;
    static bool parse_bool(std::string_view raw, bool default_value) noexcept;
    static std::string timestamp_now();

    void log_formatted(Level level, std::string_view message) noexcept;
    bool is_enabled(Level level) const noexcept;
    void emit_line(std::FILE* target, Level level, std::string_view message) const;

    mutable std::mutex mutex_;
    Level min_level_ = Level::Info;
    bool timestamps_enabled_ = true;
    Sink sink_ = Sink::Stderr;
    std::unique_ptr<std::FILE, FileCloser> file_sink_{nullptr};
};

template <typename... Args>
inline void trace(std::format_string<Args...> fmt, Args&&... args) {
    Logger::get_instance().log(Level::Trace, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void debug(std::format_string<Args...> fmt, Args&&... args) {
    Logger::get_instance().log(Level::Debug, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void info(std::format_string<Args...> fmt, Args&&... args) {
    Logger::get_instance().log(Level::Info, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void warn(std::format_string<Args...> fmt, Args&&... args) {
    Logger::get_instance().log(Level::Warn, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
inline void error(std::format_string<Args...> fmt, Args&&... args) {
    Logger::get_instance().log(Level::Error, fmt, std::forward<Args>(args)...);
}

} // namespace core::logging
