#include <catch2/catch_test_macros.hpp>

#include "core/logging/Logger.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

TEST_CASE("Logger callback sink observes formatted lines", "[logging]") {
    auto& logger = core::logging::Logger::get_instance();
    logger.set_min_level(core::logging::Level::Info);
    logger.enable_timestamps(false);

    const auto log_path =
        std::filesystem::temp_directory_path() / "filo_logger_callback_observe_test.log";
    std::filesystem::remove(log_path);
    REQUIRE(logger.use_file(log_path));

    std::vector<std::string> lines;
    std::vector<core::logging::Level> levels;
    logger.use_callback_sink(
        [&](core::logging::Level level, std::string line) {
            levels.push_back(level);
            lines.push_back(std::move(line));
        });

    core::logging::info("callback info {}", 1);
    core::logging::error("callback error {}", 2);

    CHECK(levels.size() == 2);
    REQUIRE(lines.size() == 2);
    CHECK(lines[0] == "[INFO] callback info 1");
    CHECK(lines[1] == "[ERROR] callback error 2");

    logger.clear_callback_sink();
    logger.use_stderr();
    logger.enable_timestamps(true);
    std::filesystem::remove(log_path);
}

TEST_CASE("Logger callback sink preserves file logging", "[logging]") {
    auto& logger = core::logging::Logger::get_instance();
    logger.set_min_level(core::logging::Level::Info);
    logger.enable_timestamps(false);

    const auto log_path =
        std::filesystem::temp_directory_path() / "filo_logger_callback_test.log";
    std::filesystem::remove(log_path);
    REQUIRE(logger.use_file(log_path));

    std::vector<std::string> lines;
    logger.use_callback_sink(
        [&](core::logging::Level, std::string line) {
            lines.push_back(std::move(line));
        });

    core::logging::error("file callback error");

    logger.clear_callback_sink();
    logger.use_stderr();
    logger.enable_timestamps(true);

    REQUIRE(lines.size() == 1);
    CHECK(lines[0] == "[ERROR] file callback error");

    std::ifstream file(log_path);
    REQUIRE(file.good());
    std::string file_content;
    std::getline(file, file_content);
    CHECK(file_content == "[ERROR] file callback error");
    std::filesystem::remove(log_path);
}

TEST_CASE("Logger default sink suppresses output sink but preserves callback", "[logging]") {
    const char* previous_log_file = std::getenv("FILO_LOG_FILE");
    const std::string previous_log_file_value =
        previous_log_file != nullptr ? previous_log_file : "";
#if defined(_WIN32)
    _putenv_s("FILO_LOG_FILE", "");
#else
    unsetenv("FILO_LOG_FILE");
#endif

    auto& logger = core::logging::Logger::get_instance();
    logger.set_min_level(core::logging::Level::Info);
    logger.enable_timestamps(false);
    logger.configure_from_env();

    std::vector<std::string> lines;
    logger.use_callback_sink(
        [&](core::logging::Level, std::string line) {
            lines.push_back(std::move(line));
        });

    core::logging::info("callback only");

    logger.clear_callback_sink();
    logger.use_stderr();
    logger.enable_timestamps(true);
    if (previous_log_file != nullptr) {
#if defined(_WIN32)
        _putenv_s("FILO_LOG_FILE", previous_log_file_value.c_str());
#else
        setenv("FILO_LOG_FILE", previous_log_file_value.c_str(), 1);
#endif
    }

    REQUIRE(lines.size() == 1);
    CHECK(lines[0] == "[INFO] callback only");
}
