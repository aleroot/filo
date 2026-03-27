#include "tui/MainApp.hpp"
#include "core/config/ConfigManager.hpp"
#include <thread>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <cstdio>
#include <print>

int main() {
    // Setup temporary config directory for the test
    std::string temp_config_dir = "/tmp/filo_test_config_" + std::to_string(std::rand());
    std::filesystem::create_directories(temp_config_dir);
    setenv("XDG_CONFIG_HOME", temp_config_dir.c_str(), 1);

    // Initialize configuration
    try {
        core::config::ConfigManager::get_instance().load();
    } catch (const std::exception& e) {
        std::println(stderr, "Warning: Could not load configuration: {}", e.what());
    }

    // Spawn a thread that will exit the application after a brief delay
    std::thread([temp_config_dir]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        // Cleanup
        std::filesystem::remove_all(temp_config_dir);
        std::_Exit(0);
    }).detach();

    tui::run();
    return 0;
}
