#include "Server.hpp"
#include "../core/mcp/McpDispatcher.hpp"
#include <cstdio>
#include <iostream>
#include <print>
#include <string>
#include <atomic>

namespace exec {
namespace mcp {

static std::atomic<bool> is_running{true};

void stop_server() {
    is_running = false;
}

void run_server() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Trigger skill registration on the dispatcher singleton before the loop so
    // the first tools/list response is not delayed by one-time initialisation.
    auto& dispatcher = core::mcp::McpDispatcher::get_instance();

    std::string line;
    while (is_running && std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string response = dispatcher.dispatch(line);
        if (!response.empty()) {
            std::println(stdout, "{}", response);
            std::fflush(stdout);
        }
    }
}

} // namespace mcp
} // namespace exec
