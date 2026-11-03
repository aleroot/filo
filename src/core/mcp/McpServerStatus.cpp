#include "McpServerStatus.hpp"
#include "McpOAuth.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <ranges>

namespace core::mcp {

void McpServerActivity::begin(std::string_view tool_name) {
    std::lock_guard lock(mutex_);
    ++state_.in_flight;
    state_.last_tool = std::string(tool_name);
    state_.last_call_at = std::chrono::system_clock::now();
}

void McpServerActivity::finish(std::string_view tool_name,
                               bool failed,
                               std::string_view error,
                               std::chrono::milliseconds duration) {
    std::lock_guard lock(mutex_);
    state_.in_flight = std::max(0, state_.in_flight - 1);
    ++state_.calls;
    if (failed) ++state_.failures;
    state_.last_tool = std::string(tool_name);
    state_.last_failed = failed;
    state_.last_error.clear();
    if (failed) {
        // Keep enough to explain the failure; a server can echo whole payloads.
        constexpr std::size_t kMaxErrorBytes = 480;
        std::size_t length = std::min(error.size(), kMaxErrorBytes);
        while (length < error.size() && length > 0
               && (static_cast<unsigned char>(error[length]) & 0xC0) == 0x80) {
            --length;
        }
        state_.last_error = std::string(error.substr(0, length));
        if (length < error.size()) state_.last_error += "…";
    }
    state_.last_duration = duration;
}

McpServerActivitySnapshot McpServerActivity::snapshot() const {
    std::lock_guard lock(mutex_);
    return state_;
}

std::string describe_mcp_endpoint(const core::config::McpServerConfig& config) {
    if (config.transport != "stdio") return config.url;
    std::string line = config.command;
    for (const auto& arg : config.args) line += ' ' + arg;
    return line;
}

std::string describe_mcp_auth(const core::config::McpServerConfig& config) {
    if (config.transport == "stdio") {
        return config.env.empty() ? "local process"
                                  : std::format("local process · {} env var{}",
                                                config.env.size(),
                                                config.env.size() == 1 ? "" : "s");
    }
    const auto names = config.headers | std::views::keys;
    if (std::ranges::contains(names, "authorization", core::utils::str::to_lower_ascii_copy)) {
        return "Authorization header";
    }
    if (mcp_server_wants_oauth(config)) return "OAuth";
    const auto header_names = names
        | std::views::filter([](const std::string& name) { return !name.empty(); })
        | std::ranges::to<std::vector<std::string>>();
    if (header_names.empty()) return "none";
    return std::format("{}: {}",
                       header_names.size() == 1 ? "header" : "headers",
                       header_names | std::views::join_with(std::string_view(", "))
                           | std::ranges::to<std::string>());
}

bool mcp_error_needs_login(const core::config::McpServerConfig& config,
                           std::string_view error) {
    static constexpr std::array<std::string_view, 4> kLoginMarkers{
        "401", "403", "authentication", "/mcp login"};
    return mcp_server_wants_oauth(config)
        && std::ranges::any_of(kLoginMarkers,
                               [error](std::string_view marker) { return error.contains(marker); });
}

} // namespace core::mcp
