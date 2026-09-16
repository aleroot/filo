#pragma once

#include "../core/context/SteeringLoader.hpp"

#include <string>

namespace exec::daemon {
    /// @param steering_policy startup policy applied to every session the daemon
    ///        creates, so `--steering`/`--no-steering` mean the same thing over
    ///        HTTP as they do in the TUI.
    void run_server(int port = 8080,
                    const std::string& host = "127.0.0.1",
                    bool enable_api_gateway = false,
                    bool enable_mcp_http = true,
                    const core::context::SteeringPolicy& steering_policy = {});
    void stop_server();
}
