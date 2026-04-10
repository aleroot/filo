#pragma once

#include <string>

namespace exec::daemon {
    void run_server(int port = 8080,
                    const std::string& host = "127.0.0.1",
                    bool enable_api_gateway = false,
                    bool enable_mcp_http = true);
    void stop_server();
}
