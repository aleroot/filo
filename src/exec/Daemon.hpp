#pragma once

#include <string>

namespace exec::daemon {
    void run_server(int port = 8080, const std::string& host = "127.0.0.1");
    void stop_server();
}
