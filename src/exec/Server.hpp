#pragma once

#include "../core/context/SteeringLoader.hpp"

namespace exec {
namespace mcp {
    /// @param steering_policy startup policy for the stdio session, so a host
    ///        that launches Filo with `--no-steering` gets the same tool-side
    ///        denial the TUI does.
    void run_server(const core::context::SteeringPolicy& steering_policy = {});
    void stop_server();
}
}
