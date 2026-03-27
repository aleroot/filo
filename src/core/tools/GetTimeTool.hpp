#pragma once

#include "Tool.hpp"
#include <chrono>
#include <format>

namespace core::tools {

class GetTimeTool : public Tool {
public:
    ToolDefinition get_definition() const override {
        return {
            .name        = "get_current_time",
            .title       = "Get Current Time",
            .description = "Returns the current system time in HH:MM:SS format.",
            .annotations = { .read_only_hint = true, .idempotent_hint = false },
        };
    }

    std::string execute([[maybe_unused]] const std::string& json_args) override {
        auto now = std::chrono::system_clock::now();
        std::string time_str = std::format("{:%H:%M:%S}", now);
        return "{\"time\": \"" + time_str + "\"}";
    }
};

} // namespace core::tools
