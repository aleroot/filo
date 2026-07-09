#pragma once

#include <string>

namespace core::agent {

struct SubagentEvent {
    enum class Kind {
        Started,
        TextDelta,
        ToolStarted,
        ToolFinished,
        Progress,
        Finished,
        Failed,
        Cancelled,
    };

    Kind kind = Kind::Progress;
    std::string parent_tool_call_id;
    std::string task_id;
    std::string worker_name;
    std::string description;
    std::string provider_name;
    std::string model_name;
    std::string tool_call_id;
    std::string tool_name;
    std::string tool_arguments;
    std::string tool_result;
    std::string text_delta;
    std::string summary;
    int steps = 0;
    int tool_calls = 0;
    int failed_tool_calls = 0;
};

} // namespace core::agent
