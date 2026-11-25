#pragma once

#include "CommandExecutor.hpp"

namespace core::commands {

class IssueQueueExecutor {
public:
    static void run(const CommandContext& ctx, int max_issues);
};

} // namespace core::commands
