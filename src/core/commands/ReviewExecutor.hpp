#pragma once

#include "CommandExecutor.hpp"
#include <string_view>

namespace core::commands {

class ReviewExecutor {
public:
    static void execute(const CommandContext& ctx, std::string_view raw_args);
};

} // namespace core::commands
