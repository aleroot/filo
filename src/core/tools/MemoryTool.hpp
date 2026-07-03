#pragma once

#include "Tool.hpp"
#include "../memory/MemoryStore.hpp"

#include <string_view>

namespace core::tools {

class MemoryTool final : public Tool {
public:
    explicit MemoryTool(core::memory::MemoryStore store = core::memory::MemoryStore{});

    [[nodiscard]] static bool is_mutating_action(std::string_view action) noexcept;
    [[nodiscard]] static bool committed_mutation(std::string_view tool_name,
                                                 std::string_view args,
                                                 std::string_view result);

    [[nodiscard]] ToolDefinition get_definition() const override;

    [[nodiscard]] std::string execute(const std::string& json_args,
                                      const core::context::SessionContext& context) override;

private:
    core::memory::MemoryStore store_;
};

} // namespace core::tools
