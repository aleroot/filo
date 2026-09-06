#include "MemorySystem.hpp"

#include <utility>

namespace core::memory {

MemorySystem::MemorySystem()
    : MemorySystem(MemoryStore{}, std::make_shared<NullToolRecoveryMemory>()) {}

MemorySystem::MemorySystem(MemoryStore store,
                           std::shared_ptr<ToolRecoveryMemory> tool_recovery)
    : store_(std::move(store))
    // MemoryStore is a path handle: the background service copies it and
    // both talk to the same on-disk file under the store's own lock.
    , background_(store_)
    , tool_recovery_(tool_recovery
          ? std::move(tool_recovery)
          : std::make_shared<NullToolRecoveryMemory>()) {}

std::string MemorySystem::semantic_prompt_block(
    const core::context::SessionContext& context,
    std::size_t max_entries,
    bool allow_auto_capture) const {
    if (!context.memory_policy.use_memories) return {};
    std::string error;
    const auto state = semantic(context).load(&error);
    if (!error.empty()) return {};
    return build_memory_prompt_block(state, max_entries,
        allow_auto_capture && context.memory_policy.generate_memories);
}

std::shared_ptr<MemorySystem> make_memory_system(MemoryConfig config) {
    return std::make_shared<MemorySystem>(
        MemoryStore{},
        make_tool_recovery_memory(config.tool_recovery));
}

std::shared_ptr<MemorySystem> make_memory_system(
    std::shared_ptr<ToolRecoveryMemory> tool_recovery) {
    return std::make_shared<MemorySystem>(MemoryStore{}, std::move(tool_recovery));
}

} // namespace core::memory
