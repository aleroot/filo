#pragma once

#include "MemoryBackgroundService.hpp"
#include "MemoryStore.hpp"
#include "ToolRecoveryMemory.hpp"

#include <memory>
#include <string>

namespace core::memory {

/**
 * Construction options for the process/session memory substrate.
 *
 * New memory kinds add fields here. They do not add Agent constructor
 * parameters and they do not become a singleton.
 */
struct MemoryConfig {
    bool tool_recovery = true;
};

/**
 * Composition handle for Filo's long-term memory substrate.
 *
 * This is a Facade over typed memory ports, not a generic "IMemory" bag and
 * not a service locator. Each port has its own invariants, persistence, and
 * consumers:
 *
 *   semantic()      — MemoryStore: facts, preferences, user-authored notes
 *   tool_recovery() — ToolRecoveryMemory: fail→success lessons about tools
 *   background()    — MemoryBackgroundService: consolidation / review
 *
 * New memory kinds (episodic trajectories, skill habits) are added as
 * additional typed accessors here. They do not inherit a common
 * record/recall interface: the CoALA taxonomy exists because those stores
 * are *not* interchangeable.
 *
 * Ownership: execution roots (TUI, daemon, prompter) construct one
 * MemorySystem and pass the same shared_ptr into Agent, MemoryTool,
 * ContextBuilder, /memory commands, and delegated workers. Tests inject
 * fakes through the port-wrapping factory. There is no process-global
 * instance.
 */
class MemorySystem {
public:
    /// Default: file-backed semantic store + inert tool-recovery.
    /// Used by tests and embedders that do not opt into persistence.
    MemorySystem();

    /// Full construction. A null `tool_recovery` becomes NullToolRecoveryMemory
    /// so callers never need a null check on the port.
    MemorySystem(MemoryStore store,
                 std::shared_ptr<ToolRecoveryMemory> tool_recovery);

    MemorySystem(const MemorySystem&) = delete;
    MemorySystem& operator=(const MemorySystem&) = delete;
    MemorySystem(MemorySystem&&) noexcept = default;
    MemorySystem& operator=(MemorySystem&&) noexcept = default;
    ~MemorySystem() = default;

    [[nodiscard]] MemoryStore& semantic() noexcept { return store_; }
    [[nodiscard]] const MemoryStore& semantic() const noexcept { return store_; }

    [[nodiscard]] MemoryBackgroundService& background() noexcept {
        return background_;
    }
    [[nodiscard]] const MemoryBackgroundService& background() const noexcept {
        return background_;
    }

    [[nodiscard]] ToolRecoveryMemory& tool_recovery() noexcept {
        return *tool_recovery_;
    }

    /// Prompt projection of semantic memories. ContextBuilder (and any other
    /// prompt assembler) should call this rather than constructing a store.
    [[nodiscard]] std::string semantic_prompt_block(
        std::size_t max_entries = 24,
        bool allow_auto_capture = true) const;

private:
    MemoryStore store_;
    MemoryBackgroundService background_;
    std::shared_ptr<ToolRecoveryMemory> tool_recovery_;
};

/// Execution-root factory. Semantic memory always uses the default on-disk
/// store so /memory, MemoryTool, and the prompt stay aligned when they share
/// this system.
[[nodiscard]] std::shared_ptr<MemorySystem> make_memory_system(
    MemoryConfig config = {});

/// Test/embedder factory: wrap an already-built tool-recovery port.
[[nodiscard]] std::shared_ptr<MemorySystem> make_memory_system(
    std::shared_ptr<ToolRecoveryMemory> tool_recovery);

} // namespace core::memory
