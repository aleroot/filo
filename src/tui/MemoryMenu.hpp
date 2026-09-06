#pragma once

#include "PromptComponents.hpp"
#include "core/memory/MemoryPolicy.hpp"
#include "core/memory/MemoryStore.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace tui {

enum class MemoryMenuPage { Overview, Settings, Session, Entries, Entry };

struct MemoryMenu {
    std::string title;
    std::string current;
    std::string help;
    std::vector<OptionPickerRow> options;
};

// Pure presentation model. Actions are existing slash commands; persistence
// and prompt refresh remain the command layer's responsibility.
[[nodiscard]] MemoryMenu build_memory_menu(
    MemoryMenuPage page,
    const core::memory::MemoryState& state,
    const core::memory::MemoryThreadPolicy& policy,
    std::string_view project,
    std::string_view entry_id = {});

} // namespace tui
