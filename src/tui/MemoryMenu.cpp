#include "MemoryMenu.hpp"

#include <algorithm>
#include <format>

namespace tui {
namespace {

OptionPickerRow toggle(std::string_view label, bool enabled,
                       std::string_view command, std::string_view description) {
    return {
        .value = std::format("/memory {} {}", command, enabled ? "off" : "on"),
        .label = std::format("{}: {}", label, enabled ? "On" : "Off"),
        .description = std::string(description),
        .active = enabled,
    };
}

} // namespace

MemoryMenu build_memory_menu(MemoryMenuPage page,
                              const core::memory::MemoryState& state,
                              const core::memory::MemoryThreadPolicy& policy,
                              std::string_view project,
                              std::string_view entry_id) {
    const auto count = std::ranges::count_if(state.entries,
        [](const auto& entry) { return !entry.archived; });
    MemoryMenu menu{.current = std::format("{} · {} {}", project, count,
                                          count == 1 ? "memory" : "memories")};
    switch (page) {
    case MemoryMenuPage::Overview:
        menu.title = "MEMORY";
        menu.options = {
            {.value = "page:entries", .label = "Browse memories...",
             .description = "Inspect and forget memories for this project and session."},
            {.value = "input:/memory add ", .label = "Add memory...",
             .description = "Write a durable note for this project."},
            {.value = "page:settings", .label = "Memory settings...",
             .description = std::format("Memory: {}. Automatic capture: {}. Settings apply across projects.",
                 state.settings.enabled ? "on" : "off", state.settings.auto_capture ? "on" : "off")},
            {.value = "page:session", .label = "This session...",
             .description = "Control whether this session can use or generate memories."},
            {.value = "/memory status", .label = "Show full status",
             .description = "Print settings and current project memories in the conversation."},
            {.value = "import", .label = "Import memories...",
             .description = "Choose a Markdown file to import into this project."},
            {.value = "input:/memory save ", .label = "Export memories...",
             .description = "Choose a Markdown output path for this project's memories."},
        };
        break;
    case MemoryMenuPage::Settings:
        menu.title = "MEMORY SETTINGS";
        menu.options = {
            {.value = state.settings.enabled ? "/memory off" : "/memory on",
             .label = std::format("Memory: {}", state.settings.enabled ? "On" : "Off"),
             .description = "Enable recall, or disable memory and all automatic memory features.",
             .active = state.settings.enabled},
            toggle("Automatic capture", state.settings.auto_capture, "auto",
                   "Allow the model to save durable project memories. Enabling also enables background review."),
            toggle("Background review", state.settings.background_review, "background",
                   "Extract durable notes from this project's conversation history."),
            toggle("Consolidation", state.settings.consolidation, "consolidate",
                   "Clean duplicate memories during background review."),
            toggle("Skill drafts", state.settings.skill_curation, "skills",
                   "Create disabled skill drafts for manual review."),
            {.value = "/memory review", .label = "Run review now",
             .description = "Run enabled background features for the current project."},
        };
        break;
    case MemoryMenuPage::Session:
        menu.title = "SESSION MEMORY";
        menu.options = {
            toggle("Use memories", policy.use_memories, "thread use",
                   "Include current project memories in this session's context."),
            toggle("Generate memories", policy.generate_memories, "thread generate",
                   "Allow this session to save memories when automatic capture is enabled."),
            toggle("Curate skills", policy.curate_skills, "thread skills",
                   "Allow this session to contribute skill drafts when curation is enabled."),
        };
        break;
    case MemoryMenuPage::Entries:
        menu.title = "PROJECT MEMORIES";
        for (const auto& entry : state.entries) {
            if (entry.archived) continue;
            menu.options.push_back({
                .value = "entry:" + entry.id,
                .label = std::format("{{{}}} {}", entry.id, entry.content),
                .description = std::format("{} · {} · Enter to inspect this memory.", entry.scope, entry.source),
            });
        }
        menu.options.push_back({.value = "input:/memory add ", .label = "Add memory...",
                                .description = "Write a durable note for this project."});
        menu.options.push_back({.value = "/memory clean", .label = "Clean duplicates",
                                .description = "Archive duplicate entries in this project only."});
        break;
    case MemoryMenuPage::Entry: {
        menu.title = "MEMORY DETAIL";
        const auto entry = std::ranges::find(state.entries, entry_id, &core::memory::MemoryEntry::id);
        if (entry != state.entries.end() && !entry->archived) {
            menu.help = entry->content;
            menu.options.push_back({.value = "/memory forget " + entry->id,
                                    .label = "Forget this memory",
                                    .description = "Archive this entry so it is no longer recalled."});
        } else {
            menu.help = "This memory is no longer available in the current project or session.";
        }
        menu.options.push_back({.value = "page:entries", .label = "Back to memories",
                                .description = "Return to the project memory list."});
        break;
    }
    }
    if (page != MemoryMenuPage::Overview) {
        menu.options.push_back({.value = "page:overview", .label = "Back to memory",
                                .description = "Return to the memory menu."});
    }
    return menu;
}

} // namespace tui
