#include "ConversationChanges.hpp"

#include "StoredDiff.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <ranges>
#include <utility>

namespace core::changes {
namespace {

/// One file followed back through the conversation. `content` is the file at
/// the point being unwound, absent while the file does not exist there.
struct Lineage {
    std::string current_path;
    std::optional<std::string> current_content;
    std::string path;
    std::optional<std::string> content;
    std::optional<std::string> failure;
};

constexpr std::string_view kChangedOutside = "changed outside the conversation";

[[nodiscard]] std::string_view undiffed_reason(FileChangeContent content) noexcept {
    switch (content) {
        case FileChangeContent::Binary:      return "is a binary file, whose history was never kept";
        case FileChangeContent::TooLarge:    return "was too large to diff in one of the turns";
        case FileChangeContent::BudgetSpent: return "kept no diff in a turn whose diff budget was spent";
        case FileChangeContent::Text:        break;
    }
    return "kept no diff in one of the turns";
}

/// The file as it is now, which is where unwinding starts.
[[nodiscard]] Lineage start_lineage(const std::string& path, const std::filesystem::path& root) {
    Lineage lineage{.current_path = path, .path = path};
    if (auto content = detail::read_whole(detail::absolute_in(root, path))) {
        lineage.current_content = std::move(*content);
    } else if (content.error() == detail::ReadFailure::TooLarge) {
        lineage.failure = "is too large to compare across the conversation";
    } else if (content.error() == detail::ReadFailure::Unreadable) {
        lineage.failure = "could not be read";
    }
    lineage.content = lineage.current_content;
    return lineage;
}

/// Steps `lineage` back over one turn's change: from the state the turn left
/// to the state it found. The turn's after side must be exactly what the
/// lineage holds, existence included, or the file moved on in between.
void unwind(Lineage& lineage, const FileChange& change) {
    const bool existed_before = change.kind != FileChangeKind::Added;
    const bool exists_after = change.kind != FileChangeKind::Deleted;
    if (change.kind == FileChangeKind::Renamed && !change.previous_path.empty()) {
        lineage.path = change.previous_path;
    }
    if (lineage.failure) {
        return;  // Still followed, so older turns of this file find it.
    }
    const auto fail = [&](std::string_view reason) { lineage.failure = std::string(reason); };

    if (lineage.content.has_value() != exists_after) {
        return fail(kChangedOutside);
    }
    if (change.kind == FileChangeKind::Renamed && change.previous_path.empty()) {
        return fail("kept no record of where it was moved from");
    }
    std::string restored = lineage.content.value_or(std::string{});
    if (!change.diff.empty()) {
        const auto hunks = detail::parse_hunks(change.diff);
        if (!hunks) {
            return fail("kept a diff this build cannot read");
        }
        auto before = detail::reverse_apply(restored, *hunks);
        if (!before) {
            return fail(kChangedOutside);
        }
        restored = std::move(*before);
    } else if (change.content != FileChangeContent::Text) {
        return fail(undiffed_reason(change.content));
    }
    // An empty text diff means both sides were byte-identical, so a file the
    // turn created or deleted was empty on the side where it existed.
    if (!existed_before && !restored.empty()) {
        return fail(kChangedOutside);
    }
    lineage.content = existed_before ? std::optional(std::move(restored)) : std::nullopt;
}

/// The net change a fully unwound lineage describes, or nothing when the
/// conversation left the file as it found it.
[[nodiscard]] std::optional<FileChange> net_change(const Lineage& lineage) {
    const auto& before = lineage.content;
    const auto& after = lineage.current_content;
    if (!before && !after) {
        return std::nullopt;  // Created and deleted again.
    }
    const bool moved = lineage.path != lineage.current_path;
    if (before == after && !moved) {
        return std::nullopt;
    }

    FileChange change{.path = lineage.current_path};
    if (!before) {
        change.kind = FileChangeKind::Added;
    } else if (!after) {
        change.kind = FileChangeKind::Deleted;
    } else if (moved) {
        change.kind = FileChangeKind::Renamed;
        change.previous_path = lineage.path;
    }
    if (before == after) {
        return change;  // A pure rename.
    }

    auto outcome = core::tools::detail::build_unified_diff_detailed(
        change.path, before.value_or(std::string{}), after.value_or(std::string{}));
    if (outcome.counts) {
        change.added = outcome.counts->first;
        change.deleted = outcome.counts->second;
    }
    if (!outcome.has_diff()) {
        change.content = FileChangeContent::TooLarge;
        return change;
    }
    change.diff = std::move(*outcome.diff);
    return change;
}

void merge_flags(TurnChanges& into, const TurnChanges& turn) {
    into.partial_enumeration = into.partial_enumeration || turn.partial_enumeration;
    into.ambiguous_ownership = into.ambiguous_ownership || turn.ambiguous_ownership;
    if (turn.unscoped_mutations) {
        // Workspace coverage holds only if every unscoped turn had an
        // inventory. Scoped-only turns do not weaken that coverage.
        into.workspace_observed = into.unscoped_mutations
            ? into.workspace_observed && turn.workspace_observed
            : turn.workspace_observed;
    }
    into.unscoped_mutations = into.unscoped_mutations || turn.unscoped_mutations;
    for (const auto& tool : turn.unscoped_tools) {
        if (std::ranges::find(into.unscoped_tools, tool) == into.unscoped_tools.end()) {
            into.unscoped_tools.push_back(tool);
        }
    }
}

} // namespace

ConversationChanges conversation_changes(std::span<const TurnChanges> turns,
                                         const std::filesystem::path& workspace_root) {
    std::vector<Lineage> lineages;
    // Each followed file by the path it had after the turn about to be
    // unwound. A turn's changes are independent of one another, so they are
    // all looked up before any of them is re-keyed: a turn that renamed a to b
    // and created a new a must not have its two halves mistaken for one file.
    std::map<std::string, std::size_t> by_path;
    ConversationChanges result;
    for (const auto& turn : turns) {
        if (!turn.reverted) {
            merge_flags(result.changes, turn);  // Oldest first, so tools keep their order.
        }
    }

    for (const auto& turn : std::views::reverse(turns)) {
        if (turn.reverted) {
            continue;
        }

        std::vector<std::size_t> touched;
        touched.reserve(turn.files.size());
        for (const auto& change : turn.files) {
            if (const auto found = by_path.find(change.path); found != by_path.end()) {
                touched.push_back(found->second);
                by_path.erase(found);
                continue;
            }
            touched.push_back(lineages.size());
            lineages.push_back(start_lineage(change.path, workspace_root));
        }
        for (std::size_t i = 0; i < touched.size(); ++i) {
            Lineage& lineage = lineages[touched[i]];
            unwind(lineage, turn.files[i]);
            if (!by_path.emplace(lineage.path, touched[i]).second && !lineage.failure) {
                lineage.failure = "was replaced by another file in the same turn";
            }
        }
    }

    std::ranges::sort(lineages, {}, &Lineage::current_path);
    for (const auto& lineage : lineages) {
        if (lineage.failure) {
            result.gaps.push_back({.path = lineage.current_path, .reason = *lineage.failure});
        } else if (auto change = net_change(lineage)) {
            result.changes.files.push_back(std::move(*change));
        }
    }
    return result;
}

} // namespace core::changes
