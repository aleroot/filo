#include "PatchExport.hpp"

#include <format>
#include <string_view>

namespace core::changes {
namespace {

/// The header pair `git diff` puts in front of one file's hunks.
///
/// The tracker diffs content, not paths, so the pair it stored names the
/// destination twice. A rename restores the origin on the `---` side and says
/// so explicitly, which is what lets a comparer title both revisions and what
/// makes the export readable by anything that expects a git patch.
[[nodiscard]] std::string headers(const FileChange& change) {
    const bool renamed = change.kind == FileChangeKind::Renamed
        && !change.previous_path.empty()
        && change.previous_path != change.path;
    const std::string& origin = renamed ? change.previous_path : change.path;

    std::string section = std::format("diff --git a/{} b/{}\n", origin, change.path);
    if (renamed) {
        section += std::format("rename from {}\nrename to {}\n", origin, change.path);
    }
    return section + std::format("--- a/{}\n+++ b/{}\n", origin, change.path);
}

/// The stored diff with its own header pair removed, since headers() wrote the
/// pair this export wants. A diff that does not open with a pair is returned
/// whole rather than guessed at.
[[nodiscard]] std::string_view hunks(const FileChange& change) {
    const std::string_view diff = change.diff;
    if (!diff.starts_with("--- ")) {
        return diff;
    }
    const std::size_t first = diff.find('\n');
    if (first == std::string_view::npos) {
        return diff;
    }
    const std::size_t second = diff.find('\n', first + 1);
    if (second == std::string_view::npos
        || !diff.substr(first + 1, second - first - 1).starts_with("+++ ")) {
        return diff;
    }
    return diff.substr(second + 1);
}

} // namespace

PatchExport export_patch(const FileChange& change) {
    return export_patch(std::span{&change, 1});
}

PatchExport export_patch(std::span<const FileChange> changes) {
    PatchExport result;
    for (const auto& change : changes) {
        // A change with no diff is still a change; only its line-level detail
        // is missing, so it is counted instead of silently dropped.
        if (change.diff.empty()) {
            ++result.undiffed;
            continue;
        }
        result.patch += headers(change);
        result.patch += hunks(change);
        ++result.files;
    }
    return result;
}

PatchExport export_patch(const TurnChanges& changes) {
    return export_patch(changes.files);
}

} // namespace core::changes
