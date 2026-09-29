#pragma once

#include "FileChange.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace core::changes {

/// A patch ready to hand to something outside Filo: an external comparer, a
/// pager, or the clipboard.
///
/// Only changes that carry line-level detail can be exported; the rest are
/// counted so a caller can say what it left out instead of implying the patch
/// is the whole turn.
struct PatchExport {
    /// Git-shaped unified diff. Empty when nothing could be exported.
    std::string patch;
    /// Files `patch` describes.
    std::size_t files = 0;
    /// Changed files listed without a diff: binary, beyond a size budget, or
    /// content-identical (a pure rename).
    std::size_t undiffed = 0;

    [[nodiscard]] bool empty() const noexcept { return files == 0; }
};

/// Exports one change.
[[nodiscard]] PatchExport export_patch(const FileChange& change);

/// Exports every change it can, in the order given.
[[nodiscard]] PatchExport export_patch(std::span<const FileChange> changes);

/// Exports a whole turn.
[[nodiscard]] PatchExport export_patch(const TurnChanges& changes);

} // namespace core::changes
