#pragma once

#include <filesystem>
#include <optional>
#include <string_view>

namespace tui {

/// A local file a link points at, and the line to show first.
struct DocumentTarget {
    /// As the link wrote it: possibly relative, possibly starting with `~`.
    std::filesystem::path path;
    /// One-based.
    std::optional<int> line;

    friend bool operator==(const DocumentTarget&, const DocumentTarget&) = default;
};

/// The local file a Markdown link target names, or nothing for a web or
/// other URL, an in-page anchor, or an empty target.
///
/// Models of different providers spell file links differently, so every
/// common form is read: `/abs/path`, `relative/path`, `~/path`,
/// `<path with spaces>`, `file:///abs/path`, percent-encoded characters, a
/// trailing `"title"`, and a line as `path:12`, `path:12:3`, `path#L12`,
/// `path#L12-L20` or `path#L12C3` (a column or range end is dropped).
[[nodiscard]] std::optional<DocumentTarget> parse_document_link(std::string_view href);

/// @p target with `~` expanded and a relative path made absolute against
/// @p base, lexically normalized. Nothing is read from the disk.
[[nodiscard]] DocumentTarget resolve_document_target(DocumentTarget target,
                                                     const std::filesystem::path& base);

} // namespace tui
