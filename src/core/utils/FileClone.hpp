#pragma once

#include <expected>
#include <filesystem>
#include <system_error>

namespace core::utils {

/// Makes `target`, which must not exist, a copy-on-write clone of the regular
/// file `source` (APFS clonefile, Linux FICLONE on Btrfs/XFS): it shares the
/// source's blocks, so cloning writes metadata, never contents. The final
/// component of `source` is not followed. On failure nothing is left at
/// `target`.
[[nodiscard]] std::expected<void, std::error_code> clone_file(const std::filesystem::path& source,
                                                              const std::filesystem::path& target) noexcept;

/// Whether a clone_file error means this filesystem pair cannot clone at all,
/// as opposed to a failure of this one attempt.
[[nodiscard]] bool clone_unsupported(std::error_code error) noexcept;

} // namespace core::utils
