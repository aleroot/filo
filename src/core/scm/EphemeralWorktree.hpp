#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::scm {

// A throwaway Git worktree for one isolated Boost workstream.
//
// The worktree is seeded from HEAD plus the parent's uncommitted tracked
// changes, so a candidate starts from what the user actually has rather than
// from a stale commit. It shares the object store, so creation is cheap even
// for large repositories, and it is removed on destruction.
//
// Nothing written inside ever reaches the user's checkout: the only output is
// `patch()`, which the parent agent reviews as evidence. That keeps the
// single-writer invariant intact while still allowing genuinely parallel
// implementation attempts.
class EphemeralWorktree final {
public:
  EphemeralWorktree() = default;
  EphemeralWorktree(EphemeralWorktree &&other) noexcept;
  EphemeralWorktree &operator=(EphemeralWorktree &&other) noexcept;
  EphemeralWorktree(const EphemeralWorktree &) = delete;
  EphemeralWorktree &operator=(const EphemeralWorktree &) = delete;
  ~EphemeralWorktree();

  /// Returns nullopt when `repository_root` is not a Git checkout, has no
  /// commit yet, or the worktree could not be created. Callers must treat
  /// that as "isolation unavailable", never as a failure of the turn.
  [[nodiscard]] static std::optional<EphemeralWorktree>
  create(const std::filesystem::path &repository_root, std::string_view label);

  [[nodiscard]] bool valid() const noexcept { return !root_.empty(); }
  [[nodiscard]] const std::filesystem::path &root() const noexcept {
    return root_;
  }

  /// Everything changed inside the worktree relative to its seeded state,
  /// including new files. Bounded; a truncated patch is annotated so a
  /// reviewer never mistakes it for the whole change.
  [[nodiscard]] std::optional<std::string>
  patch(std::size_t limit = 256 * 1024) const;

  /// Remove the worktree now instead of at destruction.
  void release() noexcept;

private:
  EphemeralWorktree(std::filesystem::path repository_root,
                    std::filesystem::path root)
      : repository_root_(std::move(repository_root)), root_(std::move(root)) {}

  std::filesystem::path repository_root_;
  std::filesystem::path root_;
};

} // namespace core::scm
