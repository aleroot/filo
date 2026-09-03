#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace core::scm {

struct StatusItem {
  std::string path;
  char status_code = '?'; // 'M'odified, 'A'dded, 'D'eleted, '?'Untracked, etc.
};

struct BranchRef {
  std::string name;
  std::string description;
};

struct RepositorySnapshot {
  std::filesystem::path root;
  std::string branch;
  std::string revision;
  std::vector<StatusItem> changes;
  std::uint64_t status_fingerprint = 0;
};

/**
 * @brief Abstract interface for Source Control Management systems (Git, Hg,
 * etc.).
 *
 * Follows the Strategy pattern to allow runtime selection of the SCM
 * implementation based on the repository detected in the working directory.
 */
class SourceControlProvider {
public:
  virtual ~SourceControlProvider() = default;

  /**
   * @brief Returns the name of the SCM system (e.g., "git", "hg").
   */
  [[nodiscard]] virtual std::string name() const = 0;

  /**
   * @brief Checks if the given path is ignored by the SCM (e.g., .gitignore).
   */
  [[nodiscard]] virtual bool
  is_ignored(const std::filesystem::path &path) const = 0;

  /**
   * @brief Batch variant used by bounded directory scans.
   *
   * Providers with an efficient bulk query should return one flag per input
   * path. Returning nullopt asks the caller to use is_ignored() instead.
  */
  [[nodiscard]] virtual std::optional<std::vector<bool>>
  ignored_paths(const std::vector<std::filesystem::path> &) const {
    return std::nullopt;
  }

  /**
   * @brief Returns a short textual summary of the current status (changed
   * files, branch). Designed for injection into LLM context.
   */
  [[nodiscard]] virtual std::string get_status_summary() const = 0;

  /**
   * @brief Returns branch-like refs that can be used as comparison bases.
   */
  [[nodiscard]] virtual std::vector<BranchRef> list_branch_refs() const {
    return {};
  }

  /// Stable Git-like repository identity and working-tree inventory used by
  /// AUTO's transaction coordinator. Unsupported SCMs return nullopt.
  [[nodiscard]] virtual std::optional<RepositorySnapshot>
  repository_snapshot() const {
    return std::nullopt;
  }

  /// Bounded read-only patch evidence for independent review. Includes staged
  /// and unstaged tracked changes; callers also supply the status inventory
  /// so reviewers can inspect untracked files using their normal read tools.
  [[nodiscard]] virtual std::optional<std::string> review_patch() const {
    return std::nullopt;
  }

  /**
   * @brief Returns the root directory of the repository.
   */
  [[nodiscard]] virtual std::filesystem::path get_root_dir() const = 0;
};

} // namespace core::scm
