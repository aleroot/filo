#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace core::scm {

struct StatusItem {
  std::string path;
  char status_code; // 'M'odified, 'A'dded, 'D'eleted, '?'Untracked, etc.
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
   * @brief Returns a short textual summary of the current status (changed
   * files, branch). Designed for injection into LLM context.
   */
  [[nodiscard]] virtual std::string get_status_summary() const = 0;

  /**
   * @brief Returns the root directory of the repository.
   */
  [[nodiscard]] virtual std::filesystem::path get_root_dir() const = 0;
};

} // namespace core::scm
