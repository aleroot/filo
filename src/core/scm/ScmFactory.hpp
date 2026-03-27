#pragma once

#include "SourceControlProvider.hpp"
#include <filesystem>
#include <memory>

namespace core::scm {

class ScmFactory {
public:
  /**
   * @brief Detects the SCM system for the given directory and returns the
   * appropriate provider.
   *
   * Walks up the directory tree looking for .git, .hg, etc.
   * If no SCM is found, returns a NoOpProvider.
   */
  static std::unique_ptr<SourceControlProvider>
  create(const std::filesystem::path &dir);
};

} // namespace core::scm
