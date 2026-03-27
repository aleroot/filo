#include "ScmFactory.hpp"
#include "GitProvider.hpp"
#include "NoOpProvider.hpp"
#include <filesystem>

namespace core::scm {

std::unique_ptr<SourceControlProvider>
ScmFactory::create(const std::filesystem::path &start_dir) {
  std::filesystem::path current = std::filesystem::absolute(start_dir);
  std::filesystem::path root = current.root_path();

  // Walk up looking for markers
  while (true) {
    if (std::filesystem::exists(current / ".git")) {
      return std::make_unique<GitProvider>(current);
    }
    // Future: if (exists(current / ".hg")) return
    // make_unique<HgProvider>(current);

    if (current == root || current.filename().empty()) {
      break;
    }
    current = current.parent_path();
  }

  // Default: No SCM found, treat start_dir as root for the NoOp provider
  return std::make_unique<NoOpProvider>(start_dir);
}

} // namespace core::scm
