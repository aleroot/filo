#include "FileSystemUtils.hpp"
#include <algorithm>
#include <sstream>
#include <vector>

namespace core::utils {

namespace {

void build_tree_recursive(const std::filesystem::path &current_path,
                          const core::scm::SourceControlProvider &scm,
                          int current_depth, int max_depth,
                          std::ostringstream &oss, std::string prefix) {
  if (current_depth > max_depth)
    return;

  std::vector<std::filesystem::directory_entry> entries;
  try {
    for (const auto &entry :
         std::filesystem::directory_iterator(current_path)) {
      // Use SCM to check if ignored
      if (scm.is_ignored(entry.path()))
        continue;
      entries.push_back(entry);
    }
  } catch (const std::filesystem::filesystem_error &) {
    return;
  }

  // Sort: directories first, then files
  std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
    if (a.is_directory() != b.is_directory()) {
      return a.is_directory();
    }
    return a.path().filename() < b.path().filename();
  });

  for (size_t i = 0; i < entries.size(); ++i) {
    const auto &entry = entries[i];
    bool is_last = (i == entries.size() - 1);

    oss << prefix;
    oss << (is_last ? "└── " : "├── ");
    oss << entry.path().filename().string();

    if (entry.is_directory()) {
      oss << "/\n";
      build_tree_recursive(entry.path(), scm, current_depth + 1, max_depth, oss,
                           prefix + (is_last ? "    " : "│   "));
    } else {
      oss << "\n";
    }
  }
}

} // namespace

std::string get_file_tree(const std::filesystem::path &root,
                          const core::scm::SourceControlProvider &scm,
                          int max_depth) {
  std::ostringstream oss;
  if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root)) {
    return "";
  }

  build_tree_recursive(root, scm, 1, max_depth, oss, "");
  return oss.str();
}

} // namespace core::utils
