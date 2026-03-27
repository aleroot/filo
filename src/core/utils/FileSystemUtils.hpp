#pragma once

#include "../scm/SourceControlProvider.hpp"
#include <filesystem>
#include <string>

namespace core::utils {

// Generates a tree-like string of the directory structure.
// Uses the SCM provider to respect ignore rules (like .gitignore).
std::string get_file_tree(const std::filesystem::path &root,
                          const core::scm::SourceControlProvider &scm,
                          int max_depth = 2);

} // namespace core::utils
