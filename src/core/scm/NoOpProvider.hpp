#pragma once

#include "SourceControlProvider.hpp"
#include <algorithm>
#include <filesystem>
#include <set>

namespace core::scm {

class NoOpProvider : public SourceControlProvider {
public:
  explicit NoOpProvider(std::filesystem::path root_dir)
      : root_dir_(std::move(root_dir)) {}

  [[nodiscard]] std::string name() const override { return "none"; }

  [[nodiscard]] bool
  is_ignored(const std::filesystem::path &path) const override {
    // Fallback: ignore common artifacts to keep context clean
    static const std::set<std::string> ignored_names = {
        ".git",      ".hg",          ".svn",        ".vscode", ".idea",
        ".DS_Store", "node_modules", "build",       "dist",    "target",
        "venv",      ".venv",        "__pycache__", "obj",     "bin"};

    // Check filename against blocklist
    if (ignored_names.contains(path.filename().string())) {
      return true;
    }

    // Also simple check if any part of the path is hidden (starts with .)
    // except for the current directory itself "."
    std::string filename = path.filename().string();
    if (filename.size() > 1 && filename[0] == '.' && filename != "..") {
      // Often hidden files are ignored in context unless explicitly asked for,
      // but let's be conservative and only ignore specific ones above for now
      // to avoid hiding config files the user might want.
      // Actually, usually we do want to see .env or .gitignore, so let's
      // not auto-ignore all dotfiles.
    }

    return false;
  }

  [[nodiscard]] std::string get_status_summary() const override {
    return ""; // No VCS status available
  }

  [[nodiscard]] std::filesystem::path get_root_dir() const override {
    return root_dir_;
  }

private:
  std::filesystem::path root_dir_;
};

} // namespace core::scm
