#pragma once

#include "SourceControlProvider.hpp"
#include <filesystem>
#include <string>

namespace core::scm {

class GitProvider : public SourceControlProvider {
public:
  explicit GitProvider(std::filesystem::path root_dir);

  [[nodiscard]] std::string name() const override { return "git"; }
  [[nodiscard]] bool
  is_ignored(const std::filesystem::path &path) const override;
  [[nodiscard]] std::string get_status_summary() const override;
  [[nodiscard]] std::filesystem::path get_root_dir() const override {
    return root_dir_;
  }

private:
  std::filesystem::path root_dir_;

  // Helper to run git commands in the root_dir_
  std::string run_git_command(const std::string &args) const;
};

} // namespace core::scm
