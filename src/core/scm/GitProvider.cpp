#include "GitProvider.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>

namespace core::scm {

namespace {
// Basic execution helper - in a real app, might want to use a more robust
// subprocess library or the existing ShellUtils if adapted for capturing output
// without a shell session.
std::string exec_cmd(const std::string &cmd) {
  std::array<char, 128> buffer;
  std::string result;
  // Redirect stderr to /dev/null
  std::string full_cmd = cmd + " 2>/dev/null";
  std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(full_cmd.c_str(), "r"),
                                            pclose);
  if (!pipe)
    return "";
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}
} // namespace

GitProvider::GitProvider(std::filesystem::path root_dir)
    : root_dir_(std::move(root_dir)) {}

std::string GitProvider::run_git_command(const std::string &args) const {
  // Ensure we run from the repo root
  std::string cmd = "git -C \"" + root_dir_.string() + "\" " + args;
  return exec_cmd(cmd);
}

bool GitProvider::is_ignored(const std::filesystem::path &path) const {
  // git check-ignore -q <path> returns 0 if ignored, 1 if not
  std::string cmd = "git -C \"" + root_dir_.string() + "\" check-ignore -q \"" +
                    path.string() + "\"";
  int ret = system(cmd.c_str());
  // check-ignore returns 0 if ignored
  // system returns status code (exit code in high bits usually, but strictly,
  // WEXITSTATUS) For simplicity with system(): If command succeeds (found in
  // ignore list), it returns 0. If it fails (not ignored), it returns 1.
  return (ret == 0);
}

std::string GitProvider::get_status_summary() const {
  // --branch: Show branch info
  // --short: Concise output
  // --no-optional-locks: Avoid taking locks
  std::string status = run_git_command(
      "status --short --branch --no-ahead-behind --no-optional-locks");
  if (status.empty())
    return "";

  // Optional: Trim whitespace
  while (!status.empty() && std::isspace(status.back()))
    status.pop_back();
  return status;
}

} // namespace core::scm
