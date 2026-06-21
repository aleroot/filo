#include "GitProvider.hpp"
#include <array>
#include <cctype>
#include <cstdio>
#include <memory>
#include <sstream>
#include <string_view>

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

std::string shell_single_quote(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('\'');
  for (const char ch : value) {
    if (ch == '\'') {
      out += "'\\''";
    } else {
      out.push_back(ch);
    }
  }
  out.push_back('\'');
  return out;
}

std::string git_command(const std::filesystem::path &root_dir,
                        std::string_view args) {
  return "git -C " + shell_single_quote(root_dir.string()) + " " + std::string(args);
}

std::string capture_git(const std::filesystem::path &root_dir,
                        std::string_view args) {
  return exec_cmd(git_command(root_dir, args));
}

} // namespace

GitProvider::GitProvider(std::filesystem::path root_dir)
    : root_dir_(std::move(root_dir)) {}

bool GitProvider::is_ignored(const std::filesystem::path &path) const {
  // git check-ignore -q <path> returns 0 if ignored, 1 if not
  const std::string cmd = git_command(
      root_dir_,
      "check-ignore -q -- " + shell_single_quote(path.string()));
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
  std::string status = capture_git(
      root_dir_,
      "status --short --branch --no-ahead-behind --no-optional-locks");
  if (status.empty())
    return "";

  // Optional: Trim whitespace
  while (!status.empty() && std::isspace(status.back()))
    status.pop_back();
  return status;
}

std::vector<BranchRef> GitProvider::list_branch_refs() const {
  const std::string output = capture_git(
      root_dir_,
      "for-each-ref --sort=refname --format='%(refname:short)%09%(upstream:short)' "
      "refs/heads refs/remotes");
  if (output.empty()) {
    return {};
  }

  std::vector<BranchRef> refs;
  std::stringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const std::size_t tab = line.find('\t');
    std::string name = tab == std::string::npos ? line : line.substr(0, tab);
    if (name.empty() || name.ends_with("/HEAD")) {
      continue;
    }
    const std::string upstream = tab == std::string::npos ? std::string() : line.substr(tab + 1);
    refs.push_back(BranchRef{
        .name = std::move(name),
        .description = upstream.empty() ? std::string() : "tracks " + upstream,
    });
  }

  return refs;
}

} // namespace core::scm
