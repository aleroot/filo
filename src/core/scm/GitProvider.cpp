#include "GitProvider.hpp"
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <unistd.h>
#include <sys/wait.h>

namespace core::scm {

namespace {
// Small bounded subprocess wrapper for Git's read-only plumbing commands.
struct CommandResult {
  std::string output;
  int exit_code = -1;
  bool truncated = false;
  std::uint64_t fingerprint = 14695981039346656037ULL;
};

CommandResult exec_cmd(
    const std::string &cmd,
    std::size_t max_output = std::numeric_limits<std::size_t>::max()) {
  std::array<char, 128> buffer;
  CommandResult result;
  // Redirect stderr to /dev/null
  std::string full_cmd = cmd + " 2>/dev/null";
  std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(full_cmd.c_str(), "r"),
                                            pclose);
  if (!pipe)
    return result;
  while (const auto count = std::fread(
             buffer.data(), sizeof(char), buffer.size(), pipe.get())) {
    for (std::size_t i = 0; i < count; ++i) {
      result.fingerprint ^= static_cast<unsigned char>(buffer[i]);
      result.fingerprint *= 1099511628211ULL;
    }
    const std::size_t remaining = max_output - std::min(max_output, result.output.size());
    const std::size_t kept = std::min(count, remaining);
    result.output.append(buffer.data(), kept);
    result.truncated = result.truncated || kept != count;
  }

  const int status = pclose(pipe.release());
  if (status >= 0 && WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
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
  const auto result = exec_cmd(git_command(root_dir, args));
  return result.exit_code == 0 ? result.output : std::string{};
}

class ScopedTempFile {
public:
  ScopedTempFile() = default;
  explicit ScopedTempFile(std::string path) : path_(std::move(path)) {}
  ScopedTempFile(const ScopedTempFile&) = delete;
  ScopedTempFile& operator=(const ScopedTempFile&) = delete;
  ScopedTempFile(ScopedTempFile&& other) noexcept
      : path_(std::exchange(other.path_, {})) {}
  ScopedTempFile& operator=(ScopedTempFile&& other) noexcept {
    if (this != &other) {
      remove();
      path_ = std::exchange(other.path_, {});
    }
    return *this;
  }
  ~ScopedTempFile() { remove(); }

  [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
  void remove() noexcept {
    if (path_.empty()) return;
    std::error_code ec;
    std::filesystem::remove(path_, ec);
    path_.clear();
  }

  std::string path_;
};

std::optional<ScopedTempFile> write_temp_input(std::string_view input) {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "filo-git-ignore-XXXXXX").string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');

  const int fd = mkstemp(writable.data());
  if (fd < 0) return std::nullopt;

  std::size_t written = 0;
  while (written < input.size()) {
    const auto count = ::write(
        fd,
        input.data() + written,
        input.size() - written);
    if (count <= 0) {
      ::close(fd);
      std::error_code ec;
      std::filesystem::remove(writable.data(), ec);
      return std::nullopt;
    }
    written += static_cast<std::size_t>(count);
  }
  if (::close(fd) != 0) {
    std::error_code ec;
    std::filesystem::remove(writable.data(), ec);
    return std::nullopt;
  }
  return ScopedTempFile{writable.data()};
}

} // namespace

GitProvider::GitProvider(std::filesystem::path root_dir)
    : root_dir_(std::filesystem::absolute(std::move(root_dir)).lexically_normal()) {}

bool GitProvider::is_ignored(const std::filesystem::path &path) const {
  const auto ignored = ignored_paths({path});
  return ignored.has_value() && ignored->size() == 1 && (*ignored)[0];
}

std::optional<std::vector<bool>> GitProvider::ignored_paths(
    const std::vector<std::filesystem::path> &paths) const {
  std::vector<bool> result(paths.size(), false);
  if (paths.empty()) return result;

  std::vector<std::string> relative_paths(paths.size());
  std::string input;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    const auto relative = paths[i].lexically_relative(root_dir_).lexically_normal();
    const auto first = relative.begin();
    if (relative.empty() || first == relative.end() || *first == "..") {
      result[i] = true;
      continue;
    }

    relative_paths[i] = relative.generic_string();
    input += relative_paths[i];
    input.push_back('\0');
  }

  if (input.empty()) return result;
  auto temp = write_temp_input(input);
  if (!temp.has_value()) return std::nullopt;

  const auto command = git_command(root_dir_, "check-ignore --stdin -z")
      + " < " + shell_single_quote(temp->path());
  const auto checked = exec_cmd(command);
  // check-ignore returns 1 when none of the supplied paths are ignored.
  if (checked.exit_code != 0 && checked.exit_code != 1) return std::nullopt;

  std::unordered_set<std::string> ignored;
  std::size_t cursor = 0;
  while (cursor < checked.output.size()) {
    const auto end = checked.output.find('\0', cursor);
    if (end == std::string::npos) break;
    ignored.emplace(checked.output.substr(cursor, end - cursor));
    cursor = end + 1;
  }
  for (std::size_t i = 0; i < relative_paths.size(); ++i) {
    if (!relative_paths[i].empty()) {
      result[i] = ignored.contains(relative_paths[i]);
    }
  }
  return result;
}

std::string GitProvider::get_status_summary() const {
  // --branch: Show branch info
  // --short: Concise output
  // --no-optional-locks: Avoid taking locks
  constexpr std::size_t kMaxStatusChars = 12 * 1024;
  const auto captured = exec_cmd(git_command(
      root_dir_,
      "--no-optional-locks status --porcelain=v1 --branch --no-ahead-behind "
      "--untracked-files=normal"),
      kMaxStatusChars);
  if (captured.exit_code != 0) return {};
  std::string status = captured.output;
  if (status.empty())
    return "";

  // Optional: Trim whitespace
  while (!status.empty()
         && std::isspace(static_cast<unsigned char>(status.back())))
    status.pop_back();
  if (captured.truncated) {
    char fingerprint[17]{};
    const auto [end, error] = std::to_chars(
        fingerprint, fingerprint + sizeof(fingerprint) - 1,
        captured.fingerprint, 16);
    const std::string marker = error == std::errc{}
        ? "\n… (status truncated; fingerprint "
            + std::string(fingerprint, end) + ")"
        : "\n… (status truncated)";
    if (status.size() + marker.size() > kMaxStatusChars) {
      status.resize(kMaxStatusChars - marker.size());
      if (const auto newline = status.rfind('\n'); newline != std::string::npos) {
        status.resize(newline);
      }
    }
    status += marker;
  }
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
