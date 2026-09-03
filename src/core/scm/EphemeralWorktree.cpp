#include "EphemeralWorktree.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

namespace core::scm {

namespace {

struct CommandResult {
  std::string output;
  int exit_code = -1;
  bool truncated = false;
};

CommandResult run(const std::string &command,
                  std::size_t max_output =
                      std::numeric_limits<std::size_t>::max()) {
  std::array<char, 4096> buffer{};
  CommandResult result;
  const std::string full = command + " 2>&1";
  std::unique_ptr<FILE, int (*)(FILE *)> pipe(popen(full.c_str(), "r"), pclose);
  if (!pipe)
    return result;
  while (const auto count =
             std::fread(buffer.data(), sizeof(char), buffer.size(), pipe.get())) {
    const std::size_t remaining =
        max_output - std::min(max_output, result.output.size());
    const std::size_t kept = std::min(count, remaining);
    result.output.append(buffer.data(), kept);
    result.truncated = result.truncated || kept != count;
  }
  const int status = pclose(pipe.release());
  if (status >= 0 && WIFEXITED(status))
    result.exit_code = WEXITSTATUS(status);
  return result;
}

std::string quote(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('\'');
  for (const char ch : value) {
    if (ch == '\'')
      out += "'\\''";
    else
      out.push_back(ch);
  }
  out.push_back('\'');
  return out;
}

std::string git(const std::filesystem::path &root, std::string_view args) {
  return "git -C " + quote(root.string()) + " " + std::string(args);
}

// Worktree directory names are derived from a model-supplied label, so they
// are reduced to a conservative alphabet before ever reaching a path.
std::string sanitize(std::string_view label) {
  std::string out;
  for (const unsigned char ch : label) {
    if (std::isalnum(ch))
      out.push_back(static_cast<char>(std::tolower(ch)));
    else if (!out.empty() && out.back() != '-')
      out.push_back('-');
    if (out.size() >= 24)
      break;
  }
  while (!out.empty() && out.back() == '-')
    out.pop_back();
  return out.empty() ? std::string("worker") : out;
}

std::string unique_suffix() {
  static std::atomic_uint64_t counter{0};
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::to_string(static_cast<std::uint64_t>(now)) + "-" +
         std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + "-" +
         std::to_string(static_cast<std::uint64_t>(::getpid()));
}

// Copy `.filo/` into the isolated copy. Bounded: the directory is meant to hold
// small configuration, and a runaway cache must not be duplicated per candidate.
void copy_project_config(const std::filesystem::path &repository_root,
                         const std::filesystem::path &worktree_root) {
  constexpr std::uintmax_t kMaxBytes = 8ULL * 1024 * 1024;
  std::error_code ec;
  const auto source = repository_root / ".filo";
  if (!std::filesystem::is_directory(source, ec))
    return;

  std::uintmax_t total = 0;
  for (std::filesystem::recursive_directory_iterator it(
           source, std::filesystem::directory_options::skip_permission_denied,
           ec), end;
       it != end && !ec; it.increment(ec)) {
    if (it->is_regular_file(ec))
      total += it->file_size(ec);
    if (total > kMaxBytes)
      return;
  }
  if (ec)
    return;
  std::filesystem::copy(
      source, worktree_root / ".filo",
      std::filesystem::copy_options::recursive |
          std::filesystem::copy_options::overwrite_existing |
          std::filesystem::copy_options::skip_symlinks,
      ec);
}

// Commits are made with an explicit identity so the worktree works even when
// the machine has no configured user, and never runs repository hooks.
constexpr std::string_view kCommitFlags =
    "-c user.email=boost@filo.local -c user.name=Filo "
    "-c commit.gpgsign=false -c core.hooksPath=/dev/null";

} // namespace

EphemeralWorktree::EphemeralWorktree(EphemeralWorktree &&other) noexcept
    : repository_root_(std::move(other.repository_root_)),
      root_(std::move(other.root_)) {
  other.root_.clear();
}

EphemeralWorktree &
EphemeralWorktree::operator=(EphemeralWorktree &&other) noexcept {
  if (this != &other) {
    release();
    repository_root_ = std::move(other.repository_root_);
    root_ = std::move(other.root_);
    other.root_.clear();
  }
  return *this;
}

EphemeralWorktree::~EphemeralWorktree() { release(); }

std::optional<EphemeralWorktree>
EphemeralWorktree::create(const std::filesystem::path &repository_root,
                          std::string_view label) {
  std::error_code ec;
  if (repository_root.empty() ||
      !std::filesystem::is_directory(repository_root, ec))
    return std::nullopt;

  // A repository without a commit has nothing to branch a worktree from.
  if (run(git(repository_root, "rev-parse --verify --quiet HEAD")).exit_code != 0)
    return std::nullopt;

  const auto base = std::filesystem::temp_directory_path(ec);
  if (ec)
    return std::nullopt;
  const auto root =
      base / ("filo-boost-" + sanitize(label) + "-" + unique_suffix());
  if (std::filesystem::exists(root, ec))
    return std::nullopt;

  if (run(git(repository_root,
              "worktree add --detach --quiet " + quote(root.string()) + " HEAD"))
          .exit_code != 0) {
    std::filesystem::remove_all(root, ec);
    return std::nullopt;
  }
  EphemeralWorktree worktree(repository_root, root);

  // Filo's own project configuration is conventionally gitignored, so a plain
  // worktree checkout would not contain it. Without it the copy discovers no
  // verification recipes and a candidate could never be proven — copy it in.
  // It stays ignored, so it never appears in the candidate diff.
  copy_project_config(repository_root, root);

  // Seed the parent's uncommitted tracked changes so the candidate starts from
  // the user's real state. Untracked files are deliberately left out: they are
  // frequently large build output, and a candidate that needs one can create
  // it. A seeding failure is not fatal — it only means a HEAD-clean start.
  const auto dirty = run(git(repository_root,
      "--no-pager diff --no-ext-diff --no-textconv --no-color --binary HEAD --"));
  if (dirty.exit_code == 0 && !dirty.output.empty() && !dirty.truncated) {
    const auto patch_file = root / ".filo-boost-seed.patch";
    {
      std::ofstream out(patch_file, std::ios::binary);
      out << dirty.output;
    }
    static_cast<void>(run(git(root, "apply --whitespace=nowarn " +
                                        quote(patch_file.string()))));
    std::filesystem::remove(patch_file, ec);
  }

  // Snapshot the seeded state as a commit so patch() is exactly the worker's
  // own change, including any file it adds.
  if (run(git(root, "add -A")).exit_code != 0 ||
      run(git(root, std::string(kCommitFlags) +
                        " commit --quiet --allow-empty --no-verify "
                        "-m filo-boost-baseline"))
              .exit_code != 0) {
    return std::nullopt; // destructor removes the worktree
  }
  return worktree;
}

std::optional<std::string>
EphemeralWorktree::patch(std::size_t limit) const {
  if (!valid())
    return std::nullopt;
  if (run(git(root_, "add -A")).exit_code != 0)
    return std::nullopt;
  auto diff = run(git(root_, "--no-pager diff --no-ext-diff --no-textconv "
                             "--no-color --cached HEAD --"),
                  limit);
  if (diff.exit_code != 0)
    return std::nullopt;
  if (diff.truncated)
    diff.output +=
        "\n[Candidate diff truncated: inspect the worktree change set before "
        "adopting it wholesale.]\n";
  return diff.output;
}

void EphemeralWorktree::release() noexcept {
  if (root_.empty())
    return;
  std::error_code ec;
  const auto root = std::exchange(root_, {});
  static_cast<void>(run(git(repository_root_,
                            "worktree remove --force " + quote(root.string()))));
  std::filesystem::remove_all(root, ec);
  static_cast<void>(run(git(repository_root_, "worktree prune")));
}

} // namespace core::scm
