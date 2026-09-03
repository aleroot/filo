#include "GitWorkspaceCoordinator.hpp"

#include "ScmFactory.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <mutex>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace core::scm {

struct WorkspaceLeaseState {
  std::shared_timed_mutex mutex;
  std::filesystem::path execution_root;
  // Re-locking from a thread that already owns the mutex is only safe in one
  // direction. Nested SharedRead leases are a deliberate pattern here (a
  // subtree resolves to the same execution root) and map onto a recursive
  // POSIX read lock, which libstdc++ handles. Anything involving an exclusive
  // lock on the same thread deadlocks instead: glibc reports EDEADLK, and
  // libstdc++ 16 turns that into an assert and aborts the process rather than
  // reporting a timeout. Ownership is tracked per mode so those cases fail
  // closed before they can reach the mutex.
  struct ThreadOwnership {
    int shared = 0;
    int exclusive = 0;
  };
  std::mutex owners_mutex;
  std::unordered_map<std::thread::id, ThreadOwnership> owners;
};

namespace {

[[nodiscard]] std::string ascii_lower(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const unsigned char ch : value) {
    result.push_back(static_cast<char>(std::tolower(ch)));
  }
  return result;
}

[[nodiscard]] std::string
canonical_key(const std::filesystem::path &workspace_root) {
  std::error_code error;
  auto path = std::filesystem::weakly_canonical(workspace_root, error);
  if (error) {
    std::error_code absolute_error;
    path = std::filesystem::absolute(workspace_root, absolute_error)
               .lexically_normal();
    if (absolute_error) {
      // Never collapse to an empty key: every unresolvable root would then
      // share one lease and contend with each other for no reason.
      path = workspace_root.lexically_normal();
    }
  }

  std::string key = path.string();
  while (key.size() > 1 && (key.back() == '/' || key.back() == '\\')) {
    key.pop_back();
  }
#if defined(_WIN32) || defined(__APPLE__)
  // Default filesystems on these platforms are case-insensitive, and
  // weakly_canonical does not case-fold. Without this, /repo and /Repo would
  // take two independent leases on one checkout and defeat mutual exclusion.
  key = ascii_lower(key);
#endif
  return key;
}

[[nodiscard]] std::filesystem::path
coordination_root(const std::filesystem::path &workspace_root) {
  try {
    auto provider = ScmFactory::create(workspace_root);
    if (provider && provider->name() == "git") {
      return provider->get_root_dir();
    }
  } catch (...) {
    // Locking the supplied root still provides safe local coordination when
    // repository discovery is temporarily unavailable.
  }
  return workspace_root;
}

// True when taking `access` on this thread would deadlock against a lease the
// same thread already holds. Shared-on-shared is the one safe re-entry.
[[nodiscard]] bool would_self_deadlock(WorkspaceLeaseState &state,
                                       core::goal::WorkspaceAccess access) {
  std::lock_guard lock(state.owners_mutex);
  const auto found = state.owners.find(std::this_thread::get_id());
  if (found == state.owners.end())
    return false;
  return found->second.exclusive > 0 ||
         access == core::goal::WorkspaceAccess::ExclusiveWrite;
}

void add_owner(WorkspaceLeaseState &state,
               core::goal::WorkspaceAccess access) {
  std::lock_guard lock(state.owners_mutex);
  auto &entry = state.owners[std::this_thread::get_id()];
  if (access == core::goal::WorkspaceAccess::ExclusiveWrite)
    ++entry.exclusive;
  else
    ++entry.shared;
}

void remove_owner(WorkspaceLeaseState &state,
                  core::goal::WorkspaceAccess access) {
  std::lock_guard lock(state.owners_mutex);
  const auto found = state.owners.find(std::this_thread::get_id());
  if (found == state.owners.end())
    return;
  if (access == core::goal::WorkspaceAccess::ExclusiveWrite)
    found->second.exclusive = std::max(0, found->second.exclusive - 1);
  else
    found->second.shared = std::max(0, found->second.shared - 1);
  if (found->second.shared <= 0 && found->second.exclusive <= 0)
    state.owners.erase(found);
}

[[nodiscard]] std::unordered_set<std::string>
changed_paths(const RepositorySnapshot &snapshot) {
  std::unordered_set<std::string> paths;
  for (const auto &change : snapshot.changes) {
    paths.insert(change.path);
  }
  return paths;
}

} // namespace

WorkspaceLeaseRegistry::~WorkspaceLeaseRegistry() = default;

std::shared_ptr<WorkspaceLeaseState> WorkspaceLeaseRegistry::state_for(
    const std::filesystem::path &workspace_root) {
  // Probing the SCM spawns a subprocess, so resolve the root once and reuse it
  // for both the lookup key and the execution root.
  auto root = coordination_root(workspace_root);
  const std::string key = canonical_key(root);
  std::lock_guard lock(mutex_);
  if (states_.size() > 256) {
    std::erase_if(states_,
                  [](const auto &entry) { return entry.second.expired(); });
  }
  if (const auto found = states_.find(key); found != states_.end()) {
    if (auto state = found->second.lock()) {
      return state;
    }
  }
  auto state = std::make_shared<WorkspaceLeaseState>();
  // The key is normalized (and case-folded on case-insensitive platforms) for
  // lookup only; the execution root must stay usable as a real path.
  state->execution_root = std::move(root);
  states_.insert_or_assign(key, state);
  return state;
}

GitWorkspaceCoordinator::GitWorkspaceCoordinator(
    std::shared_ptr<WorkspaceLeaseRegistry> registry)
    : registry_(registry ? std::move(registry)
                         : std::make_shared<WorkspaceLeaseRegistry>()) {}

GitWorkspaceCoordinator::Lease::Lease(
    std::shared_ptr<WorkspaceLeaseState> state,
    core::goal::WorkspaceAccess access)
    : state_(std::move(state)), access_(access) {}

GitWorkspaceCoordinator::Lease::Lease(Lease &&other) noexcept {
  *this = std::move(other);
}

GitWorkspaceCoordinator::Lease &GitWorkspaceCoordinator::Lease::operator=(
    Lease &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  // Drop the lock before releasing the mutex owner. Defaulted move-assign
  // would reset state_ first and unlock a destroyed shared_timed_mutex.
  release();
  access_ = other.access_;
  lock_ = std::move(other.lock_);
  state_ = std::move(other.state_);
  other.lock_.emplace<std::monostate>();
  other.state_.reset();
  return *this;
}

GitWorkspaceCoordinator::Lease::~Lease() { release(); }

void GitWorkspaceCoordinator::Lease::release() noexcept {
  // Deregister before dropping the lock: owns_lock() reads the live lock.
  if (state_ && owns_lock()) {
    remove_owner(*state_, access_);
  }
  lock_.emplace<std::monostate>();
  state_.reset();
}

bool GitWorkspaceCoordinator::Lease::reentrant_on_this_thread() const noexcept {
  try {
    return state_ && !owns_lock() && would_self_deadlock(*state_, access_);
  } catch (...) {
    return false;
  }
}

bool GitWorkspaceCoordinator::Lease::owns_lock() const noexcept {
  if (const auto *read = std::get_if<ReadLock>(&lock_)) {
    return read->owns_lock();
  }
  if (const auto *write = std::get_if<WriteLock>(&lock_)) {
    return write->owns_lock();
  }
  return false;
}

bool GitWorkspaceCoordinator::Lease::try_lock_for(
    std::chrono::milliseconds timeout) noexcept {
  if (!state_ || owns_lock()) {
    return owns_lock();
  }
  try {
    // Waiting here could never succeed, and the wait itself is the operation
    // that aborts on a standard library which checks for self-deadlock.
    if (would_self_deadlock(*state_, access_)) {
      return false;
    }
    if (access_ == core::goal::WorkspaceAccess::SharedRead) {
      ReadLock lock(state_->mutex, std::defer_lock);
      if (!lock.try_lock_for(timeout)) {
        return false;
      }
      lock_.emplace<ReadLock>(std::move(lock));
      add_owner(*state_, access_);
      return true;
    }
    WriteLock lock(state_->mutex, std::defer_lock);
    if (!lock.try_lock_for(timeout)) {
      return false;
    }
    lock_.emplace<WriteLock>(std::move(lock));
    add_owner(*state_, access_);
    return true;
  } catch (...) {
    return false;
  }
}

const std::filesystem::path &
GitWorkspaceCoordinator::Lease::execution_root() const noexcept {
  static const std::filesystem::path empty;
  return state_ ? state_->execution_root : empty;
}

GitWorkspaceCoordinator::Lease GitWorkspaceCoordinator::acquire(
    const std::filesystem::path &workspace_root,
    core::goal::WorkspaceAccess access) const {
  return acquire(workspace_root, access, AcquireOptions{});
}

GitWorkspaceCoordinator::Lease GitWorkspaceCoordinator::acquire(
    const std::filesystem::path &workspace_root,
    core::goal::WorkspaceAccess access, const AcquireOptions &options) const {
  if (workspace_root.empty()) {
    return {};
  }
  Lease lease(registry_->state_for(workspace_root), access);
  // A thread that already holds this lease cannot take it again; spinning to
  // the deadline would only delay the same answer.
  if (lease.reentrant_on_this_thread()) {
    return {};
  }
  const auto deadline = std::chrono::steady_clock::now() + options.timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (options.cancellation_requested && options.cancellation_requested()) {
      return {};
    }
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    const auto slice =
        remaining < kAcquirePoll ? remaining : kAcquirePoll;
    if (slice.count() <= 0) {
      break;
    }
    if (lease.try_lock_for(slice)) {
      return lease;
    }
  }
  return {};
}

GitWorkspaceCoordinator::UpgradeResult
GitWorkspaceCoordinator::upgrade_to_exclusive(
    Lease current, const std::filesystem::path &workspace_root) const {
  return upgrade_to_exclusive(std::move(current), workspace_root,
                              AcquireOptions{});
}

GitWorkspaceCoordinator::UpgradeResult
GitWorkspaceCoordinator::upgrade_to_exclusive(
    Lease current, const std::filesystem::path &workspace_root,
    const AcquireOptions &options) const {
  if (current.owns_lock() &&
      current.access() == core::goal::WorkspaceAccess::ExclusiveWrite) {
    return {.lease = std::move(current), .upgraded = true};
  }

  // A shared_timed_mutex reader cannot be promoted atomically, so the shared
  // lease has to go first. The caller is told whether exclusion was actually
  // won; a restored SharedRead lease is a coordination miss, not an upgrade.
  current = {};
  auto exclusive =
      acquire(workspace_root, core::goal::WorkspaceAccess::ExclusiveWrite,
              options);
  if (exclusive.owns_lock()) {
    return {.lease = std::move(exclusive), .upgraded = true};
  }
  return {.lease = acquire(workspace_root,
                           core::goal::WorkspaceAccess::SharedRead, options),
          .upgraded = false};
}

std::optional<RepositorySnapshot> GitWorkspaceCoordinator::capture(
    const std::filesystem::path &workspace_root) const noexcept {
  try {
    auto provider = ScmFactory::create(workspace_root);
    return provider ? provider->repository_snapshot() : std::nullopt;
  } catch (...) {
    return std::nullopt;
  }
}

WorkspaceAudit GitWorkspaceCoordinator::audit(
    const RepositorySnapshot &baseline) const noexcept {
  WorkspaceAudit result;
  result.current = capture(baseline.root);
  if (!result.current.has_value()) {
    return result;
  }
  result.repository_available = true;
  result.branch_changed = baseline.branch != result.current->branch;
  result.revision_changed = baseline.revision != result.current->revision;
  result.working_tree_changed =
      baseline.status_fingerprint != result.current->status_fingerprint;

  const auto before = changed_paths(baseline);
  const auto after = changed_paths(*result.current);
  for (const auto &path : before) {
    if (!after.contains(path)) {
      result.preexisting_changes_removed.push_back(path);
    }
  }
  std::ranges::sort(result.preexisting_changes_removed);
  return result;
}

std::string
GitWorkspaceCoordinator::render_preflight(const RepositorySnapshot &baseline) {
  std::string out =
      "\n\n[AUTO repository transaction baseline]\n"
      "Filo recorded the repository state below before this turn and audits it "
      "again at completion. Preserve all pre-existing changes and do not "
      "change branch or HEAD unless the user explicitly requested that "
      "repository transition.\n";
  out += std::format("Branch: {}\nRevision: {}\n",
                     baseline.branch.empty() ? "(detached)" : baseline.branch,
                     baseline.revision);
  if (baseline.changes.empty()) {
    out += "Pre-existing changes: none\n";
  } else {
    out += "Pre-existing changed paths:\n";
    constexpr std::size_t kMaxRenderedChanges = 64;
    const std::size_t count =
        std::min(baseline.changes.size(), kMaxRenderedChanges);
    for (std::size_t i = 0; i < count; ++i) {
      out += std::format("- {} {}\n", baseline.changes[i].status_code,
                         baseline.changes[i].path);
    }
    if (baseline.changes.size() > count) {
      out += std::format("- ... {} more path(s)\n",
                         baseline.changes.size() - count);
    }
  }
  out += "[/AUTO repository transaction baseline]";
  return out;
}

std::string GitWorkspaceCoordinator::render_audit(const WorkspaceAudit &audit) {
  if (!audit.repository_available) {
    return "repository state could not be re-read";
  }
  std::string out = std::format(
      "branch changed: {}; HEAD changed: {}; working tree changed: {}",
      audit.branch_changed ? "yes" : "no",
      audit.revision_changed ? "yes" : "no",
      audit.working_tree_changed ? "yes" : "no");
  if (!audit.preexisting_changes_removed.empty()) {
    out += "; pre-existing changed paths no longer present:";
    for (const auto &path : audit.preexisting_changes_removed) {
      out += " ";
      out += path;
    }
  }
  return out;
}

bool GitWorkspaceCoordinator::has_reconciliation_statement(
    std::string_view response) noexcept {
  return ascii_lower(response).contains("repository reconciliation:");
}

std::string
GitWorkspaceCoordinator::reconciliation_follow_up(const WorkspaceAudit &audit) {
  return std::format(
      "AUTO repository gate: the repository transaction changed branch/HEAD "
      "or removed a path that was already modified before this turn. Audit: "
      "{}. "
      "Inspect the final repository state now. If this transition was "
      "explicitly "
      "requested and correct, state `Repository reconciliation: <why the "
      "transition "
      "is intentional and how pre-existing work was preserved>` in the final "
      "answer. Otherwise stop and report the conflict without destructive "
      "cleanup.",
      render_audit(audit));
}

} // namespace core::scm
