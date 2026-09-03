#pragma once

#include "../goal/GoalTypes.hpp"
#include "SourceControlProvider.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace core::scm {

struct WorkspaceLeaseState;

// Explicitly owned synchronization scope for repository leases. Execution
// roots share one registry with their agents; tests and embedders can create
// isolated registries without touching process-global state.
class WorkspaceLeaseRegistry final {
public:
  WorkspaceLeaseRegistry() = default;
  ~WorkspaceLeaseRegistry();
  WorkspaceLeaseRegistry(const WorkspaceLeaseRegistry &) = delete;
  WorkspaceLeaseRegistry &operator=(const WorkspaceLeaseRegistry &) = delete;

private:
  friend class GitWorkspaceCoordinator;
  [[nodiscard]] std::shared_ptr<WorkspaceLeaseState>
  state_for(const std::filesystem::path &workspace_root);

  std::mutex mutex_;
  std::unordered_map<std::string, std::weak_ptr<WorkspaceLeaseState>> states_;
};

struct WorkspaceAudit {
  bool repository_available = false;
  bool branch_changed = false;
  bool revision_changed = false;
  bool working_tree_changed = false;
  std::vector<std::string> preexisting_changes_removed;
  std::optional<RepositorySnapshot> current;

  [[nodiscard]] bool requires_reconciliation() const noexcept {
    return !repository_available || branch_changed || revision_changed ||
           !preexisting_changes_removed.empty();
  }
};

// Repository lease and state auditor. Coordinators that receive the same
// WorkspaceLeaseRegistry share one lock per canonical root: exploration gets
// shared access and a writer gets exclusive access. The lease is intentionally
// separate from Git probing so a future backend can replace the checkout with
// an isolated worktree without changing the graph scheduler or Agent.
class GitWorkspaceCoordinator {
public:
  static constexpr auto kAcquireTimeout = std::chrono::milliseconds(60'000);
  static constexpr auto kAcquirePoll = std::chrono::milliseconds(50);

  struct AcquireOptions {
    std::chrono::milliseconds timeout{60'000};
    std::function<bool()> cancellation_requested;
  };

  explicit GitWorkspaceCoordinator(
      std::shared_ptr<WorkspaceLeaseRegistry> registry = {});

  class Lease {
  public:
    Lease() = default;
    Lease(Lease &&other) noexcept;
    Lease &operator=(Lease &&other) noexcept;
    ~Lease();
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;

    [[nodiscard]] bool owns_lock() const noexcept;
    /// Filesystem root assigned to this lease. Today this is the canonical
    /// checkout; an isolated-worktree backend can change it without changing
    /// graph or agent call sites.
    [[nodiscard]] const std::filesystem::path &execution_root() const noexcept;
    [[nodiscard]] core::goal::WorkspaceAccess access() const noexcept {
      return access_;
    }

  private:
    friend class GitWorkspaceCoordinator;
    using ReadLock = std::shared_lock<std::shared_timed_mutex>;
    using WriteLock = std::unique_lock<std::shared_timed_mutex>;

    Lease(std::shared_ptr<WorkspaceLeaseState> state,
          core::goal::WorkspaceAccess access);
    [[nodiscard]] bool
    try_lock_for(std::chrono::milliseconds timeout) noexcept;
    /// True when this thread already holds the lease. Re-locking a
    /// shared_timed_mutex from an owning thread is undefined behaviour, so
    /// callers must fail closed rather than wait.
    [[nodiscard]] bool reentrant_on_this_thread() const noexcept;
    /// Drop the lock and deregister this thread's ownership.
    void release() noexcept;

    std::shared_ptr<WorkspaceLeaseState> state_;
    core::goal::WorkspaceAccess access_ =
        core::goal::WorkspaceAccess::SharedRead;
    std::variant<std::monostate, ReadLock, WriteLock> lock_;
  };

  /// Interruptible acquire. Never blocks forever: the wait is polled so Stop
  /// can unwind, and `options.timeout` bounds parent-vs-parent contention.
  /// A lease that does not `owns_lock()` is a coordination miss, not a hang.
  [[nodiscard]] Lease acquire(const std::filesystem::path &workspace_root,
                              core::goal::WorkspaceAccess access) const;
  [[nodiscard]] Lease acquire(const std::filesystem::path &workspace_root,
                              core::goal::WorkspaceAccess access,
                              const AcquireOptions &options) const;

  /// Outcome of an upgrade attempt. `upgraded` is true only when the exclusive
  /// lock is actually held. Otherwise `lease` carries a restored SharedRead
  /// lease (or none) so the caller is never left worse off than before, but it
  /// must not assume writer exclusion. Returning a plain Lease would let a
  /// silent downgrade masquerade as a successful upgrade.
  struct UpgradeResult {
    Lease lease;
    bool upgraded = false;
  };

  /// Take ExclusiveWrite, releasing `current` first because a shared_timed_mutex
  /// reader cannot be promoted atomically. An already-exclusive lease is passed
  /// through untouched, so no release window is opened needlessly.
  [[nodiscard]] UpgradeResult
  upgrade_to_exclusive(Lease current,
                       const std::filesystem::path &workspace_root) const;
  [[nodiscard]] UpgradeResult
  upgrade_to_exclusive(Lease current,
                       const std::filesystem::path &workspace_root,
                       const AcquireOptions &options) const;

  [[nodiscard]] std::optional<RepositorySnapshot>
  capture(const std::filesystem::path &workspace_root) const noexcept;

  [[nodiscard]] WorkspaceAudit
  audit(const RepositorySnapshot &baseline) const noexcept;

  [[nodiscard]] static std::string
  render_preflight(const RepositorySnapshot &baseline);
  [[nodiscard]] static std::string render_audit(const WorkspaceAudit &audit);
  [[nodiscard]] static bool
  has_reconciliation_statement(std::string_view response) noexcept;
  [[nodiscard]] static std::string
  reconciliation_follow_up(const WorkspaceAudit &audit);

private:
  std::shared_ptr<WorkspaceLeaseRegistry> registry_;
};

} // namespace core::scm
