#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/SessionContext.hpp"
#include "core/scm/GitWorkspaceCoordinator.hpp"
#include "core/scm/ScmFactory.hpp"
#include "core/workspace/FileAccessScope.hpp"
#include "core/workspace/PathVisibility.hpp"
#include "core/workspace/SessionWorkspace.hpp"
#include "core/workspace/Workspace.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <string_view>
#include <system_error>
#include <vector>

TEST_CASE("Workspace bounds logic", "[Workspace]") {
    using core::workspace::Workspace;

    auto& ws = Workspace::get_instance();
    std::error_code ec;
    const auto test_dir = std::filesystem::current_path(ec) / "test_workspace_dir";
    std::filesystem::create_directories(test_dir, ec);

    SECTION("Default behavior - unenforced") {
        ws.initialize(test_dir, {}, false);
        REQUIRE_FALSE(ws.is_enforced());
        REQUIRE(ws.allows_read(test_dir / "some_file.txt"));
        REQUIRE(ws.allows_read("/etc/passwd"));
        REQUIRE(ws.allows_read("C:\\Windows\\System32\\config"));
    }

    SECTION("Enforced primary directory") {
        ws.initialize(test_dir, {}, true);
        REQUIRE(ws.is_enforced());
        REQUIRE(ws.allows_read(test_dir));
        REQUIRE(ws.allows_read(test_dir / "child_file.txt"));
        REQUIRE(ws.allows_read(test_dir / "nested" / "file.txt"));

        REQUIRE_FALSE(ws.allows_read(test_dir.parent_path()));
        REQUIRE_FALSE(ws.allows_read("/tmp/outside_file.txt"));
        REQUIRE_FALSE(ws.allows_read(test_dir / ".." / "outside.txt"));
    }

    SECTION("Shorter target path is denied safely") {
        const auto nested_root = test_dir / "nested" / "root";
        std::filesystem::create_directories(nested_root, ec);
        ws.initialize(nested_root, {}, true);

        REQUIRE_FALSE(ws.allows_read(test_dir / "nested"));
        REQUIRE_FALSE(ws.allows_read(test_dir));
    }

    SECTION("Enforced with additional directories") {
        const auto add_dir1 = std::filesystem::current_path(ec) / "test_workspace_add1";
        const auto add_dir2 = std::filesystem::current_path(ec) / "test_workspace_add2";
        std::filesystem::create_directories(add_dir1, ec);
        std::filesystem::create_directories(add_dir2, ec);

        ws.initialize(test_dir, {add_dir1, add_dir2}, true);

        REQUIRE(ws.allows_read(test_dir / "file.txt"));
        REQUIRE(ws.allows_read(add_dir1));
        REQUIRE(ws.allows_read(add_dir1 / "file1.txt"));
        REQUIRE(ws.allows_read(add_dir2 / "file2.txt"));

        REQUIRE_FALSE(ws.allows_read("/tmp/outside_file.txt"));
        REQUIRE_FALSE(ws.allows_read(test_dir.parent_path() / "another.txt"));

        std::filesystem::remove_all(add_dir1, ec);
        std::filesystem::remove_all(add_dir2, ec);
    }

    std::filesystem::remove_all(test_dir, ec);
    ws.initialize("", {}, false);
}

TEST_CASE("SessionWorkspace resolves relative paths against its primary root", "[Workspace]") {
    using core::workspace::SessionWorkspace;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto root = std::filesystem::current_path(ec) / "test_session_workspace_root";
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root / "nested", ec);

    const SessionWorkspace workspace{WorkspaceSnapshot{
        .primary = root,
        .additional = {},
        .enforce = true,
        .version = 7,
    }};

    REQUIRE(workspace.primary() == root.lexically_normal());
    REQUIRE(workspace.version() == 7);
    REQUIRE(workspace.resolve_path("nested/value.txt")
            == (root / "nested" / "value.txt").lexically_normal());
    REQUIRE(workspace.allows_read("nested/value.txt"));
    REQUIRE_FALSE(workspace.allows_read("../outside.txt"));

    std::filesystem::remove_all(root, ec);
}

TEST_CASE("SessionContext delegates to its owned SessionWorkspace", "[Workspace][SessionContext]") {
    using core::context::SessionContext;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto primary = std::filesystem::current_path(ec) / "test_session_context_root";
    const auto extra = std::filesystem::current_path(ec) / "test_session_context_extra";
    std::filesystem::remove_all(primary, ec);
    std::filesystem::remove_all(extra, ec);
    std::filesystem::create_directories(primary / "nested", ec);
    std::filesystem::create_directories(extra, ec);

    const SessionContext context = core::context::make_session_context(
        WorkspaceSnapshot{
            .primary = primary,
            .additional = {extra},
            .enforce = true,
            .version = 23,
        },
        core::context::SessionTransport::mcp_http,
        "workspace-test-session");

    REQUIRE(context.workspace_view().primary() == primary.lexically_normal());
    REQUIRE(context.effective_workspace().version == 23);
    REQUIRE(context.resolve_path("nested/value.txt")
            == (primary / "nested" / "value.txt").lexically_normal());
    REQUIRE(context.allows_read("nested/value.txt"));
    REQUIRE(context.allows_read(extra / "allowed.txt"));
    REQUIRE_FALSE(context.allows_read("../outside.txt"));

    std::filesystem::remove_all(primary, ec);
    std::filesystem::remove_all(extra, ec);
}

TEST_CASE("SessionContext grants existing absolute paths for the session", "[Workspace][SessionContext]") {
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto base = std::filesystem::temp_directory_path(ec)
        / std::format("filo-session-grant-{}", std::rand());
    const auto primary = base / "primary";
    const auto external = base / "external";
    const auto external_file = base / "single.txt";
    const auto missing = base / "missing";
    std::filesystem::create_directories(primary, ec);
    std::filesystem::create_directories(external, ec);
    {
        std::ofstream out(external_file);
        out << "single\n";
    }

    auto context = core::context::make_session_context(WorkspaceSnapshot{
        .primary = primary,
        .additional = {},
        .enforce = true,
        .version = 4,
    });
    context.path_visibility = std::make_shared<core::workspace::PathVisibility>(
        std::make_shared<core::workspace::AllowAllPathVisibilityPolicy>());

    REQUIRE(context.extend_workspace({external, external_file, missing, "relative"}) == 2);
    REQUIRE(context.effective_workspace().version == 5);
    REQUIRE_FALSE(context.path_visibility);
    REQUIRE(context.allows_read(external / "nested.txt"));
    REQUIRE(context.allows_read(external_file));
    REQUIRE_FALSE(context.allows_read(base / "sibling.txt"));
    REQUIRE(context.extend_workspace({external}) == 0);
    REQUIRE(context.effective_workspace().version == 5);

    std::filesystem::remove_all(base, ec);
}

TEST_CASE("SessionContext replaces the primary workspace root", "[Workspace][SessionContext]") {
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto base = std::filesystem::temp_directory_path(ec)
        / std::format("filo-session-change-primary-{}", std::rand());
    const auto primary = base / "primary";
    const auto nested_additional = primary / "nested-add";
    const auto sibling_additional = base / "sibling-add";
    const auto next_primary = base / "next-primary";
    const auto missing = base / "missing";
    std::filesystem::create_directories(nested_additional, ec);
    std::filesystem::create_directories(sibling_additional, ec);
    std::filesystem::create_directories(next_primary, ec);

    auto context = core::context::make_session_context(WorkspaceSnapshot{
        .primary = primary,
        .additional = {sibling_additional},
        .enforce = true,
        .version = 1,
    });
    context.path_visibility = std::make_shared<core::workspace::PathVisibility>(
        std::make_shared<core::workspace::AllowAllPathVisibilityPolicy>());

    // A missing directory or the current primary are both rejected, and
    // rejection must not disturb visibility caching or bump the version.
    REQUIRE_FALSE(context.set_workspace_primary(missing));
    REQUIRE_FALSE(context.set_workspace_primary(primary));
    REQUIRE(context.path_visibility);
    REQUIRE(context.effective_workspace().version == 1);

    REQUIRE(context.set_workspace_primary(next_primary));
    REQUIRE_FALSE(context.path_visibility);
    REQUIRE(context.effective_workspace().version == 2);
    REQUIRE(context.workspace_view().primary()
        == core::workspace::SessionWorkspace::normalize_path(next_primary));
    REQUIRE(context.allows_read(next_primary / "file.txt"));
    REQUIRE_FALSE(context.allows_read(primary / "file.txt"));
    // A sibling additional root unrelated to the old primary survives the swap.
    REQUIRE(context.allows_read(sibling_additional / "file.txt"));

    std::filesystem::remove_all(base, ec);
}

TEST_CASE("SourceControlProvider lists branch refs through abstraction", "[Workspace][SCM]") {
    if (std::system("git --version >/dev/null 2>&1") != 0) {
        SKIP("git is not available in this environment");
    }

    namespace fs = std::filesystem;
    struct CwdGuard {
        fs::path old;
        ~CwdGuard() {
            std::error_code ec;
            fs::current_path(old, ec);
        }
    } guard{fs::current_path()};

    std::error_code ec;
    const fs::path temp_dir =
        fs::temp_directory_path() / std::format("filo-test-scm-{}", std::rand());
    fs::remove_all(temp_dir, ec);
    fs::create_directories(temp_dir, ec);
    REQUIRE_FALSE(ec);
    fs::current_path(temp_dir);

    REQUIRE(std::system("git init -q") == 0);
    {
        std::ofstream readme("README.md");
        readme << "test\n";
    }
    REQUIRE(std::system("git add README.md") == 0);
    REQUIRE(std::system(
        "git -c user.name=Filo -c user.email=filo@example.invalid commit -qm initial")
        == 0);
    REQUIRE(std::system("git branch feature/ref-list") == 0);

    auto scm = core::scm::ScmFactory::create(temp_dir);
    const auto refs = scm->list_branch_refs();

    const auto has_ref = [&refs](std::string_view name) {
        return std::ranges::any_of(refs, [name](const core::scm::BranchRef& ref) {
            return ref.name == name;
        });
    };
    REQUIRE(has_ref("feature/ref-list"));

    fs::current_path(guard.old, ec);
    fs::remove_all(temp_dir, ec);
}

TEST_CASE("GitWorkspaceCoordinator serializes writers behind shared readers",
          "[Workspace][SCM][coordinator]") {
  const auto root = std::filesystem::temp_directory_path() /
                    std::format("filo-workspace-lease-{}", std::rand());
  std::error_code error;
  std::filesystem::create_directories(root / ".git", error);
  std::filesystem::create_directories(root / "nested", error);
  REQUIRE_FALSE(error);

  auto registry =
      std::make_shared<core::scm::WorkspaceLeaseRegistry>();
  core::scm::GitWorkspaceCoordinator reader_coordinator(registry);
  core::scm::GitWorkspaceCoordinator writer_coordinator(registry);
  auto first_reader =
      reader_coordinator.acquire(root, core::goal::WorkspaceAccess::SharedRead);
  auto second_reader = reader_coordinator.acquire(
      root / "nested", core::goal::WorkspaceAccess::SharedRead);
  REQUIRE(first_reader.owns_lock());
  REQUIRE(second_reader.owns_lock());
  REQUIRE(first_reader.execution_root() == second_reader.execution_root());

  // A separately owned registry is intentionally isolated. This proves that
  // coordination is dependency-injected rather than process-global.
  core::scm::GitWorkspaceCoordinator isolated_coordinator;
  auto isolated_writer = isolated_coordinator.acquire(
      root, core::goal::WorkspaceAccess::ExclusiveWrite);
  REQUIRE(isolated_writer.owns_lock());

  std::promise<void> started;
  auto started_future = started.get_future();
  auto writer = std::async(std::launch::async, [&] {
    started.set_value();
    return writer_coordinator.acquire(
        root, core::goal::WorkspaceAccess::ExclusiveWrite);
  });
  started_future.wait();
  REQUIRE(writer.wait_for(std::chrono::milliseconds(50)) ==
          std::future_status::timeout);

  first_reader = {};
  second_reader = {};
  REQUIRE(writer.wait_for(std::chrono::seconds(2)) ==
          std::future_status::ready);
  REQUIRE(writer.get().owns_lock());
  std::filesystem::remove_all(root, error);
}

TEST_CASE(
    "GitWorkspaceCoordinator audits repository transitions and dirty work",
    "[Workspace][SCM][coordinator]") {
  if (std::system("git --version >/dev/null 2>&1") != 0) {
    SKIP("git is not available in this environment");
  }

  namespace fs = std::filesystem;
  struct CwdGuard {
    fs::path old;
    ~CwdGuard() {
      std::error_code error;
      fs::current_path(old, error);
    }
  } guard{fs::current_path()};

  std::error_code error;
  const fs::path root = fs::temp_directory_path() /
                        std::format("filo-workspace-audit-{}", std::rand());
  fs::remove_all(root, error);
  fs::create_directories(root, error);
  REQUIRE_FALSE(error);
  fs::current_path(root);
  REQUIRE(std::system("git init -q") == 0);
  {
    std::ofstream file("tracked.txt");
    file << "base\n";
  }
  REQUIRE(std::system("git add tracked.txt") == 0);
  REQUIRE(std::system("git -c user.name=Filo -c "
                      "user.email=filo@example.invalid commit -qm base") == 0);
  {
    std::ofstream file("tracked.txt", std::ios::app);
    file << "pre-existing work\n";
  }

  core::scm::GitWorkspaceCoordinator coordinator;
  const auto baseline = coordinator.capture(root);
  REQUIRE(baseline.has_value());
  REQUIRE(baseline->changes.size() == 1);
  REQUIRE(baseline->changes.front().path == "tracked.txt");

  REQUIRE(std::system("git add tracked.txt") == 0);
  REQUIRE(
      std::system("git -c user.name=Filo -c user.email=filo@example.invalid "
                  "commit -qm transition") == 0);
  const core::scm::WorkspaceAudit audit = coordinator.audit(*baseline);
  REQUIRE(audit.repository_available);
  REQUIRE(audit.revision_changed);
  REQUIRE(audit.requires_reconciliation());
  REQUIRE(audit.preexisting_changes_removed ==
          std::vector<std::string>{"tracked.txt"});
  REQUIRE_THAT(
      core::scm::GitWorkspaceCoordinator::reconciliation_follow_up(audit),
      Catch::Matchers::ContainsSubstring("Repository reconciliation:"));

  fs::current_path(guard.old, error);
  fs::remove_all(root, error);
}

TEST_CASE("GitWorkspaceCoordinator fails closed when a baseline cannot be re-read",
          "[Workspace][SCM][coordinator]") {
  core::scm::RepositorySnapshot baseline{
      .root = std::filesystem::temp_directory_path() /
              std::format("filo-missing-repository-{}", std::rand()),
      .branch = "main",
      .revision = "abc123",
  };
  std::error_code error;
  std::filesystem::remove_all(baseline.root, error);

  core::scm::GitWorkspaceCoordinator coordinator;
  const auto audit = coordinator.audit(baseline);
  REQUIRE_FALSE(audit.repository_available);
  REQUIRE(audit.requires_reconciliation());
}

TEST_CASE("Scratch scope widens workspace bounds without touching project roots",
          "[Workspace][scratch]") {
    using core::workspace::FileAccessScope;
    using core::workspace::SessionWorkspace;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto project = std::filesystem::current_path(ec) / "test_workspace_scratch";
    std::filesystem::create_directories(project, ec);
    const auto scratch_root = std::filesystem::temp_directory_path(ec)
        / std::format("filo-scratch-bounds-{}", std::rand());
    std::filesystem::create_directories(scratch_root, ec);

    const SessionWorkspace workspace(WorkspaceSnapshot{
        .primary = project,
        .additional = {},
        .enforce = true,
        .version = 1,
        .scratch = FileAccessScope({scratch_root}, {scratch_root}),
    });

    const auto canonical = [](const std::filesystem::path& path) {
        return core::workspace::SessionWorkspace::normalize_path(path);
    };
    REQUIRE(workspace.allows_read(project / "file.txt"));
    REQUIRE(workspace.allows_write(project / "file.txt"));
    REQUIRE(workspace.allows_read(canonical(scratch_root / "scratch.txt")));
    REQUIRE(workspace.allows_write(canonical(scratch_root / "nested" / "deep.txt")));
    REQUIRE(workspace.is_scratch_path(canonical(scratch_root / "scratch.txt")));

    // A project path is in scope but is not scratch; the two notions stay
    // distinct so callers can tell why a path was admitted.
    REQUIRE_FALSE(workspace.is_scratch_path(project / "file.txt"));

    // Anything outside both remains denied.
    REQUIRE_FALSE(workspace.allows_read("/etc/passwd"));
    REQUIRE_FALSE(workspace.allows_write("/etc/passwd"));
    REQUIRE_FALSE(workspace.allows_read(project.parent_path() / "sibling.txt"));

    std::filesystem::remove_all(project, ec);
    std::filesystem::remove_all(scratch_root, ec);
}

TEST_CASE("A read-only scratch root admits reads but refuses writes",
          "[Workspace][scratch]") {
    // Landrun's read-only mode grants the shell a larger readable temp set than
    // writable set. A single set could not express that without either
    // under-granting reads or over-granting writes.
    using core::workspace::FileAccessScope;
    using core::workspace::SessionWorkspace;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto project = std::filesystem::current_path(ec) / "test_workspace_ro_scratch";
    std::filesystem::create_directories(project, ec);
    const auto shared_temp = std::filesystem::temp_directory_path(ec)
        / std::format("filo-scratch-ro-{}", std::rand());
    const auto private_temp = std::filesystem::temp_directory_path(ec)
        / std::format("filo-scratch-rw-{}", std::rand());
    std::filesystem::create_directories(shared_temp, ec);
    std::filesystem::create_directories(private_temp, ec);

    const SessionWorkspace workspace(WorkspaceSnapshot{
        .primary = project,
        .additional = {},
        .enforce = true,
        .version = 1,
        .scratch = FileAccessScope({shared_temp, private_temp}, {private_temp}),
    });

    const auto canonical = [](const std::filesystem::path& path) {
        return core::workspace::SessionWorkspace::normalize_path(path);
    };
    REQUIRE(workspace.allows_read(canonical(shared_temp / "readable.txt")));
    REQUIRE_FALSE(workspace.allows_write(canonical(shared_temp / "readable.txt")));
    REQUIRE(workspace.allows_read(canonical(private_temp / "both.txt")));
    REQUIRE(workspace.allows_write(canonical(private_temp / "both.txt")));

    std::filesystem::remove_all(project, ec);
    std::filesystem::remove_all(shared_temp, ec);
    std::filesystem::remove_all(private_temp, ec);
}

TEST_CASE("FileAccessScope keeps every writable root readable",
          "[Workspace][scratch]") {
    using core::workspace::FileAccessScope;

    std::error_code ec;
    const auto writable = std::filesystem::temp_directory_path(ec)
        / std::format("filo-scratch-invariant-{}", std::rand());
    std::filesystem::create_directories(writable, ec);

    // Declared writable only; the type must fold it into the readable set so a
    // caller cannot build a scope that permits a write it would refuse to read.
    const FileAccessScope scope({}, {writable});
    const auto probe = core::workspace::SessionWorkspace::normalize_path(
        writable / "x.txt");
    REQUIRE(scope.allows_write(probe));
    REQUIRE(scope.allows_read(probe));

    // Relative and unrelated paths are never in scope.
    REQUIRE_FALSE(scope.allows_read("relative.txt"));
    REQUIRE_FALSE(scope.allows_read(core::workspace::SessionWorkspace::normalize_path(
        "/etc/passwd")));

    REQUIRE(FileAccessScope{}.empty());
    REQUIRE_FALSE(FileAccessScope{}.allows_read(writable / "x.txt"));

    std::filesystem::remove_all(writable, ec);
}

TEST_CASE("FileAccessScope exclusions override an admitted ancestor",
          "[Workspace][scratch]") {
    using core::workspace::FileAccessScope;

    std::error_code ec;
    const auto scratch = std::filesystem::temp_directory_path(ec)
        / std::format("filo-scratch-exclusion-{}", std::rand());
    const auto excluded = scratch / "private";
    std::filesystem::create_directories(excluded, ec);
    REQUIRE_FALSE(ec);

    const FileAccessScope scope({scratch}, {scratch}, {excluded, excluded});
    const auto normalized = [](const std::filesystem::path& path) {
        return core::workspace::SessionWorkspace::normalize_path(path);
    };
    REQUIRE(scope.allows_read(normalized(scratch / "public.txt")));
    REQUIRE(scope.allows_write(normalized(scratch / "public.txt")));
    REQUIRE_FALSE(scope.allows_read(normalized(excluded)));
    REQUIRE_FALSE(scope.allows_read(normalized(excluded / "secret.txt")));
    REQUIRE_FALSE(scope.allows_write(normalized(excluded / "secret.txt")));
    REQUIRE(scope.excluded_roots().size() == 1);

    std::filesystem::remove_all(scratch, ec);
}

TEST_CASE("An empty scratch scope restores strict project-root bounds",
          "[Workspace][scratch]") {
    using core::workspace::SessionWorkspace;
    using core::workspace::WorkspaceSnapshot;

    std::error_code ec;
    const auto project = std::filesystem::current_path(ec) / "test_workspace_no_scratch";
    std::filesystem::create_directories(project, ec);

    // Default-constructed scratch: the scope fails closed unless a composition
    // root supplies one.
    const SessionWorkspace workspace(WorkspaceSnapshot{
        .primary = project,
        .additional = {},
        .enforce = true,
        .version = 1,
    });

    REQUIRE(workspace.allows_read(project / "file.txt"));
    REQUIRE_FALSE(workspace.allows_read("/tmp/outside_file.txt"));
    REQUIRE_FALSE(workspace.is_scratch_path("/tmp/outside_file.txt"));

    std::filesystem::remove_all(project, ec);
}

TEST_CASE("SCM review patch includes staged and unstaged changes without diff helpers", "[Workspace][SCM][boost]") {
  namespace fs = std::filesystem;
  struct WorkspaceGuard {
    fs::path old = fs::current_path();
    fs::path root = fs::temp_directory_path() /
        std::format("filo-review-patch-{}", std::chrono::steady_clock::now().time_since_epoch().count());
    ~WorkspaceGuard() {
      std::error_code ec;
      fs::current_path(old, ec);
      fs::remove_all(root, ec);
    }
  } guard;
  fs::create_directories(guard.root);
  fs::current_path(guard.root);
  REQUIRE(std::system("git init -q") == 0);
  std::ofstream("tracked.txt") << "base\n";
  REQUIRE(std::system("git add tracked.txt") == 0);
  auto scm = core::scm::ScmFactory::create(guard.root);
  auto unborn = scm->review_patch();
  REQUIRE(unborn.has_value());
  CHECK_THAT(*unborn, Catch::Matchers::ContainsSubstring("+base"));
  REQUIRE(std::system("git -c user.name=Filo -c user.email=filo@example.invalid commit -qm base") == 0);
  std::ofstream("tracked.txt") << "staged\n";
  REQUIRE(std::system("git add tracked.txt") == 0);
  std::ofstream("second.txt") << "second staged file\n";
  REQUIRE(std::system("git add second.txt") == 0);
  std::ofstream("tracked.txt") << "unstaged\n";
  REQUIRE(std::system("git config diff.external 'touch diff-helper-ran'") == 0);
  const auto patch = scm->review_patch();
  REQUIRE(patch.has_value());
  CHECK_THAT(*patch, Catch::Matchers::ContainsSubstring("+unstaged"));
  CHECK_THAT(*patch, Catch::Matchers::ContainsSubstring("+second staged file"));
  CHECK_FALSE(fs::exists("diff-helper-ran"));
}
