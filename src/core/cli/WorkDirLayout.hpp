#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace core::cli {

/// A `-w/--work-dir` entry that did not become a workspace root.
struct RejectedWorkDir {
    std::string requested;          ///< Exactly what appeared on the command line.
    std::filesystem::path resolved; ///< Absolutized against the launch directory.
    std::string reason;             ///< Single-line, human-readable explanation.
};

/// The workspace roots a process should start with, plus everything dropped on
/// the way there so the caller can report it instead of failing silently.
struct WorkDirLayout {
    std::filesystem::path primary;                 ///< Empty when no `-w` was given.
    std::vector<std::filesystem::path> additional; ///< Resolved, de-duplicated, in CLI order.
    std::vector<RejectedWorkDir> rejected;
};

/**
 * Resolve `-w/--work-dir` entries into workspace roots.
 *
 * Every entry is absolutized against @p launch_cwd — the directory Filo was
 * started in — *before* anything changes the process working directory. That
 * ordering is the entire point of this function: `main()` chdirs into the
 * primary root, so resolving the later entries afterwards reinterprets them
 * against the wrong base. `filo -w proj -w .` used to collapse `.` onto the
 * primary and quietly lose the enclosing workspace; `filo -w proj -w sibling`
 * used to produce the nonexistent `proj/sibling`.
 *
 * The result is de-duplicated the way `SessionWorkspace::set_primary`
 * de-duplicates: an additional root that *is* the primary, that sits inside the
 * primary, or that repeats an earlier entry is dropped. Nonexistent roots are
 * dropped as well, matching what `SessionWorkspace::add_additional_paths`
 * accepts at runtime. Every drop is reported through @ref WorkDirLayout::rejected.
 *
 * A root nested *above* the primary is kept: granting a parent directory is how
 * a session reaches the primary's siblings, which is a deliberate use.
 *
 * Pure — reads the filesystem but mutates no global state and never changes the
 * working directory.
 */
[[nodiscard]] WorkDirLayout resolve_work_dirs(
    const std::vector<std::string>& work_dirs,
    const std::filesystem::path& launch_cwd);

} // namespace core::cli
