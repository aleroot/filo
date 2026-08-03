#pragma once

#include <filesystem>
#include <vector>

namespace core::workspace {

/**
 * Directories outside the project roots that tools may touch, split by the
 * access they permit.
 *
 * ### Why this exists
 *
 * Filo hands the model two tools that reach the filesystem directly, without
 * passing through any workspace check: `run_terminal_command` (a persistent
 * shell) and `python` (an in-process interpreter). Denying the *native* path
 * tools (`read_file`, `list_directory`, `grep_search`, ...) access to a
 * directory those execution tools can already reach buys no confidentiality --
 * it only removes the auditable, permission-gated, diff-rendering path to the
 * same bytes and pushes the model toward `cat`.
 *
 * So this scope must mirror what the session's *execution* tools can reach:
 *
 *  - Unsandboxed, that is the host temp directories, read and write.
 *  - Under landrun, it is exactly what the compiled sandbox policy grants the
 *    process tree outside the project roots. Both sets come from one shared
 *    helper (`core::landrun::landrun_temp_scope`) so the kernel's view and the
 *    native tools' view cannot drift apart.
 *
 * ### Why read and write are separate
 *
 * Landrun's `read-only` mode genuinely grants the shell a larger readable set
 * than writable set, so a single set cannot describe it faithfully. Collapsing
 * the two would either under-grant reads (the failure this type exists to
 * prevent) or over-grant writes.
 *
 * ### Value semantics
 *
 * This is a plain value carried in WorkspaceSnapshot and therefore in
 * SessionContext. There is no global instance: the scope is built once in the
 * composition root (`main`) and flows down with the session it describes,
 * which keeps it trivially testable and free of process-wide mutable state.
 *
 * Roots are canonicalized once on construction so the query methods only need
 * one normalization pass per probe. Excluded roots are evaluated before the
 * corresponding allow set, which lets the scope represent policies such as
 * "/tmp except /tmp/private" without widening native-tool access.
 *
 * Invariant: every writable root is also readable.
 */
class FileAccessScope {
public:
    FileAccessScope() = default;

    /// Canonicalizes, de-duplicates, drops relative/empty entries, and folds
    /// the writable roots into the readable set to maintain the invariant.
    /// Excluded roots deny both reads and writes beneath an otherwise admitted
    /// root.
    FileAccessScope(std::vector<std::filesystem::path> readable_roots,
                    std::vector<std::filesystem::path> writable_roots,
                    std::vector<std::filesystem::path> excluded_roots = {});

    /// The host temp directories, read and write. The unsandboxed posture.
    [[nodiscard]] static FileAccessScope host_temp_directories();

    /// Normalized, deduplicated, empty/relative-filtered; writable folded into
    /// readable. Both sets satisfy the invariant.
    static void normalize_roots(std::vector<std::filesystem::path>& readable,
                                std::vector<std::filesystem::path>& writable);

    /// @param path Absolute path. It is normalized exactly like the roots, so
    ///        callers may pass a raw path as well as the output of
    ///        SessionWorkspace::resolve_path.
    [[nodiscard]] bool allows_read(const std::filesystem::path& path) const;
    [[nodiscard]] bool allows_write(const std::filesystem::path& path) const;

    /// Normalizes the root sets in place; invoked by
    /// SessionWorkspace::normalize_snapshot() so a snapshot's roots and later
    /// probes stay on the same footing.
    void normalize();

    [[nodiscard]] bool empty() const noexcept { return readable_roots_.empty(); }

    [[nodiscard]] const std::vector<std::filesystem::path>& readable_roots() const noexcept {
        return readable_roots_;
    }
    [[nodiscard]] const std::vector<std::filesystem::path>& writable_roots() const noexcept {
        return writable_roots_;
    }
    [[nodiscard]] const std::vector<std::filesystem::path>& excluded_roots() const noexcept {
        return excluded_roots_;
    }

    [[nodiscard]] bool operator==(const FileAccessScope&) const = default;

private:
    std::vector<std::filesystem::path> readable_roots_;
    std::vector<std::filesystem::path> writable_roots_;
    std::vector<std::filesystem::path> excluded_roots_;
};

} // namespace core::workspace
