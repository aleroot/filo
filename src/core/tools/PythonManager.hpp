#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <atomic>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <Python.h>

namespace py = pybind11;

namespace core::tools {

struct PythonExecResult {
    std::string output;       // captured stdout (+ stderr appended if non-empty)
    bool        success = true;
    std::string error;        // exception text when success == false
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

/**
 * @brief Singleton that owns the embedded Python interpreter lifetime and the
 *        persistent execution namespaces (REPL-like: variables and functions
 *        defined in one call are visible in subsequent calls).
 *
 * Namespaces are scoped: each scope key (a session id) has its own globals
 * dict, so one conversation never sees another's variables, and clearing a
 * scope releases everything that conversation kept alive. The empty key is
 * a shared default scope for callers without a session.
 *
 * Isolation: if the FILO_PYTHON_VENV environment variable points to a venv
 * directory, its site-packages are prepended to sys.path so that packages
 * installed there are used instead of (or in addition to) the system ones.
 *
 * Member declaration order matters for destruction: scopes_ is declared
 * after interpreter_ and therefore destroyed first, releasing all Python
 * references before Py_Finalize() runs inside ~scoped_interpreter().
 */
class PythonManager {
public:
    static PythonManager& get_instance();

    /** Initialise the interpreter (idempotent).
     *
     * After initialization, the GIL is released so that other threads
     * can acquire it. This is essential for multi-threaded usage.
     */
    void ensure_initialized();

    /**
     * Execute @p code in the namespace of @p scope, creating it on first use.
     * Stdout and stderr are captured; the namespace (variables, imports,
     * function definitions) survives across calls with the same scope.
     *
     * Thread-safe: can be called from any thread. The GIL is acquired
     * automatically for the duration of the execution.
     */
    PythonExecResult execute(const std::string& code, std::string_view scope = {});

    /** Drop the namespace of @p scope and free what it kept alive.
     *
     * The next execute() with this scope starts from a clean slate. A call
     * still running in the scope keeps its namespace until it returns; the
     * namespace is released then. No-op before initialization.
     */
    void clear_scope(std::string_view scope);

    /** Re-key the namespace of @p from to @p to (a session id rotation).
     *  A namespace already under @p to is cleared. */
    void move_scope(std::string_view from, std::string_view to);

    /** Number of live scopes (for tests). */
    [[nodiscard]] std::size_t scope_count();

private:
    struct Scope {
        py::dict globals;
        int      running = 0;      // execute() calls in progress
        bool     retired = false;  // removed from scopes_; release when idle
    };

    PythonManager()  = default;
    ~PythonManager();  // Defined in .cpp to ensure GIL is held during cleanup

    // All three require the GIL.
    std::shared_ptr<Scope> acquire_scope(std::string_view scope);
    void retire(std::shared_ptr<Scope> scope);
    static void release(Scope& scope);

    // interpreter_ declared first → destroyed last (calls Py_Finalize).
    std::unique_ptr<py::scoped_interpreter> interpreter_;
    // scopes_ declared second → destroyed first (releases Python refs).
    // Guarded by the GIL: only touched while it is held, and no Python code
    // runs between a lookup and the use of its result.
    std::map<std::string, std::shared_ptr<Scope>, std::less<>> scopes_;
    // Saved by PyEval_SaveThread() after init; must be restored before finalize.
    PyThreadState*                          main_thread_state_{nullptr};
    // Set once initialization finished, so clear/move can skip a cold start.
    std::atomic<bool>                       initialized_{false};

    // Guard for lazy initialization thread-safety
    std::mutex init_mutex_;
};

#pragma GCC diagnostic pop

} // namespace core::tools
