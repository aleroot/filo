#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
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
    bool        interrupted = false;  // cancelled through the predicate
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

    /**
     * Cancellable execute(): when @p cancellation_requested turns true the
     * running code gets a KeyboardInterrupt, raised at its next bytecode.
     *
     * Code blocked inside a C call (time.sleep, a socket read, a subprocess
     * wait) only sees that exception once the call returns, and no thread but
     * Python's main one can cut such a call short. So the code runs on its own
     * thread, and if it has not unwound within @p grace after the interrupt,
     * this returns an interrupted result and leaves it to finish there.
     * Later calls wait for it (each wait is itself cancellable), and code
     * whose call was cancelled before it started never runs.
     */
    PythonExecResult execute(const std::string& code,
                             std::string_view scope,
                             const std::function<bool()>& cancellation_requested,
                             std::chrono::milliseconds grace = std::chrono::milliseconds(1000));

    /** Take the process-wide Python execution lock.
     *
     * Foreign Python entry points (e.g. Python skill calls) hold this while
     * they run, so their code cannot run in the window where execute() has
     * sys.stdout/sys.stderr redirected to its capture buffers.
     */
    std::unique_lock<std::mutex> lock_execution();

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

    struct Run;
    // Runs @p code in @p scope; execution_mutex_ must be held. With @p run,
    // the executing thread is published to it for interruption.
    PythonExecResult execute_locked(const std::string& code, std::string_view scope, Run* run);
    // Raises KeyboardInterrupt in @p run's code until it has finished, from a
    // helper thread, so the caller never waits for the GIL.
    void interrupt(std::shared_ptr<Run> run);

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

    // Serialises the sys.stdout/sys.stderr swap + run + restore in execute()
    // against every other Python entry point. Python code may release the GIL
    // mid-call (sleep, join, IO, subprocess), so without this an overlapping
    // call can restore out of order and leave a dead capture buffer installed
    // as sys.stdout/sys.stderr forever. Always taken before the GIL.
    std::mutex                              execution_mutex_;

    // Guard for lazy initialization thread-safety
    std::mutex init_mutex_;
};

#pragma GCC diagnostic pop

} // namespace core::tools
