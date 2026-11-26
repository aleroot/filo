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
     * Thread-safe: can be called from any thread. Calls in one scope run one
     * at a time; calls in different scopes run side by side, taking turns on
     * the GIL, so one conversation's slow or stuck code never holds up
     * another's.
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
     * While that abandoned code still holds its scope, later calls in the
     * same scope fail at once instead of queueing behind it for as long as it
     * stays blocked; other scopes are unaffected. A call cancelled while it
     * waits for its scope, or before it started, never runs.
     */
    PythonExecResult execute(const std::string& code,
                             std::string_view scope,
                             const std::function<bool()>& cancellation_requested,
                             std::chrono::milliseconds grace = std::chrono::milliseconds(1000));

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
        // Held while a call runs in this scope, so calls in one namespace run
        // one at a time. Waited for with the GIL released; never taken while
        // holding it.
        std::timed_mutex busy;
    };

    PythonManager()  = default;
    ~PythonManager();  // Defined in .cpp to ensure GIL is held during cleanup

    // All three require the GIL.
    std::shared_ptr<Scope> acquire_scope(std::string_view scope);
    void retire(std::shared_ptr<Scope> scope);
    static void release(Scope& scope);

    struct Run;
    // Runs @p code in @p scope once no other call runs there. With @p run,
    // the executing thread is published to it for interruption, and the wait
    // for the scope ends when the run is cancelled.
    PythonExecResult execute_in_scope(const std::string& code, std::string_view scope, Run* run);
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

    // Per scope, the run last left behind by an interrupt that it outlived;
    // while it is unfinished, cancellable calls in that scope are refused
    // rather than queued. Never held while waiting for the GIL or a scope.
    std::mutex                                                     abandoned_mutex_;
    std::map<std::string, std::shared_ptr<Run>, std::less<>>       abandoned_;

    // Guard for lazy initialization thread-safety
    std::mutex init_mutex_;
};

#pragma GCC diagnostic pop

} // namespace core::tools
