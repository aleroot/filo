#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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
 * @brief Singleton that owns the embedded Python interpreter lifetime and a
 *        persistent execution namespace (REPL-like: variables and functions
 *        defined in one call are visible in subsequent calls).
 *
 * Isolation: if the FILO_PYTHON_VENV environment variable points to a venv
 * directory, its site-packages are prepended to sys.path so that packages
 * installed there are used instead of (or in addition to) the system ones.
 *
 * Member declaration order matters for destruction: namespace_ is declared
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
     * Execute @p code in the persistent namespace.
     * Stdout and stderr are captured; the interpreter state (variables,
     * imports, function definitions) survives across calls.
     *
     * Thread-safe: can be called from any thread. The GIL is acquired
     * automatically for the duration of the execution.
     */
    PythonExecResult execute(const std::string& code);

    /** Reset the persistent namespace back to a clean slate.
     *
     * Thread-safe: can be called from any thread. The GIL is acquired
     * automatically for the duration of the operation.
     */
    void reset_namespace();

private:
    PythonManager()  = default;
    ~PythonManager();  // Defined in .cpp to ensure GIL is held during cleanup

    // interpreter_ declared first → destroyed last (calls Py_Finalize).
    std::unique_ptr<py::scoped_interpreter> interpreter_;
    // namespace_ declared second → destroyed first (releases Python refs).
    std::optional<py::dict>                 namespace_;
    // Saved by PyEval_SaveThread() after init; must be restored before finalize.
    PyThreadState*                          main_thread_state_{nullptr};

    // Guard for lazy initialization thread-safety
    std::mutex init_mutex_;
};

#pragma GCC diagnostic pop

} // namespace core::tools
