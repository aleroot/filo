#include "PythonManager.hpp"
#include "../logging/Logger.hpp"
#include "../utils/DetachedWorkTracker.hpp"
#include <pybind11/eval.h>
#include <condition_variable>
#include <cstdlib>
#include <thread>
#include <utility>
#include <Python.h>

namespace py = pybind11;

namespace core::tools {

namespace {

/**
 * @brief Restores sys.stdout/sys.stderr on every exit path, including
 *        exceptions thrown while the redirected code runs.
 *
 * Pairing this with PythonManager's execution_mutex_ is what makes the
 * redirection safe: user code can release the GIL mid-run (sleep, join, IO,
 * subprocess), so without the lock two overlapping execute() calls would
 * restore each other's capture buffers and leave a dead io.StringIO installed
 * as sys.stdout/sys.stderr forever.
 */
class StreamRedirect {
public:
    StreamRedirect(py::object out, py::object err)
        : sys_(py::module_::import("sys")) {
        old_out_ = sys_.attr("stdout");
        old_err_ = sys_.attr("stderr");
        sys_.attr("stdout") = std::move(out);
        sys_.attr("stderr") = std::move(err);
    }

    ~StreamRedirect() {
        try {
            restore();
        } catch (...) {
            // Never throw from a destructor; the process keeps whatever
            // streams are installed and the next call re-establishes them.
        }
    }

    /** Put the real streams back. Idempotent, so the caller can restore as
     *  soon as the code is done and the destructor becomes a no-op. */
    void restore() {
        if (!sys_) return;
        sys_.attr("stdout") = old_out_;
        sys_.attr("stderr") = old_err_;
        sys_ = py::module_();
    }

    StreamRedirect(const StreamRedirect&) = delete;
    StreamRedirect& operator=(const StreamRedirect&) = delete;

private:
    py::module_ sys_;
    py::object  old_out_;
    py::object  old_err_;
};

} // namespace

PythonManager& PythonManager::get_instance() {
    // Intentionally leak the singleton to avoid Python finalization during
    // C++ static destruction, which is fragile in multi-threaded embed setups
    // and has caused shutdown crashes/hangs in practice.
    static PythonManager* instance = new PythonManager();
    return *instance;
}

PythonManager::~PythonManager() {
    try {
        if (!interpreter_) return;

        // If we released the GIL via PyEval_SaveThread(), we must restore the
        // original thread state before finalization. Finalizing without
        // restoring this state can crash during process teardown.
        if (main_thread_state_ != nullptr) {
            PyEval_RestoreThread(main_thread_state_);
            main_thread_state_ = nullptr;
            scopes_.clear();
            interpreter_.reset();
            return;
        }

        // Fallback path: interpreter exists but no saved main thread state.
        py::gil_scoped_acquire acquire;
        scopes_.clear();
        interpreter_.reset();
    } catch (...) {
        // Avoid throwing from static destruction. At process shutdown the OS
        // will reclaim remaining Python resources even if cleanup is partial.
    }
}

void PythonManager::ensure_initialized() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (interpreter_) return;

    interpreter_ = std::make_unique<py::scoped_interpreter>();

    // Venv isolation
    const char* venv_env = std::getenv("FILO_PYTHON_VENV");
    if (venv_env && *venv_env) {
        try {
            py::dict locals;
            py::exec(R"(
import sys, os as _os
_venv = _os.environ.get('FILO_PYTHON_VENV', '')
if _venv and _os.path.isdir(_venv):
    _ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    _sp  = _os.path.join(_venv, 'lib', f'python{_ver}', 'site-packages')
    if _os.path.isdir(_sp) and _sp not in sys.path:
        sys.path.insert(0, _sp)
    sys.prefix      = _venv
    sys.exec_prefix = _venv
)", py::globals(), locals);
        } catch (const py::error_already_set& e) {
            core::logging::warn("PythonManager: venv setup failed: {}", e.what());
        }
    }

    // Release the GIL to enable multi-threading and keep the returned main
    // thread state so we can restore it during shutdown before finalization.
    main_thread_state_ = PyEval_SaveThread();
    initialized_.store(true, std::memory_order_release);
}

std::shared_ptr<PythonManager::Scope> PythonManager::acquire_scope(std::string_view scope) {
    auto it = scopes_.find(scope);
    if (it == scopes_.end()) {
        auto created = std::make_shared<Scope>();
        created->globals["__builtins__"] = py::module_::import("builtins");
        it = scopes_.emplace(std::string(scope), std::move(created)).first;
    }
    return it->second;
}

void PythonManager::release(Scope& scope) {
    // Functions defined in the namespace reference it as their globals, so
    // the dict sits in a reference cycle: clear it to break the cycle, then
    // collect so whatever else the conversation built is freed now rather
    // than at the next automatic collection.
    try {
        scope.globals.clear();
        py::module_::import("gc").attr("collect")();
    } catch (const py::error_already_set& e) {
        core::logging::warn("PythonManager: releasing a namespace failed: {}", e.what());
    }
}

void PythonManager::retire(std::shared_ptr<Scope> scope) {
    if (!scope) return;
    if (scope->running > 0) {
        scope->retired = true;  // the last running call releases it
        return;
    }
    release(*scope);
}

void PythonManager::clear_scope(std::string_view scope) {
    if (!initialized_.load(std::memory_order_acquire)) return;
    py::gil_scoped_acquire acquire;
    const auto it = scopes_.find(scope);
    if (it == scopes_.end()) return;
    auto removed = std::move(it->second);
    scopes_.erase(it);
    retire(std::move(removed));
}

void PythonManager::move_scope(std::string_view from, std::string_view to) {
    if (from == to || !initialized_.load(std::memory_order_acquire)) return;
    py::gil_scoped_acquire acquire;
    const auto it = scopes_.find(from);
    if (it == scopes_.end()) return;
    auto moved = std::move(it->second);
    scopes_.erase(it);
    std::shared_ptr<Scope> displaced;
    if (const auto target = scopes_.find(to); target != scopes_.end()) {
        displaced = std::exchange(target->second, std::move(moved));
    } else {
        scopes_.emplace(std::string(to), std::move(moved));
    }
    retire(std::move(displaced));
}

std::size_t PythonManager::scope_count() {
    if (!initialized_.load(std::memory_order_acquire)) return 0;
    py::gil_scoped_acquire acquire;
    return scopes_.size();
}

std::unique_lock<std::mutex> PythonManager::lock_execution() {
    return std::unique_lock<std::mutex>(execution_mutex_);
}

/**
 * @brief One cancellable execution, shared by the caller that waits for it,
 *        the thread that runs it, and the thread that interrupts it.
 */
struct PythonManager::Run {
    std::mutex              mutex;
    std::condition_variable settled;
    bool                    done = false;     // guarded by mutex
    PythonExecResult        result;           // guarded by mutex
    std::atomic<bool>       cancelled{false};
    // Guarded by the GIL: true exactly while py::exec runs for this call, so
    // an interrupt can never land in code that is not this call's.
    bool                    in_exec = false;
    unsigned long           thread_id = 0;

    [[nodiscard]] bool finished() {
        std::lock_guard lock(mutex);
        return done;
    }
};

namespace {

PythonExecResult interrupted_result(std::string error, std::string output = {}) {
    return {.output = std::move(output), .success = false,
            .error = std::move(error), .interrupted = true};
}

} // namespace

PythonExecResult PythonManager::execute(const std::string& code, std::string_view scope_key) {
    // Held across the stream swap, the code, and the restore: the three only
    // form one atomic step for other Python entry points when serialised.
    std::lock_guard<std::mutex> exec_lock(execution_mutex_);
    return execute_locked(code, scope_key, nullptr);
}

PythonExecResult PythonManager::execute(const std::string& code,
                                        std::string_view scope_key,
                                        const std::function<bool()>& cancellation_requested,
                                        std::chrono::milliseconds grace) {
    if (!cancellation_requested) return execute(code, scope_key);
    if (cancellation_requested()) {
        return interrupted_result("Interrupted by the user before the code started.");
    }

    // The code runs on its own thread so that this one can always answer the
    // interrupt, even while the code is stuck in a C call. Detached and
    // counted, like every other detached agent thread.
    auto run = std::make_shared<Run>();
    std::thread([this, run, code, scope = std::string(scope_key)] {
        const core::utils::DetachedWorkTracker::Guard detached_guard{
            core::utils::DetachedWorkTracker::instance()};
        PythonExecResult result;
        try {
            std::lock_guard<std::mutex> exec_lock(execution_mutex_);
            result = run->cancelled.load(std::memory_order_acquire)
                ? interrupted_result("Interrupted by the user before the code started.")
                : execute_locked(code, scope, run.get());
        } catch (const std::exception& e) {
            result = {.success = false, .error = std::string("Python execution failed: ") + e.what()};
        } catch (...) {
            result = {.success = false, .error = "Python execution failed with an unknown error."};
        }
        {
            std::lock_guard lock(run->mutex);
            run->result = std::move(result);
            run->done = true;
        }
        run->settled.notify_all();
    }).detach();

    constexpr auto kPollInterval = std::chrono::milliseconds(50);
    std::unique_lock lock(run->mutex);
    while (!run->settled.wait_for(lock, kPollInterval, [&] { return run->done; })) {
        if (!cancellation_requested()) continue;
        run->cancelled.store(true, std::memory_order_release);
        lock.unlock();
        interrupt(run);
        lock.lock();
        if (!run->settled.wait_for(lock, grace, [&] { return run->done; })) {
            return interrupted_result(
                "Interrupted by the user. The code is blocked in a call that cannot be cut "
                "short, so it keeps running in the background until that call returns; "
                "later python calls wait for it.");
        }
        break;
    }
    return std::move(run->result);
}

void PythonManager::interrupt(std::shared_ptr<Run> run) {
    // Raising the exception needs the GIL, and code inside a C call that does
    // not release it would block whoever asks for it; so a helper thread asks,
    // never the caller. It re-raises until the code is gone, because Python
    // code may catch one KeyboardInterrupt (a bare `except:`) and carry on.
    std::thread([this, run = std::move(run)] {
        const core::utils::DetachedWorkTracker::Guard detached_guard{
            core::utils::DetachedWorkTracker::instance()};
        constexpr auto kRepeatInterval = std::chrono::milliseconds(200);
        while (!run->finished()) {
            if (initialized_.load(std::memory_order_acquire)) {
                const PyGILState_STATE gil = PyGILState_Ensure();
                if (run->in_exec) {
                    PyThreadState_SetAsyncExc(run->thread_id, PyExc_KeyboardInterrupt);
                }
                PyGILState_Release(gil);
            }
            std::unique_lock lock(run->mutex);
            run->settled.wait_for(lock, kRepeatInterval, [&] { return run->done; });
        }
    }).detach();
}

PythonExecResult PythonManager::execute_locked(const std::string& code,
                                               std::string_view scope_key,
                                               Run* run) {
    ensure_initialized();

    // Acquire the GIL before any Python C API calls.
    // This is safe to call from any thread after PyEval_SaveThread() was called
    // during initialization. The GIL is automatically released when this guard
    // goes out of scope (RAII).
    py::gil_scoped_acquire acquire;

    // Declared after the GIL guard, so the last reference to a retired
    // namespace is dropped while the GIL is still held.
    const auto scope = acquire_scope(scope_key);
    ++scope->running;
    struct RunningGuard {
        PythonManager& manager;
        Scope& scope;
        ~RunningGuard() {
            if (--scope.running == 0 && scope.retired) manager.release(scope);
        }
    } running_guard{*this, *scope};

    PythonExecResult result;

    py::module_ io     = py::module_::import("io");
    py::object buf_out = io.attr("StringIO")();
    py::object buf_err = io.attr("StringIO")();

    // The guard restores the streams on every exit path of the block below.
    StreamRedirect redirect(buf_out, buf_err);
    // Published and withdrawn under the GIL, which the interrupting thread
    // holds while it raises: the exception reaches this exec or nothing.
    struct ExecWindow {
        Run* run;
        explicit ExecWindow(Run* r) : run(r) {
            if (!run) return;
            run->thread_id = PyThread_get_thread_ident();
            run->in_exec = true;
        }
        ~ExecWindow() {
            if (!run) return;
            run->in_exec = false;
            // An exception raised as the code finished may still be pending;
            // drop it so it never surfaces in a later, unrelated call.
            PyThreadState_SetAsyncExc(run->thread_id, nullptr);
        }
    };
    try {
        const ExecWindow window{run};
        // Checked inside the window: a cancel that lands after this check
        // finds in_exec set and interrupts the exec instead.
        if (run && run->cancelled.load(std::memory_order_acquire)) {
            result.success = false;
        } else {
            // Use the scope's dict for both globals and locals so that function
            // definitions and assignments persist across calls (REPL semantics).
            py::exec(py::str(code), scope->globals, scope->globals);
            result.success = true;
        }
    } catch (const py::error_already_set& e) {
        result.success = false;
        result.error   = e.what();
    } catch (const std::exception& e) {
        result.success = false;
        result.error   = std::string("Python execution failed: ") + e.what();
    } catch (...) {
        result.success = false;
        result.error   = "Python execution failed with an unknown error.";
    }
    // Restore before reading the buffers: the real streams are back as soon
    // as the code is done, and getvalue() works on the detached buffers.
    redirect.restore();
    if (run && !result.success && run->cancelled.load(std::memory_order_acquire)) {
        result.interrupted = true;
        result.error = "Interrupted by the user (KeyboardInterrupt).";
    }

    result.output = py::str(buf_out.attr("getvalue")()).cast<std::string>();
    std::string err_str = py::str(buf_err.attr("getvalue")()).cast<std::string>();
    if (!err_str.empty()) {
        if (!result.output.empty()) result.output += '\n';
        result.output += err_str;
    }

    return result;
}

} // namespace core::tools
