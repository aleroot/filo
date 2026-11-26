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

/// Installed once as sys.stdout/sys.stderr. A write goes to the capture of
/// the execute() call running on the writing thread, or of the call that
/// started that thread while it runs; anything else reaches the real stream.
/// Swapping sys.stdout per call instead would make every call in the process
/// wait for the one before it: the streams are global, and two overlapping
/// swaps restore each other's buffers.
constexpr const char* kStreamRouterSource = R"PY(
import sys, threading

class _Capture:
    __slots__ = ('streams', 'active')
    def __init__(self, out, err):
        self.streams = (out, err)
        self.active = True

_by_thread = {}
_threads = getattr(threading, '_active', {})

def _current():
    ident = threading.get_ident()
    capture = _by_thread.get(ident)
    if capture is None:
        # Looked up without current_thread(): for a thread Python did not
        # start, that registers a dummy Thread object that is never removed.
        capture = getattr(_threads.get(ident), '_filo_capture', None)
        if capture is not None and not capture.active:
            capture = None
    return capture

class _Router:
    def __init__(self, index, target):
        self._filo_index = index
        self._filo_target = target
    def _filo_stream(self):
        capture = _current()
        return self._filo_target if capture is None else capture.streams[self._filo_index]
    def write(self, text):
        return self._filo_stream().write(text)
    def writelines(self, lines):
        return self._filo_stream().writelines(lines)
    def flush(self):
        return self._filo_stream().flush()
    def __getattr__(self, name):
        return getattr(self._filo_stream(), name)

_original_start = threading.Thread.start
def _start(self, *args, **kwargs):
    capture = _current()
    if capture is not None:
        self._filo_capture = capture
    return _original_start(self, *args, **kwargs)
threading.Thread.start = _start

_routers = (_Router(0, sys.stdout), _Router(1, sys.stderr))

def install():
    # Also undoes code that replaced the streams, as restoring them did.
    sys.stdout, sys.stderr = _routers

def begin(out, err):
    install()
    capture = _Capture(out, err)
    _by_thread[threading.get_ident()] = capture
    return capture

def end(capture):
    capture.active = False
    _by_thread.pop(threading.get_ident(), None)
    install()

install()
)PY";

constexpr const char* kStreamRouterModule = "_filo_streams";

/**
 * @brief Captures what one execute() call prints, until finish() or the end
 *        of the scope, whichever comes first. Requires the GIL.
 */
class OutputCapture {
public:
    OutputCapture()
        : router_(py::module_::import(kStreamRouterModule)),
          out_(py::module_::import("io").attr("StringIO")()),
          err_(py::module_::import("io").attr("StringIO")()),
          capture_(router_.attr("begin")(out_, err_)) {}

    ~OutputCapture() {
        try {
            finish();
        } catch (...) {
            // Never throw from a destructor; the router stays installed and
            // a capture left active only receives this thread's writes.
        }
    }

    /** Stop capturing. Idempotent, so the caller can stop as soon as the
     *  code is done and the destructor becomes a no-op. */
    void finish() {
        if (!capture_) return;
        router_.attr("end")(capture_);
        capture_ = py::object();
    }

    /** Stdout, then stderr on its own line when there is any. */
    [[nodiscard]] std::string text() const {
        auto text = py::str(out_.attr("getvalue")()).cast<std::string>();
        const auto err = py::str(err_.attr("getvalue")()).cast<std::string>();
        if (!err.empty()) {
            if (!text.empty()) text += '\n';
            text += err;
        }
        return text;
    }

    OutputCapture(const OutputCapture&) = delete;
    OutputCapture& operator=(const OutputCapture&) = delete;

private:
    py::module_ router_;
    py::object  out_;
    py::object  err_;
    py::object  capture_;
};

/// How often a call waiting for its scope checks whether it was cancelled.
constexpr auto kScopePollInterval = std::chrono::milliseconds(50);

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

    try {
        py::object router = py::module_::import("types").attr("ModuleType")(kStreamRouterModule);
        py::exec(kStreamRouterSource, router.attr("__dict__"));
        py::module_::import("sys").attr("modules")[kStreamRouterModule] = router;
    } catch (const py::error_already_set& e) {
        // Calls then fail with the import error instead of mixing output.
        core::logging::error("PythonManager: stream routing setup failed: {}", e.what());
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
    {
        // A call still stuck in the old namespace no longer stands in the
        // way: the next call starts a new one.
        std::lock_guard lock(abandoned_mutex_);
        if (const auto it = abandoned_.find(scope); it != abandoned_.end()) abandoned_.erase(it);
    }
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
    {
        // A stuck call moves with its namespace, and leaves with the one it
        // displaces. (Brief, and never held while waiting for the GIL.)
        std::lock_guard lock(abandoned_mutex_);
        std::shared_ptr<Run> stuck;
        if (const auto source = abandoned_.find(from); source != abandoned_.end()) {
            stuck = std::move(source->second);
            abandoned_.erase(source);
        }
        if (stuck) {
            abandoned_.insert_or_assign(std::string(to), std::move(stuck));
        } else if (const auto target = abandoned_.find(to); target != abandoned_.end()) {
            abandoned_.erase(target);
        }
    }
    retire(std::move(displaced));
}

std::size_t PythonManager::scope_count() {
    if (!initialized_.load(std::memory_order_acquire)) return 0;
    py::gil_scoped_acquire acquire;
    return scopes_.size();
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
    // True while the run holds its scope: cleared before the scope is let go,
    // so a call that finds it false never waits behind this run.
    std::atomic<bool>       holds_scope{false};
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
    return execute_in_scope(code, scope_key, nullptr);
}

PythonExecResult PythonManager::execute(const std::string& code,
                                        std::string_view scope_key,
                                        const std::function<bool()>& cancellation_requested,
                                        std::chrono::milliseconds grace) {
    if (!cancellation_requested) return execute(code, scope_key);
    if (cancellation_requested()) {
        return interrupted_result("Interrupted by the user before the code started.");
    }
    {
        std::lock_guard lock(abandoned_mutex_);
        if (const auto it = abandoned_.find(scope_key); it != abandoned_.end()) {
            if (it->second->holds_scope.load(std::memory_order_acquire)) {
                return {.success = false,
                        .error = "Not run: an earlier interrupted python call in this conversation "
                                 "is still blocked in a call that cannot be cut short (a "
                                 "subprocess wait, a socket or pipe read) and holds its namespace. "
                                 "It is released when that call returns; end whatever it waits "
                                 "for, start a new namespace with reset, or use another tool "
                                 "meanwhile."};
            }
            abandoned_.erase(it);
        }
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
            result = execute_in_scope(code, scope, run.get());
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
            lock.unlock();
            std::lock_guard abandoned_lock(abandoned_mutex_);
            abandoned_.insert_or_assign(std::string(scope_key), run);
            return interrupted_result(
                "Interrupted by the user. The code is blocked in a call that cannot be cut "
                "short, so it keeps running in the background until that call returns; "
                "later python calls in this conversation are refused until then, unless "
                "reset starts a new namespace.");
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

PythonExecResult PythonManager::execute_in_scope(const std::string& code,
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
    std::shared_ptr<Scope> scope;
    std::unique_lock<std::timed_mutex> busy;
    while (!busy) {
        scope = acquire_scope(scope_key);
        std::unique_lock candidate(scope->busy, std::defer_lock);
        {
            // The call running in this scope needs the GIL to finish.
            const py::gil_scoped_release released;
            while (!candidate.try_lock_for(kScopePollInterval)) {
                if (run && run->cancelled.load(std::memory_order_acquire)) {
                    return interrupted_result("Interrupted by the user before the code started.");
                }
            }
        }
        // A namespace cleared while this call waited is not run in: the
        // next lookup creates its successor.
        if (!scope->retired) busy = std::move(candidate);
    }
    if (run) run->holds_scope.store(true, std::memory_order_release);
    // Declared after `busy`, so the scope reads as free before it is.
    struct ScopeHold {
        Run* run;
        ~ScopeHold() {
            if (run) run->holds_scope.store(false, std::memory_order_release);
        }
    } const scope_hold{run};
    if (run && run->cancelled.load(std::memory_order_acquire)) {
        return interrupted_result("Interrupted by the user before the code started.");
    }

    ++scope->running;
    struct RunningGuard {
        PythonManager& manager;
        Scope& scope;
        ~RunningGuard() {
            if (--scope.running == 0 && scope.retired) manager.release(scope);
        }
    } running_guard{*this, *scope};

    PythonExecResult result;

    // Stops capturing on every exit path of the block below.
    OutputCapture capture;
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
    // Stop before reading the buffers, so nothing written afterwards (by a
    // thread the code started) lands in them.
    capture.finish();
    if (run && !result.success && run->cancelled.load(std::memory_order_acquire)) {
        result.interrupted = true;
        result.error = "Interrupted by the user (KeyboardInterrupt).";
    }

    result.output = capture.text();
    return result;
}

} // namespace core::tools
