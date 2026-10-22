#include "PythonManager.hpp"
#include "../logging/Logger.hpp"
#include <pybind11/eval.h>
#include <cstdlib>
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

PythonExecResult PythonManager::execute(const std::string& code, std::string_view scope_key) {
    // Held across the stream swap, the code, and the restore: the three only
    // form one atomic step for other Python entry points when serialised.
    std::lock_guard<std::mutex> exec_lock(execution_mutex_);
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
    try {
        // Use the scope's dict for both globals and locals so that function
        // definitions and assignments persist across calls (REPL semantics).
        py::exec(py::str(code), scope->globals, scope->globals);
        result.success = true;
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

    result.output = py::str(buf_out.attr("getvalue")()).cast<std::string>();
    std::string err_str = py::str(buf_err.attr("getvalue")()).cast<std::string>();
    if (!err_str.empty()) {
        if (!result.output.empty()) result.output += '\n';
        result.output += err_str;
    }

    return result;
}

} // namespace core::tools
