#include "PythonManager.hpp"
#include "../logging/Logger.hpp"
#include <pybind11/eval.h>
#include <cstdlib>
#include <Python.h>

namespace py = pybind11;

namespace core::tools {

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
            namespace_.reset();
            interpreter_.reset();
            return;
        }

        // Fallback path: interpreter exists but no saved main thread state.
        py::gil_scoped_acquire acquire;
        namespace_.reset();
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

    // Bootstrap the persistent namespace with builtins
    namespace_.emplace();
    (*namespace_)["__builtins__"] = py::module_::import("builtins");

    // Venv isolation
    const char* venv_env = std::getenv("FILO_PYTHON_VENV");
    if (venv_env && *venv_env) {
        try {
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
del _venv, _ver, _sp, _os
)", py::globals(), *namespace_);
        } catch (const py::error_already_set& e) {
            core::logging::warn("PythonManager: venv setup failed: {}", e.what());
        }
    }

    // Release the GIL to enable multi-threading and keep the returned main
    // thread state so we can restore it during shutdown before finalization.
    main_thread_state_ = PyEval_SaveThread();
}

PythonExecResult PythonManager::execute(const std::string& code) {
    ensure_initialized();

    // Acquire the GIL before any Python C API calls.
    // This is safe to call from any thread after PyEval_SaveThread() was called
    // during initialization. The GIL is automatically released when this guard
    // goes out of scope (RAII).
    py::gil_scoped_acquire acquire;

    PythonExecResult result;

    py::module_ sys    = py::module_::import("sys");
    py::module_ io     = py::module_::import("io");
    py::object buf_out = io.attr("StringIO")();
    py::object buf_err = io.attr("StringIO")();

    py::object old_stdout = sys.attr("stdout");
    py::object old_stderr = sys.attr("stderr");
    sys.attr("stdout") = buf_out;
    sys.attr("stderr") = buf_err;

    try {
        // Use namespace_ for both globals and locals so that function
        // definitions and assignments persist across calls (REPL semantics).
        py::exec(py::str(code), *namespace_, *namespace_);
        result.success = true;
    } catch (const py::error_already_set& e) {
        result.success = false;
        result.error   = e.what();
    }

    sys.attr("stdout") = old_stdout;
    sys.attr("stderr") = old_stderr;

    result.output = py::str(buf_out.attr("getvalue")()).cast<std::string>();
    std::string err_str = py::str(buf_err.attr("getvalue")()).cast<std::string>();
    if (!err_str.empty()) {
        if (!result.output.empty()) result.output += '\n';
        result.output += err_str;
    }

    return result;
}

void PythonManager::reset_namespace() {
    if (!interpreter_ || !namespace_) return;

    // Acquire the GIL before modifying Python objects.
    py::gil_scoped_acquire acquire;

    namespace_->clear();
    (*namespace_)["__builtins__"] = py::module_::import("builtins");
}

} // namespace core::tools
