#include "PythonTool.hpp"
#include "PythonManager.hpp"
#include "../logging/Logger.hpp"
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <stdexcept>
#include <filesystem>

namespace py = pybind11;

namespace core::tools {

PythonTool::PythonTool(const std::string& script_path, const std::string& module_name)
    : script_path_(script_path), module_name_(module_name) {
    
    PythonManager::get_instance().ensure_initialized();
    load_definition();
}

void PythonTool::load_definition() {
    try {
        py::gil_scoped_acquire acquire;

        // Add the script directory to python path
        std::filesystem::path p(script_path_);
        std::string dir = p.parent_path().string();
        
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        path.append(dir);

        // Import the module
        py::module_ mod = py::module_::import(module_name_.c_str());

        // We expect the module to have a function `get_schema()` returning a dict
        py::object get_schema = mod.attr("get_schema");
        py::dict schema = get_schema().cast<py::dict>();

        definition_.name = schema["name"].cast<std::string>();
        definition_.description = schema["description"].cast<std::string>();
        
        if (schema.contains("parameters")) {
            py::list params = schema["parameters"].cast<py::list>();
            for (auto item : params) {
                py::dict pdict = item.cast<py::dict>();
                ToolParameter param;
                param.name = pdict["name"].cast<std::string>();
                param.type = pdict["type"].cast<std::string>();
                param.description = pdict["description"].cast<std::string>();
                if (pdict.contains("required")) {
                    param.required = pdict["required"].cast<bool>();
                }
                definition_.parameters.push_back(param);
            }
        }
    } catch (const py::error_already_set& e) {
        core::logging::error("Failed to load Python skill '{}': {}", module_name_, e.what());
        throw std::runtime_error("Python skill loading failed.");
    }
}

ToolDefinition PythonTool::get_definition() const {
    return definition_;
}

std::string PythonTool::execute(const std::string& json_args) {
    try {
        py::gil_scoped_acquire acquire;

        py::module_ mod = py::module_::import(module_name_.c_str());
        py::object execute_func = mod.attr("execute");
        
        // Pass JSON string to python, let python parse it 
        // (could also parse in C++ and pass py::dict, but passing string is fast and easy)
        py::object result = execute_func(json_args);
        
        return result.cast<std::string>();
    } catch (const py::error_already_set& e) {
        return "{\"error\": \"Python exception: " + std::string(e.what()) + "\"}";
    }
}

} // namespace core::tools
