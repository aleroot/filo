#pragma once

#include "Tool.hpp"
#include <string>
#include <pybind11/pybind11.h>

namespace core::tools {

class PythonTool : public Tool {
public:
    /**
     * @brief Construct a PythonSkill using a python script file path.
     * The python script is expected to define a function that handles the arguments,
     * and a dictionary or function returning the schema.
     */
    PythonTool(const std::string& script_path, const std::string& module_name);
    
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args) override;

private:
    std::string script_path_;
    std::string module_name_;
    ToolDefinition definition_;
    
    void load_definition();
};

} // namespace core::tools
