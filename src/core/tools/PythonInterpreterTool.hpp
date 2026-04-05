#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Exposes the embedded Python interpreter as an LLM tool.
 *
 * The interpreter maintains a persistent namespace across calls: variables,
 * imports, and function definitions survive between invocations (REPL
 * semantics). Stdout and stderr are captured and returned to the model.
 *
 * For package isolation, start filo with FILO_PYTHON_VENV set to a venv
 * directory; pip installs will go there instead of the system Python.
 */
class PythonInterpreterTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
