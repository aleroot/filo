#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Exposes the embedded Python interpreter as an LLM tool.
 *
 * Each conversation (the root session of the calling context) has its own
 * persistent namespace: variables, imports, and function definitions survive
 * between its invocations (REPL semantics) and are released when the session
 * state is cleared. Stdout and stderr are captured and returned to the model.
 *
 * For package isolation, start filo with FILO_PYTHON_VENV set to a venv
 * directory; pip installs will go there instead of the system Python.
 */
class PythonInterpreterTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
    void clear_session_state(std::string_view session_id) override;
    void move_session_state(std::string_view from, std::string_view to) override;
};

} // namespace core::tools
