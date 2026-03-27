#include "PythonInterpreterTool.hpp"
#include "PythonManager.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <format>

namespace core::tools {

ToolDefinition PythonInterpreterTool::get_definition() const {
    return {
        .name  = "python",
        .title = "Python Interpreter",
        .description =
            "Execute Python code in a persistent embedded interpreter. "
            "The interpreter maintains state across calls: variables, imports, and "
            "function definitions from earlier calls are available in later ones. "
            "Stdout and stderr are captured and returned. "
            "Useful for calculations, data processing, file manipulation, and any "
            "task that benefits from a scripting environment. "
            "To keep installed packages separate from the system Python, point the "
            "FILO_PYTHON_VENV environment variable to a venv directory before starting filo.",
        .parameters = {
            {"code", "string", "Python source code to execute.", true}
        },
        .annotations = {
            .destructive_hint = true,  // can modify files, make network calls, etc.
            .open_world_hint  = true,
        },
    };
}

std::string PythonInterpreterTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments."})";
    }

    std::string_view code_view;
    if (doc["code"].get(code_view) != simdjson::SUCCESS || code_view.empty()) {
        return R"({"error":"Missing or empty 'code' argument."})";
    }

    auto result = PythonManager::get_instance().execute(std::string(code_view));

    std::string escaped_output = core::utils::escape_json_string(result.output);
    if (result.success) {
        return std::format(R"({{"output":"{}","success":true}})", escaped_output);
    }
    std::string escaped_error = core::utils::escape_json_string(result.error);
    return std::format(R"({{"output":"{}","success":false,"error":"{}"}})",
                       escaped_output, escaped_error);
}

} // namespace core::tools
