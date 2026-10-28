#include "PythonInterpreterTool.hpp"
#include "SteeringEnforcement.hpp"
#include "ToolNames.hpp"
#include "PythonManager.hpp"
#include "../context/SessionContext.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <format>

namespace core::tools {

ToolDefinition PythonInterpreterTool::get_definition() const {
    return {
        .name  = std::string(names::kPython),
        .title = "Python",
        .description =
            "Execute Python code in a persistent embedded interpreter. "
            "State persists within this conversation: variables, imports, and "
            "function definitions from earlier calls are available in later ones. "
            "Set reset to true to start from an empty namespace and free what earlier "
            "calls kept in memory. "
            "Stdout and stderr are captured and returned. "
            "Useful for calculations, data processing, file manipulation, and any "
            "task that benefits from a scripting environment. "
            "To keep installed packages separate from the system Python, point the "
            "FILO_PYTHON_VENV environment variable to a venv directory before starting filo.",
        .parameters = {
            {"code", "string", "Python source code to execute. Required unless reset is true.", false},
            {"reset", "boolean",
             "Clear this conversation's namespace before running code, or on its own. Defaults to false.", false}
        },
        .annotations = {
            .destructive_hint = true,  // can modify files, make network calls, etc.
            .open_world_hint  = true,
        },
        .network_mediation = core::tools::NetworkMediation::unmediated,
    };
}

std::string PythonInterpreterTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    return execute(json_args, ToolInvocationContext{.session_context = context});
}

std::string PythonInterpreterTool::execute(const std::string& json_args, const ToolInvocationContext& invocation) {
    const auto& context = invocation.session_context;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments."})";
    }

    bool reset = false;
    if (const auto reset_field = doc["reset"]; reset_field.error() == simdjson::SUCCESS
        && reset_field.get(reset) != simdjson::SUCCESS) {
        return R"({"error":"'reset' must be a boolean."})";
    }

    std::string_view code_view;
    if (doc["code"].get(code_view) != simdjson::SUCCESS) code_view = {};
    if (code_view.empty() && !reset) {
        return R"({"error":"Missing or empty 'code' argument."})";
    }

    // An embedded interpreter is a filesystem reader with no path argument, and
    // it runs inside Filo, which no sandbox confines. So the code is not
    // inspected and guessed about: while steering is withheld the interpreter is
    // simply unavailable, and the sandboxed shell is the way to run code.
    if (const auto steering_error =
            SteeringEnforcement::for_context(context).in_process_code_error()) {
        return *steering_error;
    }

    // Delegated workers carry their parent's root session, so a task sees the
    // variables of the conversation that started it.
    auto& manager = PythonManager::get_instance();
    const std::string_view scope = context.root_session();
    if (reset) {
        manager.clear_scope(scope);
        if (code_view.empty()) return R"({"output":"Python namespace cleared.","success":true})";
    }
    auto result = manager.execute(std::string(code_view), scope, invocation.cancellation_requested);

    std::string escaped_output = core::utils::escape_json_string(result.output);
    if (result.success) {
        return std::format(R"({{"output":"{}","success":true}})", escaped_output);
    }
    std::string escaped_error = core::utils::escape_json_string(result.error);
    return std::format(R"({{"output":"{}","success":false,"error":"{}"}})",
                       escaped_output, escaped_error);
}

void PythonInterpreterTool::clear_session_state(std::string_view session_id) {
    PythonManager::get_instance().clear_scope(session_id);
}

void PythonInterpreterTool::move_session_state(std::string_view from, std::string_view to) {
    PythonManager::get_instance().move_scope(from, to);
}

} // namespace core::tools
