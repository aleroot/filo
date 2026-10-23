#ifdef FILO_ENABLE_PYTHON

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/PythonInterpreterTool.hpp"
#include "core/tools/PythonManager.hpp"
#include "core/utils/JsonUtils.hpp"
#include "TestSessionContext.hpp"

#include <atomic>
#include <chrono>
#include <format>
#include <string>
#include <string_view>
#include <thread>

using Catch::Matchers::ContainsSubstring;

namespace {

// Session ids are unique per test case: the interpreter is process-wide and
// Catch2 runs the cases in random order.
std::string unique_session(std::string_view label) {
    static std::atomic<int> next{0};
    return std::format("py-ns-{}-{}", label, next.fetch_add(1));
}

core::context::SessionContext python_session(std::string id, std::string root = {}) {
    auto context = test_support::make_workspace_session_context(
        core::context::SessionTransport::cli, std::move(id));
    context.root_session_id = std::move(root);
    return context;
}

std::string run(core::tools::PythonInterpreterTool& tool,
                const core::context::SessionContext& context,
                std::string_view code,
                bool reset = false) {
    return tool.execute(
        std::format(R"({{"code":"{}","reset":{}}})",
                    core::utils::escape_json_string(code), reset ? "true" : "false"),
        context);
}

// Keeps a weak reference to an object that only the scope's namespace holds,
// tied into a reference cycle through a function's globals, so the probe
// reports whether clearing really released it.
constexpr std::string_view kPlantProbe =
    "import sys, weakref\n"
    "class Payload: pass\n"
    "payload = Payload()\n"
    "def keep(): return payload\n"
    "sys.modules['__main__'].__dict__.setdefault('_filo_probes', {})[PROBE] = weakref.ref(payload)\n";

std::string plant_probe_code(std::string_view probe) {
    return std::format("PROBE = '{}'\n{}", probe, kPlantProbe);
}

std::string probe_alive_code(std::string_view probe) {
    return std::format(
        "import sys\n"
        "print('alive' if sys.modules['__main__'].__dict__['_filo_probes']['{}']() is not None"
        " else 'released')\n",
        probe);
}

} // namespace

TEST_CASE("Python namespaces are separate per session", "[python][memory]") {
    core::tools::PythonInterpreterTool tool;
    const auto a = python_session(unique_session("a"));
    const auto b = python_session(unique_session("b"));

    REQUIRE_THAT(run(tool, a, "x = 41"), ContainsSubstring(R"("success":true)"));
    CHECK_THAT(run(tool, a, "print(x + 1)"), ContainsSubstring("42"));
    CHECK_THAT(run(tool, b, "print(x)"), ContainsSubstring("NameError"));
}

TEST_CASE("A delegated worker shares its root session's Python namespace",
          "[python][memory]") {
    core::tools::PythonInterpreterTool tool;
    const auto parent_id = unique_session("parent");
    const auto parent = python_session(parent_id);
    const auto worker = python_session(unique_session("worker"), parent_id);

    REQUIRE_THAT(run(tool, parent, "shared = 'from parent'"),
                 ContainsSubstring(R"("success":true)"));
    CHECK_THAT(run(tool, worker, "print(shared); answer = 7"),
               ContainsSubstring("from parent"));
    CHECK_THAT(run(tool, parent, "print(answer)"), ContainsSubstring("7"));

    // Clearing the worker's own id leaves the conversation's namespace alone.
    tool.clear_session_state(worker.session_id);
    CHECK_THAT(run(tool, parent, "print(shared)"), ContainsSubstring("from parent"));
}

TEST_CASE("Clearing a session releases what its Python namespace kept alive",
          "[python][memory]") {
    core::tools::PythonInterpreterTool tool;
    auto& manager = core::tools::PythonManager::get_instance();
    const auto id = unique_session("clear");
    const auto context = python_session(id);
    const auto observer = python_session(unique_session("observer"));
    const auto probe = id;

    REQUIRE_THAT(run(tool, context, plant_probe_code(probe)),
                 ContainsSubstring(R"("success":true)"));
    REQUIRE_THAT(run(tool, observer, probe_alive_code(probe)), ContainsSubstring("alive"));
    const auto scopes_before = manager.scope_count();

    tool.clear_session_state(id);

    CHECK(manager.scope_count() == scopes_before - 1);
    CHECK_THAT(run(tool, observer, probe_alive_code(probe)), ContainsSubstring("released"));
    CHECK_THAT(run(tool, context, "print(payload)"), ContainsSubstring("NameError"));
}

TEST_CASE("The python tool's reset argument starts from an empty namespace",
          "[python][memory]") {
    core::tools::PythonInterpreterTool tool;
    const auto context = python_session(unique_session("reset"));

    REQUIRE_THAT(run(tool, context, "kept = 1"), ContainsSubstring(R"("success":true)"));

    const auto reset_only = run(tool, context, "", true);
    CHECK_THAT(reset_only, ContainsSubstring(R"("success":true)"));
    CHECK_THAT(run(tool, context, "print('kept' in globals())"), ContainsSubstring("False"));

    REQUIRE_THAT(run(tool, context, "kept = 2"), ContainsSubstring(R"("success":true)"));
    CHECK_THAT(run(tool, context, "print('kept' in globals())", true),
               ContainsSubstring("False"));

    CHECK_THAT(run(tool, context, ""), ContainsSubstring("Missing or empty 'code'"));
}

TEST_CASE("A rotated session id keeps its Python namespace", "[python][memory]") {
    core::tools::PythonInterpreterTool tool;
    const auto old_id = unique_session("old");
    const auto new_id = unique_session("new");

    REQUIRE_THAT(run(tool, python_session(old_id), "carried = 'yes'"),
                 ContainsSubstring(R"("success":true)"));

    tool.move_session_state(old_id, new_id);

    CHECK_THAT(run(tool, python_session(new_id), "print(carried)"), ContainsSubstring("yes"));
    CHECK_THAT(run(tool, python_session(old_id), "print(carried)"), ContainsSubstring("NameError"));
}

TEST_CASE("Clearing a session with a running Python call waits for it to finish",
          "[python][memory]") {
    core::tools::PythonInterpreterTool tool;
    const auto id = unique_session("running");
    const auto context = python_session(id);
    const auto observer = python_session(unique_session("running-observer"));
    const auto probe = id;
    const auto flag = std::format("_filo_go_{}", std::hash<std::string>{}(id));

    REQUIRE_THAT(run(tool, context, plant_probe_code(probe)),
                 ContainsSubstring(R"("success":true)"));

    std::string running_result;
    std::thread running([&] {
        running_result = run(tool, context, std::format(
            "import sys, time\n"
            "sys.modules['__main__'].__dict__['{0}_started'] = True\n"
            "while not sys.modules['__main__'].__dict__.get('{0}', False):\n"
            "    time.sleep(0.005)\n"
            "print('still here', type(payload).__name__)\n",
            flag));
    });

    const auto started = std::format(
        "import sys\nprint(sys.modules['__main__'].__dict__.get('{}_started', False))", flag);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (run(tool, observer, started).find("True") == std::string::npos) {
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    tool.clear_session_state(id);

    // The running call keeps its namespace; a new call already starts fresh.
    CHECK_THAT(run(tool, observer, probe_alive_code(probe)), ContainsSubstring("alive"));
    CHECK_THAT(run(tool, context, "print('payload' in globals())"), ContainsSubstring("False"));

    run(tool, observer, std::format(
        "import sys\nsys.modules['__main__'].__dict__['{}'] = True", flag));
    running.join();

    CHECK_THAT(running_result, ContainsSubstring("still here Payload"));
    CHECK_THAT(run(tool, observer, probe_alive_code(probe)), ContainsSubstring("released"));
}

#endif // FILO_ENABLE_PYTHON
