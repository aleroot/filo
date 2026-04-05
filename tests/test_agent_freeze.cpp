#include "core/agent/Agent.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/GetTimeTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/ApplyPatchTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/ReadFileTool.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/ReplaceTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "TestSessionContext.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "core/tools/PythonTool.hpp"
#endif
#include <cstdio>
#include <print>
#include <thread>
#include <chrono>

int main() {
    core::config::ProviderConfig gemini_cfg;
    gemini_cfg.model = "gemini-2.5-flash";
    auto provider = core::llm::ProviderFactory::create_provider("gemini", gemini_cfg);
    core::tools::ToolManager& sm = core::tools::ToolManager::get_instance();
    
    sm.register_tool(std::make_shared<core::tools::GetTimeTool>());
    sm.register_tool(std::make_shared<core::tools::ShellTool>());
    sm.register_tool(std::make_shared<core::tools::ApplyPatchTool>());
    sm.register_tool(std::make_shared<core::tools::FileSearchTool>());
    sm.register_tool(std::make_shared<core::tools::ReadFileTool>());
    sm.register_tool(std::make_shared<core::tools::WriteFileTool>());
    sm.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    sm.register_tool(std::make_shared<core::tools::ReplaceTool>());
    sm.register_tool(std::make_shared<core::tools::GrepSearchTool>());
#ifdef FILO_ENABLE_PYTHON
    sm.register_tool(std::make_shared<core::tools::PythonTool>("src/core/tools/sample_skill.py", "sample_skill"));
    #endif

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        sm,
        test_support::make_workspace_session_context());

    std::println(stdout, "Sending message...");
    bool done = false;

    agent->send_message("Hello, world!",
        [](const std::string& chunk) { std::println(stdout, "Text: {}", chunk); },
        [](const std::string& name, const std::string& args) { std::println(stdout, "Tool: {} args: {}", name, args); },
        [&done]() {
            std::println(stdout, "Done callback.");
            done = true;
        }
    );
    std::println(stdout, "send_message returned. Waiting for async completion...");

    while (!done) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::println(stdout, "All done.");
    return 0;
}
