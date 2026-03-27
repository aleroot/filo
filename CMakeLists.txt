cmake_minimum_required(VERSION 3.28...4.3)
project(filo VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

option(FILO_ENABLE_PYTHON "Enable embedded Python support via pybind11" ON)
option(FILO_ENABLE_LLAMACPP "Enable embedded local GGUF inference via llama.cpp" OFF)
option(FILO_ENABLE_IPO "Enable interprocedural optimization for Release-like builds" ON)
option(FILO_ENABLE_NATIVE_TUNING "Enable host-specific CPU tuning (-march=native)" OFF)
option(FILO_ENABLE_TSAN "Enable ThreadSanitizer instrumentation" OFF)
option(FILO_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

if(CMAKE_GENERATOR STREQUAL "Xcode")
    set(CMAKE_XCODE_GENERATE_SCHEME ON CACHE BOOL "Generate Xcode schemes" FORCE)
    set(CMAKE_XCODE_GENERATE_TOP_LEVEL_PROJECT_ONLY ON CACHE BOOL
        "Generate only the top-level Xcode project" FORCE)
endif()

include(cmake/CPM.cmake)
include(CheckIPOSupported)

set(_filo_saved_folder "${CMAKE_FOLDER}")
set(CMAKE_FOLDER "Dependencies")

# Add CLI11 for arg parsing
CPMAddPackage("gh:CLIutils/CLI11@2.6.2")

# Add simdjson for MCP communication (replacing nlohmann/json)
CPMAddPackage("gh:simdjson/simdjson@4.2.4")

# Terminal UI
CPMAddPackage("gh:ArthurSonzogni/FTXUI@6.1.9")
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    foreach(ftxui_target screen dom component)
        if(TARGET ${ftxui_target})
            target_compile_options(${ftxui_target} PRIVATE -Wno-shorten-64-to-32)
        endif()
    endforeach()
endif()

# Add CPR for HTTP requests.
# Use the system-installed libcurl so it has the distro's CA bundle paths
# already baked in — avoids "unable to get local issuer certificate" errors
# that occur when CPR builds its own curl from source without a CA path.
set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
CPMAddPackage(
    NAME cpr
    GITHUB_REPOSITORY libcpr/cpr
    GIT_TAG 1.13.0
    VERSION 1.13.0
)

# Add cpp-httplib for HTTP server support
CPMAddPackage("gh:yhirose/cpp-httplib@0.28.0")

if(FILO_ENABLE_PYTHON)
    CPMAddPackage(
        NAME pybind11
        GITHUB_REPOSITORY pybind/pybind11
        GIT_TAG v3.0.2
        VERSION 3.0.2
    )
endif()

if(FILO_ENABLE_LLAMACPP)
    CPMAddPackage(
        NAME llama_cpp
        GITHUB_REPOSITORY ggml-org/llama.cpp
        GIT_TAG bd6992180bd1
        OPTIONS
            "BUILD_SHARED_LIBS OFF"
            "LLAMA_BUILD_COMMON ON"
            "LLAMA_BUILD_TESTS OFF"
            "LLAMA_BUILD_TOOLS OFF"
            "LLAMA_BUILD_EXAMPLES OFF"
            "LLAMA_BUILD_SERVER OFF"
            "LLAMA_OPENSSL OFF"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        if(TARGET llama)
            # Upstream bd6992180bd1 misses <algorithm> in llama-quant.cpp on this toolchain.
            target_compile_options(llama PRIVATE -include algorithm)
        endif()
        if(TARGET common)
            # Upstream common sources also rely on std algorithms without including <algorithm>.
            target_compile_options(common PRIVATE -include algorithm)
        endif()
    endif()
endif()

set(CMAKE_FOLDER "${_filo_saved_folder}")
unset(_filo_saved_folder)

set(FILO_CORE_SOURCES
    src/exec/Server.cpp
    src/exec/Daemon.cpp
    src/exec/Prompter.cpp
    src/core/mcp/McpDispatcher.cpp
    src/core/mcp/McpClientSession.cpp
    src/core/mcp/McpConnectionManager.cpp
    src/tui/Autocomplete.cpp
    src/tui/Conversation.cpp
    src/tui/DiffPreview.cpp
    src/tui/MarkdownRenderer.cpp
    src/tui/MainApp.cpp
    src/tui/PromptInput.cpp
    src/tui/PromptComponents.cpp
    src/core/config/ConfigManager.cpp
    src/core/llm/providers/RouterProvider.cpp
    src/core/llm/HttpLLMProvider.cpp
    src/core/llm/ModelRegistry.cpp
    src/core/llm/protocols/OpenAIProtocol.cpp
    src/core/llm/protocols/OpenAIResponsesProtocol.cpp
    src/core/llm/protocols/KimiProtocol.cpp
    src/core/llm/protocols/GrokProtocol.cpp
    src/core/llm/protocols/DashScopeProtocol.cpp
    src/core/llm/protocols/AnthropicProtocol.cpp
    src/core/llm/protocols/GeminiProtocol.cpp
    src/core/llm/protocols/OllamaProtocol.cpp
    src/core/llm/ProviderFactory.cpp
    src/core/llm/routing/PolicyConfig.cpp
    src/core/llm/routing/PolicyLoader.cpp
    src/core/llm/routing/RouterEngine.cpp
    src/core/llm/routing/AutoClassifier.cpp
    src/core/auth/FileTokenStore.cpp
    src/core/auth/GoogleOAuthFlow.cpp
    src/core/auth/ClaudeOAuthFlow.cpp
    src/core/auth/OpenAIAuthFlow.cpp
    src/core/auth/OpenAIOAuthFlow.cpp
    src/core/auth/KimiOAuthFlow.cpp
    src/core/auth/QwenOAuthFlow.cpp
    src/core/auth/OAuthTokenManager.cpp
    src/core/auth/OAuthCredentialSource.cpp
    src/core/auth/AuthenticationManager.cpp
    src/core/auth/ui/ConsoleAuthUI.cpp
    src/core/agent/Agent.cpp
    src/core/agent/SubagentOrchestrator.cpp
    src/core/scm/GitProvider.cpp
    src/core/scm/ScmFactory.cpp
    src/core/utils/FileSystemUtils.cpp
    src/core/context/ContextMentions.cpp
    src/core/tools/ShellTool.cpp
    src/core/tools/ApplyPatchTool.cpp
    src/core/tools/FileSearchTool.cpp
    src/core/tools/ReadFileTool.cpp
    src/core/tools/WriteFileTool.cpp
    src/core/tools/ListDirectoryTool.cpp
    src/core/tools/ReplaceTool.cpp
    src/core/tools/GrepSearchTool.cpp
    src/core/tools/DeleteFileTool.cpp
    src/core/tools/MoveFileTool.cpp
    src/core/tools/CreateDirectoryTool.cpp
    src/core/tools/SearchReplaceTool.cpp
    src/core/tools/AskUserQuestionTool.cpp
    src/core/utils/JsonUtils.cpp
    src/core/logging/Logger.cpp
    src/core/commands/CommandExecutor.cpp
    src/core/commands/SkillCommandLoader.cpp
    src/core/tools/SkillLoader.cpp
    src/core/session/SessionStore.cpp
    src/core/session/SessionReport.cpp
    src/core/history/PromptHistoryStore.cpp
    src/core/permissions/PermissionSystem.cpp
)

file(GLOB_RECURSE FILO_HEADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)
list(SORT FILO_HEADER_FILES)

if(FILO_ENABLE_PYTHON)
    list(APPEND FILO_CORE_SOURCES
        src/core/tools/PythonManager.cpp
        src/core/tools/PythonTool.cpp
        src/core/tools/PythonInterpreterTool.cpp
    )
endif()

if(FILO_ENABLE_LLAMACPP)
    list(APPEND FILO_CORE_SOURCES
        src/core/llm/providers/LlamaCppProvider.cpp
    )
endif()

add_library(filo_build_options INTERFACE)
target_compile_features(filo_build_options INTERFACE cxx_std_26)
set_target_properties(filo_build_options PROPERTIES FOLDER "Filo/Internal")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(filo_build_options INTERFACE
        -Wall
        -Wextra
        -Wpedantic
    )

    if(FILO_WARNINGS_AS_ERRORS)
        target_compile_options(filo_build_options INTERFACE -Werror)
    endif()

    if(FILO_ENABLE_NATIVE_TUNING)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            message(WARNING "FILO_ENABLE_NATIVE_TUNING is not supported with AppleClang; leaving host-specific tuning disabled.")
        else()
            target_compile_options(filo_build_options INTERFACE -march=native)
        endif()
    endif()

    if(FILO_ENABLE_TSAN)
        target_compile_options(filo_build_options INTERFACE
            -fsanitize=thread
            -fno-omit-frame-pointer
        )
        target_link_options(filo_build_options INTERFACE -fsanitize=thread)
    endif()
endif()

set(FILO_IPO_SUPPORTED OFF)
if(FILO_ENABLE_IPO)
    check_ipo_supported(
        RESULT filo_ipo_supported
        OUTPUT filo_ipo_output
        LANGUAGES CXX
    )
    if(filo_ipo_supported)
        set(FILO_IPO_SUPPORTED ON)
    else()
        message(STATUS "IPO/LTO disabled: ${filo_ipo_output}")
    endif()
endif()

add_library(filo_core STATIC ${FILO_CORE_SOURCES})
target_sources(filo_core PRIVATE ${FILO_HEADER_FILES})
target_include_directories(filo_core PUBLIC src)
target_link_libraries(filo_core
    PUBLIC
        filo_build_options
        simdjson::simdjson
        ftxui::screen
        ftxui::dom
        ftxui::component
        cpr::cpr
        httplib::httplib
)

if(APPLE)
    find_library(CFNETWORK_FRAMEWORK CFNetwork REQUIRED)
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
    target_link_libraries(filo_core
        PUBLIC
            ${CFNETWORK_FRAMEWORK}
            ${COREFOUNDATION_FRAMEWORK}
    )
endif()

set_target_properties(filo_core PROPERTIES
    FOLDER "Filo"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${FILO_IPO_SUPPORTED}
)

if(FILO_ENABLE_PYTHON)
    target_link_libraries(filo_core PUBLIC pybind11::embed)
    target_compile_definitions(filo_core PUBLIC FILO_ENABLE_PYTHON)
endif()

if(FILO_ENABLE_LLAMACPP)
    target_link_libraries(filo_core PUBLIC llama PRIVATE common)
    target_compile_definitions(filo_core PUBLIC FILO_ENABLE_LLAMACPP)
endif()

add_executable(filo src/main.cpp)
target_link_libraries(filo PRIVATE filo_core CLI11::CLI11)
set_target_properties(filo PROPERTIES
    FOLDER "Filo"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${FILO_IPO_SUPPORTED}
)

enable_testing()
set(_filo_saved_folder "${CMAKE_FOLDER}")
set(CMAKE_FOLDER "Dependencies")
CPMAddPackage("gh:catchorg/Catch2@3.11.0")
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    foreach(catch2_target Catch2 Catch2WithMain)
        if(TARGET ${catch2_target})
            target_compile_options(${catch2_target} PRIVATE -Wno-shorten-64-to-32)
        endif()
    endforeach()
endif()
set(CMAKE_FOLDER "${_filo_saved_folder}")
unset(_filo_saved_folder)

add_executable(filo_tests
    tests/test_tools.cpp
    tests/test_commands.cpp
    tests/test_history.cpp
    tests/test_config_manager.cpp
    tests/test_context_mentions.cpp
    tests/test_openai_provider.cpp
    tests/test_openai_responses_protocol.cpp
    tests/test_model_registry.cpp
    tests/test_grok_provider.cpp
    tests/test_grok_enhanced.cpp
    tests/test_gemini_provider.cpp
    tests/test_claude_provider.cpp
    tests/test_mistral_provider.cpp
    tests/test_kimi_provider.cpp
    tests/test_qwen_provider.cpp
    tests/test_qwen_oauth.cpp
    tests/test_ollama_provider.cpp
    tests/test_llamacpp_provider.cpp
    tests/test_mcp_server.cpp
    tests/test_oauth.cpp
    tests/test_tier1.cpp
    tests/test_routing.cpp
    tests/test_router_provider.cpp
    tests/test_auto_classifier.cpp
    tests/test_prompt_input.cpp
    tests/test_http_response_lifecycle.cpp
    tests/test_session.cpp
    tests/test_sessions_lifecycle.cpp
    tests/test_budget_tracker.cpp
    tests/test_agent_permissions.cpp
    tests/test_safety_policy.cpp
    tests/test_subagent_orchestrator.cpp
    tests/test_skill_loader.cpp
    tests/test_skill_commands.cpp
    tests/test_conversation.cpp
    tests/test_markdown_renderer.cpp
    tests/test_prompt_components.cpp
    tests/test_permission_system.cpp
    tests/test_permission_gate_deadlock.cpp
    tests/test_concurrent_queue.cpp
    tests/test_main_exit.cpp
    tests/test_prompter_mode.cpp
)
target_link_libraries(filo_tests PRIVATE filo_core Catch2::Catch2WithMain)
set_target_properties(filo_tests PROPERTIES
    FOLDER "Tests"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${FILO_IPO_SUPPORTED}
)

add_executable(test_agent_freeze tests/test_agent_freeze.cpp)
target_link_libraries(test_agent_freeze PRIVATE filo_core)
set_target_properties(test_agent_freeze PROPERTIES
    FOLDER "Tests"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${FILO_IPO_SUPPORTED}
)

add_executable(test_tui_freeze tests/test_tui_freeze.cpp)
target_link_libraries(test_tui_freeze PRIVATE filo_core)
set_target_properties(test_tui_freeze PROPERTIES
    FOLDER "Tests"
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${FILO_IPO_SUPPORTED}
    INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${FILO_IPO_SUPPORTED}
)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(filo_tests DISCOVERY_MODE PRE_TEST)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PREFIX "Source Files"
    FILES
        src/main.cpp
        ${FILO_CORE_SOURCES}
        ${FILO_HEADER_FILES}
)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    PREFIX "Test Files"
    FILES
        tests/test_tools.cpp
        tests/test_commands.cpp
        tests/test_history.cpp
        tests/test_config_manager.cpp
        tests/test_context_mentions.cpp
        tests/test_openai_provider.cpp
        tests/test_openai_responses_protocol.cpp
        tests/test_grok_provider.cpp
    tests/test_grok_enhanced.cpp
        tests/test_gemini_provider.cpp
        tests/test_claude_provider.cpp
        tests/test_mistral_provider.cpp
        tests/test_kimi_provider.cpp
        tests/test_qwen_provider.cpp
        tests/test_qwen_oauth.cpp
        tests/test_ollama_provider.cpp
        tests/test_llamacpp_provider.cpp
        tests/test_mcp_server.cpp
        tests/test_oauth.cpp
        tests/test_tier1.cpp
        tests/test_routing.cpp
        tests/test_router_provider.cpp
        tests/test_auto_classifier.cpp
        tests/test_prompt_input.cpp
        tests/test_session.cpp
        tests/test_sessions_lifecycle.cpp
        tests/test_budget_tracker.cpp
        tests/test_agent_permissions.cpp
        tests/test_safety_policy.cpp
        tests/test_subagent_orchestrator.cpp
        tests/test_skill_loader.cpp
        tests/test_conversation.cpp
        tests/test_markdown_renderer.cpp
        tests/test_prompt_components.cpp
        tests/test_agent_freeze.cpp
        tests/test_tui_freeze.cpp
        tests/test_permission_system.cpp
        tests/test_main_exit.cpp
        tests/test_prompter_mode.cpp
)
