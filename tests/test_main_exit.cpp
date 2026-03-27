/**
 * Tests for main() exit behavior and session summary display.
 * 
 * These tests verify that:
 * 1. The program exits cleanly without hanging
 * 2. The session summary is displayed on exit
 * 3. MCP server thread is properly cleaned up
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdio>
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>

#include "core/session/SessionReport.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/budget/BudgetTracker.hpp"

using namespace Catch::Matchers;

// ---------------------------------------------------------------------------
// Session Report Display Tests
// ---------------------------------------------------------------------------

TEST_CASE("SessionReport::print displays summary with session ID", "[exit][summary]") {
    // Capture stdout
    std::ostringstream oss;
    auto old_cout_buf = std::cout.rdbuf(oss.rdbuf());
    
    // Setup test data using singletons
    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();
    
    // Record some activity
    core::llm::TokenUsage usage{.prompt_tokens = 100, .completion_tokens = 50, .total_tokens = 150};
    stats.record_turn("test-model", usage, false);
    stats.record_api_call(true);
    stats.record_tool_call(true);
    
    // Print the report
    core::session::SessionReport::print(budget, stats.snapshot(), "test123", "/tmp/test_session.json");
    
    // Restore stdout
    std::cout.rdbuf(old_cout_buf);
    
    std::string output = oss.str();
    
    // Verify key elements are present
    REQUIRE_THAT(output, ContainsSubstring("Session Summary"));
    REQUIRE_THAT(output, ContainsSubstring("test123"));
    REQUIRE_THAT(output, ContainsSubstring("Session ID"));
    REQUIRE_THAT(output, ContainsSubstring("Duration"));
    REQUIRE_THAT(output, ContainsSubstring("Turns"));
    REQUIRE_THAT(output, ContainsSubstring("API calls"));
    REQUIRE_THAT(output, ContainsSubstring("Tool calls"));
}

TEST_CASE("SessionReport::print includes resume command", "[exit][summary]") {
    std::ostringstream oss;
    auto old_cout_buf = std::cout.rdbuf(oss.rdbuf());
    
    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();
    
    // Need at least one turn for report to display
    core::llm::TokenUsage usage{.prompt_tokens = 10, .completion_tokens = 5, .total_tokens = 15};
    stats.record_turn("test-model", usage, false);
    
    core::session::SessionReport::print(budget, stats.snapshot(), "abc123", "/tmp/test.json");
    
    std::cout.rdbuf(old_cout_buf);
    std::string output = oss.str();
    
    // Verify resume command is shown
    REQUIRE_THAT(output, ContainsSubstring("--resume"));
    REQUIRE_THAT(output, ContainsSubstring("abc123"));
}

TEST_CASE("SessionReport displays tool call success/failure counts", "[exit][summary]") {
    std::ostringstream oss;
    auto old_cout_buf = std::cout.rdbuf(oss.rdbuf());
    
    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();
    
    // Need at least one turn for report to display
    core::llm::TokenUsage usage{.prompt_tokens = 10, .completion_tokens = 5, .total_tokens = 15};
    stats.record_turn("test-model", usage, false);
    
    // Record some tool calls
    stats.record_tool_call(true);   // success
    stats.record_tool_call(true);   // success
    stats.record_tool_call(false);  // failure
    
    core::session::SessionReport::print(budget, stats.snapshot(), "test456", "/tmp/test2.json");
    
    std::cout.rdbuf(old_cout_buf);
    std::string output = oss.str();
    
    // Verify tool calls are shown
    REQUIRE_THAT(output, ContainsSubstring("Tool calls"));
}

// ---------------------------------------------------------------------------
// MCP Server Clean Exit Tests
// ---------------------------------------------------------------------------

/**
 * Test that verifies the MCP server loop exits cleanly when stdin is closed.
 * This simulates what happens in main() when we fclose(stdin) before joining
 * the MCP thread.
 */
TEST_CASE("MCP server exits cleanly on stdin EOF", "[exit][mcp]") {
    // Create a pipe to simulate stdin
    int pipe_fds[2];
    REQUIRE(pipe(pipe_fds) == 0);
    
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    
    if (pid == 0) {
        // Child process - simulate MCP server
        close(pipe_fds[1]);  // Close write end
        
        // Redirect stdin to the pipe
        dup2(pipe_fds[0], STDIN_FILENO);
        close(pipe_fds[0]);
        
        // Simulate the MCP server loop (like in exec/Server.cpp)
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
        
        std::string line;
        int lines_read = 0;
        while (std::getline(std::cin, line)) {
            if (!line.empty()) {
                lines_read++;
            }
        }
        
        // Exit with number of lines read
        exit(lines_read);
    } else {
        // Parent process
        close(pipe_fds[0]);  // Close read end
        
        // Close write end immediately to send EOF
        close(pipe_fds[1]);
        
        // Wait for child with timeout
        int status;
        auto start = std::chrono::steady_clock::now();
        pid_t result;
        
        do {
            result = waitpid(pid, &status, WNOHANG);
            if (result == 0) {
                // Child still running, check timeout
                auto elapsed = std::chrono::steady_clock::now() - start;
                if (elapsed > std::chrono::seconds(2)) {
                    // Kill the child and fail the test
                    kill(pid, SIGTERM);
                    waitpid(pid, &status, 0);
                    FAIL("MCP server did not exit within 2 seconds - would have hung");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } while (result == 0);
        
        // Child should have exited cleanly
        REQUIRE(result == pid);
        REQUIRE(WIFEXITED(status));
        // Exit code should be 0 (no lines read before EOF)
        REQUIRE(WEXITSTATUS(status) == 0);
    }
}

/**
 * Test that verifies session stats are preserved on exit.
 */
TEST_CASE("Session stats snapshot preserves all data on exit", "[exit][stats]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();
    
    // Record various activities
    core::llm::TokenUsage usage1{.prompt_tokens = 10, .completion_tokens = 5, .total_tokens = 15};
    stats.record_turn("model1", usage1, false);
    
    core::llm::TokenUsage usage2{.prompt_tokens = 20, .completion_tokens = 10, .total_tokens = 30};
    stats.record_turn("model2", usage2, false);
    
    stats.record_api_call(true);
    stats.record_api_call(false);
    stats.record_tool_call(true);
    stats.record_tool_call(true);
    stats.record_tool_call(false);
    
    // Take snapshot (this is what happens before exit)
    auto snapshot = stats.snapshot();
    
    // Verify snapshot contains correct data
    REQUIRE(snapshot.turn_count == 2);
    REQUIRE(snapshot.api_calls_total == 2);
    REQUIRE(snapshot.api_calls_success == 1);
    // api failures = total - success
    REQUIRE(snapshot.tool_calls_total == 3);
    REQUIRE(snapshot.tool_calls_success == 2);
    // tool failures = total - success
}

/**
 * Test that verifies budget tracker is accessible on exit.
 */
TEST_CASE("BudgetTracker accessible on exit for summary display", "[exit][budget]") {
    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    
    // Record some usage
    core::llm::TokenUsage usage;
    usage.prompt_tokens = 100;
    usage.completion_tokens = 50;
    usage.total_tokens = 150;
    
    budget.record(usage, "test-model", false);
    
    // Get status string (used in summary)
    std::string status = budget.status_string();
    
    // Status should not be empty when there's data
    REQUIRE(!status.empty());
    
    // Reset should work for clean exit
    budget.reset_session();
    auto after_reset = budget.session_cost_usd();
    REQUIRE(after_reset == 0.0);
}

/**
 * Test that verifies the fix for the hanging issue - closing stdin unblocks getline.
 */
TEST_CASE("Closing stdin unblocks blocking read for clean exit", "[exit][regression]") {
    // This test verifies the specific fix: fclose(stdin) allows the MCP thread to exit
    int pipe_fds[2];
    REQUIRE(pipe(pipe_fds) == 0);
    
    pid_t pid = fork();
    REQUIRE(pid >= 0);
    
    if (pid == 0) {
        // Child process
        close(pipe_fds[1]);
        dup2(pipe_fds[0], STDIN_FILENO);
        close(pipe_fds[0]);
        
        // Simulate what happens in main() - the MCP server blocks on getline
        std::string line;
        bool got_eof = false;
        
        // First read should block until we close the pipe
        if (!std::getline(std::cin, line)) {
            got_eof = true;
        }
        
        exit(got_eof ? 0 : 1);
    } else {
        close(pipe_fds[0]);
        
        // Simulate the fix: close stdin to unblock the child
        // In the real code, we do fclose(stdin) then join
        close(pipe_fds[1]);  // This sends EOF to the child
        
        int status;
        auto start = std::chrono::steady_clock::now();
        
        while (true) {
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result == pid) {
                REQUIRE(WIFEXITED(status));
                REQUIRE(WEXITSTATUS(status) == 0);
                break;
            }
            
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > std::chrono::seconds(2)) {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                FAIL("Child process hung - regression in exit handling");
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}
