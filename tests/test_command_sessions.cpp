#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/tools/shell/ShellSessions.hpp"

#include <chrono>
#include <filesystem>
#include <future>
#include <string>
#include <thread>

using Catch::Matchers::ContainsSubstring;
using core::tools::shell::ShellSessions;
using namespace std::chrono_literals;

namespace {

constexpr std::string_view kOwner = "conversation-a";

[[nodiscard]] ShellSessions::CommandStep start(ShellSessions& sessions,
                                               std::string command,
                                               std::chrono::milliseconds yield) {
    return sessions.start_command(
        {.session_id = std::string(kOwner), .command = std::move(command)},
        std::filesystem::temp_directory_path(), {}, yield);
}

} // namespace

TEST_CASE("A command that ends within its yield returns its output and exit code",
          "[shell][command_session]") {
    ShellSessions sessions;
    const auto step = start(sessions, "printf 'hi'; pwd; exit 3", 10s);
    REQUIRE(step.error.empty());
    CHECK(step.exit_code == 3);
    CHECK_FALSE(step.process_id.has_value());
    CHECK_THAT(step.output, ContainsSubstring("hi"));
    // Without a working directory it runs in the workspace it was given.
    CHECK_THAT(step.output, ContainsSubstring(
        std::filesystem::canonical(std::filesystem::temp_directory_path()).filename().string()));
    CHECK(step.wall_time < 5s);
}

TEST_CASE("A command still running keeps a session a later call reads and writes",
          "[shell][command_session]") {
    ShellSessions sessions;
    const auto first = start(sessions, "echo ready; read line; echo got:$line", 300ms);
    REQUIRE(first.error.empty());
    REQUIRE(first.process_id.has_value());
    CHECK_FALSE(first.exit_code.has_value());
    CHECK_THAT(first.output, ContainsSubstring("ready"));

    const auto second = sessions.continue_command(kOwner, *first.process_id, "hello\n", 5s);
    REQUIRE(second.error.empty());
    CHECK(second.exit_code == 0);
    CHECK_FALSE(second.process_id.has_value());
    CHECK_THAT(second.output, ContainsSubstring("got:hello"));
    CHECK_FALSE(second.output.contains("ready"));

    // The ended session is gone.
    CHECK_THAT(sessions.continue_command(kOwner, *first.process_id, "", 10ms).error,
               ContainsSubstring("Unknown process id"));
}

TEST_CASE("A session belongs to the conversation that started it",
          "[shell][command_session]") {
    ShellSessions sessions;
    const auto step = start(sessions, "sleep 30", 50ms);
    REQUIRE(step.process_id.has_value());
    CHECK_THAT(sessions.continue_command("conversation-b", *step.process_id, "", 10ms).error,
               ContainsSubstring("Unknown process id"));

    // Moving the conversation moves its sessions.
    sessions.move(kOwner, "conversation-b");
    CHECK(sessions.continue_command("conversation-b", *step.process_id, "", 10ms).error.empty());

    // Clearing it kills them.
    sessions.clear("conversation-b");
    CHECK_THAT(sessions.continue_command("conversation-b", *step.process_id, "", 10ms).error,
               ContainsSubstring("Unknown process id"));
}

TEST_CASE("Interrupting a conversation stops the session a call is reading",
          "[shell][command_session]") {
    ShellSessions sessions;
    const auto step = start(sessions, "sleep 30", 50ms);
    REQUIRE(step.process_id.has_value());
    auto reading = std::async(std::launch::async, [&] {
        return sessions.continue_command(kOwner, *step.process_id, "", 20s);
    });
    // Wait until the poll is the running command.
    for (int i = 0; i < 200 && sessions.active_commands().empty(); ++i) {
        std::this_thread::sleep_for(5ms);
    }
    REQUIRE(sessions.active_commands().size() == 1);
    CHECK(sessions.active_commands()[0].session_id == kOwner);
    CHECK(sessions.interrupt(kOwner));
    REQUIRE(reading.wait_for(5s) == std::future_status::ready);
    const auto stopped = reading.get();
    CHECK(stopped.error.empty());
    CHECK(sessions.active_commands().empty());
}

TEST_CASE("Quiescing a conversation stops its background sessions",
          "[shell][command_session]") {
    ShellSessions sessions;
    const auto step = start(sessions, "sleep 30", 50ms);
    REQUIRE(step.process_id.has_value());
    CHECK(sessions.quiesce_session(kOwner));
    CHECK_THAT(sessions.continue_command(kOwner, *step.process_id, "", 10ms).error,
               ContainsSubstring("Unknown process id"));
}
